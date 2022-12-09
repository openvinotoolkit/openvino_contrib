// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <arm_compute/runtime/NEON/functions/NEScale.h>
#include <ngraph/runtime/reference/interpolate.hpp>
#include "arm_converter/arm_converter.hpp"


using Transform_mode  = ngraph::op::v4::Interpolate::CoordinateTransformMode;

namespace ArmPlugin {
static void pad_input_data(const uint8_t* data_ptr,
                           uint8_t* padded_data_ptr,
                           size_t type_size,
                           const ngraph::Shape& input_shape,
                           const ngraph::Shape& padded_input_shape,
                           const std::vector<size_t>& pads_begin) {
    ngraph::CoordinateTransform input_transform(input_shape);
    ngraph::CoordinateTransform padded_transform(padded_input_shape);

    for (const ngraph::Coordinate& input_coord : input_transform) {
        auto padded_coord = input_coord;
        size_t i = 0;
        for (size_t pad : pads_begin) {
            padded_coord[i] += pad;
            ++i;
        }
        uint8_t* dst_ptr = padded_data_ptr + type_size * padded_transform.index(padded_coord);
        const uint8_t* src_ptr = data_ptr + type_size * input_transform.index(input_coord);
        std::memcpy(dst_ptr, src_ptr, type_size);
    }
}

template <typename T, typename V, typename U>
void wrap_interpolate(const T* input_data,
                      const ngraph::Shape& input_shape,
                      const V* scales,
                      const ngraph::Shape& scales_shape,
                      const U* axes,
                      const ngraph::Shape& axes_shape,
                      T* out,
                      const ngraph::Shape& out_shape,
                      const ngraph::op::v4::Interpolate::InterpolateAttrs& attrs) {
    auto& pads_begin = attrs.pads_begin;
    auto& pads_end   = attrs.pads_end;
    ngraph::Shape padded_shape = input_shape;
    for (size_t i = 0; i < pads_begin.size(); i++) {
        padded_shape[i] += pads_begin[i] + pads_end[i];
    }

    auto type_size = sizeof(T);
    std::vector<uint8_t> padded_input_data(ngraph::shape_size(padded_shape) * type_size, 0);
    const uint8_t* data_ptr  = reinterpret_cast<const uint8_t*>(input_data);
    uint8_t* padded_data_ptr = padded_input_data.data();

    pad_input_data(data_ptr, padded_data_ptr, type_size, input_shape, padded_shape, pads_begin);

    auto scales_size = ngraph::shape_size(scales_shape);
    std::vector<float> scales_vec;
    if (attrs.shape_calculation_mode == ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES) {
        for (size_t i = 0; i < scales_size; i++) {
            auto axis = axes[i];
            scales_vec.push_back(static_cast<float>(out_shape[axis]) / padded_shape[axis]);
        }
    } else {
        scales_vec = std::vector<float>(scales, scales + scales_size);
    }

    auto axes_size = ngraph::shape_size(axes_shape);
    std::vector<int64_t> axes_vec(axes, axes + axes_size);
    ngraph::runtime::reference::interpolate<T>(reinterpret_cast<T*>(padded_data_ptr),
                                                padded_shape,
                                                scales_vec,
                                                axes_vec,
                                                out,
                                                out_shape,
                                                attrs);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Interpolate& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.get_input_shape(0),
                                    node.input(2),
                                    node.get_input_shape(2),
                                    node.input(3),
                                    node.get_input_shape(3),
                                    node.output(0),
                                    node.get_output_shape(0),
                                    node.get_attrs());
    };
    return CallSwitch(
        AP_WRAP(make, wrap_interpolate),
        node.get_input_element_type(0), allTypes,
        node.get_input_element_type(2), floatTypes,
        node.get_input_element_type(1), indexTypes);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ArmInterpolate& node) {
    auto& attrs = node.get_attrs();
    auto& coord_mode = attrs.coordinate_transformation_mode;

    arm_compute::SamplingPolicy coord = arm_compute::SamplingPolicy::TOP_LEFT;
    auto& out_shape = node.get_output_shape(0);
    if ((coord_mode == Transform_mode::PYTORCH_HALF_PIXEL && out_shape[2] > 1 && out_shape[3] > 1) ||
        coord_mode == Transform_mode::HALF_PIXEL) {
        coord = arm_compute::SamplingPolicy::CENTER;
    }

    arm_compute::InterpolationPolicy policy;
    switch (attrs.mode) {
        case opset::Interpolate::InterpolateMode::LINEAR:
        case opset::Interpolate::InterpolateMode::LINEAR_ONNX:
            policy = arm_compute::InterpolationPolicy::BILINEAR;
            break;
        case opset::Interpolate::InterpolateMode::NEAREST:
            policy = arm_compute::InterpolationPolicy::NEAREST_NEIGHBOR;
            break;
        default:
            IE_THROW() << "Unsupported interpolate mode";
    }

    return MakeConversion<arm_compute::NEScale>(node.input(0),
                                                node.output(0),
                                                arm_compute::ScaleKernelInfo(policy,
                                                                             arm_compute::BorderMode::REPLICATE,
                                                                             arm_compute::PixelValue(),
                                                                             coord,
                                                                             false,
                                                                             coord_mode == Transform_mode::ALIGN_CORNERS));
}
} //  namespace ArmPlugin
