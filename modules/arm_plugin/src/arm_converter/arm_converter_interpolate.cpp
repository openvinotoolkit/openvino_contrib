// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include <arm_compute/runtime/NEON/functions/NEScale.h>
#include <ngraph/runtime/reference/interpolate.hpp>
#include "arm_converter/arm_converter.hpp"


using Nearest_mode    = ngraph::op::v4::Interpolate::NearestMode;
using Transform_mode  = ngraph::op::v4::Interpolate::CoordinateTransformMode;
using InterpolateMode = ngraph::op::v4::Interpolate::InterpolateMode;


bool isSupportedConfiguration(const ngraph::op::v4::Interpolate& node) {
    auto& inp_shape = node.get_input_shape(0);
    auto& out_shape = node.get_output_shape(0);

    float scale_h = out_shape[2] / inp_shape[2];
    float scale_w = out_shape[3] / inp_shape[3];
    bool is_upsample = scale_h > 1 && scale_w > 1;

    auto& attrs = node.get_attrs();
    auto& coord_mode = attrs.coordinate_transformation_mode;
    auto& nearest_mode = attrs.nearest_mode;

    if (coord_mode == Transform_mode::asymmetric && nearest_mode == Nearest_mode::floor) {
        return true;
    }

    if (coord_mode == Transform_mode::align_corners && nearest_mode == Nearest_mode::round_prefer_ceil) {
        return true;
    }

    if (is_upsample) {
        if (coord_mode == Transform_mode::asymmetric && nearest_mode == Nearest_mode::simple) {
            return true;
        }

        bool int_factor = scale_h == static_cast<int>(scale_h) && scale_w == static_cast<int>(scale_w);
        if (int_factor && coord_mode != Transform_mode::asymmetric &&
            (nearest_mode == Nearest_mode::round_prefer_ceil || nearest_mode == Nearest_mode::round_prefer_floor)) {
            return true;
        }
    } else {
        float down_scale_h = inp_shape[2] / out_shape[2];
        float down_scale_w = inp_shape[3] / out_shape[3];
        bool int_factor = down_scale_h == static_cast<int>(down_scale_h) && down_scale_w == static_cast<int>(down_scale_w);

        if (int_factor && coord_mode != Transform_mode::align_corners && nearest_mode == Nearest_mode::simple) {
            return true;
        }

        if (int_factor && nearest_mode == Nearest_mode::round_prefer_ceil &&
            ((out_shape[2] > 1 && out_shape[3] > 1) || coord_mode != Transform_mode::half_pixel)) {
            return true;
        }
    }

    return false;
}


namespace ArmPlugin {
template <typename T, typename U>
void wrap_interpolate(const T* input_data,
                      const ngraph::Shape& input_shape,
                      const T* scales,
                      const ngraph::Shape& scales_shape,
                      const U* axes,
                      const ngraph::Shape& axes_shape,
                      T* out,
                      const ngraph::Shape& out_shape,
                      const ngraph::op::v4::Interpolate::InterpolateAttrs& attrs) {
    auto scales_size = ngraph::shape_size(scales_shape);
    std::vector<float> scales_vec;
    if (attrs.shape_calculation_mode == ngraph::op::v4::Interpolate::ShapeCalcMode::sizes) {
        for (size_t i = 0; i < scales_size; i++) {
            auto axis = axes[i];
            scales_vec.push_back(static_cast<float>(out_shape[axis]) / input_shape[axis]);
        }
    } else {
        scales_vec = std::vector<float>(scales, scales + scales_size);
    }

    auto axes_size = ngraph::shape_size(axes_shape);
    std::vector<int64_t> axes_vec(axes, axes + axes_size);
    ngraph::runtime::reference::interpolate<T>(input_data,
                                               input_shape,
                                               scales_vec,
                                               axes_vec,
                                               out,
                                               out_shape,
                                               attrs);
}
template<> Converter::Conversion::Ptr Converter::Convert(const opset::Interpolate& node) {
    auto& attrs = node.get_attrs();
    auto& inp_shape = node.get_input_shape(0);
    auto& out_shape = node.get_output_shape(0);
    bool isSupported = true;

    if (inp_shape.size() != 4 || inp_shape[0] != out_shape[0] || inp_shape[1] != out_shape[1]) {
        isSupported = false;
    }

    auto& pads_begin = attrs.pads_begin;
    auto& pads_end   = attrs.pads_end;
    if (!std::all_of(pads_begin.begin(), pads_begin.end(), [](int i){return i == 0;}) ||
        !std::all_of(pads_end.begin(), pads_end.end(), [](int i){return i == 0;})) {
        isSupported = false;
    }

    auto& nearest_mode = attrs.nearest_mode;
    auto& coord_mode   = attrs.coordinate_transformation_mode;
    if (attrs.antialias || coord_mode == Transform_mode::tf_half_pixel_for_nn || nearest_mode == Nearest_mode::ceil) {
        isSupported = false;
    }

    arm_compute::InterpolationPolicy policy;
    switch (attrs.mode) {
        case opset::Interpolate::InterpolateMode::linear:
        case opset::Interpolate::InterpolateMode::linear_onnx:
            policy = arm_compute::InterpolationPolicy::BILINEAR;
            break;
        case opset::Interpolate::InterpolateMode::nearest:
            policy = arm_compute::InterpolationPolicy::NEAREST_NEIGHBOR;
            break;
        default:
            isSupported = false;
    }

    if (policy == arm_compute::InterpolationPolicy::NEAREST_NEIGHBOR && isSupported && !isSupportedConfiguration(node)) {
        isSupported = false;
    }

    arm_compute::SamplingPolicy coord = arm_compute::SamplingPolicy::TOP_LEFT;
    if ((coord_mode == Transform_mode::pytorch_half_pixel && out_shape.size() == 4 && out_shape[2] > 1 && out_shape[3] > 1) ||
        coord_mode == Transform_mode::half_pixel) {
        coord = arm_compute::SamplingPolicy::CENTER;
    }

    if (isSupported) {
        return MakeConversion<arm_compute::NEScale>(node.input(0),
                                                    node.output(0),
                                                    arm_compute::ScaleKernelInfo(policy,
                                                                                arm_compute::BorderMode::REPLICATE,
                                                                                arm_compute::PixelValue(),
                                                                                coord,
                                                                                false,
                                                                                coord_mode == Transform_mode::align_corners));
    } else {
        auto make = [&] (auto refFunction) {
            return MakeConversion(refFunction,
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

        switch (node.get_input_element_type(0)) {
            case ngraph::element::Type_t::f16 :
                if (node.get_input_element_type(3) == ngraph::element::i32) {
                    return make(wrap_interpolate<half_float::half, std::int32_t>);
                }
                return make(wrap_interpolate<half_float::half, std::int64_t>);
            case ngraph::element::Type_t::f32 :
                if (node.get_input_element_type(3) == ngraph::element::i32) {
                    return make(wrap_interpolate<float, std::int32_t>);
                }
                return make(wrap_interpolate<float, std::int64_t>);
            default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
        }
    }
}
} //  namespace ArmPlugin
