// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <ngraph/runtime/reference/convert_color_nv12.hpp>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template <typename T>
void wrap_color_convert_i420(const T* arg_y,
                             const ngraph::Shape& arg_y_shape,
                             const T* arg_u,
                             const T* arg_v,
                             T* out_ptr,
                             ov::op::util::ConvertColorI420Base::ColorConversion color_format,
                             const bool single_plane) {
    enum Arg_y {N_DIM, H_DIM, W_DIM, C_DIM};
    size_t batch_size = arg_y_shape[Arg_y::N_DIM];
    size_t image_w = arg_y_shape[Arg_y::W_DIM];
    size_t image_h = arg_y_shape[Arg_y::H_DIM];
    size_t stride_y =  image_w * image_h;
    size_t stride_uv =  image_w * image_h / 4;
    if (single_plane) {
        image_h = image_h * 2 / 3;
        stride_uv = stride_uv * 4;
    }
    ngraph::runtime::reference::color_convert_i420<T>(arg_y,
                                                      single_plane ? arg_u + image_w * image_h : arg_u,
                                                      single_plane ? arg_v + 5 * image_w * image_h / 4 : arg_v,
                                                      out_ptr,
                                                      batch_size,
                                                      image_h,
                                                      image_w,
                                                      stride_y,
                                                      stride_uv,
                                                      color_format);
}

template<> Converter::Conversion::Ptr Converter::Convert(const ngraph::op::v8::I420toBGR& node) {
    const bool single_plane = node.get_input_size() == 1;
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.get_input_shape(0),
                                    single_plane ? node.input(0) : node.input(1),
                                    single_plane ? node.input(0) : node.input(2),
                                    node.output(0),
                                    ov::op::util::ConvertColorI420Base::ColorConversion::I420_TO_BGR,
                                    single_plane);
        };
    return CallSwitch(
        AP_WRAP(make, wrap_color_convert_i420),
        node.get_input_element_type(0),
        std::tuple<std::uint8_t, ngraph::float16, float>{});
}

template<> Converter::Conversion::Ptr Converter::Convert(const ngraph::op::v8::I420toRGB& node) {
    const bool single_plane = node.get_input_size() == 1;
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.get_input_shape(0),
                                    single_plane ? node.input(0) : node.input(1),
                                    single_plane ? node.input(0) : node.input(2),
                                    node.output(0),
                                    ov::op::util::ConvertColorI420Base::ColorConversion::I420_TO_RGB,
                                    single_plane);
        };
    return CallSwitch(
        AP_WRAP(make, wrap_color_convert_i420),
        node.get_input_element_type(0),
        std::tuple<std::uint8_t, ngraph::float16, float>{});
}

} // namespace ArmPlugin