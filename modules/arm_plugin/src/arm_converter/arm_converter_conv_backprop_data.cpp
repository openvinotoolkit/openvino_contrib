// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/convolution.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::ConvolutionBackpropData& node) {
    auto make = [&] (auto refFunction) {
        auto out_shape = node.get_shape();
        ngraph::Strides in_dilation(std::vector<size_t>(node.get_input_shape(0).size() - 2));
        std::fill(in_dilation.begin(), in_dilation.end(), 1);
        return MakeConversion(refFunction,
                              node.input(0),
                              node.input(1),
                              node.output(0),
                              node.get_input_shape(0),
                              node.get_input_shape(1),
                              out_shape,
                              in_dilation,
                              node.get_dilations(),
                              node.get_pads_begin(),
                              node.get_pads_end(),
                              node.get_strides());
    };

    if (node.input(0).get_element_type() != ngraph::element::f32) {
        THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type();
    }
    return make(ngraph::runtime::reference::convolution_backprop_in<float, float, float>);
}
}  //  namespace ArmPlugin
