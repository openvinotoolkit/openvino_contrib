// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/group_convolution.hpp>
#include <ngraph/runtime/reference/group_convolution_backprop_data.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::GroupConvolutionBackpropData& node) {
    auto make = [&] (auto refFunction) {
        auto out_shape = node.get_shape();
        ngraph::Strides in_dilation(std::vector<size_t>(node.get_input_shape(0).size() - 2));
        std::fill(in_dilation.begin(), in_dilation.end(), 1);
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.input(1),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_input_shape(1),
                                    out_shape,
                                    node.get_strides(),
                                    node.get_dilations(),
                                    node.get_pads_begin(),
                                    node.get_pads_end());
    };

    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::f16 :
            return make(ngraph::runtime::reference::group_convolution_backprop_data<ngraph::float16, ngraph::float16, ngraph::float16>);
        case ngraph::element::Type_t::f32 :
            return make(ngraph::runtime::reference::group_convolution_backprop_data<float, float, float>);
        default: IE_THROW() << "Unsupported Type: " << node.get_input_element_type(0); return {};
    }
}
}  //  namespace ArmPlugin
