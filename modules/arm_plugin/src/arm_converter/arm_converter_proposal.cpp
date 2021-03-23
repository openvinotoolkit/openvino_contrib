// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/proposal.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::Proposal& node) {
    if (node.get_input_element_type(0) != ngraph::element::f32) {
        IE_THROW() << "Unsupported Type: " << node.get_input_element_type(0);
    }

    auto func = ngraph::runtime::reference::proposal_v4<float>;
    return MakeConversion(func,
                          node.input(0),
                          node.input(1),
                          node.input(2),
                          node.output(0),
                          node.output(1),
                          node.get_input_shape(0),
                          node.get_input_shape(1),
                          node.get_input_shape(2),
                          node.get_output_shape(0),
                          node.get_output_shape(1),
                          node.get_attrs());
}

template<> Converter::Conversion::Ptr Converter::Convert(const ngraph::op::v0::Proposal& node) {
    if (node.get_input_element_type(0) != ngraph::element::f32) {
        IE_THROW() << "Unsupported Type: " << node.get_input_element_type(0);
    }

    auto func = ngraph::runtime::reference::proposal_v0<float>;
    return MakeConversion(func,
                          node.input(0),
                          node.input(1),
                          node.input(2),
                          node.output(0),
                          node.get_input_shape(0),
                          node.get_input_shape(1),
                          node.get_input_shape(2),
                          node.get_output_shape(0),
                          node.get_attrs());
}

}  //  namespace ArmPlugin
