// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/proposal.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::Proposal& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
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
    };
    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::f16 : return make(ngraph::runtime::reference::proposal_v4<ngraph::float16>);
        case ngraph::element::Type_t::f32 : return make(ngraph::runtime::reference::proposal_v4<float>);
        default: IE_THROW() << "Unsupported Type: " << node.get_input_element_type(0); return {};
    }
}

template<> Converter::Conversion::Ptr Converter::Convert(const ngraph::op::v0::Proposal& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.input(1),
                                    node.input(2),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_input_shape(1),
                                    node.get_input_shape(2),
                                    node.get_output_shape(0),
                                    node.get_attrs());
    };
    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::f16 : return make(ngraph::runtime::reference::proposal_v0<ngraph::float16>);
        case ngraph::element::Type_t::f32 : return make(ngraph::runtime::reference::proposal_v0<float>);
        default: IE_THROW() << "Unsupported Type: " << node.get_input_element_type(0); return {};
    }
}

}  //  namespace ArmPlugin
