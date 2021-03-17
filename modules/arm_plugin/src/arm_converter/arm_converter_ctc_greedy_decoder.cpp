// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/ctc_greedy_decoder.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::CTCGreedyDecoder& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.input(1),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_input_shape(1),
                                    node.get_output_shape(0),
                                    node.get_ctc_merge_repeated());
    };
    switch (node.input(0).get_element_type()) {
        case ngraph::element::Type_t::f16 : return make(ngraph::runtime::reference::ctc_greedy_decoder<half_float::half>);
        case ngraph::element::Type_t::f32 : return make(ngraph::runtime::reference::ctc_greedy_decoder<float>);
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
    }
}

}  //  namespace ArmPlugin
