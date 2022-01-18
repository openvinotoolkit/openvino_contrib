// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/ctc_greedy_decoder_seq_len.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::CTCGreedyDecoderSeqLen& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.input(1),
                                    node.input(2),
                                    node.output(0),
                                    node.output(1),
                                    node.get_input_shape(0),
                                    node.get_output_shape(0),
                                    node.get_merge_repeated());
    };

    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::ctc_greedy_decoder_seq_len),
        node.get_input_element_type(0),  floatTypes,
        node.get_input_element_type(1),  indexTypes,
        node.get_classes_index_type(),   indexTypes,
        node.get_sequence_length_type(), indexTypes);
}

}  //  namespace ArmPlugin
