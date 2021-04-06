// Copyright (C) 2020-2021 Intel Corporation
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

    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::f16 :
            switch (node.get_input_element_type(1)) {
                case ngraph::element::Type_t::i32 :
                    switch (node.get_classes_index_type()) {
                        case ngraph::element::Type_t::i32 :
                            switch (node.get_sequence_length_type()) {
                                case ngraph::element::Type_t::i32 :
                                    return make(ngraph::runtime::reference::ctc_greedy_decoder_seq_len<ngraph::float16,
                                                                                                       std::int32_t,
                                                                                                       std::int32_t,
                                                                                                       std::int32_t>);
                                case ngraph::element::Type_t::i64 :
                                    return make(ngraph::runtime::reference::ctc_greedy_decoder_seq_len<ngraph::float16,
                                                                                                       std::int32_t,
                                                                                                       std::int32_t,
                                                                                                       std::int64_t>);
                                default: IE_THROW() << "Unsupported Seq Len Output Type: " << node.get_sequence_length_type(); return {};
                            }
                        case ngraph::element::Type_t::i64 :
                            switch (node.get_sequence_length_type()) {
                                case ngraph::element::Type_t::i32 :
                                    return make(ngraph::runtime::reference::ctc_greedy_decoder_seq_len<ngraph::float16,
                                                                                                       std::int32_t,
                                                                                                       std::int64_t,
                                                                                                       std::int32_t>);
                                case ngraph::element::Type_t::i64 :
                                    return make(ngraph::runtime::reference::ctc_greedy_decoder_seq_len<ngraph::float16,
                                                                                                       std::int32_t,
                                                                                                       std::int64_t,
                                                                                                       std::int64_t>);
                                default: IE_THROW() << "Unsupported Seq Len Output Type: " << node.get_sequence_length_type(); return {};
                            }
                        default: IE_THROW() << "Unsupported Class Index Type: " << node.get_classes_index_type(); return {};
                    }
                case ngraph::element::Type_t::i64 :
                    switch (node.get_classes_index_type()) {
                        case ngraph::element::Type_t::i32 :
                            switch (node.get_sequence_length_type()) {
                                case ngraph::element::Type_t::i32 :
                                    return make(ngraph::runtime::reference::ctc_greedy_decoder_seq_len<ngraph::float16,
                                                                                                       std::int64_t,
                                                                                                       std::int32_t,
                                                                                                       std::int32_t>);
                                case ngraph::element::Type_t::i64 :
                                    return make(ngraph::runtime::reference::ctc_greedy_decoder_seq_len<ngraph::float16,
                                                                                                       std::int64_t,
                                                                                                       std::int32_t,
                                                                                                       std::int64_t>);
                                default: IE_THROW() << "Unsupported Seq Len Output Type: " << node.get_sequence_length_type(); return {};
                            }
                        case ngraph::element::Type_t::i64 :
                            switch (node.get_sequence_length_type()) {
                                case ngraph::element::Type_t::i32 :
                                    return make(ngraph::runtime::reference::ctc_greedy_decoder_seq_len<ngraph::float16,
                                                                                                       std::int64_t,
                                                                                                       std::int64_t,
                                                                                                       std::int32_t>);
                                case ngraph::element::Type_t::i64 :
                                    return make(ngraph::runtime::reference::ctc_greedy_decoder_seq_len<ngraph::float16,
                                                                                                       std::int64_t,
                                                                                                       std::int64_t,
                                                                                                       std::int64_t>);
                                default: IE_THROW() << "Unsupported Seq Len Output Type: " << node.get_sequence_length_type(); return {};
                            }
                        default: IE_THROW() << "Unsupported Class Index Type: " << node.get_classes_index_type(); return {};
                    }
                default: IE_THROW() << "Unsupported Seq Len Type: " << node.get_input_element_type(1); return {};
            }
        case ngraph::element::Type_t::f32 :
            switch (node.get_input_element_type(1)) {
                case ngraph::element::Type_t::i32 :
                    switch (node.get_classes_index_type()) {
                        case ngraph::element::Type_t::i32 :
                            switch (node.get_sequence_length_type()) {
                                case ngraph::element::Type_t::i32 :
                                    return make(ngraph::runtime::reference::ctc_greedy_decoder_seq_len<float,
                                                                                                       std::int32_t,
                                                                                                       std::int32_t,
                                                                                                       std::int32_t>);
                                case ngraph::element::Type_t::i64 :
                                    return make(ngraph::runtime::reference::ctc_greedy_decoder_seq_len<float,
                                                                                                       std::int32_t,
                                                                                                       std::int32_t,
                                                                                                       std::int64_t>);
                                default: IE_THROW() << "Unsupported Seq Len Output Type: " << node.get_sequence_length_type(); return {};
                            }
                        case ngraph::element::Type_t::i64 :
                            switch (node.get_sequence_length_type()) {
                                case ngraph::element::Type_t::i32 :
                                    return make(ngraph::runtime::reference::ctc_greedy_decoder_seq_len<float,
                                                                                                       std::int32_t,
                                                                                                       std::int64_t,
                                                                                                       std::int32_t>);
                                case ngraph::element::Type_t::i64 :
                                    return make(ngraph::runtime::reference::ctc_greedy_decoder_seq_len<float,
                                                                                                       std::int32_t,
                                                                                                       std::int64_t,
                                                                                                       std::int64_t>);
                                default: IE_THROW() << "Unsupported Seq Len Output Type: " << node.get_sequence_length_type(); return {};
                            }
                        default: IE_THROW() << "Unsupported Class Index Type: " << node.get_classes_index_type(); return {};
                    }
                case ngraph::element::Type_t::i64 :
                    switch (node.get_classes_index_type()) {
                        case ngraph::element::Type_t::i32 :
                            switch (node.get_sequence_length_type()) {
                                case ngraph::element::Type_t::i32 :
                                    return make(ngraph::runtime::reference::ctc_greedy_decoder_seq_len<float,
                                                                                                       std::int64_t,
                                                                                                       std::int32_t,
                                                                                                       std::int32_t>);
                                case ngraph::element::Type_t::i64 :
                                    return make(ngraph::runtime::reference::ctc_greedy_decoder_seq_len<float,
                                                                                                       std::int64_t,
                                                                                                       std::int32_t,
                                                                                                       std::int64_t>);
                                default: IE_THROW() << "Unsupported Seq Len Output Type: " << node.get_sequence_length_type(); return {};
                            }
                        case ngraph::element::Type_t::i64 :
                            switch (node.get_sequence_length_type()) {
                                case ngraph::element::Type_t::i32 :
                                    return make(ngraph::runtime::reference::ctc_greedy_decoder_seq_len<float,
                                                                                                       std::int64_t,
                                                                                                       std::int64_t,
                                                                                                       std::int32_t>);
                                case ngraph::element::Type_t::i64 :
                                    return make(ngraph::runtime::reference::ctc_greedy_decoder_seq_len<float,
                                                                                                       std::int64_t,
                                                                                                       std::int64_t,
                                                                                                       std::int64_t>);
                                default: IE_THROW() << "Unsupported Seq Len Output Type: " << node.get_sequence_length_type(); return {};
                            }
                        default: IE_THROW() << "Unsupported Class Index Type: " << node.get_classes_index_type(); return {};
                    }
                default: IE_THROW() << "Unsupported Seq Len Type: " << node.get_input_element_type(1); return {};
            }
        default: IE_THROW() << "Unsupported Data Type: " << node.get_input_element_type(0); return {};
    }
}

}  //  namespace ArmPlugin
