// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "wordpiece_tokenizer.hpp"
#include "utils.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace ov;
using namespace ov::opset10;


WordpieceTokenizer::WordpieceTokenizer(
    const ov::OutputVector& arguments,
    const std::string& suffix_indicator,
    int max_bytes_per_word
) :
    ov::op::Op(arguments),
    m_suffix_indicator(suffix_indicator),
    m_max_bytes_per_word(max_bytes_per_word) {

    constructor_validate_and_infer_types();
}

WordpieceTokenizer::WordpieceTokenizer(
    const ov::OutputVector& arguments,
    const std::shared_ptr<models::FastWordPiece>& tokenizer,
    const std::string& suffix_indicator,
    int max_bytes_per_word
) :
    ov::op::Op(arguments),
    m_tokenizer(tokenizer),
    m_suffix_indicator(suffix_indicator),
    m_max_bytes_per_word(max_bytes_per_word) {

    if (m_tokenizer == nullptr) {
        // vocab constant folding doesn't work, get packed constant
        auto packed_vocab_const = as_type_ptr<Constant>(arguments[5].get_node_shared_ptr()->get_input_node_shared_ptr(0));
        auto packed_vocab_buf = static_cast<const char*>(packed_vocab_const->get_data_ptr());
        auto vocab_size = *reinterpret_cast<const int32_t*>(packed_vocab_buf + 0);
        auto vocab_begins = reinterpret_cast<const int32_t*>(packed_vocab_buf + 4);
        auto vocab_ends = reinterpret_cast<const int32_t*>(packed_vocab_buf + 4 + 4);
        auto vocab_chars = packed_vocab_buf + 4 + 4 + 4 * vocab_size;

        auto unk_token_id_const = as_type_ptr<Constant>(arguments[8].get_node_shared_ptr());
        auto unk_token_id  = *static_cast<const int32_t*>(unk_token_id_const->get_data_ptr());

        core::Vocab vocab;
        std::string unk_token;
        if(unk_token_id < 0)
            unk_token_id += vocab_size;
        for(size_t id = 0; id < vocab_size; ++id) {
            auto token = std::string(vocab_chars + vocab_begins[id], vocab_chars + vocab_ends[id]);
            vocab[token] = int32_t(id); // TODO: Check range
            if(id == unk_token_id)
                unk_token = token;
        }
        m_tokenizer = std::make_shared<models::FastWordPiece>(vocab, unk_token, m_max_bytes_per_word, m_suffix_indicator, true);
    }
    constructor_validate_and_infer_types();
}


void WordpieceTokenizer::validate_and_infer_types() {
    check_ragged_string_input(this, 0);
    check_string_input(this, 5);
    set_ragged_output(this, 0, get_input_partial_shape(0), element::i32);
}


bool WordpieceTokenizer::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto ragged_begins = inputs[0].data<const int32_t>();
    auto ragged_ends   = inputs[1].data<const int32_t>();
    auto begins = inputs[2].data<const int32_t>();
    auto ends   = inputs[3].data<const int32_t>();
    auto chars  = inputs[4].data<const uint8_t>();

    // Set output shapes
    outputs[0].set_shape(inputs[0].get_shape());
    outputs[1].set_shape(inputs[1].get_shape());
    const size_t num_rows = inputs[0].get_size();


    // FIXME: Not accurate estimation as there is theoretical possibility for re-use the same symbol area
    // to represent different elements in ragged tensor
    outputs[2].set_shape({inputs[4].get_size()});

    // Get pointers in the output tensors
    auto new_begins = outputs[0].data<int32_t>();
    auto new_ends   = outputs[1].data<int32_t>();
    auto new_elems  = outputs[2].data<int32_t>();
    int32_t ragged_offset = 0;

    for(size_t seq = 0; seq < num_rows; ++seq) {
        new_begins[seq] = ragged_offset;

        for(size_t ragged_col = ragged_begins[seq]; ragged_col < ragged_ends[seq]; ++ragged_col) {

            auto str = std::string(chars + begins[ragged_col], chars + ends[ragged_col]);
            std::vector<core::Token> results = m_tokenizer->Tokenize(str);

            for (const core::Token& token : results) {
                OPENVINO_ASSERT(ragged_offset < outputs[2].get_size());
                new_elems[ragged_offset++] = token.id_;
            };
        }
        new_ends[seq] = ragged_offset;
    }
    outputs[2].set_shape({size_t(ragged_offset)});
    return true;
}

