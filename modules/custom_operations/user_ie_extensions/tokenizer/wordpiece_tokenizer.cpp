// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//#include "fast_tokenizer/normalizers/normalizers.h"
//
//#include "fast_tokenizer/pretokenizers/pretokenizers.h"

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

//    std::cerr << "Slow\n";

    using namespace paddlenlp::fast_tokenizer;

    auto vocab_begins_const = as_type_ptr<Constant>(arguments[5].get_node_shared_ptr());
    auto vocab_begins = static_cast<const int32_t*>(vocab_begins_const->get_data_ptr());
    auto vocab_size = vocab_begins_const->get_shape()[0];

    auto vocab_ends_const = as_type_ptr<Constant>(arguments[6].get_node_shared_ptr());
    auto vocab_ends = static_cast<const int32_t*>(vocab_ends_const->get_data_ptr());

    auto vocab_chars_const = as_type_ptr<Constant>(arguments[7].get_node_shared_ptr());
    auto vocab_chars = static_cast<const char*>(vocab_chars_const->get_data_ptr());

    auto unk_token_id_const = as_type_ptr<Constant>(arguments[8].get_node_shared_ptr());
    auto unk_token_id  = *static_cast<const int32_t*>(vocab_begins_const->get_data_ptr());

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
    constructor_validate_and_infer_types();
}

WordpieceTokenizer::WordpieceTokenizer(
    const ov::OutputVector& arguments,
    const std::shared_ptr<models::FastWordPiece> tokenizer,
    const std::string& suffix_indicator,
    int max_bytes_per_word
) :
    ov::op::Op(arguments),
    m_tokenizer(tokenizer),
    m_suffix_indicator(suffix_indicator),
    m_max_bytes_per_word(max_bytes_per_word) {

//    std::cerr << "Fast\n";
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

    //const size_t num_parts = inputs[2].get_size();
    //size_t new_num_parts = num_parts;

    // FIXME: Not accurate estimation as there is theoretical possibility for re-use the same symbol area
    // to represent different elements in ragged tensor
    outputs[2].set_shape({inputs[4].get_size()});

    // Get pointers in the output tensors
    auto new_begins = outputs[0].data<int32_t>();
    auto new_ends   = outputs[1].data<int32_t>();
    auto new_elems  = outputs[2].data<int32_t>();
    int32_t ragged_offset = 0;

    using namespace paddlenlp::fast_tokenizer;

    for(size_t seq = 0; seq < num_rows; ++seq) {
        new_begins[seq] = ragged_offset;

        for(size_t ragged_col = ragged_begins[seq]; ragged_col < ragged_ends[seq]; ++ragged_col) {

            auto str = std::string(chars + begins[ragged_col], chars + ends[ragged_col]);
            std::vector<core::Token> results = m_tokenizer->Tokenize(str);

//            std::cerr << "[ WordpieceTokenizer ] String bytes: ";
            for (auto i = begins[ragged_col]; i < ends[ragged_col]; ++i) {
                std::cerr << static_cast<int> (chars[i]) << " ";
            }
//            std::cerr << "\n";
//            std::cerr << "[ WordpieceTokenizer ] String: '" << str << "'\n";
//            std::cerr << "[ WordpieceTokenizer ] String len: " << ends[ragged_col] - begins[ragged_col]  << "\n";
            for (const core::Token& token : results) {
//                std::cout << "[ WordpieceTokenizer ]     id: " << token.id_ << ", value: " << token.value_
//                          << ", offset: (" << token.offset_.first << ", "
//                          << token.offset_.second << ")." << std::endl;
                OPENVINO_ASSERT(ragged_offset < outputs[2].get_size());
                new_elems[ragged_offset++] = token.id_;
            };
        }
        new_ends[seq] = ragged_offset;
    }
    outputs[2].set_shape({size_t(ragged_offset)});
    return true;
}

