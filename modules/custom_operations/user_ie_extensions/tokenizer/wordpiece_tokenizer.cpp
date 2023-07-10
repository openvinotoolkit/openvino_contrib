// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fast_tokenizer/normalizers/normalizers.h"
#include "fast_tokenizer/models/models.h"
#include "fast_tokenizer/pretokenizers/pretokenizers.h"

#include "wordpiece_tokenizer.hpp"
#include "utils.hpp"

using namespace ov;


void WordpieceTokenizer::validate_and_infer_types() {
    check_ragged_string_input(this, 0);
    check_string_input(this, 5);
    set_ragged_output(this, 0, get_input_partial_shape(0), element::i32);
}

#undef tokenizer

bool WordpieceTokenizer::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto ragged_begins = inputs[0].data<const int32_t>();
    auto ragged_ends   = inputs[1].data<const int32_t>();
    auto begins = inputs[2].data<const int32_t>();
    auto ends   = inputs[3].data<const int32_t>();
    auto chars  = inputs[4].data<const uint8_t>();

    auto vocab_begins = inputs[5].data<const int32_t>();
    auto vocab_ends   = inputs[6].data<const int32_t>();
    auto vocab_chars  = inputs[7].data<const uint8_t>();

    auto vocab_size = inputs[5].get_size();

    OPENVINO_ASSERT(inputs.size() == 9, "Too few inputs passed to WordpieceTokenizer, it means it is not converted properly or it is not used in the supported pattern");

    auto unk_token_id  = *inputs[8].data<const int32_t>();

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

//    std::cerr << "[ WordpieceTokenizer ] Start vocab reading\n";
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

//    std::cerr << "[ WordpieceTokenizer ] Finish vocab reading\n";
//    std::cerr << "[ WordpieceTokenizer ] unk_token = " << unk_token << "\n";
//    std::cerr << "[ WordpieceTokenizer ] Start tokenizer initialization\n";

    auto tokenizer = models::FastWordPiece(vocab, unk_token, m_max_bytes_per_word, m_suffix_indicator, true);   // FIXME: why true?

//    std::cerr << "[ WordpieceTokenizer ] Finish tokenizer initialization\n";


    for(size_t seq = 0; seq < num_rows; ++seq) {
        new_begins[seq] = ragged_offset;

        for(size_t ragged_col = ragged_begins[seq]; ragged_col < ragged_ends[seq]; ++ragged_col) {

            auto str = std::string(chars + begins[ragged_col], chars + ends[ragged_col]);
            std::vector<core::Token> results = tokenizer.Tokenize(str);

//            std::cerr << "[ WordpieceTokenizer ] String bytes: ";
//            for (auto i = begins[ragged_col]; i < ends[ragged_col]; ++i) {
//                std::cerr << static_cast<int> (chars[i]) << " ";
//            }
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

