// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fast_tokenizer/normalizers/normalizers.h"
#include "fast_tokenizer/models/models.h"
#include "fast_tokenizer/pretokenizers/pretokenizers.h"

#include "bpe_tokenizer.hpp"
#include "utils.hpp"

using namespace ov;

#undef tokenizer

void BPETokenizer::validate_and_infer_types() {
    check_ragged_string_input(this, 0);
    check_string_input(this, 5);
    check_string_input(this, 8);
    set_ragged_output(this, 0, get_input_partial_shape(0), element::i32);
}

bool BPETokenizer::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto ragged_begins = inputs[0].data<const int32_t>();
    auto ragged_ends   = inputs[1].data<const int32_t>();
    auto begins = inputs[2].data<const int32_t>();
    auto ends   = inputs[3].data<const int32_t>();
    auto chars  = inputs[4].data<const uint8_t>();

    auto vocab_begins = inputs[5].data<const int32_t>();
    auto vocab_ends   = inputs[6].data<const int32_t>();
    auto vocab_chars  = inputs[7].data<const uint8_t>();

    auto merges_begins = inputs[8].data<const int32_t>();
    auto merges_ends   = inputs[9].data<const int32_t>();
    auto merges_chars  = inputs[10].data<const uint8_t>();

    auto vocab_size = inputs[5].get_size();
    auto merges_size = inputs[8].get_size();

    OPENVINO_ASSERT(inputs.size() == 11, "Too few inputs passed to BPETokenizer, it means it is not converted properly or it is not used in the supported pattern");

#if 1
    // Set output shapes
    outputs[0].set_shape(inputs[0].get_shape());
    outputs[1].set_shape(inputs[1].get_shape());
    const size_t num_rows = inputs[0].get_size();

    // FIXME: Not accurate estimation as there is theoretical possibility for re-use the same symbol area
    // to represent different elements in ragged tensor
    outputs[2].set_shape({inputs[4].get_size()});

    using namespace paddlenlp::fast_tokenizer;

//    std::cerr << "[ BPETokenizer ] Start vocab reading\n";
    core::Vocab vocab;
    int32_t unk_token_id = -1;

//    std::cerr << "[ BPETokenizer ] Vocab size is " << vocab_size << "\n";

    for(size_t id = 0; id < vocab_size; ++id) {
        auto token = std::string(vocab_chars + vocab_begins[id], vocab_chars + vocab_ends[id]);
        vocab[token] = int32_t(id); // TODO: Check range
    }

//    std::cerr << "[ BPETokenizer ] Finish vocab reading\n";
//
//    std::cerr << "[ BPETokenizer ] Start merges reading\n";
//    std::cerr << "[ BPETokenizer ] Merges Size: " << merges_size << "\n";
    core::Merges merges;
    std::string delim = " ";


    for(size_t id = 0; id < merges_size; ++id) {
        auto merge = std::string(merges_chars + merges_begins[id], merges_chars + merges_ends[id]);
        const int delim_pos = merge.find(delim);

        std::pair<std::string, std::string> merge_pair = {
            merge.substr(0, delim_pos), merge.substr(delim_pos + 1)
        };
        merges.emplace_back(merge_pair);
    }

//    std::cerr << "[ BPETokenizer ] Finish merges reading\n";


//    std::cerr << "[ BPETokenizer ] Start tokenizer initialization\n";

    std::vector<std::string> unk_token = {};
    if (m_unk_token.size() > 0) {
        unk_token.push_back(m_unk_token);
    };
    std::vector<std::string> suffix_indicator = {};
    if (m_suffix_indicator.size() > 0) {
        suffix_indicator.push_back(m_suffix_indicator);
    };
    std::vector<std::string> end_suffix = {};
    if (m_end_suffix.size() > 0) {
        end_suffix.push_back(m_end_suffix);
    };

    models::BPE tokenizer(vocab, merges, 10000 /* default cache size */, {} /* dropout - don't use dropout for inference */,
                          unk_token, suffix_indicator, end_suffix, m_fuse_unk);

//    std::cerr << "[ BPETokenizer ] Finish tokenizer initialization\n";

    // Get pointers in the output tensors
    auto new_begins = outputs[0].data<int32_t>();
    auto new_ends   = outputs[1].data<int32_t>();
    auto new_elems  = outputs[2].data<int32_t>();
    int32_t ragged_offset = 0;

//    std::cerr << "Ragged Begins and ends:\n";
//    for (size_t i = 0; i < inputs[0].get_size(); ++i) {
//        std::cerr << inputs[0].data<int32_t>()[i] << ", ";
//    }
//    std::cerr << "\n";
//    for (size_t i = 0; i < inputs[1].get_size(); ++i) {
//        std::cerr << inputs[1].data<int32_t>()[i] << ", ";
//    }
//    std::cerr << "\n";


    for(size_t seq = 0; seq < num_rows; ++seq) {
        new_begins[seq] = ragged_offset;
        for(size_t ragged_col = ragged_begins[seq]; ragged_col < ragged_ends[seq]; ++ragged_col) {
            auto str = std::string(chars + begins[ragged_col], chars + ends[ragged_col]);

            std::cerr << "[ BPETokenizer ] String: '" << str << "'\n";
//            std::cerr << "[ BPETokenizer ] String len: " << ends[ragged_col] - begins[ragged_col]  << "\n";

            std::vector<core::Token> results = tokenizer.Tokenize(str);

            for (const core::Token& token : results) {
                std::cout << "[ BPETokenizer ]     id: " << token.id_ << ", value: " << token.value_
                          << ", offset: (" << token.offset_.first << ", "
                          << token.offset_.second << ")." << std::endl;
                OPENVINO_ASSERT(ragged_offset < outputs[2].get_size());
                new_elems[ragged_offset++] = token.id_;
            };
        }

        new_ends[seq] = ragged_offset;
    }
    outputs[2].set_shape({size_t(ragged_offset)});
    return true;

#else
    // Stub implementation that transforms each input string to its length duplicating element if the length is odd
    // End of stub implementation
#endif
}

