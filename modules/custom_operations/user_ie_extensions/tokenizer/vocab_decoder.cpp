// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef _MSC_VER
#    pragma warning(disable : 4251)
#    pragma warning(disable : 4275)
#endif

#include <algorithm>

#include "vocab_decoder.hpp"
#include "utils.hpp"

using namespace ov;

void VocabDecoder::validate_and_infer_types() {
    check_string_input(this, 1);
    const auto shape = get_input_partial_shape(0);
    set_ragged_string_output(this, 0, {shape[0]});
}

bool VocabDecoder::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto batch_size = inputs[0].get_shape()[0];
    auto seq_len    = inputs[0].get_shape()[1];
    auto input_data = inputs[0].data<const int32_t>();

    auto vocab_begins = inputs[1].data<const int32_t>();
    auto vocab_ends   = inputs[2].data<const int32_t>();
    auto vocab_chars  = inputs[3].data<const uint8_t>();
    auto vocab_size   = inputs[1].get_size();

    std::vector<std::vector<uint8_t>> vocab;
    vocab.resize(vocab_size);

    std::vector<uint8_t> empty = {};

    OPENVINO_ASSERT(inputs.size() == 4, "Too few inputs passed to VocabDecoder, it means it is not converted properly or it is not used in the supported pattern");

    for(size_t id = 0; id < vocab_size; ++id) {
        vocab[id] = std::vector<uint8_t>(vocab_chars + vocab_begins[id], vocab_chars + vocab_ends[id]);
    }
    // Set output shapes
    outputs[0].set_shape({batch_size});
    outputs[1].set_shape({batch_size});
    outputs[2].set_shape({batch_size * seq_len});
    outputs[3].set_shape({batch_size * seq_len});
    outputs[4].set_shape({batch_size * seq_len * 100});  // 100 chars - max token length
    const size_t num_rows = inputs[0].get_size();

    // Get pointers in the output tensors
    auto new_ragged_begins = outputs[0].data<int32_t>();
    auto new_ragged_ends = outputs[1].data<int32_t>();
    auto new_begins = outputs[2].data<int32_t>();
    auto new_ends   = outputs[3].data<int32_t>();
    auto new_chars  = outputs[4].data<uint8_t>();
    uint32_t char_offset = 0;

    for(size_t batch = 0; batch < batch_size; ++batch) {
        new_ragged_begins[batch] = batch * seq_len;
        new_ragged_ends[batch]   = new_ragged_begins[batch] + seq_len;

        for(size_t seq = new_ragged_begins[batch]; seq < new_ragged_ends[batch]; ++seq) {
            auto token_id = input_data[seq];
            std::vector<uint8_t> token;
            if (std::find(m_skip_tokens.begin(), m_skip_tokens.end(), token_id) == m_skip_tokens.end()) {
                token = vocab[token_id];
            } else {
                token = empty;
            }

            std::copy(token.begin(), token.end(), &new_chars[char_offset]);

            new_begins[seq] = char_offset;
            char_offset += token.size();
            new_ends[seq] = char_offset;
        }
    }
    outputs[4].set_shape({char_offset});
    return true;
}
