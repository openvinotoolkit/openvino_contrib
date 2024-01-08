// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "chars_to_bytes.hpp"
#include "bytes_to_chars.hpp"
#include "utils.hpp"

using namespace ov;

void CharsToBytes::validate_and_infer_types() {
    check_ragged_string_input(this, 0);
//    set_ragged_string_output(this, 0, get_input_partial_shape(0));
    set_string_output(this, 0, get_input_partial_shape(0));
}

std::array<std::array<uint8_t, 64>, 4> CharsToBytes::create_pair_map() {
    auto bytes_to_chars = create_bytes_to_chars_map();
    std::array<std::array<uint8_t, 64>, 4> pair_map;

    for (int i=0; i < bytes_to_chars.size(); ++i) {
        std::vector<uint8_t> chars = bytes_to_chars[i];
        if (chars.size() == 2) {
            pair_map[chars[0] - 194][chars[1] - 128] = i;
        };
    };

    return pair_map;
}

bool CharsToBytes::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto ragged_begins = inputs[0].data<const int32_t>();
    auto ragged_ends   = inputs[1].data<const int32_t>();
    auto begins = inputs[2].data<const int32_t>();
    auto ends   = inputs[3].data<const int32_t>();
    auto chars  = inputs[4].data<const uint8_t>();

    outputs[0].set_shape(inputs[0].get_shape());
    outputs[1].set_shape(inputs[1].get_shape());
    outputs[2].set_shape(Shape({inputs[4].get_size()}));
    const size_t num_rows = inputs[0].get_size();

    // Get pointers in the output tensors
    auto new_begins = outputs[0].data<int32_t>();
    auto new_ends   = outputs[1].data<int32_t>();
    auto new_chars  = outputs[2].data<uint8_t>();
    uint32_t char_pointer = 0;

    for(size_t row = 0; row < num_rows; ++row) {
        new_begins[row] = char_pointer;
        for(size_t col = ragged_begins[row]; col < ragged_ends[row]; ++col) {
            const auto word_len = ends[col] - begins[col];

            for (size_t k = 0; k < word_len; ++k) {
                const auto first_byte = chars[begins[col] + k];
                if (first_byte < m_one_byte_border) {
                    new_chars[char_pointer++] = first_byte;
                } else {
                    const auto second_byte = chars[begins[col] + (++k)];
                    new_chars[char_pointer++] = m_pair_map[first_byte - m_first_byte_offset][second_byte - m_second_byte_offset];
                }
            }
        };
        new_ends[row] = char_pointer;
    }
    outputs[2].set_shape({char_pointer});
    return true;
}
