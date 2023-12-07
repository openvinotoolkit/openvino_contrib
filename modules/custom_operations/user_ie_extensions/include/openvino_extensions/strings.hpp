// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/runtime/tensor.hpp>

namespace openvino_extensions {
// Pack any container with string to ov::Tensor with element type u8
// Requirements for BatchOfStrings: .size() with size and .begin(), .end() as iterators, elements with .begin(), .end() and .size()
// so basically any STL container with std::string is compatible
// Tensor destination will be reshaped according the input data
template <typename BatchOfStrings>
void pack_strings(const BatchOfStrings& strings, ov::Tensor& destination) {
    size_t batch_size = strings.size();

    // First run over all elements: calculate total memory required to hold all strings
    size_t symbols_size = std::accumulate(
        strings.begin(), strings.end(), size_t(0),
        [](size_t accum, typename BatchOfStrings::const_reference str)
        { return accum + str.size(); });

    size_t total_size = 4 * (1 + 1 + batch_size) + symbols_size;
    destination.set_shape({total_size});

    int32_t* pindices = reinterpret_cast<int32_t*>(destination.data<uint8_t>());
    pindices[0] = int32_t(batch_size);
    pindices[1] = 0;
    pindices += 2;
    char* psymbols = reinterpret_cast<char*>(pindices + batch_size);
    size_t current_symbols_pos = 0;

    for (const auto& str: strings) {
        psymbols = std::copy(str.begin(), str.end(), psymbols);
        current_symbols_pos += str.size();
        *pindices = int32_t(current_symbols_pos);
        ++pindices;
    }
}

std::vector<std::string> unpack_strings(const ov::Tensor& source) {
    size_t length = source.get_byte_size();
    // check the format of the input bitstream representing the string tensor
    OPENVINO_ASSERT(length >= 4, "Incorrect packed string tensor format: no batch size in the packed string tensor");
    const int32_t* pindices = reinterpret_cast<const int32_t*>(source.data<const uint8_t>());
    int32_t batch_size = pindices[0];
    OPENVINO_ASSERT(int32_t(length) >= 4 + 4 + 4 * batch_size,
        "Incorrect packed string tensor format: the packed string tensor must contain first string offset and end indices");
    const int32_t* begin_ids = pindices + 1;
    const int32_t* end_ids = pindices + 2;
    const char* symbols = reinterpret_cast<const char*>(pindices + 2 + batch_size);

    std::vector<std::string> result;
    result.reserve(size_t(batch_size));
    for (int32_t idx = 0; idx < batch_size; ++idx) {
        result.emplace_back(symbols + begin_ids[idx], symbols + end_ids[idx]);
    }
    return result;
}
}
