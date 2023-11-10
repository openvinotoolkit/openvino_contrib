// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/runtime/tensor.hpp>

// Pack any container with string to ov::Tensor with element type u8
// Requirements for BatchOfStrings: .size() with size and .begin(), .end() as iterators, elements with .begin(), .end() and .length()
// so basically any STL container with std::string is compatible
// Tensor destination will be reshaped according the input data
template <typename BatchOfStrings>
void pack_strings(const BatchOfStrings& strings, ov::Tensor& destination) {
    auto batch_size = strings.size();

    // First run over all elements: calculate total memory required to hold all strings
    auto symbols_size = std::accumulate(
        strings.begin(), strings.end(), size_t(0),
        [](size_t accum, typename BatchOfStrings::const_reference s)
        { return accum + s.length(); });

    auto total_size = 4*(1 + 1 + batch_size) + symbols_size;
    destination.set_shape({total_size});

    auto data = destination.data<uint8_t>();
    auto pbatch_size = reinterpret_cast<int32_t*>(data);
    auto pindices = pbatch_size + 1;
    auto psymbols = reinterpret_cast<char*>(pindices + 1 + batch_size);
    size_t current_symbols_pos = 0;

    *pbatch_size = batch_size;
    *pindices = 0;

    for(auto s: strings) {
        psymbols = std::copy(s.begin(), s.end(), psymbols);
        current_symbols_pos += s.length();
        *++pindices = current_symbols_pos;
    }
}

std::vector<std::string> unpack_strings(const ov::Tensor& source) {
    auto strings = source.data<const uint8_t>();
    auto length = source.get_byte_size();
    // check the format of the input bitstream representing the string tensor
    OPENVINO_ASSERT(length >= 4, "Incorrect packed string tensor format: no batch size in the packed string tensor");
    auto batch_size = *reinterpret_cast<const int32_t*>(strings + 0);
    OPENVINO_ASSERT(length >= 4 + 4 + 4 * batch_size,
        "Incorrect packed string tensor format: the packed string tensor must contain first string offset and end indices");
    auto begin_ids = reinterpret_cast<const int32_t*>(strings + 4);
    auto end_ids = begin_ids + 1;
    auto symbols = strings + 4 + 4 + 4 * batch_size;

    std::vector<std::string> result;
    result.reserve(batch_size);
    for(size_t i = 0; i < batch_size; ++i) {
        result.push_back(std::string(symbols + begin_ids[i], symbols + end_ids[i]));
    }
    return result;
}
