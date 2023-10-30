// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <openvino/runtime/tensor.hpp>
#include <openvino/frontend/node_context.hpp>


#ifndef OPENVINO_ELEMENT_STRING_SUPPORTED
    #define OPENVINO_ELEMENT_STRING_SUPPORTED 0
#endif

#ifndef OPENVINO_USE_INPUT_OUTPUT_STRING_TENSOR_HACK
    #define OPENVINO_USE_INPUT_OUTPUT_STRING_TENSOR_HACK 0
#endif

#define USE_STRING_TENSORS 0    // modify this depending on willingness to use explicit string tensors

#if USE_STRING_TENSORS && !OPENVINO_ELEMENT_STRING_SUPPORTED
    #error "USE_STRING_TENSORS = 1 can be used only when OpenVINO supports element::string that is determined by OPENVINO_ELEMENT_STRING_SUPPORTED == 1"
#endif

#define SENTENCE_PIECE_EXTENSION_DECOMPOSED_STRINGS 0


void parse_packed_strings (
    const ov::Tensor& packed,
    int32_t& batch_size,
    const int32_t*& begin_ids,
    const int32_t*& end_ids,
    const uint8_t*& symbols);


void check_string_input(const ov::Node* node, size_t input_index);

void check_string_scalar_input(const ov::Node* node, size_t input_index);

void check_ragged_input(const ov::Node* node, size_t input_index);

void check_ragged_string_input(const ov::Node* node, size_t input_index);

void set_string_output(ov::Node* node, size_t output_index, const ov::PartialShape& shape);

void set_ragged_string_output(ov::Node* node, size_t output_index, const ov::PartialShape& shape);

void set_ragged_output(ov::Node* node, size_t output_index, const ov::PartialShape& shape, ov::element::Type type);

void unpack_strings_to_tensors(const std::string* strings, const ov::Shape shape, ov::Tensor& begins, ov::Tensor& ends, ov::Tensor& chars);

void override_parameter (std::shared_ptr<ov::Node> node, ov::element::Type type, const ov::PartialShape& shape);

ov::OutputVector pre_translate_string_tensor_input(ov::Output<ov::Node> input);

ov::OutputVector pre_translate_ragged_tensor_input(ov::Output<ov::Node> input);

ov::OutputVector pre_translate_ragged_string_tensor_input(ov::Output<ov::Node> input);

ov::Output<ov::Node> post_translate_string_tensor_output(const ov::OutputVector& outputs);

ov::Output<ov::Node> post_translate_ragged_tensor_output(const ov::OutputVector& outputs);

bool evaluate_normalization_helper (
    ov::TensorVector& outputs,
    const ov::TensorVector& inputs,
    std::function<std::string(const std::string&)> normalizer);

std::shared_ptr<ov::Node> string_attribute_to_constant (const ov::frontend::NodeContext& node, const std::string& name);

// Pack any container with string to ov::Tensor with element type u8
// Requirements for BatchOfStrings: .size() with size and .begin(), .end() as iterators, elements with .begin(), .end() and .length()
// so basically any STL container with std::string is compatible
// Tensor destination will be reshaped according the input data
template <typename BatchOfStrings>
void pack_strings (const BatchOfStrings& strings, ov::Tensor& destination) {
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

std::vector<std::string> unpack_strings(const ov::Tensor& source);
