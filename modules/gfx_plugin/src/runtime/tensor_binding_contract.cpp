// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/tensor_binding_contract.hpp"

#include <cctype>
#include <limits>

#include "common/runtime_param_descriptor.hpp"
#include "runtime/executable_descriptor.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool consume_whitespace(std::string_view text, size_t& pos) {
    while (pos < text.size() &&
           std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    return pos < text.size();
}

bool descriptor_static_input_shape(
    const RuntimeStageExecutableDescriptor &descriptor, size_t input_idx,
    ov::Shape &shape) {
    if (input_idx >= descriptor.input_bindings.size()) {
        return false;
    }
    return parse_static_shape_contract(
        descriptor.input_bindings[input_idx].partial_shape, shape);
}

bool descriptor_has_binding_contracts(
    const RuntimeStageExecutableDescriptor &descriptor, size_t input_count,
    size_t output_count = 1) {
    return descriptor.input_bindings.size() >= input_count &&
           descriptor.output_bindings.size() >= output_count;
}

bool runtime_param_metadata_axes_valid(
    const RuntimeStageExecutableDescriptor &descriptor, const ov::Shape &shape) {
    const auto rank = static_cast<int64_t>(shape.size());
    if (rank < 0 || rank > 8 ||
        descriptor.runtime_param_i64_metadata.size() > shape.size()) {
        return false;
    }
    uint64_t seen_axes = 0;
    for (auto axis : descriptor.runtime_param_i64_metadata) {
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= rank) {
            return false;
        }
        const uint64_t bit = uint64_t{1} << static_cast<uint64_t>(axis);
        if ((seen_axes & bit) != 0u) {
            return false;
        }
        seen_axes |= bit;
    }
    return !descriptor.runtime_param_i64_metadata.empty();
}

bool runtime_param_metadata_axes_schema_valid(
    const RuntimeStageExecutableDescriptor &descriptor) {
    if (descriptor.runtime_param_i64_metadata.empty() ||
        descriptor.runtime_param_i64_metadata.size() > 8) {
        return false;
    }
    for (size_t i = 0; i < descriptor.runtime_param_i64_metadata.size(); ++i) {
        for (size_t j = i + 1; j < descriptor.runtime_param_i64_metadata.size();
             ++j) {
            if (descriptor.runtime_param_i64_metadata[i] ==
                descriptor.runtime_param_i64_metadata[j]) {
                return false;
            }
        }
    }
    return true;
}

bool runtime_param_metadata_permutation_valid(
    const RuntimeStageExecutableDescriptor &descriptor, const ov::Shape &shape) {
    if (descriptor.runtime_param_i64_metadata.size() != shape.size()) {
        return false;
    }
    std::vector<bool> seen(shape.size(), false);
    for (auto axis : descriptor.runtime_param_i64_metadata) {
        if (axis < 0 || static_cast<size_t>(axis) >= shape.size()) {
            return false;
        }
        const auto index = static_cast<size_t>(axis);
        if (seen[index]) {
            return false;
        }
        seen[index] = true;
    }
    return true;
}

bool runtime_param_metadata_permutation_schema_valid(
    const RuntimeStageExecutableDescriptor &descriptor) {
    const auto rank = descriptor.runtime_param_i64_metadata.size();
    if (rank == 0 || rank > 8) {
        return false;
    }
    std::vector<bool> seen(rank, false);
    for (auto axis : descriptor.runtime_param_i64_metadata) {
        if (axis < 0 || static_cast<size_t>(axis) >= rank) {
            return false;
        }
        const auto index = static_cast<size_t>(axis);
        if (seen[index]) {
            return false;
        }
        seen[index] = true;
    }
    return true;
}

bool runtime_param_metadata_positive_triplet(
    const RuntimeStageExecutableDescriptor &descriptor) {
    return descriptor.runtime_param_i64_metadata.size() == 3 &&
           descriptor.runtime_param_i64_metadata[0] > 0 &&
           descriptor.runtime_param_i64_metadata[1] > 0 &&
           descriptor.runtime_param_i64_metadata[2] > 0;
}

bool runtime_shape_metadata_interpolate_schema_valid(
    const RuntimeStageExecutableDescriptor &descriptor) {
    return descriptor.runtime_shape_i64_metadata.size() >= 3 &&
           descriptor.runtime_shape_i64_metadata[2] >= 0 &&
           descriptor.runtime_shape_i64_metadata[2] <= 2;
}

}  // namespace

ov::element::Type element_type_from_contract(std::string_view name) {
    if (name == "f32" || name == "float32") {
        return ov::element::f32;
    }
    if (name == "f16" || name == "float16") {
        return ov::element::f16;
    }
    if (name == "bf16") {
        return ov::element::bf16;
    }
    if (name == "i64") {
        return ov::element::i64;
    }
    if (name == "i32") {
        return ov::element::i32;
    }
    if (name == "i16") {
        return ov::element::i16;
    }
    if (name == "i8") {
        return ov::element::i8;
    }
    if (name == "u64") {
        return ov::element::u64;
    }
    if (name == "u32") {
        return ov::element::u32;
    }
    if (name == "u16") {
        return ov::element::u16;
    }
    if (name == "u8") {
        return ov::element::u8;
    }
    if (name == "boolean" || name == "bool") {
        return ov::element::boolean;
    }
    return ov::element::dynamic;
}

bool parse_static_shape_contract(std::string_view text, ov::Shape& shape) {
    shape.clear();
    size_t pos = 0;
    if (!consume_whitespace(text, pos)) {
        return false;
    }

    const char open = text[pos];
    const char close = open == '{' ? '}' : (open == '[' ? ']' : '\0');
    if (close == '\0') {
        return false;
    }
    ++pos;

    consume_whitespace(text, pos);
    if (pos < text.size() && text[pos] == close) {
        ++pos;
        consume_whitespace(text, pos);
        return pos == text.size();
    }

    while (pos < text.size()) {
        consume_whitespace(text, pos);
        if (pos >= text.size() ||
            !std::isdigit(static_cast<unsigned char>(text[pos]))) {
            return false;
        }

        size_t value = 0;
        while (pos < text.size() &&
               std::isdigit(static_cast<unsigned char>(text[pos]))) {
            const size_t digit = static_cast<size_t>(text[pos] - '0');
            if (value >
                (std::numeric_limits<size_t>::max() - digit) / 10u) {
                return false;
            }
            value = value * 10u + digit;
            ++pos;
        }
        shape.push_back(value);

        consume_whitespace(text, pos);
        if (pos >= text.size()) {
            return false;
        }
        if (text[pos] == close) {
            ++pos;
            consume_whitespace(text, pos);
            return pos == text.size();
        }
        if (text[pos] != ',') {
            return false;
        }
        ++pos;
    }

    return false;
}

bool descriptor_has_static_shape_contracts(
    const RuntimeStageExecutableDescriptor &descriptor, size_t input_count,
    size_t output_count) {
    ov::Shape shape;
    for (size_t i = 0; i < input_count; ++i) {
        if (i >= descriptor.input_bindings.size() ||
            !parse_static_shape_contract(descriptor.input_bindings[i].partial_shape,
                                         shape)) {
            return false;
        }
    }
    for (size_t i = 0; i < output_count; ++i) {
        if (i >= descriptor.output_bindings.size() ||
            !parse_static_shape_contract(descriptor.output_bindings[i].partial_shape,
                                         shape)) {
            return false;
        }
    }
    return true;
}

bool descriptor_owns_runtime_param_payload(
    const RuntimeStageExecutableDescriptor &descriptor) {
    const size_t runtime_param_count = descriptor.runtime_param_buffer_count;
    switch (descriptor.runtime_param_payload_kind) {
    case RuntimeParamDescriptorPayloadKind::None:
        return runtime_param_descriptor_buffer_count_matches(
                   descriptor.runtime_param_payload_kind, runtime_param_count);
    case RuntimeParamDescriptorPayloadKind::BinaryBroadcast:
        return runtime_param_descriptor_buffer_count_matches(
                   descriptor.runtime_param_payload_kind, runtime_param_count) &&
               descriptor_has_binding_contracts(descriptor, 2);
    case RuntimeParamDescriptorPayloadKind::Broadcast:
        return runtime_param_descriptor_buffer_count_matches(
                   descriptor.runtime_param_payload_kind, runtime_param_count) &&
               descriptor_has_binding_contracts(descriptor, 1);
    case RuntimeParamDescriptorPayloadKind::Select:
        return runtime_param_descriptor_buffer_count_matches(
                   descriptor.runtime_param_payload_kind, runtime_param_count) &&
               descriptor_has_binding_contracts(descriptor, 3);
    case RuntimeParamDescriptorPayloadKind::Tile:
        return runtime_param_descriptor_buffer_count_matches(
                   descriptor.runtime_param_payload_kind, runtime_param_count) &&
               descriptor_has_binding_contracts(descriptor, 1);
    case RuntimeParamDescriptorPayloadKind::Interpolate:
        return runtime_param_descriptor_buffer_count_matches(
                   descriptor.runtime_param_payload_kind, runtime_param_count) &&
               descriptor_has_binding_contracts(descriptor, 1) &&
               runtime_shape_metadata_interpolate_schema_valid(descriptor);
    case RuntimeParamDescriptorPayloadKind::Softmax:
        return runtime_param_descriptor_buffer_count_matches(
                   descriptor.runtime_param_payload_kind, runtime_param_count) &&
               descriptor_has_binding_contracts(descriptor, 1) &&
               runtime_param_metadata_positive_triplet(descriptor);
    case RuntimeParamDescriptorPayloadKind::Transpose: {
        ov::Shape input_shape;
        if (!runtime_param_descriptor_buffer_count_matches(
                descriptor.runtime_param_payload_kind, runtime_param_count) ||
            !descriptor_has_binding_contracts(descriptor, 1) ||
            !runtime_param_metadata_permutation_schema_valid(descriptor)) {
            return false;
        }
        return !descriptor_static_input_shape(descriptor, 0, input_shape) ||
               runtime_param_metadata_permutation_valid(descriptor, input_shape);
    }
    case RuntimeParamDescriptorPayloadKind::Reduce: {
        ov::Shape input_shape;
        if (!runtime_param_descriptor_buffer_count_matches(
                descriptor.runtime_param_payload_kind, runtime_param_count) ||
            !descriptor_has_binding_contracts(descriptor, 1) ||
            !descriptor.runtime_param_reduce_keep_dims_valid ||
            !runtime_param_metadata_axes_schema_valid(descriptor)) {
            return false;
        }
        return !descriptor_static_input_shape(descriptor, 0, input_shape) ||
               runtime_param_metadata_axes_valid(descriptor, input_shape);
    }
    }
    return false;
}

}  // namespace gfx_plugin
}  // namespace ov
