// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

#include "kernel_ir/gfx_codegen_backend.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "backends/metal/runtime/mpsrt/gfx_mpsrt_abi.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "openvino/core/node.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace gfx_plugin {

inline std::optional<ov::Tensor> gfx_evaluate_constant_source_tensor(const ov::Output<ov::Node>& source) {
    auto node = source.get_node_shared_ptr();
    if (!node) {
        return std::nullopt;
    }
    if (auto constant = ov::as_type_ptr<const ov::op::v0::Constant>(node)) {
        return constant->get_tensor_view();
    }
    if (!node->has_evaluate()) {
        return std::nullopt;
    }

    ov::TensorVector inputs;
    inputs.reserve(node->get_input_size());
    for (const auto& input_value : node->input_values()) {
        auto input_tensor = gfx_evaluate_constant_source_tensor(input_value);
        if (!input_tensor.has_value()) {
            return std::nullopt;
        }
        inputs.push_back(*input_tensor);
    }

    ov::TensorVector outputs;
    outputs.reserve(node->get_output_size());
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        if (node->get_output_partial_shape(i).is_dynamic()) {
            return std::nullopt;
        }
        outputs.emplace_back(node->get_output_element_type(i), node->get_output_shape(i));
    }
    if (!node->evaluate(outputs, inputs)) {
        return std::nullopt;
    }
    return outputs.at(source.get_index());
}

inline bool gfx_mpsrt_const_payload_already_attached(const KernelSource& source,
                                                     GfxMpsrtValue value) {
    for (const auto& payload : source.const_tensor_sources) {
        if (payload.value_id == value) {
            return true;
        }
    }
    return false;
}

inline bool gfx_mpsrt_program_input_is_const(const GfxMpsrtProgram& program,
                                             size_t input_idx) {
    return input_idx < program.inputs.size() &&
           (program.inputs[input_idx].flags & GfxMpsrtTensorFlagConst) != 0;
}

inline void gfx_attach_mpsrt_const_tensor_sources(
    KernelSource& source, const std::shared_ptr<const ov::Node>& node) {
    if (!node || !source.module) {
        return;
    }
    GfxMpsrtProgram program{};
    if (!read_module_mpsrt_program(source.module, program) || !program.valid) {
        return;
    }
    const size_t input_count = std::min(node->get_input_size(), program.inputs.size());
    for (size_t input_idx = 0; input_idx < input_count; ++input_idx) {
        if (!gfx_mpsrt_program_input_is_const(program, input_idx)) {
            continue;
        }
        auto const_tensor = gfx_evaluate_constant_source_tensor(node->input_value(input_idx));
        if (!const_tensor.has_value() || const_tensor->get_byte_size() == 0) {
            continue;
        }
        const auto value = static_cast<GfxMpsrtValue>(input_idx);
        if (gfx_mpsrt_const_payload_already_attached(source, value)) {
            continue;
        }
        KernelConstTensorSource payload{};
        payload.value_id = value;
        payload.bytes.resize(const_tensor->get_byte_size());
        std::memcpy(payload.bytes.data(), const_tensor->data(), payload.bytes.size());
        source.const_tensor_sources.push_back(std::move(payload));
    }
}

}  // namespace gfx_plugin
}  // namespace ov
