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
#include "runtime/gfx_mpsrt_abi.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "openvino/core/node.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"

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

inline bool gfx_module_uses_mpsrt_conv_stage(mlir::ModuleOp module) {
    if (!module) {
        return false;
    }
    GfxMpsrtModuleStagePlan stage_plan;
    if (read_module_mpsrt_stage_plan(module, stage_plan)) {
        return stage_plan.stage.kind == GfxMpsrtStageKind::MPSConv2D ||
               stage_plan.stage.kind == GfxMpsrtStageKind::MPSGroupConv2D;
    }
    return false;
}

inline bool gfx_should_attach_mpsrt_conv_const_weights(const std::shared_ptr<const ov::Node>& node,
                                                       size_t input_idx,
                                                       const ov::Tensor& tensor) {
    if (input_idx != 1 || tensor.get_byte_size() == 0) {
        return false;
    }
    const auto et = tensor.get_element_type();
    if (et != ov::element::f16 && et != ov::element::f32) {
        return false;
    }
    if (auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
        return conv->get_input_partial_shape(1).is_static() &&
               conv->get_input_shape(1).size() == 4;
    }
    if (auto group_conv = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) {
        return group_conv->get_input_partial_shape(1).is_static() &&
               group_conv->get_input_shape(1).size() == 5;
    }
    return false;
}

inline void gfx_attach_mpsrt_conv_const_tensors(KernelSource& source,
                                                const std::shared_ptr<const ov::Node>& node) {
    if (!node || !gfx_module_uses_mpsrt_conv_stage(source.module)) {
        return;
    }
    for (size_t input_idx = 0; input_idx < node->get_input_size(); ++input_idx) {
        auto const_tensor = gfx_evaluate_constant_source_tensor(node->input_value(input_idx));
        if (!const_tensor.has_value() ||
            !gfx_should_attach_mpsrt_conv_const_weights(node, input_idx, *const_tensor)) {
            continue;
        }
        MpsrtConstTensorSource payload{};
        payload.value = static_cast<GfxMpsrtValue>(input_idx);
        payload.bytes.resize(const_tensor->get_byte_size());
        std::memcpy(payload.bytes.data(), const_tensor->data(), payload.bytes.size());
        source.mpsrt_const_tensors.push_back(std::move(payload));
    }
}

}  // namespace gfx_plugin
}  // namespace ov
