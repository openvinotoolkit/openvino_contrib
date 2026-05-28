// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_apple_msl_activation.hpp"

#include <optional>
#include <utility>
#include <vector>

#include "mlir/codegen_common.hpp"
#include "mlir/gfx_backend_custom_kernel_adapter.hpp"
#include "mlir/msl_codegen_apple_msl_common.hpp"
#include "mlir/msl_codegen_apple_msl_op_kinds.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool activation_msl_type_supported(const ov::element::Type& type) {
    return type == ov::element::f32 || type == ov::element::f16;
}

bool activation_shapes_supported(const ov::Node& op) {
    return op.get_input_size() >= 1 &&
           op.get_output_size() == 1 &&
           op.get_input_partial_shape(0).is_static() &&
           op.get_output_partial_shape(0).is_static() &&
           op.get_input_element_type(0) == op.get_output_element_type(0) &&
           ov::shape_size(op.get_input_shape(0)) == ov::shape_size(op.get_output_shape(0));
}

std::optional<float> scalar_float_constant_input(const std::shared_ptr<const ov::Node>& node,
                                                 size_t input_idx) {
    if (!node || input_idx >= node->get_input_size()) {
        return std::nullopt;
    }
    const auto constant = ov::as_type_ptr<const ov::op::v0::Constant>(
        node->input_value(input_idx).get_node_shared_ptr());
    if (!constant ||
        constant->get_output_element_type(0) != node->get_input_element_type(0) ||
        !constant->get_output_partial_shape(0).is_static() ||
        ov::shape_size(constant->get_output_shape(0)) != 1) {
        return std::nullopt;
    }
    const auto values = constant->cast_vector<float>();
    if (values.empty()) {
        return std::nullopt;
    }
    return values.front();
}

bool scalar_float_input(const std::shared_ptr<const ov::Node>& node,
                        size_t input_idx) {
    if (!node || input_idx >= node->get_input_size()) {
        return false;
    }
    return node->get_input_element_type(input_idx) == node->get_input_element_type(0) &&
           node->get_input_partial_shape(input_idx).is_static() &&
           ov::shape_size(node->get_input_shape(input_idx)) == 1;
}

bool activation_input_abi_supported(const std::shared_ptr<const ov::Node>& node) {
    if (ov::as_type_ptr<const ov::op::v4::Swish>(node)) {
        return node->get_input_size() == 1 ||
               (node->get_input_size() == 2 && scalar_float_input(node, 1));
    }
    return ov::as_type_ptr<const ov::op::util::UnaryElementwiseArithmetic>(node) &&
           node->get_input_size() == 1;
}

bool swish_uses_runtime_beta_input(const std::shared_ptr<const ov::Node>& node) {
    return ov::as_type_ptr<const ov::op::v4::Swish>(node) &&
           node->get_input_size() == 2 &&
           scalar_float_input(node, 1) &&
           !scalar_float_constant_input(node, 1).has_value();
}

}  // namespace

GfxMslGeneratedKernelSourcePlan make_activation_msl_kernel_source_plan(
    const std::shared_ptr<const ov::Node>& node,
    mlir::ModuleOp module) {
    const auto activation = node ? unary_activation_kind_from_node(*node) : std::nullopt;
    if (!node ||
        !activation ||
        !activation_input_abi_supported(node) ||
        !activation_msl_type_supported(node->get_output_element_type(0)) ||
        !activation_shapes_supported(*node)) {
        return {};
    }

    UnaryCodegenDesc desc{};
    desc.element_type = node->get_output_element_type(0);
    desc.activation = *activation;
    desc.swish_beta_runtime_input = swish_uses_runtime_beta_input(node);
    desc.entry_point = desc.swish_beta_runtime_input
                           ? "activation_swish_runtime_beta_kernel"
                           : "activation_kernel";
    if (auto elu = ov::as_type_ptr<const ov::op::v0::Elu>(node)) {
        desc.alpha = static_cast<float>(elu->get_alpha());
    }
    if (auto clamp = ov::as_type_ptr<const ov::op::v0::Clamp>(node)) {
        desc.clamp_min = clamp->get_min();
        desc.clamp_max = clamp->get_max();
    }
    if (auto gelu = ov::as_type_ptr<const ov::op::v7::Gelu>(node)) {
        desc.gelu_tanh_approximation =
            gelu->get_approximation_mode() == ov::op::GeluApproximationMode::TANH;
    }
    if (auto swish = ov::as_type_ptr<const ov::op::v4::Swish>(node)) {
        desc.alpha = 1.0f;
        if (swish->get_input_size() == 2 &&
            !desc.swish_beta_runtime_input) {
            desc.alpha = *scalar_float_constant_input(node, 1);
        }
    }

    const auto out_shape = node->get_output_shape(0);
    const std::vector<int32_t> scalar_args{
        static_cast<int32_t>(ov::shape_size(out_shape))};
    auto binding =
        desc.swish_beta_runtime_input
            ? make_backend_custom_kernel_roles_binding_plan(
                  "Activation",
                  desc.entry_point,
                  {GfxKernelBufferRole::TensorInput,
                   GfxKernelBufferRole::TensorInput,
                   GfxKernelBufferRole::TensorOutput,
                   GfxKernelBufferRole::ScalarParam})
            : make_backend_custom_kernel_binding_plan(
                  "Activation",
                  desc.entry_point,
                  scalar_args);
    if (!binding.valid) {
        return {};
    }
    if (desc.swish_beta_runtime_input) {
        if (binding.scalar_arg_count != scalar_args.size()) {
            return {};
        }
        binding.runtime_binding.scalar_args = scalar_args;
        binding.stage_manifest.custom_kernel.scalar_args = scalar_args;
    }

    auto source = make_kernel_source(module,
                                     desc.entry_point,
                                     generate_msl_from_mlir(module, desc));
    auto plan = make_msl_generated_custom_kernel_source_plan(
        std::move(source), binding);
    plan.source.module = {};
    return plan;
}

}  // namespace gfx_plugin
}  // namespace ov
