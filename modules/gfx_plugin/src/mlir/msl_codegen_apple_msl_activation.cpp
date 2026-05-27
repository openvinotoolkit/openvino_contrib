// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_apple_msl_activation.hpp"

#include <utility>

#include "mlir/codegen_common.hpp"
#include "mlir/gfx_backend_custom_kernel_adapter.hpp"
#include "mlir/msl_codegen_apple_msl_common.hpp"
#include "mlir/msl_codegen_apple_msl_op_kinds.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool activation_msl_type_supported(const ov::element::Type& type) {
    return type == ov::element::f32 || type == ov::element::f16;
}

bool activation_shapes_supported(const ov::op::util::UnaryElementwiseArithmetic& op) {
    return op.get_input_partial_shape(0).is_static() &&
           op.get_output_partial_shape(0).is_static() &&
           op.get_input_element_type(0) == op.get_output_element_type(0) &&
           ov::shape_size(op.get_input_shape(0)) == ov::shape_size(op.get_output_shape(0));
}

}  // namespace

GfxMslGeneratedKernelSourcePlan make_activation_msl_kernel_source_plan(
    const std::shared_ptr<const ov::Node>& node,
    mlir::ModuleOp module) {
    auto op = ov::as_type_ptr<const ov::op::util::UnaryElementwiseArithmetic>(node);
    const auto activation = node ? unary_activation_kind_from_node(*node) : std::nullopt;
    if (!op ||
        !activation ||
        op->get_input_size() != 1 ||
        op->get_output_size() != 1 ||
        !activation_msl_type_supported(op->get_output_element_type(0)) ||
        !activation_shapes_supported(*op)) {
        return {};
    }

    UnaryCodegenDesc desc{};
    desc.element_type = op->get_output_element_type(0);
    desc.activation = *activation;
    desc.entry_point = "activation_kernel";
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

    const auto out_shape = op->get_output_shape(0);
    auto binding = make_backend_custom_kernel_binding_plan(
        "Activation",
        desc.entry_point,
        {static_cast<int32_t>(ov::shape_size(out_shape))});
    if (!binding.valid) {
        return {};
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
