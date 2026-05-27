// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_apple_msl_eltwise.hpp"

#include <algorithm>
#include <utility>

#include "mlir/codegen_common.hpp"
#include "mlir/gfx_backend_custom_kernel_adapter.hpp"
#include "mlir/msl_codegen_apple_msl_common.hpp"
#include "mlir/msl_codegen_apple_msl_op_kinds.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool eltwise_msl_type_supported(const ov::element::Type& type) {
    return type == ov::element::f32 ||
           type == ov::element::f16 ||
           type == ov::element::i32;
}

bool uses_numpy_aligned_broadcast(const ov::op::util::BinaryElementwiseArithmetic& op) {
    return op.get_autob().m_type == ov::op::AutoBroadcastType::NUMPY;
}

bool shapes_are_static(const ov::op::util::BinaryElementwiseArithmetic& op) {
    return op.get_input_partial_shape(0).is_static() &&
           op.get_input_partial_shape(1).is_static() &&
           op.get_output_partial_shape(0).is_static();
}

EltwiseCodegenDesc make_eltwise_msl_codegen_desc(
    const ov::op::util::BinaryElementwiseArithmetic& op,
    EltwiseKind kind) {
    OPENVINO_ASSERT(shapes_are_static(op),
                    "GFX Metal Eltwise: static input and output shapes are required");
    OPENVINO_ASSERT(eltwise_msl_type_supported(op.get_output_element_type(0)),
                    "GFX Metal Eltwise: unsupported element type");

    EltwiseCodegenDesc desc{};
    desc.element_type = op.get_output_element_type(0);
    desc.input0_type = op.get_input_element_type(0);
    desc.input1_type = op.get_input_element_type(1);
    desc.output_type = op.get_output_element_type(0);
    desc.eltwise_kind = kind;

    const auto out_shape = op.get_output_shape(0);
    const auto input0_shape = op.get_input_shape(0);
    const auto input1_shape = op.get_input_shape(1);
    desc.out_shape = to_i64_shape(out_shape);
    desc.num_elements = static_cast<uint32_t>(ov::shape_size(out_shape));
    desc.is_broadcast = input0_shape != out_shape || input1_shape != out_shape;
    if (desc.is_broadcast) {
        OPENVINO_ASSERT(uses_numpy_aligned_broadcast(op),
                        "GFX Metal Eltwise: only OpenVINO NUMPY broadcast is supported");
    }
    fill_broadcast_strides(out_shape, input0_shape, desc.stride0);
    fill_broadcast_strides(out_shape, input1_shape, desc.stride1);
    return desc;
}

}  // namespace

GfxMslGeneratedKernelSourcePlan make_eltwise_msl_kernel_source_plan(
    const std::shared_ptr<const ov::Node>& node,
    mlir::ModuleOp module) {
    auto op = ov::as_type_ptr<const ov::op::util::BinaryElementwiseArithmetic>(node);
    const auto kind = node ? eltwise_kind_from_node(*node) : std::nullopt;
    if (!op ||
        !kind ||
        op->get_input_size() != 2 ||
        op->get_output_size() != 1 ||
        op->get_output_element_type(0) != op->get_input_element_type(0) ||
        op->get_output_element_type(0) != op->get_input_element_type(1) ||
        !eltwise_msl_type_supported(op->get_output_element_type(0)) ||
        !shapes_are_static(*op)) {
        return {};
    }

    const auto out_shape = op->get_output_shape(0);
    const bool broadcast =
        op->get_input_shape(0) != out_shape || op->get_input_shape(1) != out_shape;
    if (broadcast && !uses_numpy_aligned_broadcast(*op)) {
        return {};
    }

    const auto desc = make_eltwise_msl_codegen_desc(*op, *kind);
    auto binding = make_backend_custom_kernel_binding_plan(
        "Eltwise",
        "eltwise_kernel",
        {static_cast<int32_t>(desc.num_elements),
         static_cast<int32_t>(std::max<size_t>(out_shape.size(), 1))});
    if (!binding.valid) {
        return {};
    }

    auto source = make_kernel_source(module,
                                     "eltwise_kernel",
                                     generate_msl_from_mlir(module, desc));
    auto plan = make_msl_generated_custom_kernel_source_plan(
        std::move(source), binding);
    plan.source.module = {};
    return plan;
}

}  // namespace gfx_plugin
}  // namespace ov
