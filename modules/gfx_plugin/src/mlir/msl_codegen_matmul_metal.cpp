// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_matmul_metal.hpp"

#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "compiler/stage_compiler_policy.hpp"
#include "mlir/msl_codegen_matmul_mpsrt.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/matmul.hpp"
#include "runtime/gfx_compile_profiling.hpp"
#include "runtime/gfx_stage_policy.hpp"
#include "transforms/mlir_fused_ops.hpp"

#include "openvino/core/except.hpp"

#include <utility>

namespace ov {
namespace gfx_plugin {
namespace {

ov::element::Type
resolve_matmul_buffer_type(const ov::element::Type &type,
                           const ov::element::Type &fallback) {
  if (type != ov::element::dynamic) {
    return type;
  }
  return fallback == ov::element::dynamic ? ov::element::f32 : fallback;
}

GfxMpsrtKernelSourcePlan lower_matmul_node_to_metal_kernel_source_plan(
    mlir::MLIRContext &ctx, const GpuBufferManager *buffer_manager,
    const std::shared_ptr<const ov::Node> &node, MatMulCodegenDesc desc,
    const ov::Shape &shape_a, const ov::Shape &shape_b) {
  if (!node) {
    return {};
  }

  auto module = build_mlir_for_node(node, ctx);
  if (!module) {
    return {};
  }
  if (desc.has_activation) {
    const bool applied =
        apply_fused_activation(module, desc.activation, desc.alpha);
    if (!applied) {
      return {};
    }
  }

  const auto output_type = node->get_output_element_type(0);
  desc.element_type =
      resolve_matmul_buffer_type(desc.element_type, output_type);
  desc.input_a_type =
      resolve_matmul_buffer_type(desc.input_a_type, desc.element_type);
  desc.input_b_type =
      resolve_matmul_buffer_type(desc.input_b_type, desc.element_type);
  desc.output_type = output_type;

  const auto stage_compiler_policy =
      compiler::resolve_stage_compiler_policy(GpuBackend::Metal);
  const auto placement = select_stage_optimization_plan(
      buffer_manager, GpuBackend::Metal, "MatMul", node, desc.output_type,
      desc.has_bias, desc.has_activation,
      /*has_batchnorm=*/false, GfxStageRuntimeTraits{},
      &stage_compiler_policy);
  auto mpsrt_source = lower_matmul_module_to_mpsrt_plan(
      module, placement, desc, shape_a, shape_b);
  if (mpsrt_source.valid()) {
    increment_compile_counter("matmul_metal_source_plan_mpsrt_count");
    return std::move(mpsrt_source.mpsrt_plan);
  }

  increment_compile_counter("matmul_metal_source_plan_mpsrt_reject_count");
  OPENVINO_THROW(
      "GFX Metal: MatMul/GEMM could not be materialized through the "
      "MPS/MPSGraph-family MPSRT route. Fix or extend that route before using "
      "an MSL custom MatMul kernel.");
}

} // namespace

std::optional<MatMulCodegenDesc> make_static_matmul_codegen_desc_for_node(
    const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return std::nullopt;
  }
  auto matmul = ov::as_type_ptr<const ov::op::v0::MatMul>(node);
  if (!matmul || !matmul->get_input_partial_shape(0).is_static() ||
      !matmul->get_input_partial_shape(1).is_static() ||
      !matmul->get_output_partial_shape(0).is_static()) {
    return std::nullopt;
  }

  const ov::Shape a_shape = matmul->get_input_shape(0);
  const ov::Shape b_shape = matmul->get_input_shape(1);
  const ov::Shape out_shape = matmul->get_output_shape(0);
  if (a_shape.size() < 2 || b_shape.size() < 2 || out_shape.size() < 2) {
    return std::nullopt;
  }

  const bool ta = matmul->get_transpose_a();
  const bool tb = matmul->get_transpose_b();
  const size_t a_rank = a_shape.size();
  const size_t out_rank = out_shape.size();
  const int64_t m = static_cast<int64_t>(out_shape[out_rank - 2]);
  const int64_t n = static_cast<int64_t>(out_shape[out_rank - 1]);
  const int64_t k =
      static_cast<int64_t>(ta ? a_shape[a_rank - 2] : a_shape[a_rank - 1]);
  if (m <= 0 || n <= 0 || k <= 0) {
    return std::nullopt;
  }

  MatMulCodegenDesc desc{};
  desc.element_type = matmul->get_output_element_type(0);
  desc.input_a_type = matmul->get_input_element_type(0);
  desc.input_b_type = matmul->get_input_element_type(1);
  desc.output_type = matmul->get_output_element_type(0);
  desc.a_transpose = ta;
  desc.b_transpose = tb;
  desc.b_is_nk_layout = tb;
  desc.M = m;
  desc.N = n;
  desc.K = k;
  desc.batch = static_cast<int64_t>(ov::shape_size(out_shape) /
                                    static_cast<uint64_t>(m * n));
  desc.batch_a = static_cast<int64_t>(ov::shape_size(a_shape) /
                                      static_cast<uint64_t>(m * k));
  desc.batch_b = static_cast<int64_t>(ov::shape_size(b_shape) /
                                      static_cast<uint64_t>(k * n));
  return desc;
}

GfxMpsrtKernelSourcePlan make_apple_metal_runtime_matmul_kernel_source_plan(
    mlir::MLIRContext &ctx, const GpuBufferManager *buffer_manager,
    const std::shared_ptr<const ov::Node> &node, MatMulCodegenDesc desc,
    const ov::Shape &shape_a, const ov::Shape &shape_b,
    std::string_view stage_name) {
  auto source_plan = lower_matmul_node_to_metal_kernel_source_plan(
      ctx, buffer_manager, node, desc, shape_a, shape_b);
  OPENVINO_ASSERT(
      source_plan.valid(),
      "MetalStage: failed to create runtime MatMul source plan for ",
      stage_name);
  return source_plan;
}

} // namespace gfx_plugin
} // namespace ov
