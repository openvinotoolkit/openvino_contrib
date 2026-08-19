// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/compiler/msl_codegen_apple_msl_reduction.hpp"

#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "kernel_ir/metal_kernels/reduction_kernels.hpp"
#include "mlir/gfx_backend_custom_kernel_adapter.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_binding.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_op_kinds.hpp"
#include "openvino/core/axis_set.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool reduction_msl_type_supported(const std::shared_ptr<const ov::Node> &node,
                                  ReduceKind kind) {
  if (!node || node->get_input_size() < 1 || node->get_output_size() != 1 ||
      !node->get_input_partial_shape(0).is_static() ||
      !node->get_output_partial_shape(0).is_static() ||
      node->get_input_shape(0).size() > 8) {
    return false;
  }

  if (reduction_kind_is_logical(kind)) {
    return node->get_input_element_type(0) == ov::element::boolean &&
           node->get_output_element_type(0) == ov::element::boolean;
  }
  return node->get_input_element_type(0) == ov::element::f32 &&
         node->get_output_element_type(0) == ov::element::f32;
}

std::vector<GfxKernelBufferRole> reduction_msl_roles() {
  return {
      GfxKernelBufferRole::TensorInput,   GfxKernelBufferRole::TensorOutput,
      GfxKernelBufferRole::ScalarParam,   GfxKernelBufferRole::ScalarParam,
      GfxKernelBufferRole::ScalarParam,   GfxKernelBufferRole::RuntimeParams,
      GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
      GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams};
}

const GfxKernelSource &reduction_msl_source(ReduceKind kind) {
  if (reduction_kind_is_logical(kind)) {
    return metal_generated_reduction_logical_bool_kernel_source();
  }
  return metal_generated_reduction_f32_kernel_source();
}

struct ReductionRuntimeParamMetadata {
  std::vector<int64_t> axes;
  bool keep_dims = false;
};

template <typename ReduceOp>
std::optional<ReductionRuntimeParamMetadata>
reduction_runtime_param_metadata_as(
    const std::shared_ptr<const ov::Node> &node) {
  auto reduce = ov::as_type_ptr<const ReduceOp>(node);
  if (!reduce || !reduce->reduction_axes_constant()) {
    return std::nullopt;
  }
  ReductionRuntimeParamMetadata metadata{};
  metadata.keep_dims = reduce->get_keep_dims();
  const ov::AxisSet axes = reduce->get_reduction_axes();
  metadata.axes.reserve(axes.size());
  for (const auto axis : axes) {
    metadata.axes.push_back(static_cast<int64_t>(axis));
  }
  return metadata;
}

std::optional<ReductionRuntimeParamMetadata>
reduction_runtime_param_metadata(
    const std::shared_ptr<const ov::Node> &node) {
  if (auto metadata =
          reduction_runtime_param_metadata_as<ov::op::v1::ReduceSum>(node)) {
    return metadata;
  }
  if (auto metadata =
          reduction_runtime_param_metadata_as<ov::op::v1::ReduceMean>(node)) {
    return metadata;
  }
  if (auto metadata =
          reduction_runtime_param_metadata_as<ov::op::v1::ReduceMax>(node)) {
    return metadata;
  }
  if (auto metadata =
          reduction_runtime_param_metadata_as<ov::op::v1::ReduceMin>(node)) {
    return metadata;
  }
  if (auto metadata =
          reduction_runtime_param_metadata_as<ov::op::v1::ReduceProd>(node)) {
    return metadata;
  }
  if (auto metadata =
          reduction_runtime_param_metadata_as<ov::op::v4::ReduceL1>(node)) {
    return metadata;
  }
  if (auto metadata =
          reduction_runtime_param_metadata_as<ov::op::v4::ReduceL2>(node)) {
    return metadata;
  }
  if (auto metadata = reduction_runtime_param_metadata_as<
          ov::op::v1::ReduceLogicalAnd>(node)) {
    return metadata;
  }
  if (auto metadata = reduction_runtime_param_metadata_as<
          ov::op::v1::ReduceLogicalOr>(node)) {
    return metadata;
  }
  return std::nullopt;
}

} // namespace

std::optional<ReduceKind>
reduction_kind_from_node(const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return std::nullopt;
  }
  return reduce_kind_from_node(*node);
}

uint32_t reduction_kernel_op_code(ReduceKind kind) noexcept {
  switch (kind) {
  case ReduceKind::Sum:
    return 0u;
  case ReduceKind::Mean:
    return 1u;
  case ReduceKind::Max:
    return 2u;
  case ReduceKind::Min:
    return 3u;
  case ReduceKind::Prod:
    return 4u;
  case ReduceKind::L1:
    return 5u;
  case ReduceKind::L2:
    return 6u;
  case ReduceKind::LogicalAnd:
    return 7u;
  case ReduceKind::LogicalOr:
    return 8u;
  }
  return 0u;
}

bool reduction_kind_is_logical(ReduceKind kind) noexcept {
  return kind == ReduceKind::LogicalAnd || kind == ReduceKind::LogicalOr;
}

std::string_view reduction_msl_kernel_unit_id(ReduceKind kind) noexcept {
  if (reduction_kind_is_logical(kind)) {
    return "metal/generated/reduction_logical_bool";
  }
  return "metal/generated/reduction_f32";
}

std::string_view reduction_msl_kernel_entry_point(ReduceKind kind) noexcept {
  if (reduction_kind_is_logical(kind)) {
    return "gfx_metal_generated_reduction_logical_bool";
  }
  return "gfx_metal_generated_reduction_f32";
}

GfxMslGeneratedKernelSourcePlan make_reduction_msl_kernel_source_plan(
    const std::shared_ptr<const ov::Node> &node, mlir::ModuleOp module) {
  const auto kind = reduction_kind_from_node(node);
  if (!kind || !reduction_msl_type_supported(node, *kind)) {
    return {};
  }

  const auto input_shape = node->get_input_shape(0);
  const auto output_shape = node->get_output_shape(0);
  const auto runtime_metadata = reduction_runtime_param_metadata(node);
  if (!runtime_metadata) {
    return {};
  }
  const std::vector<int32_t> scalar_args{
      static_cast<int32_t>(ov::shape_size(output_shape)),
      static_cast<int32_t>(input_shape.size()),
      static_cast<int32_t>(reduction_kernel_op_code(*kind))};

  auto binding = make_backend_custom_kernel_roles_binding_plan(
      node->get_type_name(), reduction_msl_kernel_entry_point(*kind),
      reduction_msl_roles(), GfxKernelBackendDomain::AppleMsl);
  if (!binding.valid || binding.scalar_arg_count != scalar_args.size()) {
    return {};
  }
  binding.runtime_binding.scalar_args = scalar_args;
  binding.runtime_binding.runtime_param_i64_metadata =
      runtime_metadata->axes;
  binding.runtime_binding.runtime_param_reduce_keep_dims =
      runtime_metadata->keep_dims;
  binding.runtime_binding.runtime_param_reduce_keep_dims_valid = true;
  binding.stage_manifest.custom_kernel.scalar_args = scalar_args;

  const auto &kernel_source = reduction_msl_source(*kind);
  auto source =
      make_kernel_source(module, std::string(kernel_source.entry_point),
                         std::string(kernel_source.source),
                         /*arg_count=*/10u);
  auto plan =
      make_msl_generated_custom_kernel_source_plan(std::move(source), binding);
  plan.source.module = module;
  return plan;
}

std::optional<KernelSource> make_apple_metal_reduction_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node) {
  auto plan = make_reduction_msl_kernel_source_plan(node, source.module);
  if (!plan.valid()) {
    return std::nullopt;
  }
  plan.source.const_tensor_sources = std::move(source.const_tensor_sources);
  return plan.source;
}

} // namespace gfx_plugin
} // namespace ov
