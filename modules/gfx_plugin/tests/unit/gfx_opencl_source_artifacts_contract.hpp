// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "backends/opencl/compiler/opencl_kernel_artifacts.hpp"
#include "backends/opencl/compiler/opencl_operation_support.hpp"
#include "compiler/executable_bundle.hpp"
#include "compiler/kernel_registry.hpp"
#include "compiler/lowering_planner.hpp"
#include "compiler/manifest.hpp"
#include "compiler/operation_support.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "mlir/mlir_support.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "unit/gfx_opencl_catalog_artifact_resolver.hpp"

using namespace ov::gfx_plugin;
using ov::gfx_plugin::compiler::BackendCapabilities;
using ov::gfx_plugin::compiler::BackendTarget;
using ov::gfx_plugin::compiler::KernelRegistry;
using ov::gfx_plugin::compiler::KernelUnitKind;
using ov::gfx_plugin::compiler::LoweringPlanner;
using ov::gfx_plugin::compiler::LoweringRouteKind;
using ov::gfx_plugin::compiler::make_opencl_kernel_registry;
using ov::gfx_plugin::compiler::make_opencl_operation_support_policy;
using ov::gfx_plugin::compiler::ManifestBuilder;
using ov::gfx_plugin::compiler::OperationLegalizer;
using ov::gfx_plugin::test::resolve_opencl_catalog_source_artifact;

namespace {

std::shared_ptr<ov::op::v0::Parameter> param(ov::element::Type type,
                                             ov::Shape shape) {
  return std::make_shared<ov::op::v0::Parameter>(type, std::move(shape));
}

std::shared_ptr<ov::op::v0::Constant> i64_const(ov::Shape shape,
                                                std::vector<int64_t> values) {
  return ov::op::v0::Constant::create(ov::element::i64, std::move(shape),
                                      std::move(values));
}

std::shared_ptr<ov::op::v0::Constant> f32_const(ov::Shape shape,
                                                std::vector<float> values) {
  return ov::op::v0::Constant::create(ov::element::f32, std::move(shape),
                                      std::move(values));
}

std::shared_ptr<ov::op::v0::Constant> f16_const(ov::Shape shape,
                                                std::vector<float> values) {
  return ov::op::v0::Constant::create(ov::element::f16, std::move(shape),
                                      std::move(values));
}

bool opencl_compiler_supports_node(
    const std::shared_ptr<const ov::Node> &node) {
  const auto target = BackendTarget::from_backend(GpuBackend::OpenCL);
  const BackendCapabilities capabilities(
      target, make_opencl_operation_support_policy());
  return capabilities.query_operation({node}).semantic_legal;
}

void expect_opencl_compiler_supports_generated_unit(
    const std::shared_ptr<const ov::Node> &node,
    const std::string &expected_unit_id) {
  const auto target = BackendTarget::from_backend(GpuBackend::OpenCL);
  const BackendCapabilities capabilities(
      target, make_opencl_operation_support_policy());
  const auto support = capabilities.query_operation({node});
  ASSERT_TRUE(support.semantic_legal);
  EXPECT_EQ(support.preferred_route_kind, LoweringRouteKind::GeneratedKernel);
  EXPECT_EQ(support.preferred_route, expected_unit_id);
}

LoweringRouteKind
opencl_artifact_route_kind(const GfxOpenClSourceArtifact &artifact) {
  const auto origin = compiler::classify_opencl_kernel_artifact_origin(
      artifact.artifact_ref.source_id);
  switch (origin) {
  case KernelArtifactOrigin::Generated:
    return LoweringRouteKind::GeneratedKernel;
  case KernelArtifactOrigin::HandwrittenException:
    return LoweringRouteKind::HandwrittenKernelException;
  default:
    return LoweringRouteKind::Unsupported;
  }
}

compiler::KernelUnit
resolve_opencl_artifact_kernel_unit(const KernelRegistry &registry,
                                    const GfxOpenClSourceArtifact &artifact) {
  const auto route_kind = opencl_artifact_route_kind(artifact);
  if (route_kind == LoweringRouteKind::Unsupported) {
    return {};
  }
  return registry.resolve(route_kind, artifact.artifact_ref.source_id);
}

void expect_opencl_compiler_support_matches_kernel_registry(
    const std::shared_ptr<const ov::Node> &node) {
  const auto artifact = resolve_opencl_catalog_source_artifact(node);
  if (!artifact || !artifact->valid) {
    EXPECT_FALSE(opencl_compiler_supports_node(node));
    return;
  }

  const auto target = BackendTarget::from_backend(GpuBackend::OpenCL);
  const auto registry = make_opencl_kernel_registry(target);
  const auto kernel_unit =
      resolve_opencl_artifact_kernel_unit(registry, *artifact);
  const BackendCapabilities capabilities(
      target, make_opencl_operation_support_policy(registry));
  const auto support = capabilities.query_operation({node});

  ASSERT_TRUE(kernel_unit.valid())
      << "OpenCL source artifacts require a registered KernelUnit: "
      << artifact->artifact_ref.source_id;
  ASSERT_TRUE(support.semantic_legal) << support.semantic_reason;
  EXPECT_EQ(support.preferred_route_kind, kernel_unit.route_kind());
  EXPECT_EQ(support.preferred_route, kernel_unit.id());
  EXPECT_EQ(kernel_unit.backend_domain(), "opencl");
}

void expect_opencl_missing_kernel_unit(
    const std::shared_ptr<const ov::Node> &node,
    const std::string &expected_reason) {
  EXPECT_FALSE(resolve_opencl_catalog_source_artifact(node).has_value());

  const auto target = BackendTarget::from_backend(GpuBackend::OpenCL);
  const BackendCapabilities capabilities(
      target, make_opencl_operation_support_policy());
  const auto support = capabilities.query_operation({node});

  EXPECT_FALSE(support.semantic_legal);
  EXPECT_EQ(support.preferred_route_kind, LoweringRouteKind::Unsupported);
  EXPECT_EQ(support.semantic_reason, expected_reason);
  EXPECT_FALSE(opencl_compiler_supports_node(node));
}

void expect_opencl_artifact(const std::shared_ptr<const ov::Node> &node,
                            GfxKernelStageFamily family,
                            const std::string &source_id,
                            const std::string &entry_point, uint32_t arg_count,
                            uint32_t direct_input_count,
                            std::vector<GfxOpenClSourceScalarArg> scalar_args =
                                {GfxOpenClSourceScalarArg::ElementCount,
                                 GfxOpenClSourceScalarArg::OpCode},
                            std::vector<size_t> direct_input_indices = {},
                            std::vector<uint32_t> static_u32_scalars = {},
                            uint32_t direct_output_count = 1,
                            uint32_t input_chunk_size = 0,
                            uint32_t output_chunk_size = 0) {
  if (direct_input_indices.empty() && direct_input_count != 0) {
    for (size_t i = 0; i < direct_input_count; ++i) {
      direct_input_indices.push_back(i);
    }
  }
  auto artifact = resolve_opencl_catalog_source_artifact(node, source_id);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_TRUE(artifact->valid);
  EXPECT_EQ(artifact->stage_manifest.stage_family, family);
  EXPECT_EQ(artifact->stage_manifest.backend_domain,
            GfxKernelBackendDomain::OpenCl);
  EXPECT_EQ(artifact->stage_manifest.execution_kind,
            GfxKernelExecutionKind::CustomKernel);
  EXPECT_EQ(artifact->stage_manifest.storage, GfxKernelStorageKind::Buffer);
  EXPECT_TRUE(artifact->stage_manifest.custom_kernel.valid);
  EXPECT_EQ(artifact->stage_manifest.custom_kernel.entry_point, entry_point);
  EXPECT_EQ(artifact->artifact_ref.kind, GfxKernelArtifactKind::OpenClSource);
  EXPECT_EQ(artifact->artifact_ref.backend_domain,
            GfxKernelBackendDomain::OpenCl);
  EXPECT_EQ(artifact->artifact_ref.source_id, source_id);
  EXPECT_EQ(artifact->artifact_ref.entry_point, entry_point);
  EXPECT_EQ(artifact->arg_count, arg_count);
  EXPECT_EQ(artifact->direct_input_count, direct_input_count);
  EXPECT_EQ(artifact->direct_output_count, direct_output_count);
  EXPECT_EQ(artifact->input_chunk_size, input_chunk_size);
  EXPECT_EQ(artifact->output_chunk_size, output_chunk_size);
  EXPECT_EQ(artifact->direct_input_indices, direct_input_indices);
  EXPECT_EQ(artifact->local_size_hint, 64u);
  EXPECT_EQ(artifact->scalar_args, scalar_args);
  EXPECT_EQ(artifact->static_u32_scalars, static_u32_scalars);
  if (input_chunk_size != 0) {
    ASSERT_FALSE(artifact->planned_chunks.empty());
    uint32_t input_begin = 0;
    for (const auto &chunk : artifact->planned_chunks) {
      const uint32_t chunk_count = std::min<uint32_t>(
          input_chunk_size, direct_input_count - input_begin);
      EXPECT_EQ(chunk.binding_begin, input_begin);
      EXPECT_EQ(chunk.binding_count, chunk_count);
      EXPECT_EQ(chunk.binding_role,
                GfxOpenClSourceChunkBindingRole::DirectInputs);
      EXPECT_NE(chunk.element_count_multiplier, 0u);
      EXPECT_NE(chunk.element_count_divisor, 0u);
      ASSERT_TRUE(chunk.artifact);
      EXPECT_TRUE(chunk.artifact->valid);
      EXPECT_TRUE(chunk.artifact->planned_chunks.empty());
      EXPECT_EQ(chunk.artifact->direct_input_count, chunk_count);
      EXPECT_EQ(chunk.artifact->direct_output_count, 1u);
      input_begin += chunk_count;
    }
    EXPECT_EQ(input_begin, direct_input_count);
  } else if (output_chunk_size != 0) {
    ASSERT_FALSE(artifact->planned_chunks.empty());
    uint32_t output_begin = 0;
    for (const auto &chunk : artifact->planned_chunks) {
      const uint32_t chunk_count = std::min<uint32_t>(
          output_chunk_size, direct_output_count - output_begin);
      EXPECT_EQ(chunk.binding_begin, output_begin);
      EXPECT_EQ(chunk.binding_count, chunk_count);
      EXPECT_EQ(chunk.binding_role,
                GfxOpenClSourceChunkBindingRole::DirectOutputs);
      EXPECT_EQ(chunk.element_count_multiplier, 1u);
      EXPECT_EQ(chunk.element_count_divisor, 1u);
      ASSERT_TRUE(chunk.artifact);
      EXPECT_TRUE(chunk.artifact->valid);
      EXPECT_TRUE(chunk.artifact->planned_chunks.empty());
      EXPECT_EQ(chunk.artifact->direct_input_count, 1u);
      EXPECT_EQ(chunk.artifact->direct_output_count, chunk_count);
      output_begin += chunk_count;
    }
    EXPECT_EQ(output_begin, direct_output_count);
  } else {
    EXPECT_TRUE(artifact->planned_chunks.empty());
  }

  const auto roles =
      artifact->stage_manifest.custom_kernel.external_buffer_abi.roles;
  ASSERT_EQ(roles.size(), arg_count);
  for (size_t i = 0; i < direct_input_count; ++i) {
    EXPECT_EQ(roles[i], GfxKernelBufferRole::TensorInput);
  }
  for (size_t i = 0; i < direct_output_count; ++i) {
    EXPECT_EQ(roles[direct_input_count + i], GfxKernelBufferRole::TensorOutput);
  }
  for (size_t i = direct_input_count + direct_output_count; i < roles.size();
       ++i) {
    EXPECT_EQ(roles[i], GfxKernelBufferRole::ScalarParam);
  }
}

void expect_opencl_range_compiler_contract(
    const std::shared_ptr<ov::op::v4::Range> &range,
    const std::string &expected_unit_id,
    const std::string &expected_entry_point, uint32_t expected_abi_arg_count,
    ov::ParameterVector parameters = {}) {
  const auto target = BackendTarget::from_backend(GpuBackend::OpenCL);
  const BackendCapabilities capabilities(
      target, make_opencl_operation_support_policy());
  const OperationLegalizer legalizer(capabilities);
  const LoweringPlanner planner(target, make_opencl_kernel_registry(target));
  const auto result = std::make_shared<ov::op::v0::Result>(range);
  const auto model =
      std::make_shared<ov::Model>(ov::ResultVector{result}, parameters);

  const auto lowering_plan = planner.plan(model, legalizer);
  ASSERT_TRUE(lowering_plan.executable());

  std::optional<compiler::PlannedOperation> planned_range;
  for (const auto &op : lowering_plan.operations) {
    if (op.type_name == "Range") {
      planned_range = op;
      break;
    }
  }
  ASSERT_TRUE(planned_range.has_value());
  EXPECT_EQ(planned_range->kernel_unit.route_kind(),
            LoweringRouteKind::GeneratedKernel);
  EXPECT_EQ(planned_range->kernel_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(planned_range->kernel_unit.backend_domain(), "opencl");
  EXPECT_EQ(planned_range->kernel_unit.op_family(), "Range");
  EXPECT_EQ(planned_range->kernel_unit.id(), expected_unit_id);

  const auto manifest = ManifestBuilder{}.build(lowering_plan);
  ASSERT_TRUE(manifest.valid());

  std::optional<compiler::StageRecord> range_stage;
  for (const auto &stage : manifest.stages) {
    if (stage.kernel_unit_id == expected_unit_id) {
      range_stage = stage;
      break;
    }
  }
  ASSERT_TRUE(range_stage.has_value());
  EXPECT_EQ(range_stage->execution_kind, LoweringRouteKind::GeneratedKernel);
  EXPECT_EQ(range_stage->backend_domain, "opencl");
  EXPECT_EQ(range_stage->kernel_unit_kind, "generated_kernel");
  EXPECT_EQ(range_stage->dispatch.backend_domain, range_stage->backend_domain);
  EXPECT_EQ(range_stage->dispatch.kernel_unit_id, range_stage->kernel_unit_id);
  EXPECT_EQ(range_stage->memory.hidden_host_copy_allowed, false);

  const auto executable =
      compiler::ExecutableBundleBuilder(
          compiler::make_opencl_kernel_artifact_descriptor_resolver(),
          compiler::make_opencl_kernel_artifact_payload_resolver())
          .build(manifest, lowering_plan);
  ASSERT_TRUE(executable.verify().valid());
  ASSERT_EQ(executable.artifact_payloads.size(), 1u);

  const auto descriptor_index =
      executable.artifact_payloads.front().artifact_descriptor_index;
  ASSERT_LT(descriptor_index, executable.artifact_descriptors.size());
  const auto &descriptor = executable.artifact_descriptors[descriptor_index];
  EXPECT_EQ(descriptor.kernel.kernel_id, expected_unit_id);
  EXPECT_EQ(descriptor.kernel.op_family, "Range");
  EXPECT_EQ(descriptor.kernel.backend_domain, "opencl");
  EXPECT_EQ(descriptor.payload_kind,
            compiler::KernelArtifactPayloadKind::OpenClSource);
  EXPECT_EQ(descriptor.entry_point, expected_entry_point);
  EXPECT_EQ(descriptor.abi_arg_count, expected_abi_arg_count);
  EXPECT_EQ(descriptor.abi_output_arg_count, 1u);
  ASSERT_TRUE(executable.artifact_payloads.front().payload);
  EXPECT_EQ(executable.artifact_payloads.front().payload->source_id(),
            expected_unit_id);
  EXPECT_EQ(executable.artifact_payloads.front().payload->entry_point(),
            expected_entry_point);
}

void expect_opencl_generated_compiler_contract(
    const std::shared_ptr<ov::Node> &node,
    const std::string &expected_type_name,
    const std::string &expected_op_family, const std::string &expected_unit_id,
    const std::string &expected_entry_point, uint32_t expected_abi_arg_count,
    ov::ParameterVector parameters = {}) {
  const auto target = BackendTarget::from_backend(GpuBackend::OpenCL);
  const BackendCapabilities capabilities(
      target, make_opencl_operation_support_policy());
  const OperationLegalizer legalizer(capabilities);
  const LoweringPlanner planner(target, make_opencl_kernel_registry(target));
  const auto result = std::make_shared<ov::op::v0::Result>(node);
  const auto model =
      std::make_shared<ov::Model>(ov::ResultVector{result}, parameters);

  const auto lowering_plan = planner.plan(model, legalizer);
  ASSERT_TRUE(lowering_plan.executable());

  std::optional<compiler::PlannedOperation> planned_op;
  for (const auto &op : lowering_plan.operations) {
    if (op.type_name == expected_type_name) {
      planned_op = op;
      break;
    }
  }
  ASSERT_TRUE(planned_op.has_value());
  EXPECT_EQ(planned_op->kernel_unit.route_kind(),
            LoweringRouteKind::GeneratedKernel);
  EXPECT_EQ(planned_op->kernel_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(planned_op->kernel_unit.backend_domain(), "opencl");
  EXPECT_EQ(planned_op->kernel_unit.op_family(), expected_op_family);
  EXPECT_EQ(planned_op->kernel_unit.id(), expected_unit_id);

  const auto manifest = ManifestBuilder{}.build(lowering_plan);
  ASSERT_TRUE(manifest.valid());

  std::optional<compiler::StageRecord> stage_record;
  for (const auto &stage : manifest.stages) {
    if (stage.kernel_unit_id == expected_unit_id) {
      stage_record = stage;
      break;
    }
  }
  ASSERT_TRUE(stage_record.has_value());
  EXPECT_EQ(stage_record->execution_kind, LoweringRouteKind::GeneratedKernel);
  EXPECT_EQ(stage_record->backend_domain, "opencl");
  EXPECT_EQ(stage_record->kernel_unit_kind, "generated_kernel");
  EXPECT_EQ(stage_record->dispatch.backend_domain,
            stage_record->backend_domain);
  EXPECT_EQ(stage_record->dispatch.kernel_unit_id,
            stage_record->kernel_unit_id);
  EXPECT_EQ(stage_record->memory.hidden_host_copy_allowed, false);

  const auto executable =
      compiler::ExecutableBundleBuilder(
          compiler::make_opencl_kernel_artifact_descriptor_resolver(),
          compiler::make_opencl_kernel_artifact_payload_resolver())
          .build(manifest, lowering_plan);
  ASSERT_TRUE(executable.verify().valid());
  ASSERT_EQ(executable.artifact_payloads.size(), 1u);

  const auto descriptor_index =
      executable.artifact_payloads.front().artifact_descriptor_index;
  ASSERT_LT(descriptor_index, executable.artifact_descriptors.size());
  const auto &descriptor = executable.artifact_descriptors[descriptor_index];
  EXPECT_EQ(descriptor.kernel.kernel_id, expected_unit_id);
  EXPECT_EQ(descriptor.kernel.op_family, expected_op_family);
  EXPECT_EQ(descriptor.kernel.backend_domain, "opencl");
  EXPECT_EQ(descriptor.payload_kind,
            compiler::KernelArtifactPayloadKind::OpenClSource);
  EXPECT_EQ(descriptor.entry_point, expected_entry_point);
  EXPECT_EQ(descriptor.abi_arg_count, expected_abi_arg_count);
  EXPECT_EQ(descriptor.abi_output_arg_count, 1u);
  ASSERT_TRUE(executable.artifact_payloads.front().payload);
  EXPECT_EQ(executable.artifact_payloads.front().payload->source_id(),
            expected_unit_id);
  EXPECT_EQ(executable.artifact_payloads.front().payload->entry_point(),
            expected_entry_point);
}

void expect_opencl_source_excludes(const std::shared_ptr<const ov::Node> &node,
                                   const std::vector<std::string> &needles) {
  const auto artifact = resolve_opencl_catalog_source_artifact(node);
  ASSERT_TRUE(artifact.has_value());
  for (const auto &needle : needles) {
    EXPECT_EQ(artifact->source.find(needle), std::string::npos) << needle;
  }
}

} // namespace
