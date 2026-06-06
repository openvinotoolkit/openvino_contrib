// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
#include "kernel_ir/opencl_kernels/interpolate_f16_kernel.hpp"
#include "kernel_ir/opencl_kernels/interpolate_f32_kernel.hpp"
#include "kernel_ir/opencl_kernels/matmul_f32_kernel.hpp"
#include "kernel_ir/opencl_kernels/range_kernel.hpp"
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
#include "runtime/gpu_buffer.hpp"

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

bool opencl_artifact_has_registered_kernel_unit(
    const std::shared_ptr<const ov::Node> &node) {
  const auto artifact = resolve_gfx_opencl_source_artifact(node);
  if (!artifact || !artifact->valid) {
    return false;
  }
  const auto registry = make_opencl_kernel_registry(
      BackendTarget::from_backend(GpuBackend::OpenCL));
  return resolve_opencl_artifact_kernel_unit(registry, *artifact).valid();
}

void expect_opencl_compiler_support_matches_kernel_registry(
    const std::shared_ptr<const ov::Node> &node) {
  const auto artifact = resolve_gfx_opencl_source_artifact(node);
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

  EXPECT_EQ(support.semantic_legal, kernel_unit.valid())
      << support.semantic_reason;
  if (!kernel_unit.valid()) {
    EXPECT_EQ(support.preferred_route_kind, LoweringRouteKind::Unsupported);
    return;
  }

  ASSERT_TRUE(support.semantic_legal) << support.semantic_reason;
  EXPECT_EQ(support.preferred_route_kind, kernel_unit.route_kind());
  EXPECT_EQ(support.preferred_route, kernel_unit.id());
  EXPECT_EQ(kernel_unit.backend_domain(), "opencl");
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
  auto artifact = resolve_gfx_opencl_source_artifact(node);
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
  const auto artifact = resolve_gfx_opencl_source_artifact(node);
  ASSERT_TRUE(artifact.has_value());
  for (const auto &needle : needles) {
    EXPECT_EQ(artifact->source.find(needle), std::string::npos) << needle;
  }
}

} // namespace

TEST(GfxOpenClSourceArtifactsTest, BackendTargetIsStableAndCapabilityDriven) {
  const auto target = BackendTarget::from_backend(GpuBackend::OpenCL);
  EXPECT_EQ(target.backend(), GpuBackend::OpenCL);
  EXPECT_NE(target.fingerprint().find("backend=opencl"), std::string::npos);
  EXPECT_TRUE(target.is_compatible_with_fingerprint(target.fingerprint()));

  BackendCapabilities capabilities(target,
                                   make_opencl_operation_support_policy());
  const auto kernel_registry = make_opencl_kernel_registry(target);
  const auto audit = kernel_registry.audit();
  ASSERT_TRUE(audit.valid());
  EXPECT_EQ(audit.handwritten_exception_count, 0u);
  EXPECT_EQ(kernel_registry.route_count(LoweringRouteKind::GeneratedKernel),
            167u);
  EXPECT_EQ(kernel_registry.route_count(
                LoweringRouteKind::HandwrittenKernelException),
            0u);
  OperationLegalizer legalizer(capabilities);
  LoweringPlanner planner(target, kernel_registry);
  auto lhs = param(ov::element::f32, ov::Shape{2, 3});
  auto rhs = param(ov::element::f32, ov::Shape{3, 4});
  auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, false);
  const auto support = capabilities.query_operation({matmul});
  EXPECT_TRUE(support.semantic_legal);
  EXPECT_EQ(support.preferred_route_kind, LoweringRouteKind::GeneratedKernel);
  EXPECT_EQ(support.preferred_route, "opencl/generated/matmul_f32");

  auto result = std::make_shared<ov::op::v0::Result>(matmul);
  auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                           ov::ParameterVector{lhs, rhs});
  const auto plan = planner.plan(model, legalizer);
  EXPECT_TRUE(plan.executable());
  EXPECT_EQ(plan.route_count(LoweringRouteKind::GeneratedKernel), 1u);
  bool found_generated_kernel_unit = false;
  for (const auto &op : plan.operations) {
    if (op.kernel_unit.route_kind() == LoweringRouteKind::GeneratedKernel) {
      found_generated_kernel_unit = true;
      EXPECT_TRUE(op.kernel_unit.valid());
      EXPECT_EQ(op.kernel_unit.kind(), KernelUnitKind::GeneratedKernel);
      EXPECT_EQ(op.kernel_unit.backend_domain(), target.backend_id());
      EXPECT_EQ(op.kernel_unit.id(), "opencl/generated/matmul_f32");
    }
  }
  EXPECT_TRUE(found_generated_kernel_unit);

  const auto manifest = ManifestBuilder{}.build(plan);
  EXPECT_TRUE(manifest.verify().valid());
  EXPECT_EQ(manifest.schema_version, 2u);
  EXPECT_EQ(manifest.route_count(LoweringRouteKind::GeneratedKernel), 1u);
  for (const auto &stage : manifest.stages) {
    EXPECT_FALSE(stage.memory.hidden_host_copy_allowed);
    EXPECT_EQ(stage.dispatch.execution_kind, stage.execution_kind);
    EXPECT_EQ(stage.dispatch.backend_domain, stage.backend_domain);
    EXPECT_EQ(stage.dispatch.kernel_unit_id, stage.kernel_unit_id);
    EXPECT_EQ(stage.dispatch.kernel_unit_kind, stage.kernel_unit_kind);
    if (stage.execution_kind == LoweringRouteKind::GeneratedKernel) {
      EXPECT_EQ(stage.kernel_unit_kind, "generated_kernel");
    }
  }
}

TEST(GfxOpenClSourceArtifactsTest,
     LayoutArtifactsIgnoreShapeOperandsAndUseDataInputOnly) {
  const auto data = param(ov::element::f32, ov::Shape{1, 2, 3});
  const auto reshape = std::make_shared<ov::op::v1::Reshape>(
      data, i64_const(ov::Shape{1}, {6}), false);
  const auto squeeze =
      std::make_shared<ov::op::v0::Squeeze>(data, i64_const(ov::Shape{1}, {0}));
  const auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
      param(ov::element::f32, ov::Shape{2, 3}), i64_const(ov::Shape{1}, {0}));

  expect_opencl_artifact(reshape, GfxKernelStageFamily::Layout,
                         "opencl/baseline/linear_copy_f32",
                         "gfx_opencl_baseline_unary_f32",
                         /*arg_count=*/4,
                         /*direct_input_count=*/1);
  expect_opencl_artifact(squeeze, GfxKernelStageFamily::Layout,
                         "opencl/baseline/linear_copy_f32",
                         "gfx_opencl_baseline_unary_f32",
                         /*arg_count=*/4,
                         /*direct_input_count=*/1);
  expect_opencl_artifact(unsqueeze, GfxKernelStageFamily::Layout,
                         "opencl/baseline/linear_copy_f32",
                         "gfx_opencl_baseline_unary_f32",
                         /*arg_count=*/4,
                         /*direct_input_count=*/1);

  EXPECT_TRUE(opencl_compiler_supports_node(reshape));
  EXPECT_TRUE(opencl_compiler_supports_node(squeeze));
  EXPECT_TRUE(opencl_compiler_supports_node(unsqueeze));
}

TEST(GfxOpenClSourceArtifactsTest,
     BaselineArtifactsDoNotBecomeCompilerSupportWithoutKernelUnit) {
  const auto data = param(ov::element::f32, ov::Shape{1, 2, 3});
  const auto convert =
      std::make_shared<ov::op::v0::Convert>(data, ov::element::i32);

  const auto artifact = resolve_gfx_opencl_source_artifact(convert);
  ASSERT_TRUE(artifact.has_value());
  ASSERT_TRUE(artifact->valid);

  const auto target = BackendTarget::from_backend(GpuBackend::OpenCL);
  const BackendCapabilities capabilities(
      target, make_opencl_operation_support_policy());
  const auto support = capabilities.query_operation({convert});

  EXPECT_FALSE(support.semantic_legal);
  EXPECT_EQ(support.semantic_reason, "missing_opencl_registered_kernel_unit");
  EXPECT_FALSE(opencl_artifact_has_registered_kernel_unit(convert));
  EXPECT_FALSE(opencl_compiler_supports_node(convert));
}

TEST(GfxOpenClSourceArtifactsTest,
     ConvertArtifactsUseTypedSourceKernelsForF32I32AndI64) {
  const auto f32 = param(ov::element::f32, ov::Shape{2, 3});
  const auto i32 = param(ov::element::i32, ov::Shape{2, 3});
  const auto i64 = param(ov::element::i64, ov::Shape{2, 3});
  const auto f32_to_f32 =
      std::make_shared<ov::op::v0::Convert>(f32, ov::element::f32);
  const auto f32_to_i32 =
      std::make_shared<ov::op::v0::Convert>(f32, ov::element::i32);
  const auto i64_to_f32 =
      std::make_shared<ov::op::v0::Convert>(i64, ov::element::f32);
  const auto i32_to_i64 =
      std::make_shared<ov::op::v0::Convert>(i32, ov::element::i64);

  expect_opencl_artifact(f32_to_f32, GfxKernelStageFamily::Convert,
                         "opencl/baseline/convert_f32_to_f32",
                         "gfx_opencl_baseline_convert_f32_to_f32",
                         /*arg_count=*/3,
                         /*direct_input_count=*/1,
                         {GfxOpenClSourceScalarArg::ElementCount});
  expect_opencl_artifact(f32_to_i32, GfxKernelStageFamily::Convert,
                         "opencl/baseline/convert_f32_to_i32",
                         "gfx_opencl_baseline_convert_f32_to_i32",
                         /*arg_count=*/3,
                         /*direct_input_count=*/1,
                         {GfxOpenClSourceScalarArg::ElementCount});
  expect_opencl_artifact(i64_to_f32, GfxKernelStageFamily::Convert,
                         "opencl/baseline/convert_i64_to_f32",
                         "gfx_opencl_baseline_convert_i64_to_f32",
                         /*arg_count=*/3,
                         /*direct_input_count=*/1,
                         {GfxOpenClSourceScalarArg::ElementCount});
  expect_opencl_artifact(i32_to_i64, GfxKernelStageFamily::Convert,
                         "opencl/baseline/convert_i32_to_i64",
                         "gfx_opencl_baseline_convert_i32_to_i64",
                         /*arg_count=*/3,
                         /*direct_input_count=*/1,
                         {GfxOpenClSourceScalarArg::ElementCount});

  expect_opencl_compiler_support_matches_kernel_registry(f32_to_f32);
  expect_opencl_compiler_support_matches_kernel_registry(f32_to_i32);
  expect_opencl_compiler_support_matches_kernel_registry(i64_to_f32);
  expect_opencl_compiler_support_matches_kernel_registry(i32_to_i64);
}

TEST(GfxOpenClSourceArtifactsTest,
     MatMulArtifactsUseGeneratedF32KernelUnitMetadata) {
  const auto lhs = param(ov::element::f32, ov::Shape{2, 3});
  const auto rhs = param(ov::element::f32, ov::Shape{3, 4});
  const auto matmul =
      std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, false);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 9, GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      2, 4, 3, // M, N, K
      0, 0,    // lhs/rhs batch strides
      3, 1,    // lhs row/col strides
      4, 1,    // rhs row/col strides
  };

  expect_opencl_artifact(
      matmul, GfxKernelStageFamily::Gemm, "opencl/generated/matmul_f32",
      "gfx_opencl_generated_matmul_f32",
      /*arg_count=*/13,
      /*direct_input_count=*/2, scalar_args, {0, 1}, static_u32_scalars);
  const auto artifact = resolve_gfx_opencl_source_artifact(matmul);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->source,
            opencl_generated_matmul_f32_kernel_source().source);
  EXPECT_EQ(artifact->source.find("__global long*"), std::string::npos);
  expect_opencl_compiler_support_matches_kernel_registry(matmul);
}

TEST(GfxOpenClSourceArtifactsTest,
     MatMulArtifactsCarryTransposeAndBatchBroadcastMetadata) {
  const auto lhs = param(ov::element::f32, ov::Shape{2, 1, 3, 2});
  const auto rhs = param(ov::element::f32, ov::Shape{1, 3, 4});
  const auto matmul =
      std::make_shared<ov::op::v0::MatMul>(lhs, rhs, true, false);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 9, GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      2, 4, 3, // M, N, K
      6, 0,    // lhs batches advance, rhs broadcasts
      1, 2,    // transposed lhs row/col strides
      4, 1,    // rhs row/col strides
  };

  expect_opencl_artifact(
      matmul, GfxKernelStageFamily::Gemm, "opencl/generated/matmul_f32",
      "gfx_opencl_generated_matmul_f32",
      /*arg_count=*/13,
      /*direct_input_count=*/2, scalar_args, {0, 1}, static_u32_scalars);
  expect_opencl_compiler_support_matches_kernel_registry(matmul);
}

TEST(GfxOpenClSourceArtifactsTest,
     MatMulRejectsUnsupportedOpenClKernelUnitContract) {
  const auto lhs = param(ov::element::f16, ov::Shape{2, 3});
  const auto rhs = param(ov::element::f16, ov::Shape{3, 4});
  const auto matmul =
      std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, false);

  EXPECT_FALSE(resolve_gfx_opencl_source_artifact(matmul).has_value());
  const auto target = BackendTarget::from_backend(GpuBackend::OpenCL);
  const BackendCapabilities capabilities(
      target, make_opencl_operation_support_policy());
  const auto support = capabilities.query_operation({matmul});
  EXPECT_FALSE(support.semantic_legal);
  EXPECT_EQ(support.semantic_reason, "missing_opencl_matmul_kernel_unit");
  EXPECT_FALSE(opencl_compiler_supports_node(matmul));
}

TEST(GfxOpenClSourceArtifactsTest,
     InterpolateArtifactsUseGeneratedKernelUnits) {
  const auto data = param(ov::element::f32, ov::Shape{1, 4, 16, 16});
  const auto output_shape = i64_const(ov::Shape{2}, {32, 32});
  ov::op::v0::Interpolate::Attributes attrs;
  attrs.axes = ov::AxisSet{2, 3};
  attrs.mode = "linear";
  attrs.align_corners = false;
  const auto interpolate =
      std::make_shared<ov::op::v0::Interpolate>(data, output_shape, attrs);

  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::StaticU32,
      GfxOpenClSourceScalarArg::StaticU32,
      GfxOpenClSourceScalarArg::StaticU32,
      GfxOpenClSourceScalarArg::StaticU32,
      GfxOpenClSourceScalarArg::Input0Dim0,
      GfxOpenClSourceScalarArg::Input0Dim1,
      GfxOpenClSourceScalarArg::Input0Dim2,
      GfxOpenClSourceScalarArg::Input0Dim3,
      GfxOpenClSourceScalarArg::Output0Dim2,
      GfxOpenClSourceScalarArg::Output0Dim3};
  const std::vector<uint32_t> bilinear_half_pixel_scalars = {
      0, // nearest
      0, // align_corners
      1, // use_half_pixel
      0, // nearest_mode
  };

  expect_opencl_artifact(interpolate, GfxKernelStageFamily::Layout,
                         "opencl/generated/interpolate_f32",
                         "gfx_opencl_generated_interpolate_f32",
                         /*arg_count=*/13,
                         /*direct_input_count=*/1, scalar_args, {0},
                         bilinear_half_pixel_scalars);
  const auto artifact = resolve_gfx_opencl_source_artifact(interpolate);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->source,
            opencl_generated_interpolate_f32_kernel_source().source);
  expect_opencl_compiler_support_matches_kernel_registry(interpolate);

  const auto f16_data = param(ov::element::f16, ov::Shape{1, 4, 16, 16});
  const auto f16_output_shape = i64_const(ov::Shape{2}, {32, 32});
  attrs.mode = "nearest";
  const auto f16_interpolate = std::make_shared<ov::op::v0::Interpolate>(
      f16_data, f16_output_shape, attrs);
  const std::vector<uint32_t> nearest_half_pixel_scalars = {
      1, // nearest
      0, // align_corners
      1, // use_half_pixel
      0, // nearest_mode
  };
  expect_opencl_artifact(f16_interpolate, GfxKernelStageFamily::Layout,
                         "opencl/generated/interpolate_f16",
                         "gfx_opencl_generated_interpolate_f16",
                         /*arg_count=*/13,
                         /*direct_input_count=*/1, scalar_args, {0},
                         nearest_half_pixel_scalars);
  const auto f16_artifact = resolve_gfx_opencl_source_artifact(f16_interpolate);
  ASSERT_TRUE(f16_artifact.has_value());
  EXPECT_EQ(f16_artifact->source,
            opencl_generated_interpolate_f16_kernel_source().source);
  EXPECT_NE(f16_artifact->source.find("gfx_f16_bits_to_f32"),
            std::string::npos);
  EXPECT_EQ(f16_artifact->source.find("__global half"), std::string::npos);
  expect_opencl_compiler_support_matches_kernel_registry(f16_interpolate);
}

TEST(GfxOpenClSourceArtifactsTest,
     InterpolateRejectsUnsupportedOpenClKernelUnitContract) {
  const auto data = param(ov::element::f32, ov::Shape{1, 4, 16, 16});
  const auto output_shape = i64_const(ov::Shape{2}, {1, 4});
  const auto scales = f32_const(ov::Shape{2}, {1.0f, 1.0f});
  const auto axes = i64_const(ov::Shape{2}, {0, 1});
  using Base = ov::op::util::InterpolateBase;
  ov::op::v4::Interpolate::InterpolateAttrs attrs;
  attrs.mode = Base::InterpolateMode::LINEAR;
  attrs.shape_calculation_mode = Base::ShapeCalcMode::SIZES;
  attrs.coordinate_transformation_mode =
      Base::CoordinateTransformMode::HALF_PIXEL;
  const auto interpolate = std::make_shared<ov::op::v4::Interpolate>(
      data, output_shape, scales, axes, attrs);

  EXPECT_FALSE(resolve_gfx_opencl_source_artifact(interpolate).has_value());
  EXPECT_FALSE(opencl_compiler_supports_node(interpolate));
}

TEST(GfxOpenClSourceArtifactsTest,
     TransposeArtifactsCarryShapeStrideAndPermutationMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto transpose = std::make_shared<ov::op::v1::Transpose>(
      data, i64_const(ov::Shape{3}, {1, 2, 0}));
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 13,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      3,           // rank
      3,  4, 2, 1, // output dims padded to rank 4
      12, 4, 1, 1, // input strides padded to rank 4
      1,  2, 0, 0, // permutation padded to rank 4
  };

  expect_opencl_artifact(
      transpose, GfxKernelStageFamily::Transpose,
      "opencl/generated/transpose_f32", "gfx_opencl_generated_transpose_f32",
      /*arg_count=*/16,
      /*direct_input_count=*/1, scalar_args, {0}, static_u32_scalars);
  expect_opencl_source_excludes(
      transpose, {"long", "__global long*", "gfx_opencl_generated_range_i64"});
  expect_opencl_compiler_support_matches_kernel_registry(transpose);
}

TEST(GfxOpenClSourceArtifactsTest,
     SliceArtifactsCarryShapeStrideBeginAndStepMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto slice = std::make_shared<ov::op::v8::Slice>(
      data, i64_const(ov::Shape{3}, {0, 1, 0}),
      i64_const(ov::Shape{3}, {2, 3, 4}), i64_const(ov::Shape{3}, {1, 1, 2}),
      i64_const(ov::Shape{3}, {0, 1, 2}));
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 17,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      3,           // rank
      2,  2, 2, 1, // output dims padded to rank 4
      12, 4, 1, 1, // input strides padded to rank 4
      0,  1, 0, 0, // begin coordinate padded to rank 4
      1,  1, 2, 1, // step coordinate padded to rank 4
  };

  expect_opencl_artifact(
      slice, GfxKernelStageFamily::GatherScatter, "opencl/baseline/slice_f32",
      "gfx_opencl_baseline_slice_f32",
      /*arg_count=*/20,
      /*direct_input_count=*/1, scalar_args, {0}, static_u32_scalars);
  expect_opencl_source_excludes(
      slice, {"long", "__global long*", "gfx_opencl_generated_range_i64"});
  expect_opencl_compiler_support_matches_kernel_registry(slice);
}

TEST(GfxOpenClSourceArtifactsTest,
     StridedSliceArtifactsReuseSliceKernelAndStaticMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto slice = std::make_shared<ov::op::v1::StridedSlice>(
      data, i64_const(ov::Shape{3}, {0, 1, 0}),
      i64_const(ov::Shape{3}, {2, 3, 4}), i64_const(ov::Shape{3}, {1, 1, 2}),
      std::vector<int64_t>{0, 0, 0}, std::vector<int64_t>{0, 0, 0});
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 17,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      3,           // rank
      2,  2, 2, 1, // output dims padded to rank 4
      12, 4, 1, 1, // input strides padded to rank 4
      0,  1, 0, 0, // begin coordinate padded to rank 4
      1,  1, 2, 1, // step coordinate padded to rank 4
  };

  expect_opencl_artifact(
      slice, GfxKernelStageFamily::GatherScatter, "opencl/baseline/slice_f32",
      "gfx_opencl_baseline_slice_f32",
      /*arg_count=*/20,
      /*direct_input_count=*/1, scalar_args, {0}, static_u32_scalars);
  expect_opencl_source_excludes(
      slice, {"long", "__global long*", "gfx_opencl_generated_range_i64"});
  expect_opencl_compiler_support_matches_kernel_registry(slice);
}

TEST(GfxOpenClSourceArtifactsTest,
     RangeF32ArtifactsUseDirectScalarInputsAndElementCountOnly) {
  const auto range = std::make_shared<ov::op::v4::Range>(
      f32_const(ov::Shape{}, {1.5f}), f32_const(ov::Shape{}, {6.5f}),
      f32_const(ov::Shape{}, {1.0f}), ov::element::f32);

  expect_opencl_artifact(range, GfxKernelStageFamily::GatherScatter,
                         "opencl/generated/range_f32",
                         "gfx_opencl_generated_range_f32",
                         /*arg_count=*/5,
                         /*direct_input_count=*/3,
                         {GfxOpenClSourceScalarArg::ElementCount}, {0, 1, 2});
  expect_opencl_compiler_supports_generated_unit(range,
                                                 "opencl/generated/range_f32");
  expect_opencl_range_compiler_contract(range, "opencl/generated/range_f32",
                                        "gfx_opencl_generated_range_f32",
                                        /*expected_abi_arg_count=*/5);
  EXPECT_FALSE(
      make_opencl_range_source_artifact(range, "opencl/generated/range_i64")
          .has_value());
  expect_opencl_compiler_support_matches_kernel_registry(range);
}

TEST(GfxOpenClSourceArtifactsTest,
     RangePayloadResolverRejectsMismatchedKernelUnitWithoutSourceFallback) {
  const auto range = std::make_shared<ov::op::v4::Range>(
      f32_const(ov::Shape{}, {1.5f}), f32_const(ov::Shape{}, {6.5f}),
      f32_const(ov::Shape{}, {1.0f}), ov::element::f32);

  compiler::KernelArtifactDescriptor descriptor;
  descriptor.kernel.backend_domain = "opencl";
  descriptor.kernel.kernel_id = "opencl/generated/range_i64";
  descriptor.payload_kind = compiler::KernelArtifactPayloadKind::OpenClSource;

  compiler::PlannedOperation planned_range;
  planned_range.source_node = range;
  planned_range.type_name = "Range";

  const auto resolver =
      compiler::make_opencl_kernel_artifact_payload_resolver();
  EXPECT_FALSE(static_cast<bool>(resolver(descriptor, planned_range)));
  EXPECT_TRUE(descriptor.entry_point.empty());
}

TEST(GfxOpenClSourceArtifactsTest,
     GeneratedPayloadResolverRejectsMismatchedKernelUnitWithoutSourceFallback) {
  const auto lhs = param(ov::element::f32, ov::Shape{2, 3});
  const auto rhs = param(ov::element::f32, ov::Shape{3, 4});
  const auto matmul =
      std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, false);

  compiler::KernelArtifactDescriptor descriptor;
  descriptor.kernel.backend_domain = "opencl";
  descriptor.kernel.kernel_id = "opencl/generated/shapeof_i32";
  descriptor.kernel.origin = compiler::KernelArtifactOrigin::Generated;
  descriptor.payload_kind = compiler::KernelArtifactPayloadKind::OpenClSource;

  compiler::PlannedOperation planned_matmul;
  planned_matmul.source_node = matmul;
  planned_matmul.type_name = "MatMul";

  const auto resolver =
      compiler::make_opencl_kernel_artifact_payload_resolver();
  EXPECT_FALSE(static_cast<bool>(resolver(descriptor, planned_matmul)));
  EXPECT_TRUE(descriptor.entry_point.empty());
}

TEST(GfxOpenClSourceArtifactsTest,
     GeneratedPayloadResolverRejectsBaselineSourceArtifact) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3});
  const auto convert =
      std::make_shared<ov::op::v0::Convert>(data, ov::element::i32);

  compiler::KernelArtifactDescriptor descriptor;
  descriptor.kernel.backend_domain = "opencl";
  descriptor.kernel.kernel_id = "opencl/baseline/convert_f32_to_i32";
  descriptor.kernel.origin = compiler::KernelArtifactOrigin::Generated;
  descriptor.payload_kind = compiler::KernelArtifactPayloadKind::OpenClSource;

  compiler::PlannedOperation planned_convert;
  planned_convert.source_node = convert;
  planned_convert.type_name = "Convert";

  const auto resolver =
      compiler::make_opencl_kernel_artifact_payload_resolver();
  EXPECT_FALSE(static_cast<bool>(resolver(descriptor, planned_convert)));
  EXPECT_TRUE(descriptor.entry_point.empty());
}

TEST(GfxOpenClSourceArtifactsTest,
     RangeF16ArtifactsUsePackedF16KernelWithDirectScalarInputs) {
  const auto range = std::make_shared<ov::op::v4::Range>(
      f16_const(ov::Shape{}, {1.0f}), f16_const(ov::Shape{}, {4.0f}),
      f16_const(ov::Shape{}, {0.5f}), ov::element::f16);

  expect_opencl_artifact(range, GfxKernelStageFamily::GatherScatter,
                         "opencl/generated/range_f16",
                         "gfx_opencl_generated_range_f16",
                         /*arg_count=*/5,
                         /*direct_input_count=*/3,
                         {GfxOpenClSourceScalarArg::ElementCount}, {0, 1, 2});
  expect_opencl_compiler_supports_generated_unit(range,
                                                 "opencl/generated/range_f16");
  expect_opencl_range_compiler_contract(range, "opencl/generated/range_f16",
                                        "gfx_opencl_generated_range_f16",
                                        /*expected_abi_arg_count=*/5);
  expect_opencl_compiler_support_matches_kernel_registry(range);
}

TEST(GfxOpenClSourceArtifactsTest,
     RangeI64ArtifactsUseTheSameManifestAbiWithI64OutputKernel) {
  const auto range = std::make_shared<ov::op::v4::Range>(
      i64_const(ov::Shape{}, {0}), i64_const(ov::Shape{}, {10}),
      i64_const(ov::Shape{}, {2}), ov::element::i64);

  expect_opencl_artifact(range, GfxKernelStageFamily::GatherScatter,
                         "opencl/generated/range_i64",
                         "gfx_opencl_generated_range_i64",
                         /*arg_count=*/5,
                         /*direct_input_count=*/3,
                         {GfxOpenClSourceScalarArg::ElementCount}, {0, 1, 2});
  expect_opencl_compiler_supports_generated_unit(range,
                                                 "opencl/generated/range_i64");
  expect_opencl_range_compiler_contract(range, "opencl/generated/range_i64",
                                        "gfx_opencl_generated_range_i64",
                                        /*expected_abi_arg_count=*/5);
  expect_opencl_compiler_support_matches_kernel_registry(range);
}

TEST(GfxOpenClSourceArtifactsTest,
     TileF32ArtifactsCarryStaticShapeAndStrideMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{2, 1, 3});
  const auto tile = std::make_shared<ov::op::v0::Tile>(
      data, i64_const(ov::Shape{3}, {1, 4, 2}));
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 17,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      3,           // rank
      2,  4, 6, 1, // output dims padded to rank 4
      2,  1, 3, 1, // input dims padded to rank 4
      24, 6, 1, 1, // output strides padded to rank 4
      3,  3, 1, 1, // input strides padded to rank 4
  };

  expect_opencl_artifact(
      tile, GfxKernelStageFamily::Layout, "opencl/generated/tile_f32",
      "gfx_opencl_generated_tile_f32",
      /*arg_count=*/20,
      /*direct_input_count=*/1, scalar_args, {0}, static_u32_scalars);
  expect_opencl_source_excludes(
      tile, {"long", "__global long*", "gfx_opencl_generated_range_i64"});
  expect_opencl_compiler_support_matches_kernel_registry(tile);
  expect_opencl_generated_compiler_contract(
      tile, "Tile", "Tile", "opencl/generated/tile_f32",
      "gfx_opencl_generated_tile_f32",
      /*expected_abi_arg_count=*/20, {data});
}

TEST(GfxOpenClSourceArtifactsTest,
     TileF16ArtifactsReuseStaticShapeAndStrideMetadata) {
  const auto data = param(ov::element::f16, ov::Shape{2, 1, 3});
  const auto tile = std::make_shared<ov::op::v0::Tile>(
      data, i64_const(ov::Shape{3}, {1, 4, 2}));
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 17,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      3,           // rank
      2,  4, 6, 1, // output dims padded to rank 4
      2,  1, 3, 1, // input dims padded to rank 4
      24, 6, 1, 1, // output strides padded to rank 4
      3,  3, 1, 1, // input strides padded to rank 4
  };

  expect_opencl_artifact(
      tile, GfxKernelStageFamily::Layout, "opencl/generated/tile_f16",
      "gfx_opencl_generated_tile_f16",
      /*arg_count=*/20,
      /*direct_input_count=*/1, scalar_args, {0}, static_u32_scalars);
  expect_opencl_compiler_support_matches_kernel_registry(tile);
  expect_opencl_generated_compiler_contract(
      tile, "Tile", "Tile", "opencl/generated/tile_f16",
      "gfx_opencl_generated_tile_f16",
      /*expected_abi_arg_count=*/20, {data});
}

TEST(GfxOpenClSourceArtifactsTest,
     TilePayloadResolverRejectsMismatchedKernelUnitWithoutSourceFallback) {
  const auto data = param(ov::element::f32, ov::Shape{2, 1, 3});
  const auto tile = std::make_shared<ov::op::v0::Tile>(
      data, i64_const(ov::Shape{3}, {1, 4, 2}));

  compiler::KernelArtifactDescriptor descriptor;
  descriptor.kernel.backend_domain = "opencl";
  descriptor.kernel.kernel_id = "opencl/generated/tile_f16";
  descriptor.kernel.origin = compiler::KernelArtifactOrigin::Generated;
  descriptor.payload_kind = compiler::KernelArtifactPayloadKind::OpenClSource;

  compiler::PlannedOperation planned_tile;
  planned_tile.source_node = tile;
  planned_tile.type_name = "Tile";

  const auto resolver =
      compiler::make_opencl_kernel_artifact_payload_resolver();
  EXPECT_FALSE(static_cast<bool>(resolver(descriptor, planned_tile)));
  EXPECT_TRUE(descriptor.entry_point.empty());
}

TEST(GfxOpenClSourceArtifactsTest,
     DynamicF16TileUsesRuntimeInputAndOutputDimsUnderSourceManifest) {
  const auto data = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 3});
  const auto repeats = std::make_shared<ov::op::v0::Parameter>(
      ov::element::i64, ov::PartialShape{3});
  const auto tile = std::make_shared<ov::op::v0::Tile>(data, repeats);

  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::StaticU32,
      GfxOpenClSourceScalarArg::Output0Dim0,
      GfxOpenClSourceScalarArg::Output0Dim1,
      GfxOpenClSourceScalarArg::Output0Dim2,
      GfxOpenClSourceScalarArg::Output0Dim3,
      GfxOpenClSourceScalarArg::Input0Dim0,
      GfxOpenClSourceScalarArg::Input0Dim1,
      GfxOpenClSourceScalarArg::Input0Dim2,
      GfxOpenClSourceScalarArg::Input0Dim3};

  expect_opencl_artifact(tile, GfxKernelStageFamily::Layout,
                         "opencl/generated/tile_dynamic_f16",
                         "gfx_opencl_generated_tile_dynamic_f16",
                         /*arg_count=*/12,
                         /*direct_input_count=*/1, scalar_args, {0}, {3});
  expect_opencl_compiler_support_matches_kernel_registry(tile);
}

TEST(GfxOpenClSourceArtifactsTest,
     DynamicF32TileUsesRuntimeInputAndOutputDimsUnderSourceManifest) {
  const auto data = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f32, ov::PartialShape{1, -1, 3});
  const auto repeats = std::make_shared<ov::op::v0::Parameter>(
      ov::element::i64, ov::PartialShape{3});
  const auto tile = std::make_shared<ov::op::v0::Tile>(data, repeats);

  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::StaticU32,
      GfxOpenClSourceScalarArg::Output0Dim0,
      GfxOpenClSourceScalarArg::Output0Dim1,
      GfxOpenClSourceScalarArg::Output0Dim2,
      GfxOpenClSourceScalarArg::Output0Dim3,
      GfxOpenClSourceScalarArg::Input0Dim0,
      GfxOpenClSourceScalarArg::Input0Dim1,
      GfxOpenClSourceScalarArg::Input0Dim2,
      GfxOpenClSourceScalarArg::Input0Dim3};

  expect_opencl_artifact(tile, GfxKernelStageFamily::Layout,
                         "opencl/generated/tile_dynamic_f32",
                         "gfx_opencl_generated_tile_dynamic_f32",
                         /*arg_count=*/12,
                         /*direct_input_count=*/1, scalar_args, {0}, {3});
  expect_opencl_source_excludes(
      tile, {"long", "__global long*", "gfx_opencl_generated_range_i64"});
  expect_opencl_compiler_support_matches_kernel_registry(tile);
}

TEST(GfxOpenClSourceArtifactsTest, GatherI64ArtifactsCarryLinearDimsMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto indices = param(ov::element::i64, ov::Shape{2});
  const auto gather = std::make_shared<ov::op::v8::Gather>(
      data, indices, i64_const(ov::Shape{}, {1}));
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 4, GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      2, // outer product before axis
      4, // inner product after axis
      3, // gathered axis extent
      2, // flattened indices count
  };

  expect_opencl_artifact(
      gather, GfxKernelStageFamily::GatherScatter,
      "opencl/baseline/gather_i64_f32", "gfx_opencl_baseline_gather_i64_f32",
      /*arg_count=*/8,
      /*direct_input_count=*/2, scalar_args, {0, 1}, static_u32_scalars);
  expect_opencl_compiler_support_matches_kernel_registry(gather);
}

TEST(GfxOpenClSourceArtifactsTest,
     GatherI32ArtifactsNormalizeNegativeAxisInMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto indices = param(ov::element::i32, ov::Shape{2});
  const auto gather = std::make_shared<ov::op::v8::Gather>(
      data, indices, i64_const(ov::Shape{}, {-1}));
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 4, GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      6, // outer product before axis
      1, // inner product after axis
      4, // gathered axis extent
      2, // flattened indices count
  };

  expect_opencl_artifact(
      gather, GfxKernelStageFamily::GatherScatter,
      "opencl/baseline/gather_i32_f32", "gfx_opencl_baseline_gather_i32_f32",
      /*arg_count=*/8,
      /*direct_input_count=*/2, scalar_args, {0, 1}, static_u32_scalars);
  expect_opencl_source_excludes(gather,
                                {"long", "__global const long*",
                                 "gfx_opencl_baseline_gather_i64_f32",
                                 "gfx_opencl_baseline_gather_elements_i64_f32",
                                 "gfx_opencl_baseline_gather_nd_i64_f32"});
  expect_opencl_compiler_support_matches_kernel_registry(gather);
}

TEST(GfxOpenClSourceArtifactsTest,
     GatherElementsI64ArtifactsCarryRankAxisShapeAndStrideMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto indices = param(ov::element::i64, ov::Shape{2, 2, 4});
  const auto gather =
      std::make_shared<ov::op::v6::GatherElements>(data, indices, 1);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 18,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      3,           // rank
      1,           // axis
      2,  2, 4, 1, // output dims padded to rank 4
      8,  4, 1, 1, // output strides padded to rank 4
      2,  3, 4, 1, // data dims padded to rank 4
      12, 4, 1, 1, // data strides padded to rank 4
  };

  expect_opencl_artifact(gather, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/gather_elements_i64_f32",
                         "gfx_opencl_baseline_gather_elements_i64_f32",
                         /*arg_count=*/22,
                         /*direct_input_count=*/2, scalar_args, {0, 1},
                         static_u32_scalars);
  expect_opencl_compiler_support_matches_kernel_registry(gather);
}

TEST(GfxOpenClSourceArtifactsTest,
     GatherElementsI32ArtifactsNormalizeNegativeAxisInMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto indices = param(ov::element::i32, ov::Shape{2, 3, 2});
  const auto gather =
      std::make_shared<ov::op::v6::GatherElements>(data, indices, -1);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 18,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      3,           // rank
      2,           // axis
      2,  3, 2, 1, // output dims padded to rank 4
      6,  2, 1, 1, // output strides padded to rank 4
      2,  3, 4, 1, // data dims padded to rank 4
      12, 4, 1, 1, // data strides padded to rank 4
  };

  expect_opencl_artifact(gather, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/gather_elements_i32_f32",
                         "gfx_opencl_baseline_gather_elements_i32_f32",
                         /*arg_count=*/22,
                         /*direct_input_count=*/2, scalar_args, {0, 1},
                         static_u32_scalars);
  expect_opencl_source_excludes(gather,
                                {"long", "__global const long*",
                                 "gfx_opencl_baseline_gather_i64_f32",
                                 "gfx_opencl_baseline_gather_elements_i64_f32",
                                 "gfx_opencl_baseline_gather_nd_i64_f32"});
  expect_opencl_compiler_support_matches_kernel_registry(gather);
}

TEST(GfxOpenClSourceArtifactsTest,
     GatherNDI64ArtifactsCarryIndexDepthSliceAndStrideMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto indices = param(ov::element::i64, ov::Shape{2, 2});
  const auto gather = std::make_shared<ov::op::v8::GatherND>(data, indices);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 11,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      2,           // index depth
      1,           // slice rank
      4,           // flattened slice size
      2,  3, 4, 1, // data dims padded to rank 4
      12, 4, 1, 1, // data strides padded to rank 4
  };

  expect_opencl_artifact(gather, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/gather_nd_i64_f32",
                         "gfx_opencl_baseline_gather_nd_i64_f32",
                         /*arg_count=*/15,
                         /*direct_input_count=*/2, scalar_args, {0, 1},
                         static_u32_scalars);
  expect_opencl_compiler_support_matches_kernel_registry(gather);
}

TEST(GfxOpenClSourceArtifactsTest, GatherNDI32ArtifactsCarryFullSliceMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto indices = param(ov::element::i32, ov::Shape{2, 1});
  const auto gather = std::make_shared<ov::op::v8::GatherND>(data, indices);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 11,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      1,           // index depth
      2,           // slice rank
      12,          // flattened slice size
      2,  3, 4, 1, // data dims padded to rank 4
      12, 4, 1, 1, // data strides padded to rank 4
  };

  expect_opencl_artifact(gather, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/gather_nd_i32_f32",
                         "gfx_opencl_baseline_gather_nd_i32_f32",
                         /*arg_count=*/15,
                         /*direct_input_count=*/2, scalar_args, {0, 1},
                         static_u32_scalars);
  expect_opencl_source_excludes(gather,
                                {"long", "__global const long*",
                                 "gfx_opencl_baseline_gather_i64_f32",
                                 "gfx_opencl_baseline_gather_elements_i64_f32",
                                 "gfx_opencl_baseline_gather_nd_i64_f32"});
  expect_opencl_compiler_support_matches_kernel_registry(gather);
}

TEST(GfxOpenClSourceArtifactsTest,
     ScatterUpdateI64ArtifactsCarryAxisIndexAndUpdateMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto indices = param(ov::element::i64, ov::Shape{2});
  const auto updates = param(ov::element::f32, ov::Shape{2, 2, 4});
  const auto scatter = std::make_shared<ov::op::v3::ScatterUpdate>(
      data, indices, updates, i64_const(ov::Shape{}, {1}));
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 28,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      3,                    // data rank
      1,                    // indices rank
      3,                    // updates rank
      1,                    // axis
      2,                    // flattened indices count
      2,  3, 4, 1,          // data dims padded to rank 4
      12, 4, 1, 1,          // data strides padded to rank 4
      2,  1, 1, 1,          // indices dims padded to rank 4
      1,  1, 1, 1,          // indices strides padded to rank 4
      8,  4, 1, 1, 1, 1, 1, // update strides padded to rank 7
  };

  expect_opencl_artifact(scatter, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/scatter_update_i64_f32",
                         "gfx_opencl_baseline_scatter_update_i64_f32",
                         /*arg_count=*/33,
                         /*direct_input_count=*/3, scalar_args, {0, 1, 2},
                         static_u32_scalars);
  expect_opencl_compiler_support_matches_kernel_registry(scatter);
}

TEST(GfxOpenClSourceArtifactsTest,
     ScatterElementsI32ArtifactsCarryAxisShapeAndStrideMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto indices = param(ov::element::i32, ov::Shape{2, 2, 4});
  const auto updates = param(ov::element::f32, ov::Shape{2, 2, 4});
  const auto scatter = std::make_shared<ov::op::v3::ScatterElementsUpdate>(
      data, indices, updates, i64_const(ov::Shape{}, {1}));
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 19,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      3,           // rank
      1,           // axis
      16,          // flattened update count
      2,  2, 4, 1, // update dims padded to rank 4
      8,  4, 1, 1, // update strides padded to rank 4
      2,  3, 4, 1, // data dims padded to rank 4
      12, 4, 1, 1, // data strides padded to rank 4
  };

  expect_opencl_artifact(scatter, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/scatter_elements_i32_f32",
                         "gfx_opencl_baseline_scatter_elements_i32_f32",
                         /*arg_count=*/24,
                         /*direct_input_count=*/3, scalar_args, {0, 1, 2},
                         static_u32_scalars);
  expect_opencl_compiler_support_matches_kernel_registry(scatter);
}

TEST(GfxOpenClSourceArtifactsTest,
     ScatterNDI64ArtifactsCarryIndexDepthSliceAndTupleMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto indices = param(ov::element::i64, ov::Shape{2, 2});
  const auto updates = param(ov::element::f32, ov::Shape{2, 4});
  const auto scatter =
      std::make_shared<ov::op::v3::ScatterNDUpdate>(data, indices, updates);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 11,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      2,           // index depth
      4,           // flattened slice size
      2,           // flattened tuple count
      2,  3, 4, 1, // data dims padded to rank 4
      12, 4, 1, 1, // data strides padded to rank 4
  };

  expect_opencl_artifact(scatter, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/scatter_nd_i64_f32",
                         "gfx_opencl_baseline_scatter_nd_i64_f32",
                         /*arg_count=*/16,
                         /*direct_input_count=*/3, scalar_args, {0, 1, 2},
                         static_u32_scalars);
  expect_opencl_compiler_support_matches_kernel_registry(scatter);
}

TEST(GfxOpenClSourceArtifactsTest, ShapeOfI32ArtifactsUseRuntimeShapeMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto shape_of =
      std::make_shared<ov::op::v3::ShapeOf>(data, ov::element::i32);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  for (uint32_t axis = 0; axis < 8; ++axis) {
    scalar_args.push_back(static_cast<GfxOpenClSourceScalarArg>(
        static_cast<uint32_t>(GfxOpenClSourceScalarArg::Input0Dim0) + axis));
  }

  expect_opencl_artifact(shape_of, GfxKernelStageFamily::GatherScatter,
                         "opencl/generated/shapeof_i32",
                         "gfx_opencl_generated_shapeof_i32",
                         /*arg_count=*/11,
                         /*direct_input_count=*/1, scalar_args, {0});
  const auto unit = make_opencl_kernel_registry(
                        BackendTarget::from_backend(GpuBackend::OpenCL))
                        .resolve(LoweringRouteKind::GeneratedKernel,
                                 "opencl/generated/shapeof_i32");
  ASSERT_TRUE(unit.valid());
  EXPECT_EQ(unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(unit.op_family(), "ShapeOf");
  expect_opencl_compiler_support_matches_kernel_registry(shape_of);
}

TEST(GfxOpenClSourceArtifactsTest,
     ShapeOfI64ArtifactsUseSameManifestAbiWithI64OutputKernel) {
  const auto data = param(ov::element::f32, ov::Shape{5, 6});
  const auto shape_of =
      std::make_shared<ov::op::v3::ShapeOf>(data, ov::element::i64);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  for (uint32_t axis = 0; axis < 8; ++axis) {
    scalar_args.push_back(static_cast<GfxOpenClSourceScalarArg>(
        static_cast<uint32_t>(GfxOpenClSourceScalarArg::Input0Dim0) + axis));
  }

  expect_opencl_artifact(shape_of, GfxKernelStageFamily::GatherScatter,
                         "opencl/generated/shapeof_i64",
                         "gfx_opencl_generated_shapeof_i64",
                         /*arg_count=*/11,
                         /*direct_input_count=*/1, scalar_args, {0});
  const auto unit = make_opencl_kernel_registry(
                        BackendTarget::from_backend(GpuBackend::OpenCL))
                        .resolve(LoweringRouteKind::GeneratedKernel,
                                 "opencl/generated/shapeof_i64");
  ASSERT_TRUE(unit.valid());
  EXPECT_EQ(unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(unit.op_family(), "ShapeOf");
  expect_opencl_compiler_support_matches_kernel_registry(shape_of);
}

TEST(GfxOpenClSourceArtifactsTest,
     DynamicShapeOfKeepsOpenClArtifactWhenInputDimsAreRuntimeKnown) {
  auto data = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 64});
  const auto shape_of =
      std::make_shared<ov::op::v3::ShapeOf>(data, ov::element::i64);
  auto artifact = resolve_gfx_opencl_source_artifact(shape_of);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->stage_manifest.stage_family,
            GfxKernelStageFamily::GatherScatter);
  EXPECT_EQ(artifact->artifact_ref.source_id, "opencl/generated/shapeof_i64");
  EXPECT_EQ(artifact->source.find("__global long*"), std::string::npos);
  expect_opencl_compiler_support_matches_kernel_registry(shape_of);
}

TEST(GfxOpenClSourceArtifactsTest,
     DynamicF16ConcatUsesRuntimeAxisLengthsUnderSourceManifest) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  const auto concat =
      std::make_shared<ov::op::v0::Concat>(ov::OutputVector{lhs, rhs}, 1);
  const std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::StaticU32, GfxOpenClSourceScalarArg::Input0Dim1,
      GfxOpenClSourceScalarArg::Input1Dim1};

  expect_opencl_artifact(concat, GfxKernelStageFamily::ConcatSplit,
                         "opencl/generated/concat2_f16_dynamic",
                         "gfx_opencl_generated_concat2_f16",
                         /*arg_count=*/7,
                         /*direct_input_count=*/2, scalar_args, {0, 1}, {4});
  expect_opencl_compiler_support_matches_kernel_registry(concat);
}

TEST(GfxOpenClSourceArtifactsTest,
     DynamicF16ThreeInputConcatUsesRuntimeAxisLengthsUnderSourceManifest) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  auto mid = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  const auto concat =
      std::make_shared<ov::op::v0::Concat>(ov::OutputVector{lhs, mid, rhs}, 1);
  const std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::StaticU32, GfxOpenClSourceScalarArg::Input0Dim1,
      GfxOpenClSourceScalarArg::Input1Dim1,
      GfxOpenClSourceScalarArg::Input2Dim1};

  expect_opencl_artifact(concat, GfxKernelStageFamily::ConcatSplit,
                         "opencl/generated/concat3_f16_dynamic",
                         "gfx_opencl_generated_concat3_f16",
                         /*arg_count=*/9,
                         /*direct_input_count=*/3, scalar_args, {0, 1, 2}, {4});
  expect_opencl_compiler_support_matches_kernel_registry(concat);
}

TEST(GfxOpenClSourceArtifactsTest,
     DynamicF16BroadcastBindsTargetShapeAndInputRuntimeDims) {
  auto data = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  auto target_shape = std::make_shared<ov::op::v0::Parameter>(
      ov::element::i64, ov::PartialShape{3});
  const auto broadcast = std::make_shared<ov::op::v3::Broadcast>(
      data, target_shape, ov::op::BroadcastType::BIDIRECTIONAL);
  const std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::StaticU32,
      GfxOpenClSourceScalarArg::StaticU32,
      GfxOpenClSourceScalarArg::Input0Dim0,
      GfxOpenClSourceScalarArg::Input0Dim1,
      GfxOpenClSourceScalarArg::Input0Dim2,
      GfxOpenClSourceScalarArg::Input0Dim3};

  expect_opencl_artifact(broadcast, GfxKernelStageFamily::Layout,
                         "opencl/baseline/broadcast_f16_i64shape_dynamic",
                         "gfx_opencl_baseline_broadcast_f16_i64shape",
                         /*arg_count=*/10,
                         /*direct_input_count=*/2, scalar_args, {0, 1}, {3, 3});
  EXPECT_EQ(resolve_gfx_opencl_source_artifact(broadcast)->source.find(
                "__global long*"),
            std::string::npos);
  expect_opencl_compiler_support_matches_kernel_registry(broadcast);
}

TEST(GfxOpenClSourceArtifactsTest,
     DynamicF16StridedSliceUsesRuntimeInputAndOutputDims) {
  auto data = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  auto end = std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
                                                     ov::PartialShape{3});
  const auto begin = i64_const(ov::Shape{3}, {0, 0, 0});
  const auto strides = i64_const(ov::Shape{3}, {1, 1, 1});
  const auto slice = std::make_shared<ov::op::v1::StridedSlice>(
      data, begin, end, strides, std::vector<int64_t>{0, 0, 0},
      std::vector<int64_t>{0, 0, 0});
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::StaticU32,
      GfxOpenClSourceScalarArg::Output0Dim0,
      GfxOpenClSourceScalarArg::Output0Dim1,
      GfxOpenClSourceScalarArg::Output0Dim2,
      GfxOpenClSourceScalarArg::Output0Dim3,
      GfxOpenClSourceScalarArg::Input0Dim0,
      GfxOpenClSourceScalarArg::Input0Dim1,
      GfxOpenClSourceScalarArg::Input0Dim2,
      GfxOpenClSourceScalarArg::Input0Dim3};
  scalar_args.insert(scalar_args.end(), 8, GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {3, 0, 0, 0, 0, 1, 1, 1, 1};

  expect_opencl_artifact(
      slice, GfxKernelStageFamily::GatherScatter,
      "opencl/baseline/slice_f16_dynamic", "gfx_opencl_baseline_slice_f16",
      /*arg_count=*/21,
      /*direct_input_count=*/2, scalar_args, {0, 2}, static_u32_scalars);
  expect_opencl_compiler_support_matches_kernel_registry(slice);
}

TEST(GfxOpenClSourceArtifactsTest,
     DynamicF16StridedSliceUsesRuntimeBeginEndStepsAndDims) {
  auto data = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  auto begin = std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
                                                       ov::PartialShape{3});
  auto end = std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
                                                     ov::PartialShape{3});
  auto strides = std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
                                                         ov::PartialShape{3});
  const auto slice = std::make_shared<ov::op::v1::StridedSlice>(
      data, begin, end, strides, std::vector<int64_t>{0, 0, 0},
      std::vector<int64_t>{0, 0, 0});
  const std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::StaticU32,
      GfxOpenClSourceScalarArg::Output0Dim0,
      GfxOpenClSourceScalarArg::Output0Dim1,
      GfxOpenClSourceScalarArg::Output0Dim2,
      GfxOpenClSourceScalarArg::Output0Dim3,
      GfxOpenClSourceScalarArg::Input0Dim0,
      GfxOpenClSourceScalarArg::Input0Dim1,
      GfxOpenClSourceScalarArg::Input0Dim2,
      GfxOpenClSourceScalarArg::Input0Dim3};

  expect_opencl_artifact(slice, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/strided_slice_f16_dynamic_runtime",
                         "gfx_opencl_baseline_slice_v8_f16",
                         /*arg_count=*/15,
                         /*direct_input_count=*/4, scalar_args, {0, 1, 2, 3},
                         {3});
  expect_opencl_compiler_support_matches_kernel_registry(slice);
}

TEST(GfxOpenClSourceArtifactsTest,
     DynamicF16SliceUsesRuntimeStartsEndsStepsAndDims) {
  auto data = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  auto starts = std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
                                                        ov::PartialShape{3});
  auto ends = std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
                                                      ov::PartialShape{3});
  auto steps = std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
                                                       ov::PartialShape{3});
  const auto slice =
      std::make_shared<ov::op::v8::Slice>(data, starts, ends, steps);
  const std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::StaticU32,
      GfxOpenClSourceScalarArg::Output0Dim0,
      GfxOpenClSourceScalarArg::Output0Dim1,
      GfxOpenClSourceScalarArg::Output0Dim2,
      GfxOpenClSourceScalarArg::Output0Dim3,
      GfxOpenClSourceScalarArg::Input0Dim0,
      GfxOpenClSourceScalarArg::Input0Dim1,
      GfxOpenClSourceScalarArg::Input0Dim2,
      GfxOpenClSourceScalarArg::Input0Dim3};

  expect_opencl_artifact(slice, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/slice_v8_f16_dynamic",
                         "gfx_opencl_baseline_slice_v8_f16",
                         /*arg_count=*/15,
                         /*direct_input_count=*/4, scalar_args, {0, 1, 2, 3},
                         {3});
  EXPECT_EQ(
      resolve_gfx_opencl_source_artifact(slice)->source.find("__global long*"),
      std::string::npos);
  expect_opencl_compiler_support_matches_kernel_registry(slice);
}

TEST(GfxOpenClSourceArtifactsTest,
     DynamicI64UnitRangeUsesManifestAbiAndRegisteredKernelUnit) {
  auto stop = std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
                                                      ov::PartialShape{});
  const auto start = i64_const(ov::Shape{}, {0});
  const auto step = i64_const(ov::Shape{}, {1});
  const auto range =
      std::make_shared<ov::op::v4::Range>(start, stop, step, ov::element::i64);

  expect_opencl_artifact(range, GfxKernelStageFamily::GatherScatter,
                         "opencl/generated/range_i64_unit_dynamic",
                         "gfx_opencl_generated_range_i64_unit",
                         /*arg_count=*/3,
                         /*direct_input_count=*/1,
                         {GfxOpenClSourceScalarArg::ElementCount}, {1});
  expect_opencl_compiler_supports_generated_unit(
      range, "opencl/generated/range_i64_unit_dynamic");
  expect_opencl_range_compiler_contract(
      range, "opencl/generated/range_i64_unit_dynamic",
      "gfx_opencl_generated_range_i64_unit",
      /*expected_abi_arg_count=*/3, ov::ParameterVector{stop});
  expect_opencl_compiler_support_matches_kernel_registry(range);
}

TEST(GfxOpenClSourceArtifactsTest, BinaryConcatArtifactsUseStaticAxisMetadata) {
  const auto lhs = param(ov::element::f32, ov::Shape{1, 2, 3});
  const auto rhs = param(ov::element::f32, ov::Shape{1, 4, 3});
  const auto concat =
      std::make_shared<ov::op::v0::Concat>(ov::OutputVector{lhs, rhs}, 1);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  const std::vector<uint32_t> static_u32_scalars = {
      6,    // output axis extent
      3,    // inner contiguous block
      0, 2, // input 0 offset/axis extent
      2, 4, // input 1 offset/axis extent
  };

  expect_opencl_artifact(concat, GfxKernelStageFamily::ConcatSplit,
                         "opencl/generated/concat2_f32",
                         "gfx_opencl_generated_concat2_f32",
                         /*arg_count=*/4,
                         /*direct_input_count=*/2, scalar_args, {0, 1}, {});
  auto artifact = resolve_gfx_opencl_source_artifact(concat);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->source_static_u32_scalars, static_u32_scalars);
  EXPECT_NE(artifact->source.find("const uint axis_total = 6u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint inner = 3u;"), std::string::npos);
  EXPECT_NE(
      artifact->source.find("chunk_axis_idx >= 2u && chunk_axis_idx < 6u"),
      std::string::npos);
  EXPECT_NE(artifact->source.find("outer_idx * 4u + src_axis_idx"),
            std::string::npos);
  expect_opencl_source_excludes(
      concat, {"long", "__global long*", "gfx_opencl_generated_shapeof_i64"});
  expect_opencl_compiler_support_matches_kernel_registry(concat);
}

TEST(GfxOpenClSourceArtifactsTest,
     ThreeInputConcatArtifactsUseTheSameStaticAxisMetadata) {
  const auto src0 = param(ov::element::f32, ov::Shape{1, 2, 3});
  const auto src1 = param(ov::element::f32, ov::Shape{1, 4, 3});
  const auto src2 = param(ov::element::f32, ov::Shape{1, 1, 3});
  const auto concat = std::make_shared<ov::op::v0::Concat>(
      ov::OutputVector{src0, src1, src2}, 1);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  const std::vector<uint32_t> static_u32_scalars = {
      7,    // output axis extent
      3,    // inner contiguous block
      0, 2, // input 0 offset/axis extent
      2, 4, // input 1 offset/axis extent
      6, 1, // input 2 offset/axis extent
  };

  expect_opencl_artifact(concat, GfxKernelStageFamily::ConcatSplit,
                         "opencl/generated/concat3_f32",
                         "gfx_opencl_generated_concat3_f32",
                         /*arg_count=*/5,
                         /*direct_input_count=*/3, scalar_args, {0, 1, 2}, {});
  auto artifact = resolve_gfx_opencl_source_artifact(concat);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->source_static_u32_scalars, static_u32_scalars);
  EXPECT_NE(artifact->source.find("const uint axis_total = 7u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint inner = 3u;"), std::string::npos);
  EXPECT_NE(
      artifact->source.find("chunk_axis_idx >= 6u && chunk_axis_idx < 7u"),
      std::string::npos);
  EXPECT_NE(artifact->source.find("outer_idx * 1u + src_axis_idx"),
            std::string::npos);
  expect_opencl_source_excludes(
      concat, {"long", "__global long*", "gfx_opencl_generated_shapeof_i64"});
  expect_opencl_compiler_support_matches_kernel_registry(concat);
}

TEST(GfxOpenClSourceArtifactsTest,
     StaticConcatArtifactsGenerateSharedLayerThirtyInputSource) {
  ov::OutputVector inputs;
  std::vector<std::shared_ptr<ov::op::v0::Parameter>> params;
  params.reserve(30);
  for (size_t input_idx = 0; input_idx < 30; ++input_idx) {
    params.push_back(param(ov::element::f32, ov::Shape{1, 1, 2}));
    inputs.push_back(params.back());
  }
  const auto concat = std::make_shared<ov::op::v0::Concat>(inputs, 1);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};

  expect_opencl_artifact(concat, GfxKernelStageFamily::ConcatSplit,
                         "opencl/generated/concat30_f32",
                         "gfx_opencl_generated_concat30_f32",
                         /*arg_count=*/32,
                         /*direct_input_count=*/30, scalar_args, {}, {},
                         /*direct_output_count=*/1,
                         /*input_chunk_size=*/1);
  auto artifact = resolve_gfx_opencl_source_artifact(concat);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_NE(artifact->source.find("__global const float* src29"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint axis_total = 30u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint inner = 2u;"), std::string::npos);
  EXPECT_NE(
      artifact->source.find("chunk_axis_idx >= 29u && chunk_axis_idx < 30u"),
      std::string::npos);
  expect_opencl_source_excludes(concat, {"__global long*",
                                         "gfx_opencl_generated_shapeof_i64",
                                         "gfx_opencl_generated_concat4_f32",
                                         "__global const float* src30"});
  expect_opencl_compiler_support_matches_kernel_registry(concat);
}

TEST(GfxOpenClSourceArtifactsTest,
     StaticConcatArtifactsPreplanArtifactSizedChunksForLargeConcat) {
  ov::OutputVector inputs;
  std::vector<std::shared_ptr<ov::op::v0::Parameter>> params;
  params.reserve(30);
  for (size_t input_idx = 0; input_idx < 30; ++input_idx) {
    params.push_back(param(ov::element::f32, ov::Shape{1, 1, 2}));
    inputs.push_back(params.back());
  }
  const auto concat = std::make_shared<ov::op::v0::Concat>(inputs, 1);
  auto base = resolve_gfx_opencl_source_artifact(concat);
  ASSERT_TRUE(base.has_value());
  ASSERT_EQ(base->input_chunk_size, 1u);
  ASSERT_EQ(base->planned_chunks.size(), 30u);
  GfxOpenClSourceArtifactPayload payload(*base);
  EXPECT_TRUE(payload.valid());
  auto missing_plan = *base;
  missing_plan.planned_chunks.clear();
  EXPECT_FALSE(GfxOpenClSourceArtifactPayload(std::move(missing_plan)).valid());
  EXPECT_FALSE(make_gfx_opencl_concat_chunk_source_artifact(*base, 0, 4));

  ASSERT_TRUE(base->planned_chunks[0].artifact);
  EXPECT_EQ(base->planned_chunks[0].binding_begin, 0u);
  EXPECT_EQ(base->planned_chunks[0].binding_count, 1u);
  EXPECT_EQ(base->planned_chunks[0].binding_role,
            GfxOpenClSourceChunkBindingRole::DirectInputs);
  EXPECT_EQ(base->planned_chunks[0].element_count_multiplier, 1u);
  EXPECT_EQ(base->planned_chunks[0].element_count_divisor, 30u);
  const auto &chunk0 = *base->planned_chunks[0].artifact;
  EXPECT_EQ(chunk0.artifact_ref.entry_point,
            "gfx_opencl_generated_concat1_f32");
  EXPECT_EQ(chunk0.arg_count, 3u);
  EXPECT_EQ(chunk0.direct_input_count, 1u);
  EXPECT_EQ(chunk0.direct_output_count, 1u);
  EXPECT_EQ(chunk0.scalar_args, std::vector<GfxOpenClSourceScalarArg>{
                                    GfxOpenClSourceScalarArg::ElementCount});
  EXPECT_EQ(chunk0.direct_input_indices, std::vector<size_t>({0}));
  EXPECT_TRUE(chunk0.static_u32_scalars.empty());
  EXPECT_NE(chunk0.source.find("__global const float* src0"),
            std::string::npos);
  EXPECT_NE(chunk0.source.find("chunk_axis_idx >= 0u && chunk_axis_idx < 1u"),
            std::string::npos);
  EXPECT_EQ(chunk0.source.find("__global const float* src1"),
            std::string::npos);

  ASSERT_TRUE(base->planned_chunks[28].artifact);
  EXPECT_EQ(base->planned_chunks[28].binding_begin, 28u);
  EXPECT_EQ(base->planned_chunks[28].binding_count, 1u);
  EXPECT_EQ(base->planned_chunks[28].binding_role,
            GfxOpenClSourceChunkBindingRole::DirectInputs);
  EXPECT_EQ(base->planned_chunks[28].element_count_multiplier, 1u);
  EXPECT_EQ(base->planned_chunks[28].element_count_divisor, 30u);
  const auto &chunk_tail = *base->planned_chunks[28].artifact;
  EXPECT_EQ(chunk_tail.artifact_ref.entry_point,
            "gfx_opencl_generated_concat1_f32");
  EXPECT_EQ(chunk_tail.arg_count, 3u);
  EXPECT_EQ(chunk_tail.direct_input_count, 1u);
  EXPECT_EQ(chunk_tail.direct_input_indices, std::vector<size_t>({28}));
  EXPECT_NE(
      chunk_tail.source.find("chunk_axis_idx >= 0u && chunk_axis_idx < 1u"),
      std::string::npos);
  EXPECT_EQ(chunk_tail.source.find("__global const float* src1"),
            std::string::npos);

  ASSERT_TRUE(base->planned_chunks[29].artifact);
  EXPECT_EQ(base->planned_chunks[29].binding_begin, 29u);
  EXPECT_EQ(base->planned_chunks[29].binding_count, 1u);
  const auto &single_tail = *base->planned_chunks[29].artifact;
  EXPECT_EQ(single_tail.artifact_ref.entry_point,
            "gfx_opencl_generated_concat1_f32");
  EXPECT_EQ(single_tail.arg_count, 3u);
  EXPECT_EQ(single_tail.direct_input_count, 1u);
  EXPECT_EQ(single_tail.direct_input_indices, std::vector<size_t>({29}));
  EXPECT_NE(
      single_tail.source.find("chunk_axis_idx >= 0u && chunk_axis_idx < 1u"),
      std::string::npos);
  EXPECT_EQ(single_tail.source.find("__global const float* src1"),
            std::string::npos);
}

TEST(GfxOpenClSourceArtifactsTest,
     StaticF16ConcatArtifactsGenerateFiveInputSource) {
  const auto src0 = param(ov::element::f16, ov::Shape{1, 1, 2});
  const auto src1 = param(ov::element::f16, ov::Shape{1, 2, 2});
  const auto src2 = param(ov::element::f16, ov::Shape{1, 3, 2});
  const auto src3 = param(ov::element::f16, ov::Shape{1, 4, 2});
  const auto src4 = param(ov::element::f16, ov::Shape{1, 5, 2});
  const auto concat = std::make_shared<ov::op::v0::Concat>(
      ov::OutputVector{src0, src1, src2, src3, src4}, 1);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};

  expect_opencl_artifact(concat, GfxKernelStageFamily::ConcatSplit,
                         "opencl/generated/concat5_f16",
                         "gfx_opencl_generated_concat5_f16",
                         /*arg_count=*/7,
                         /*direct_input_count=*/5, scalar_args, {}, {},
                         /*direct_output_count=*/1,
                         /*input_chunk_size=*/4);
  auto artifact = resolve_gfx_opencl_source_artifact(concat);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_NE(artifact->source.find("__global const uint* src4"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint axis_total = 15u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint inner = 2u;"), std::string::npos);
  EXPECT_NE(
      artifact->source.find("chunk_axis_idx0 >= 10u && chunk_axis_idx0 < 15u"),
      std::string::npos);
  EXPECT_NE(
      artifact->source.find("GFX_STORE_F16_PAIR(dst, dst_elem >> 1u, lo, hi);"),
      std::string::npos);
  EXPECT_EQ(artifact->source.find("chunk_axis_idx1"), std::string::npos);
  expect_opencl_source_excludes(concat,
                                {"__global long*", "(long)", "__global half",
                                 "gfx_opencl_generated_concat4_f16",
                                 "__global const uint* src5"});
  expect_opencl_compiler_support_matches_kernel_registry(concat);
}

TEST(GfxOpenClSourceArtifactsTest,
     StaticF16ConcatArtifactsPreplanFourInputChunksFromArtifactContract) {
  const auto src0 = param(ov::element::f16, ov::Shape{1, 1, 2});
  const auto src1 = param(ov::element::f16, ov::Shape{1, 2, 2});
  const auto src2 = param(ov::element::f16, ov::Shape{1, 3, 2});
  const auto src3 = param(ov::element::f16, ov::Shape{1, 4, 2});
  const auto src4 = param(ov::element::f16, ov::Shape{1, 5, 2});
  const auto concat = std::make_shared<ov::op::v0::Concat>(
      ov::OutputVector{src0, src1, src2, src3, src4}, 1);
  auto base = resolve_gfx_opencl_source_artifact(concat);
  ASSERT_TRUE(base.has_value());
  ASSERT_EQ(base->input_chunk_size, 4u);
  ASSERT_EQ(base->planned_chunks.size(), 2u);

  ASSERT_TRUE(base->planned_chunks[0].artifact);
  EXPECT_EQ(base->planned_chunks[0].binding_begin, 0u);
  EXPECT_EQ(base->planned_chunks[0].binding_count, 4u);
  EXPECT_EQ(base->planned_chunks[0].binding_role,
            GfxOpenClSourceChunkBindingRole::DirectInputs);
  EXPECT_EQ(base->planned_chunks[0].element_count_multiplier, 10u);
  EXPECT_EQ(base->planned_chunks[0].element_count_divisor, 15u);
  const auto &chunk0 = *base->planned_chunks[0].artifact;
  EXPECT_EQ(chunk0.artifact_ref.entry_point,
            "gfx_opencl_generated_concat4_f16");
  EXPECT_EQ(chunk0.arg_count, 6u);
  EXPECT_EQ(chunk0.direct_input_count, 4u);
  EXPECT_EQ(chunk0.direct_input_indices, std::vector<size_t>({0, 1, 2, 3}));
  EXPECT_NE(chunk0.source.find("__global const uint* src3"), std::string::npos);
  EXPECT_EQ(chunk0.source.find("__global const uint* src4"), std::string::npos);

  ASSERT_TRUE(base->planned_chunks[1].artifact);
  EXPECT_EQ(base->planned_chunks[1].binding_begin, 4u);
  EXPECT_EQ(base->planned_chunks[1].binding_count, 1u);
  EXPECT_EQ(base->planned_chunks[1].binding_role,
            GfxOpenClSourceChunkBindingRole::DirectInputs);
  EXPECT_EQ(base->planned_chunks[1].element_count_multiplier, 5u);
  EXPECT_EQ(base->planned_chunks[1].element_count_divisor, 15u);
  const auto &chunk_tail = *base->planned_chunks[1].artifact;
  EXPECT_EQ(chunk_tail.artifact_ref.entry_point,
            "gfx_opencl_generated_concat1_f16");
  EXPECT_EQ(chunk_tail.arg_count, 3u);
  EXPECT_EQ(chunk_tail.direct_input_count, 1u);
  EXPECT_EQ(chunk_tail.direct_input_indices, std::vector<size_t>({4}));
  EXPECT_EQ(chunk_tail.source.find("__global const uint* src1"),
            std::string::npos);
}

TEST(GfxOpenClSourceArtifactsTest,
     EqualSplitArtifactsUseMultiOutputStaticAxisMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{1, 6, 2});
  const auto split =
      std::make_shared<ov::op::v1::Split>(data, i64_const(ov::Shape{}, {1}), 3);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  expect_opencl_artifact(split, GfxKernelStageFamily::ConcatSplit,
                         "opencl/generated/split3_f32",
                         "gfx_opencl_generated_split3_f32",
                         /*arg_count=*/5,
                         /*direct_input_count=*/1, scalar_args, {0}, {},
                         /*direct_output_count=*/3);
  auto artifact = resolve_gfx_opencl_source_artifact(split);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->element_count_source,
            GfxOpenClSourceElementCountSource::Input0);
  EXPECT_NE(artifact->source.find("const uint axis_total = 6u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint inner = 2u;"), std::string::npos);
  EXPECT_NE(artifact->source.find(", 4u, 2u);"), std::string::npos);
  expect_opencl_source_excludes(
      split, {"long", "__global long*", "gfx_opencl_generated_shapeof_i64"});
  expect_opencl_compiler_support_matches_kernel_registry(split);
}

TEST(GfxOpenClSourceArtifactsTest,
     F16EqualSplitArtifactsReuseMultiOutputStaticAxisMetadata) {
  const auto data = param(ov::element::f16, ov::Shape{1, 6, 2});
  const auto split =
      std::make_shared<ov::op::v1::Split>(data, i64_const(ov::Shape{}, {1}), 3);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  expect_opencl_artifact(split, GfxKernelStageFamily::ConcatSplit,
                         "opencl/generated/split3_f16",
                         "gfx_opencl_generated_split3_f16",
                         /*arg_count=*/5,
                         /*direct_input_count=*/1, scalar_args, {0}, {},
                         /*direct_output_count=*/3);
  auto artifact = resolve_gfx_opencl_source_artifact(split);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->element_count_source,
            GfxOpenClSourceElementCountSource::Input0);
  EXPECT_NE(artifact->source.find("const uint axis_total = 6u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint inner = 2u;"), std::string::npos);
  EXPECT_NE(artifact->source.find(", 4u, 2u);"), std::string::npos);
  expect_opencl_source_excludes(split,
                                {"__global long*", "(long)", "__global half"});
  expect_opencl_compiler_support_matches_kernel_registry(split);
}

TEST(GfxOpenClSourceArtifactsTest,
     VariadicSplitArtifactsReuseMultiOutputStaticAxisMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{1, 7, 2});
  const auto split = std::make_shared<ov::op::v1::VariadicSplit>(
      data, i64_const(ov::Shape{}, {1}), i64_const(ov::Shape{3}, {2, 3, 2}));
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  expect_opencl_artifact(split, GfxKernelStageFamily::ConcatSplit,
                         "opencl/generated/split3_f32",
                         "gfx_opencl_generated_split3_f32",
                         /*arg_count=*/5,
                         /*direct_input_count=*/1, scalar_args, {0}, {},
                         /*direct_output_count=*/3);
  auto artifact = resolve_gfx_opencl_source_artifact(split);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->element_count_source,
            GfxOpenClSourceElementCountSource::Input0);
  EXPECT_NE(artifact->source.find("const uint axis_total = 7u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint inner = 2u;"), std::string::npos);
  EXPECT_NE(artifact->source.find(", 5u, 2u);"), std::string::npos);
  expect_opencl_source_excludes(
      split, {"long", "__global long*", "gfx_opencl_generated_shapeof_i64"});
  expect_opencl_compiler_support_matches_kernel_registry(split);
}

TEST(GfxOpenClSourceArtifactsTest,
     F16VariadicSplitArtifactsReuseMultiOutputStaticAxisMetadata) {
  const auto data = param(ov::element::f16, ov::Shape{1, 7, 2});
  const auto split = std::make_shared<ov::op::v1::VariadicSplit>(
      data, i64_const(ov::Shape{}, {1}), i64_const(ov::Shape{3}, {2, 3, 2}));
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  expect_opencl_artifact(split, GfxKernelStageFamily::ConcatSplit,
                         "opencl/generated/split3_f16",
                         "gfx_opencl_generated_split3_f16",
                         /*arg_count=*/5,
                         /*direct_input_count=*/1, scalar_args, {0}, {},
                         /*direct_output_count=*/3);
  auto artifact = resolve_gfx_opencl_source_artifact(split);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->element_count_source,
            GfxOpenClSourceElementCountSource::Input0);
  EXPECT_NE(artifact->source.find("const uint axis_total = 7u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint inner = 2u;"), std::string::npos);
  EXPECT_NE(artifact->source.find(", 5u, 2u);"), std::string::npos);
  expect_opencl_source_excludes(split,
                                {"__global long*", "(long)", "__global half"});
  expect_opencl_compiler_support_matches_kernel_registry(split);
}

TEST(GfxOpenClSourceArtifactsTest,
     EqualSplitArtifactsGenerateSharedLayerThirtyOutputSource) {
  const auto data = param(ov::element::f32, ov::Shape{30, 30, 30, 30});
  const auto split = std::make_shared<ov::op::v1::Split>(
      data, i64_const(ov::Shape{}, {0}), 30);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};

  expect_opencl_artifact(split, GfxKernelStageFamily::ConcatSplit,
                         "opencl/generated/split30_f32",
                         "gfx_opencl_generated_split30_f32",
                         /*arg_count=*/32,
                         /*direct_input_count=*/1, scalar_args, {0}, {},
                         /*direct_output_count=*/30,
                         /*input_chunk_size=*/0,
                         /*output_chunk_size=*/4);
  auto artifact = resolve_gfx_opencl_source_artifact(split);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->element_count_source,
            GfxOpenClSourceElementCountSource::Input0);
  EXPECT_NE(artifact->source.find("__global float* dst29"), std::string::npos);
  EXPECT_NE(artifact->source.find("const uint axis_total = 30u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint inner = 27000u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find(", 29u, 1u);"), std::string::npos);
  expect_opencl_source_excludes(
      split, {"__global long*", "gfx_opencl_generated_shapeof_i64",
              "gfx_opencl_generated_split4_f32", "__global float* dst30"});
  expect_opencl_compiler_support_matches_kernel_registry(split);
}

TEST(GfxOpenClSourceArtifactsTest,
     EqualSplitArtifactsPreplanFourOutputChunksForLargeSplit) {
  const auto data = param(ov::element::f32, ov::Shape{30, 30, 30, 30});
  const auto split = std::make_shared<ov::op::v1::Split>(
      data, i64_const(ov::Shape{}, {0}), 30);
  auto base = resolve_gfx_opencl_source_artifact(split);
  ASSERT_TRUE(base.has_value());
  ASSERT_EQ(base->output_chunk_size, 4u);
  ASSERT_EQ(base->planned_chunks.size(), 8u);

  ASSERT_TRUE(base->planned_chunks[0].artifact);
  EXPECT_EQ(base->planned_chunks[0].binding_begin, 0u);
  EXPECT_EQ(base->planned_chunks[0].binding_count, 4u);
  EXPECT_EQ(base->planned_chunks[0].binding_role,
            GfxOpenClSourceChunkBindingRole::DirectOutputs);
  EXPECT_EQ(base->planned_chunks[0].element_count_multiplier, 1u);
  EXPECT_EQ(base->planned_chunks[0].element_count_divisor, 1u);
  const auto &chunk0 = *base->planned_chunks[0].artifact;
  EXPECT_EQ(chunk0.artifact_ref.entry_point, "gfx_opencl_generated_split4_f32");
  EXPECT_EQ(chunk0.arg_count, 6u);
  EXPECT_EQ(chunk0.direct_output_count, 4u);
  EXPECT_EQ(chunk0.scalar_args, std::vector<GfxOpenClSourceScalarArg>{
                                    GfxOpenClSourceScalarArg::ElementCount});
  EXPECT_TRUE(chunk0.static_u32_scalars.empty());
  EXPECT_NE(chunk0.source.find("__global float* dst3"), std::string::npos);
  EXPECT_NE(chunk0.source.find(", 3u, 1u);"), std::string::npos);
  EXPECT_EQ(chunk0.source.find("__global float* dst4"), std::string::npos);

  ASSERT_TRUE(base->planned_chunks[7].artifact);
  EXPECT_EQ(base->planned_chunks[7].binding_begin, 28u);
  EXPECT_EQ(base->planned_chunks[7].binding_count, 2u);
  EXPECT_EQ(base->planned_chunks[7].binding_role,
            GfxOpenClSourceChunkBindingRole::DirectOutputs);
  EXPECT_EQ(base->planned_chunks[7].element_count_multiplier, 1u);
  EXPECT_EQ(base->planned_chunks[7].element_count_divisor, 1u);
  const auto &chunk_tail = *base->planned_chunks[7].artifact;
  EXPECT_EQ(chunk_tail.artifact_ref.entry_point,
            "gfx_opencl_generated_split2_f32");
  EXPECT_EQ(chunk_tail.arg_count, 4u);
  EXPECT_EQ(chunk_tail.direct_output_count, 2u);
  EXPECT_NE(chunk_tail.source.find(", 28u, 1u);"), std::string::npos);
  EXPECT_NE(chunk_tail.source.find(", 29u, 1u);"), std::string::npos);
  EXPECT_EQ(chunk_tail.source.find("__global float* dst2"), std::string::npos);
}

TEST(GfxOpenClSourceArtifactsTest,
     SplitChunkBuilderRejectsRequestsOutsideArtifactContract) {
  const auto data = param(ov::element::f32, ov::Shape{30, 30, 30, 30});
  const auto split = std::make_shared<ov::op::v1::Split>(
      data, i64_const(ov::Shape{}, {0}), 30);
  auto base = resolve_gfx_opencl_source_artifact(split);
  ASSERT_TRUE(base.has_value());
  auto two_output_contract = *base;
  two_output_contract.output_chunk_size = 2;

  EXPECT_FALSE(
      make_gfx_opencl_split_chunk_source_artifact(two_output_contract, 0, 3));
  auto chunk0 =
      make_gfx_opencl_split_chunk_source_artifact(two_output_contract, 0, 2);
  ASSERT_TRUE(chunk0.has_value());
  EXPECT_EQ(chunk0->artifact_ref.entry_point,
            "gfx_opencl_generated_split2_f32");
  EXPECT_EQ(chunk0->direct_output_count, 2u);
}

TEST(GfxOpenClSourceArtifactsTest,
     F16VariadicSplitArtifactsGenerateFiveOutputSource) {
  const auto data = param(ov::element::f16, ov::Shape{1, 15, 2});
  const auto split = std::make_shared<ov::op::v1::VariadicSplit>(
      data, i64_const(ov::Shape{}, {1}),
      i64_const(ov::Shape{5}, {1, 2, 3, 4, 5}));
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};

  expect_opencl_artifact(split, GfxKernelStageFamily::ConcatSplit,
                         "opencl/generated/split5_f16",
                         "gfx_opencl_generated_split5_f16",
                         /*arg_count=*/7,
                         /*direct_input_count=*/1, scalar_args, {0}, {},
                         /*direct_output_count=*/5,
                         /*input_chunk_size=*/0,
                         /*output_chunk_size=*/4);
  auto artifact = resolve_gfx_opencl_source_artifact(split);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->element_count_source,
            GfxOpenClSourceElementCountSource::Input0);
  EXPECT_NE(artifact->source.find("__global uint* dst4"), std::string::npos);
  EXPECT_NE(artifact->source.find("const uint axis_total = 15u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint inner = 2u;"), std::string::npos);
  EXPECT_NE(artifact->source.find(", 10u, 5u);"), std::string::npos);
  expect_opencl_source_excludes(
      split, {"__global long*", "(long)", "__global half",
              "gfx_opencl_generated_split4_f16", "__global uint* dst5"});
  expect_opencl_compiler_support_matches_kernel_registry(split);
}

TEST(GfxOpenClSourceArtifactsTest,
     MissingOpenClArtifactsRejectEvenWhenMlirCoversThem) {
  const auto f32 = param(ov::element::f32, ov::Shape{2, 3});
  const auto i32 = param(ov::element::i32, ov::Shape{2, 3});
  const auto high_rank_lhs = param(ov::element::f32, ov::Shape{1, 1, 1, 1, 3});
  const auto high_rank_rhs = param(ov::element::f32, ov::Shape{3});

  const auto i32_abs = std::make_shared<ov::op::v0::Abs>(i32);
  const auto high_rank_broadcast_add =
      std::make_shared<ov::op::v1::Add>(high_rank_lhs, high_rank_rhs);
  const auto convert_to_f16 =
      std::make_shared<ov::op::v0::Convert>(f32, ov::element::f16);

  EXPECT_FALSE(resolve_gfx_opencl_source_artifact(i32_abs).has_value());
  EXPECT_FALSE(
      resolve_gfx_opencl_source_artifact(high_rank_broadcast_add).has_value());
  EXPECT_FALSE(resolve_gfx_opencl_source_artifact(convert_to_f16).has_value());

  EXPECT_TRUE(mlir_supports_node(i32_abs));
  EXPECT_TRUE(mlir_supports_node(high_rank_broadcast_add));
  EXPECT_TRUE(mlir_supports_node(convert_to_f16));
  EXPECT_FALSE(opencl_compiler_supports_node(i32_abs));
  EXPECT_FALSE(opencl_compiler_supports_node(high_rank_broadcast_add));
  EXPECT_FALSE(opencl_compiler_supports_node(convert_to_f16));
}
