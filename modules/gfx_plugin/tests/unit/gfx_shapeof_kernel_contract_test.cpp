// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backends/metal/compiler/metal_kernel_artifacts.hpp"
#include "backends/metal/compiler/metal_operation_support.hpp"
#include "backends/opencl/compiler/opencl_kernel_artifacts.hpp"
#include "backends/opencl/compiler/opencl_operation_support.hpp"
#include "backends/opencl/compiler/opencl_shapeof_kernel_unit.hpp"
#include "compiler/executable_bundle.hpp"
#include "compiler/kernel_registry.hpp"
#include "compiler/manifest.hpp"
#include "compiler/operation_legalizer.hpp"
#include "kernel_ir/opencl_kernels/shapeof_kernel.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

using compiler::ExecutableBundle;
using compiler::KernelArtifactDescriptor;
using compiler::KernelArtifactOrigin;
using compiler::KernelArtifactPayloadKind;
using compiler::LoweringPlan;
using compiler::LoweringRouteKind;
using compiler::ManifestBundle;

std::shared_ptr<ov::op::v0::Parameter> param(const ov::element::Type &type,
                                             ov::PartialShape shape) {
  return std::make_shared<ov::op::v0::Parameter>(type, std::move(shape));
}

std::shared_ptr<ov::Node> shapeof_node(const ov::element::Type &input_type,
                                       ov::PartialShape input_shape,
                                       const ov::element::Type &output_type) {
  return std::make_shared<ov::op::v3::ShapeOf>(
      param(input_type, std::move(input_shape)), output_type);
}

std::shared_ptr<ov::Model>
model_from_node(const std::shared_ptr<ov::Node> &node,
                ov::ParameterVector params) {
  return std::make_shared<ov::Model>(
      ov::ResultVector{std::make_shared<ov::op::v0::Result>(node)},
      std::move(params));
}

std::vector<GfxOpenClSourceScalarArg> shapeof_scalar_args() {
  std::vector<GfxOpenClSourceScalarArg> args = {
      GfxOpenClSourceScalarArg::ElementCount};
  for (uint32_t axis = 0; axis < 8; ++axis) {
    args.push_back(static_cast<GfxOpenClSourceScalarArg>(
        static_cast<uint32_t>(GfxOpenClSourceScalarArg::Input0Dim0) + axis));
  }
  return args;
}

void expect_opencl_shapeof_artifact(const std::shared_ptr<ov::Node> &node,
                                    const std::string &source_id,
                                    const std::string &entry_point) {
  auto artifact = make_opencl_shapeof_source_artifact(node);
  ASSERT_TRUE(artifact.has_value());
  ASSERT_TRUE(artifact->valid);
  EXPECT_EQ(artifact->stage_manifest.stage_family,
            GfxKernelStageFamily::GatherScatter);
  EXPECT_EQ(artifact->stage_manifest.backend_domain,
            GfxKernelBackendDomain::OpenCl);
  EXPECT_EQ(artifact->stage_manifest.execution_kind,
            GfxKernelExecutionKind::CustomKernel);
  EXPECT_EQ(artifact->artifact_ref.kind, GfxKernelArtifactKind::OpenClSource);
  EXPECT_EQ(artifact->artifact_ref.backend_domain,
            GfxKernelBackendDomain::OpenCl);
  EXPECT_EQ(artifact->artifact_ref.source_id, source_id);
  EXPECT_EQ(artifact->artifact_ref.entry_point, entry_point);
  EXPECT_EQ(artifact->arg_count, 11u);
  EXPECT_EQ(artifact->direct_input_count, 1u);
  EXPECT_EQ(artifact->direct_output_count, 1u);
  EXPECT_EQ(artifact->direct_input_indices, std::vector<size_t>{0});
  EXPECT_EQ(artifact->local_size_hint, 64u);
  EXPECT_EQ(artifact->scalar_args, shapeof_scalar_args());
  EXPECT_EQ(artifact->op, GfxOpenClArtifactOp::Identity);
  EXPECT_EQ(artifact->input_mode, GfxOpenClArtifactInputMode::Direct);

  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  const auto registry = compiler::make_opencl_kernel_registry(target);
  const auto unit =
      compiler::resolve_opencl_shapeof_kernel_unit(node, registry);
  ASSERT_TRUE(unit.valid());
  EXPECT_EQ(unit.route_kind(), LoweringRouteKind::GeneratedKernel);
  EXPECT_EQ(unit.id(), source_id);
  EXPECT_EQ(unit.op_family(), "ShapeOf");
  const compiler::BackendCapabilities capabilities(
      target, compiler::make_opencl_operation_support_policy(registry));
  const auto support = capabilities.query_operation({node});
  ASSERT_TRUE(support.semantic_legal) << support.semantic_reason;
  EXPECT_EQ(support.preferred_route_kind, unit.route_kind());
  EXPECT_EQ(support.preferred_route, unit.id());
}

TEST(ShapeOfOpenClArtifactContract, I32UsesFamilyOwnedKernelUnit) {
  expect_opencl_shapeof_artifact(
      shapeof_node(ov::element::f32, ov::Shape{2, 3, 4}, ov::element::i32),
      "opencl/generated/shapeof_i32", "gfx_opencl_generated_shapeof_i32");
}

TEST(ShapeOfOpenClArtifactContract, I64UsesFamilyOwnedKernelUnit) {
  expect_opencl_shapeof_artifact(
      shapeof_node(ov::element::f32, ov::Shape{5, 6}, ov::element::i64),
      "opencl/generated/shapeof_i64", "gfx_opencl_generated_shapeof_i64");
}

TEST(ShapeOfOpenClArtifactContract,
     DynamicInputDimsUseShapeMetadataWithoutGenericSourceFallback) {
  const auto node = shapeof_node(ov::element::f16, ov::PartialShape{1, -1, 64},
                                 ov::element::i64);
  expect_opencl_shapeof_artifact(node, "opencl/generated/shapeof_i64",
                                 "gfx_opencl_generated_shapeof_i64");
}

TEST(ShapeOfOpenClArtifactContract,
     PayloadResolverRejectsMismatchedKernelUnitWithoutFallback) {
  const auto node =
      shapeof_node(ov::element::f32, ov::Shape{2, 3}, ov::element::i64);

  compiler::KernelArtifactDescriptor descriptor;
  descriptor.kernel.backend_domain = "opencl";
  descriptor.kernel.kernel_id = "opencl/generated/shapeof_i32";
  descriptor.kernel.origin = compiler::KernelArtifactOrigin::Generated;
  descriptor.payload_kind = compiler::KernelArtifactPayloadKind::OpenClSource;

  compiler::PlannedOperation planned_shapeof;
  planned_shapeof.source_node = node;
  planned_shapeof.type_name = "ShapeOf";

  const auto resolver =
      compiler::make_opencl_kernel_artifact_payload_resolver();
  EXPECT_FALSE(static_cast<bool>(resolver(descriptor, planned_shapeof)));
  EXPECT_TRUE(descriptor.entry_point.empty());
}

struct ShapeOfRouteCase {
  std::string name;
  compiler::BackendTarget target;
  std::shared_ptr<const compiler::OperationSupportPolicy> policy;
  compiler::KernelRegistry kernel_registry;
  compiler::KernelArtifactDescriptorResolver descriptor_resolver;
  compiler::KernelArtifactPayloadResolver payload_resolver;
  KernelArtifactOrigin expected_origin = KernelArtifactOrigin::Unknown;
  KernelArtifactPayloadKind expected_payload = KernelArtifactPayloadKind::None;
  std::string expected_kernel_id;
  std::string expected_entry_point;
  std::function<std::shared_ptr<ov::Model>()> make_model;
};

struct ShapeOfCompiledContract {
  LoweringPlan plan;
  ManifestBundle manifest;
  ExecutableBundle executable;
};

class ShapeOfRouteContract final {
public:
  explicit ShapeOfRouteContract(ShapeOfRouteCase route)
      : m_route(std::move(route)) {}

  ShapeOfCompiledContract compile() const {
    const compiler::BackendCapabilities capabilities(m_route.target,
                                                     m_route.policy);
    const compiler::OperationLegalizer legalizer(capabilities);
    const compiler::LoweringPlanner planner(m_route.target,
                                            m_route.kernel_registry);
    ShapeOfCompiledContract compiled;
    compiled.plan = planner.plan(m_route.make_model(), legalizer);
    compiled.manifest = compiler::ManifestBuilder{}.build(compiled.plan);
    compiled.executable =
        compiler::ExecutableBundleBuilder(m_route.descriptor_resolver,
                                          m_route.payload_resolver)
            .build(compiled.manifest, compiled.plan);
    return compiled;
  }

  void verify(const ShapeOfCompiledContract &compiled) const {
    ASSERT_TRUE(compiled.plan.executable());
    EXPECT_EQ(compiled.plan.route_count(LoweringRouteKind::GeneratedKernel),
              1u);
    ASSERT_TRUE(compiled.manifest.verify().valid());
    ASSERT_TRUE(compiled.executable.verify().valid());

    const auto &stage = find_shapeof_stage(compiled.manifest);
    EXPECT_EQ(stage.execution_kind, LoweringRouteKind::GeneratedKernel);
    EXPECT_EQ(stage.backend_domain, m_route.target.backend_id());
    EXPECT_EQ(stage.kernel_unit_id, m_route.expected_kernel_id);
    EXPECT_FALSE(stage.memory.hidden_host_copy_allowed);

    const auto &artifact = find_shapeof_artifact(compiled.executable);
    EXPECT_EQ(artifact.kernel.origin, m_route.expected_origin);
    EXPECT_EQ(artifact.payload_kind, m_route.expected_payload);
    EXPECT_EQ(artifact.kernel.kernel_id, m_route.expected_kernel_id);
    EXPECT_EQ(artifact.entry_point, m_route.expected_entry_point);
    EXPECT_GT(artifact.abi_arg_count, 0u);
    EXPECT_EQ(artifact.abi_output_arg_count, 1u);

    const auto payload =
        compiled.executable.find_artifact_payload(artifact.artifact_key);
    ASSERT_TRUE(payload);
    EXPECT_EQ(payload->payload_kind(), m_route.expected_payload);
    EXPECT_EQ(payload->source_id(), m_route.expected_kernel_id);
    EXPECT_EQ(payload->entry_point(), m_route.expected_entry_point);
  }

private:
  const compiler::StageRecord &
  find_shapeof_stage(const ManifestBundle &manifest) const {
    const auto it =
        std::find_if(manifest.stages.begin(), manifest.stages.end(),
                     [](const compiler::StageRecord &stage) {
                       return stage.normalized_op_family == "ShapeOf";
                     });
    OPENVINO_ASSERT(it != manifest.stages.end(),
                    "ShapeOf stage is missing from manifest");
    return *it;
  }

  const KernelArtifactDescriptor &
  find_shapeof_artifact(const ExecutableBundle &executable) const {
    const auto it = std::find_if(
        executable.artifact_descriptors.begin(),
        executable.artifact_descriptors.end(),
        [this](const KernelArtifactDescriptor &artifact) {
          return artifact.kernel.kernel_id == m_route.expected_kernel_id;
        });
    OPENVINO_ASSERT(it != executable.artifact_descriptors.end(),
                    "ShapeOf artifact descriptor is missing");
    return *it;
  }

  ShapeOfRouteCase m_route;
};

std::shared_ptr<ov::Model> opencl_shapeof_model() {
  auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  return model_from_node(
      std::make_shared<ov::op::v3::ShapeOf>(data, ov::element::i64), {data});
}

std::shared_ptr<ov::Model> metal_shapeof_model() {
  auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  return model_from_node(
      std::make_shared<ov::op::v3::ShapeOf>(data, ov::element::i64), {data});
}

ShapeOfRouteCase opencl_shapeof_case() {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  return {"OpenClGeneratedShapeOfI64",
          target,
          compiler::make_opencl_operation_support_policy(),
          compiler::make_opencl_kernel_registry(target),
          compiler::make_opencl_kernel_artifact_descriptor_resolver(),
          compiler::make_opencl_kernel_artifact_payload_resolver(),
          KernelArtifactOrigin::Generated,
          KernelArtifactPayloadKind::OpenClSource,
          "opencl/generated/shapeof_i64",
          "gfx_opencl_generated_shapeof_i64",
          opencl_shapeof_model};
}

ShapeOfRouteCase metal_shapeof_case() {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::Metal);
  return {"MetalGeneratedShapeOf",
          target,
          compiler::make_metal_operation_support_policy(),
          compiler::make_metal_kernel_registry(target),
          compiler::make_metal_kernel_artifact_descriptor_resolver(),
          compiler::make_metal_kernel_artifact_payload_resolver(),
          KernelArtifactOrigin::Generated,
          KernelArtifactPayloadKind::MslSource,
          "metal/generated/shapeof",
          "shapeof_kernel",
          metal_shapeof_model};
}

std::string shapeof_route_case_name(
    const ::testing::TestParamInfo<ShapeOfRouteCase> &info) {
  return info.param.name;
}

class ShapeOfRouteContractTest
    : public ::testing::TestWithParam<ShapeOfRouteCase> {};

TEST_P(ShapeOfRouteContractTest, CompilesThroughExpectedKernelUnit) {
  const ShapeOfRouteContract contract(GetParam());
  contract.verify(contract.compile());
}

INSTANTIATE_TEST_SUITE_P(ShapeOfBackends, ShapeOfRouteContractTest,
                         ::testing::Values(opencl_shapeof_case(),
                                           metal_shapeof_case()),
                         shapeof_route_case_name);

} // namespace
} // namespace gfx_plugin
} // namespace ov
