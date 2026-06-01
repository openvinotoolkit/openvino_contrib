// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backends/metal/compiler/metal_kernel_artifacts.hpp"
#include "backends/metal/compiler/metal_operation_support.hpp"
#include "backends/opencl/compiler/opencl_kernel_artifacts.hpp"
#include "backends/opencl/compiler/opencl_operation_support.hpp"
#include "compiler/executable_bundle.hpp"
#include "compiler/kernel_registry.hpp"
#include "compiler/manifest.hpp"
#include "compiler/operation_legalizer.hpp"
#include "kernel_ir/gfx_kernel_source.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "kernel_ir/opencl_kernels/eltwise_kernel.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "runtime/executable_descriptor.hpp"
#include "unit/gfx_eltwise_contract_cases.hpp"

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
                                             ov::Shape shape) {
  return std::make_shared<ov::op::v0::Parameter>(type, std::move(shape));
}

class EltwiseOpenClArtifactContract final {
public:
  explicit EltwiseOpenClArtifactContract(EltwiseOpenClArtifactCase test_case)
      : m_case(std::move(test_case)) {}

  void verify() const {
    const auto node = m_case.make_node();
    const auto artifact = resolve_gfx_opencl_source_artifact(node);
    ASSERT_TRUE(artifact.has_value());
    ASSERT_TRUE(artifact->valid);
    EXPECT_EQ(artifact->stage_manifest.stage_family,
              GfxKernelStageFamily::Eltwise);
    EXPECT_EQ(artifact->artifact_ref.source_id, m_case.expected_source_id);
    EXPECT_EQ(artifact->artifact_ref.entry_point, m_case.expected_entry_point);
    EXPECT_EQ(artifact->arg_count, m_case.expected_arg_count);
    EXPECT_EQ(artifact->direct_input_count, m_case.expected_direct_input_count);
    EXPECT_EQ(artifact->direct_input_indices, m_case.expected_direct_inputs);
    EXPECT_EQ(artifact->scalar_args, m_case.expected_scalar_args);
    EXPECT_EQ(artifact->static_u32_scalars, m_case.expected_static_u32_scalars);
    EXPECT_EQ(artifact->op, m_case.expected_op);
    EXPECT_EQ(artifact->input_mode, m_case.expected_input_mode);
    EXPECT_EQ(artifact->source,
              opencl_generated_eltwise_kernel_source().source);
    EXPECT_NE(
        artifact->source.find("__kernel void " + m_case.expected_entry_point),
        std::string::npos);
    EXPECT_NE(artifact->source.find("uint op"), std::string::npos);
    EXPECT_EQ(artifact->source.find("gfx_opencl_generated_add"),
              std::string::npos);
  }

private:
  EltwiseOpenClArtifactCase m_case;
};

std::string eltwise_opencl_case_name(
    const ::testing::TestParamInfo<EltwiseOpenClArtifactCase> &info) {
  return info.param.name;
}

class EltwiseOpenClArtifactContractTest
    : public ::testing::TestWithParam<EltwiseOpenClArtifactCase> {};

TEST_P(EltwiseOpenClArtifactContractTest, UsesFamilyOwnedGeneratedKernelUnit) {
  EltwiseOpenClArtifactContract(GetParam()).verify();
}

INSTANTIATE_TEST_SUITE_P(Eltwise, EltwiseOpenClArtifactContractTest,
                         ::testing::ValuesIn(eltwise_opencl_artifact_cases()),
                         eltwise_opencl_case_name);

struct EltwiseRouteCase {
  std::string name;
  compiler::BackendTarget target;
  std::shared_ptr<const compiler::OperationSupportPolicy> policy;
  compiler::KernelRegistry kernel_registry;
  compiler::KernelArtifactPayloadResolver payload_resolver;
  KernelArtifactPayloadKind expected_payload = KernelArtifactPayloadKind::None;
  std::string expected_kernel_id;
  std::string expected_entry_point;
  uint32_t expected_abi_arg_count = 0;
};

struct EltwiseCompiledContract {
  LoweringPlan plan;
  ManifestBundle manifest;
  ExecutableBundle executable;
};

class EltwiseModelFactory final {
public:
  std::shared_ptr<ov::Model> f32_same_shape_add() const {
    auto lhs = param(ov::element::f32, ov::Shape{2, 3, 4});
    auto rhs = param(ov::element::f32, ov::Shape{2, 3, 4});
    auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);
    auto result = std::make_shared<ov::op::v0::Result>(add);
    return std::make_shared<ov::Model>(ov::ResultVector{result},
                                       ov::ParameterVector{lhs, rhs});
  }
};

class EltwiseRouteContract final {
public:
  explicit EltwiseRouteContract(EltwiseRouteCase route)
      : m_route(std::move(route)) {}

  EltwiseCompiledContract
  compile(const std::shared_ptr<const ov::Model> &model) const {
    const compiler::BackendCapabilities capabilities(m_route.target,
                                                     m_route.policy);
    const compiler::OperationLegalizer legalizer(capabilities);
    const compiler::LoweringPlanner planner(m_route.target,
                                            m_route.kernel_registry);

    EltwiseCompiledContract compiled;
    compiled.plan = planner.plan(model, legalizer);
    compiled.manifest = compiler::ManifestBuilder{}.build(compiled.plan);
    compiled.executable =
        compiler::ExecutableBundleBuilder(m_route.payload_resolver)
            .build(compiled.manifest, compiled.plan);
    return compiled;
  }

  void verify(const EltwiseCompiledContract &compiled) const {
    ASSERT_TRUE(compiled.plan.executable());
    EXPECT_EQ(compiled.plan.route_count(LoweringRouteKind::GeneratedKernel),
              1u);
    ASSERT_TRUE(compiled.manifest.verify().valid());
    ASSERT_TRUE(compiled.executable.verify().valid());

    const auto &stage = find_eltwise_stage(compiled.manifest);
    EXPECT_EQ(stage.execution_kind, LoweringRouteKind::GeneratedKernel);
    EXPECT_EQ(stage.backend_domain, m_route.target.backend_id());
    EXPECT_EQ(stage.kernel_unit_id, m_route.expected_kernel_id);
    EXPECT_FALSE(stage.memory.hidden_host_copy_allowed);

    const auto &artifact = find_eltwise_artifact(compiled.executable);
    EXPECT_EQ(artifact.kernel.origin, KernelArtifactOrigin::Generated);
    EXPECT_EQ(artifact.payload_kind, m_route.expected_payload);
    EXPECT_EQ(artifact.kernel.kernel_id, m_route.expected_kernel_id);
    EXPECT_EQ(artifact.entry_point, m_route.expected_entry_point);
    EXPECT_EQ(artifact.abi_arg_count, m_route.expected_abi_arg_count);
    EXPECT_EQ(artifact.abi_output_arg_count, 1u);

    const auto payload =
        compiled.executable.find_artifact_payload(artifact.artifact_key);
    ASSERT_TRUE(payload);
    EXPECT_EQ(payload->payload_kind(), m_route.expected_payload);
    EXPECT_EQ(payload->source_id(), m_route.expected_kernel_id);
    EXPECT_EQ(payload->entry_point(), m_route.expected_entry_point);
    verify_payload(payload.get());
  }

private:
  const compiler::StageRecord &
  find_eltwise_stage(const ManifestBundle &manifest) const {
    const auto it = std::find_if(manifest.stages.begin(), manifest.stages.end(),
                                 [this](const compiler::StageRecord &stage) {
                                   return stage.kernel_unit_id ==
                                          m_route.expected_kernel_id;
                                 });
    OPENVINO_ASSERT(it != manifest.stages.end(),
                    "Eltwise stage is missing from manifest");
    return *it;
  }

  const KernelArtifactDescriptor &
  find_eltwise_artifact(const ExecutableBundle &executable) const {
    const auto it = std::find_if(
        executable.artifact_descriptors.begin(),
        executable.artifact_descriptors.end(),
        [this](const KernelArtifactDescriptor &artifact) {
          return artifact.kernel.kernel_id == m_route.expected_kernel_id;
        });
    OPENVINO_ASSERT(it != executable.artifact_descriptors.end(),
                    "Eltwise artifact descriptor is missing");
    return *it;
  }

  void verify_payload(const compiler::KernelArtifactPayload *payload) const {
    if (m_route.expected_payload == KernelArtifactPayloadKind::OpenClSource) {
      const auto *source_payload =
          dynamic_cast<const GfxOpenClSourceArtifactPayload *>(payload);
      ASSERT_NE(source_payload, nullptr);
      EXPECT_EQ(source_payload->artifact().source,
                opencl_generated_eltwise_kernel_source().source);
      return;
    }
    const auto *source_payload =
        dynamic_cast<const GfxKernelSourcePayload *>(payload);
    ASSERT_NE(source_payload, nullptr);
    EXPECT_NE(std::string(source_payload->source().source)
                  .find("kernel void eltwise_kernel"),
              std::string::npos);
    EXPECT_NE(std::string(source_payload->source().source).find(" + "),
              std::string::npos);
  }

  EltwiseRouteCase m_route;
};

EltwiseRouteCase opencl_eltwise_case() {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  return EltwiseRouteCase{"OpenClGeneratedF32",
                          target,
                          compiler::make_opencl_operation_support_policy(),
                          compiler::make_opencl_kernel_registry(target),
                          compiler::make_opencl_kernel_artifact_payload_resolver(),
                          KernelArtifactPayloadKind::OpenClSource,
                          "opencl/generated/eltwise_binary_f32",
                          "gfx_opencl_generated_eltwise_binary_f32",
                          5u};
}

EltwiseRouteCase metal_eltwise_case() {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::Metal);
  return EltwiseRouteCase{
      "MetalGeneratedMslF32",
      target,
      compiler::make_metal_operation_support_policy(),
      compiler::make_metal_kernel_registry(target),
      compiler::make_metal_kernel_artifact_payload_resolver(),
      KernelArtifactPayloadKind::MslSource,
      "metal/generated/eltwise",
      "eltwise_kernel",
      8u};
}

std::string eltwise_route_case_name(
    const ::testing::TestParamInfo<EltwiseRouteCase> &info) {
  return info.param.name;
}

class EltwiseRouteContractTest
    : public ::testing::TestWithParam<EltwiseRouteCase> {
protected:
  EltwiseModelFactory models;
};

TEST_P(EltwiseRouteContractTest, CompilesThroughExpectedKernelUnit) {
  EltwiseRouteContract(GetParam())
      .verify(EltwiseRouteContract(GetParam())
                  .compile(models.f32_same_shape_add()));
}

INSTANTIATE_TEST_SUITE_P(EltwiseBackends, EltwiseRouteContractTest,
                         ::testing::Values(opencl_eltwise_case(),
                                           metal_eltwise_case()),
                         eltwise_route_case_name);

TEST(EltwiseUnsupportedContractTest,
     OpenClHighRankBroadcastRejectsMissingEltwiseKernelUnit) {
  const auto lhs = param(ov::element::f32, ov::Shape{1, 1, 1, 1, 3});
  const auto rhs = param(ov::element::f32, ov::Shape{3});
  const auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);

  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  const compiler::BackendCapabilities capabilities(
      target, compiler::make_opencl_operation_support_policy());
  const auto support = capabilities.query_operation({add});
  EXPECT_FALSE(support.semantic_legal);
  EXPECT_EQ(support.semantic_reason, "missing_opencl_eltwise_kernel_unit");
}

} // namespace
} // namespace gfx_plugin
} // namespace ov
