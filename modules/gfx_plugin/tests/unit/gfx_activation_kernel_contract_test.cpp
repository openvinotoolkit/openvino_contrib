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
#include "backends/metal/compiler/msl_codegen_apple_msl_activation.hpp"
#include "backends/opencl/compiler/opencl_activation_kernel_unit.hpp"
#include "backends/opencl/compiler/opencl_kernel_artifacts.hpp"
#include "backends/opencl/compiler/opencl_operation_support.hpp"
#include "compiler/executable_bundle.hpp"
#include "compiler/kernel_registry.hpp"
#include "compiler/lowering_planner.hpp"
#include "compiler/manifest.hpp"
#include "compiler/operation_legalizer.hpp"
#include "kernel_ir/gfx_kernel_source.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "kernel_ir/opencl_kernels/activation_kernel.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/swish.hpp"
#include "unit/gfx_activation_contract_cases.hpp"

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

class ActivationOpenClArtifactContract final {
public:
  explicit ActivationOpenClArtifactContract(
      ActivationOpenClArtifactCase test_case)
      : m_case(std::move(test_case)) {}

  void verify() const {
    const auto node = m_case.make_node();
    const auto artifact =
        make_opencl_activation_source_artifact(node, m_case.expected_source_id);
    ASSERT_TRUE(artifact.has_value());
    ASSERT_TRUE(artifact->valid);
    EXPECT_EQ(artifact->stage_manifest.stage_family,
              GfxKernelStageFamily::Activation);
    EXPECT_EQ(artifact->artifact_ref.source_id, m_case.expected_source_id);
    EXPECT_EQ(artifact->artifact_ref.entry_point, m_case.expected_entry_point);
    EXPECT_EQ(artifact->arg_count, 6u);
    EXPECT_EQ(artifact->direct_input_count, 1u);
    EXPECT_EQ(artifact->direct_output_count, 1u);
    EXPECT_EQ(artifact->direct_input_indices, std::vector<size_t>{0});
    EXPECT_EQ(artifact->scalar_args, (std::vector<GfxOpenClSourceScalarArg>{
                                         GfxOpenClSourceScalarArg::ElementCount,
                                         GfxOpenClSourceScalarArg::OpCode,
                                         GfxOpenClSourceScalarArg::StaticF32,
                                         GfxOpenClSourceScalarArg::StaticF32}));
    EXPECT_EQ(artifact->static_u32_scalars, std::vector<uint32_t>{});
    EXPECT_EQ(artifact->static_f32_scalars, m_case.expected_static_f32_scalars);
    EXPECT_EQ(artifact->op, m_case.expected_op);
    EXPECT_EQ(artifact->scalar_constant_f32, 0.0f);
    EXPECT_EQ(artifact->input_mode, GfxOpenClArtifactInputMode::Direct);
    EXPECT_EQ(artifact->source,
              opencl_generated_activation_kernel_source().source);
    EXPECT_NE(
        artifact->source.find("__kernel void " + m_case.expected_entry_point),
        std::string::npos);
    EXPECT_NE(artifact->source.find("gfx_activation_f32"), std::string::npos);
    EXPECT_EQ(artifact->source.find("gfx_opencl_baseline_unary_f32"),
              std::string::npos);
  }

private:
  ActivationOpenClArtifactCase m_case;
};

std::string activation_opencl_case_name(
    const ::testing::TestParamInfo<ActivationOpenClArtifactCase> &info) {
  return info.param.name;
}

class ActivationOpenClArtifactContractTest
    : public ::testing::TestWithParam<ActivationOpenClArtifactCase> {};

TEST_P(ActivationOpenClArtifactContractTest,
       UsesFamilyOwnedGeneratedKernelUnit) {
  ActivationOpenClArtifactContract(GetParam()).verify();
}

INSTANTIATE_TEST_SUITE_P(
    Activation, ActivationOpenClArtifactContractTest,
    ::testing::ValuesIn(activation_opencl_artifact_cases()),
    activation_opencl_case_name);

class ActivationMslArtifactContract final {
public:
  explicit ActivationMslArtifactContract(ActivationMslArtifactCase test_case)
      : m_case(std::move(test_case)) {}

  void verify() const {
    const auto node = m_case.make_node();
    const auto target =
        compiler::BackendTarget::from_backend(GpuBackend::Metal);
    const compiler::BackendCapabilities capabilities(
        target, compiler::make_metal_operation_support_policy());
    const auto support = capabilities.query_operation({node});
    ASSERT_TRUE(support.semantic_legal) << support.semantic_reason;
    EXPECT_EQ(support.preferred_route_kind, LoweringRouteKind::GeneratedKernel);
    EXPECT_EQ(support.preferred_route, "metal/generated/activation");

    const auto plan = make_activation_msl_kernel_source_plan(node);
    ASSERT_TRUE(plan.valid());
    EXPECT_EQ(plan.binding.stage_manifest.stage_family,
              GfxKernelStageFamily::Activation);
    EXPECT_EQ(plan.source.entry_point, "activation_kernel");
    EXPECT_EQ(plan.source.signature.arg_count, 3u);
    EXPECT_EQ(plan.source.signature.output_arg_count, 1u);
    EXPECT_NE(plan.source.msl_source.find("kernel void activation_kernel"),
              std::string::npos);
    EXPECT_NE(plan.source.msl_source.find(m_case.expected_source_snippet),
              std::string::npos);
    if (m_case.expected_source_snippet.find("gfx_msl_erf_approx") !=
        std::string::npos) {
      EXPECT_NE(plan.source.msl_source.find("inline float gfx_msl_erf_approx"),
                std::string::npos);
      EXPECT_EQ(plan.source.msl_source.find("erf("), std::string::npos);
    }
  }

private:
  ActivationMslArtifactCase m_case;
};

std::string activation_msl_case_name(
    const ::testing::TestParamInfo<ActivationMslArtifactCase> &info) {
  return info.param.name;
}

class ActivationMslArtifactContractTest
    : public ::testing::TestWithParam<ActivationMslArtifactCase> {};

TEST_P(ActivationMslArtifactContractTest, UsesGeneratedMslKernelUnit) {
  ActivationMslArtifactContract(GetParam()).verify();
}

INSTANTIATE_TEST_SUITE_P(Activation, ActivationMslArtifactContractTest,
                         ::testing::ValuesIn(activation_msl_artifact_cases()),
                         activation_msl_case_name);

struct ActivationRouteCase {
  std::string name;
  compiler::BackendTarget target;
  std::shared_ptr<const compiler::OperationSupportPolicy> policy;
  compiler::KernelRegistry kernel_registry;
  compiler::KernelArtifactDescriptorResolver descriptor_resolver;
  compiler::KernelArtifactPayloadResolver payload_resolver;
  KernelArtifactPayloadKind expected_payload = KernelArtifactPayloadKind::None;
  std::string expected_kernel_id;
  std::string expected_entry_point;
  uint32_t expected_abi_arg_count = 0;
};

struct ActivationCompiledContract {
  LoweringPlan plan;
  ManifestBundle manifest;
  ExecutableBundle executable;
};

class ActivationModelFactory final {
public:
  std::shared_ptr<ov::Model> f32_relu() const {
    auto input = param(ov::element::f32, ov::Shape{2, 3, 4});
    auto relu = std::make_shared<ov::op::v0::Relu>(input);
    auto result = std::make_shared<ov::op::v0::Result>(relu);
    return std::make_shared<ov::Model>(ov::ResultVector{result},
                                       ov::ParameterVector{input});
  }

  std::shared_ptr<ov::Model> f32_elu() const {
    auto input = param(ov::element::f32, ov::Shape{2, 3, 4});
    auto elu = std::make_shared<ov::op::v0::Elu>(input, 0.5);
    auto result = std::make_shared<ov::op::v0::Result>(elu);
    return std::make_shared<ov::Model>(ov::ResultVector{result},
                                       ov::ParameterVector{input});
  }
};

class ActivationRouteContract final {
public:
  explicit ActivationRouteContract(ActivationRouteCase route)
      : m_route(std::move(route)) {}

  ActivationCompiledContract
  compile(const std::shared_ptr<const ov::Model> &model) const {
    const compiler::BackendCapabilities capabilities(m_route.target,
                                                     m_route.policy);
    const compiler::OperationLegalizer legalizer(capabilities);
    const compiler::LoweringPlanner planner(m_route.target,
                                            m_route.kernel_registry);

    ActivationCompiledContract compiled;
    compiled.plan = planner.plan(model, legalizer);
    compiled.manifest = compiler::ManifestBuilder{}.build(compiled.plan);
    compiled.executable =
        compiler::ExecutableBundleBuilder(m_route.descriptor_resolver,
                                          m_route.payload_resolver)
            .build(compiled.manifest, compiled.plan);
    return compiled;
  }

  void verify(const ActivationCompiledContract &compiled) const {
    ASSERT_TRUE(compiled.plan.executable());
    EXPECT_EQ(compiled.plan.route_count(LoweringRouteKind::GeneratedKernel),
              1u);
    ASSERT_TRUE(compiled.manifest.verify().valid());
    ASSERT_TRUE(compiled.executable.verify().valid());

    const auto &stage = find_activation_stage(compiled.manifest);
    EXPECT_EQ(stage.execution_kind, LoweringRouteKind::GeneratedKernel);
    EXPECT_EQ(stage.backend_domain, m_route.target.backend_id());
    EXPECT_EQ(stage.kernel_unit_id, m_route.expected_kernel_id);
    EXPECT_FALSE(stage.memory.hidden_host_copy_allowed);

    const auto &artifact = find_activation_artifact(compiled.executable);
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
  find_activation_stage(const ManifestBundle &manifest) const {
    const auto it = std::find_if(manifest.stages.begin(), manifest.stages.end(),
                                 [this](const compiler::StageRecord &stage) {
                                   return stage.kernel_unit_id ==
                                          m_route.expected_kernel_id;
                                 });
    OPENVINO_ASSERT(it != manifest.stages.end(),
                    "Activation stage is missing from manifest");
    return *it;
  }

  const KernelArtifactDescriptor &
  find_activation_artifact(const ExecutableBundle &executable) const {
    const auto it = std::find_if(
        executable.artifact_descriptors.begin(),
        executable.artifact_descriptors.end(),
        [this](const KernelArtifactDescriptor &artifact) {
          return artifact.kernel.kernel_id == m_route.expected_kernel_id;
        });
    OPENVINO_ASSERT(it != executable.artifact_descriptors.end(),
                    "Activation artifact descriptor is missing");
    return *it;
  }

  void verify_payload(const compiler::KernelArtifactPayload *payload) const {
    if (m_route.expected_payload == KernelArtifactPayloadKind::OpenClSource) {
      const auto *source_payload =
          dynamic_cast<const GfxOpenClSourceArtifactPayload *>(payload);
      ASSERT_NE(source_payload, nullptr);
      EXPECT_EQ(source_payload->artifact().stage_manifest.stage_family,
                GfxKernelStageFamily::Activation);
      EXPECT_EQ(source_payload->artifact().source,
                opencl_generated_activation_kernel_source().source);
      return;
    }
    const auto *source_payload =
        dynamic_cast<const GfxKernelSourcePayload *>(payload);
    ASSERT_NE(source_payload, nullptr);
    const std::string source(source_payload->source().source);
    EXPECT_NE(source.find("kernel void activation_kernel"), std::string::npos);
    EXPECT_NE(source.find("0.500000"), std::string::npos);
    EXPECT_NE(source.find("exp(x) - 1.0f"), std::string::npos);
  }

  ActivationRouteCase m_route;
};

ActivationRouteCase opencl_activation_case() {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  return ActivationRouteCase{
      "OpenClGeneratedReluF32",
      target,
      compiler::make_opencl_operation_support_policy(),
      compiler::make_opencl_kernel_registry(target),
      compiler::make_opencl_kernel_artifact_descriptor_resolver(),
      compiler::make_opencl_kernel_artifact_payload_resolver(),
      KernelArtifactPayloadKind::OpenClSource,
      "opencl/generated/activation_f32",
      "gfx_opencl_generated_activation_f32",
      6u};
}

ActivationRouteCase metal_activation_case() {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::Metal);
  return ActivationRouteCase{
      "MetalGeneratedMslReluF32",
      target,
      compiler::make_metal_operation_support_policy(),
      compiler::make_metal_kernel_registry(target),
      compiler::make_metal_kernel_artifact_descriptor_resolver(),
      compiler::make_metal_kernel_artifact_payload_resolver(),
      KernelArtifactPayloadKind::MslSource,
      "metal/generated/activation",
      "activation_kernel",
      3u};
}

std::string activation_route_case_name(
    const ::testing::TestParamInfo<ActivationRouteCase> &info) {
  return info.param.name;
}

class ActivationRouteContractTest
    : public ::testing::TestWithParam<ActivationRouteCase> {
protected:
  ActivationModelFactory models;
};

TEST_P(ActivationRouteContractTest, CompilesThroughExpectedKernelUnit) {
  const ActivationRouteContract contract(GetParam());
  const auto model =
      GetParam().expected_payload == KernelArtifactPayloadKind::MslSource
          ? models.f32_elu()
          : models.f32_relu();
  contract.verify(contract.compile(model));
}

INSTANTIATE_TEST_SUITE_P(ActivationBackends, ActivationRouteContractTest,
                         ::testing::Values(opencl_activation_case(),
                                           metal_activation_case()),
                         activation_route_case_name);

TEST(ActivationRouteContractTest,
     MetalStaticBetaSwishUsesGeneratedActivationMslKernelUnit) {
  const auto input = param(ov::element::f32, ov::Shape{2, 3});
  const auto beta =
      ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.5f});
  const auto swish = std::make_shared<ov::op::v4::Swish>(input, beta);

  const auto target = compiler::BackendTarget::from_backend(GpuBackend::Metal);
  const compiler::BackendCapabilities capabilities(
      target, compiler::make_metal_operation_support_policy());
  const auto support = capabilities.query_operation({swish});
  EXPECT_TRUE(support.semantic_legal);
  EXPECT_EQ(support.preferred_route_kind, LoweringRouteKind::GeneratedKernel);
  EXPECT_EQ(support.preferred_route, "metal/generated/activation");

  const auto plan = make_activation_msl_kernel_source_plan(swish);
  ASSERT_TRUE(plan.valid());
  EXPECT_EQ(plan.source.entry_point, "activation_kernel");
  EXPECT_NE(plan.source.msl_source.find("0.500000f * x"), std::string::npos);
}

TEST(ActivationRouteContractTest,
     OpenClDynamicBetaSwishUsesGeneratedActivationRuntimeScalarKernelUnit) {
  const auto input = param(ov::element::f32, ov::Shape{2, 3});
  const auto beta = param(ov::element::f32, ov::Shape{});
  const auto swish = std::make_shared<ov::op::v4::Swish>(input, beta);

  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  const compiler::BackendCapabilities capabilities(
      target, compiler::make_opencl_operation_support_policy());
  const auto support = capabilities.query_operation({swish});
  EXPECT_TRUE(support.semantic_legal);
  EXPECT_EQ(support.preferred_route_kind, LoweringRouteKind::GeneratedKernel);
  EXPECT_EQ(support.preferred_route,
            "opencl/generated/activation_runtime_beta_f32");

  const auto artifact = make_opencl_activation_source_artifact(
      swish, "opencl/generated/activation_runtime_beta_f32");
  ASSERT_TRUE(artifact.has_value());
  ASSERT_TRUE(artifact->valid);
  EXPECT_EQ(artifact->stage_manifest.stage_family,
            GfxKernelStageFamily::Activation);
  EXPECT_EQ(artifact->artifact_ref.source_id,
            "opencl/generated/activation_runtime_beta_f32");
  EXPECT_EQ(artifact->artifact_ref.entry_point,
            "gfx_opencl_generated_activation_runtime_beta_f32");
  EXPECT_EQ(artifact->arg_count, 5u);
  EXPECT_EQ(artifact->direct_input_count, 2u);
  EXPECT_EQ(artifact->direct_output_count, 1u);
  EXPECT_EQ(artifact->direct_input_indices, (std::vector<size_t>{0, 1}));
  EXPECT_EQ(artifact->scalar_args, (std::vector<GfxOpenClSourceScalarArg>{
                                       GfxOpenClSourceScalarArg::ElementCount,
                                       GfxOpenClSourceScalarArg::OpCode}));
  EXPECT_EQ(artifact->static_f32_scalars, std::vector<float>{});
  EXPECT_EQ(artifact->op, GfxOpenClArtifactOp::Swish);
  EXPECT_EQ(artifact->source,
            opencl_generated_activation_kernel_source().source);
  EXPECT_NE(
      artifact->source.find("__kernel void "
                            "gfx_opencl_generated_activation_runtime_beta_f32"),
      std::string::npos);
  EXPECT_NE(artifact->source.find("runtime_beta[0]"), std::string::npos);
}

TEST(ActivationRouteContractTest, OpenClReluUsesActivationKernelUnitOwner) {
  const auto input = param(ov::element::f32, ov::Shape{2, 3});
  const auto relu = std::make_shared<ov::op::v0::Relu>(input);

  const auto artifact = make_opencl_activation_source_artifact(
      relu, "opencl/generated/activation_f32");
  ASSERT_TRUE(artifact.has_value());
  ASSERT_TRUE(artifact->valid);
  EXPECT_EQ(artifact->stage_manifest.stage_family,
            GfxKernelStageFamily::Activation);
  EXPECT_FALSE(make_opencl_activation_source_artifact(
                   relu, "opencl/generated/eltwise_binary_f32")
                   .has_value());

  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  const compiler::BackendCapabilities capabilities(
      target, compiler::make_opencl_operation_support_policy());
  const auto support = capabilities.query_operation({relu});
  ASSERT_TRUE(support.semantic_legal) << support.semantic_reason;
  EXPECT_EQ(support.semantic_reason,
            "registered_opencl_activation_kernel_unit");
  EXPECT_EQ(support.preferred_route_kind, LoweringRouteKind::GeneratedKernel);
  EXPECT_EQ(support.preferred_route, "opencl/generated/activation_f32");
}

TEST(ActivationRouteContractTest,
     OpenClActivationCompilerUsesOpOwnedKernelUnit) {
  const auto input = param(ov::element::f32, ov::Shape{2, 3});
  const auto relu = std::make_shared<ov::op::v0::Relu>(input);

  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  const auto registry = compiler::make_opencl_kernel_registry(target);
  const auto unit =
      compiler::resolve_opencl_activation_kernel_unit(relu, registry);
  ASSERT_TRUE(unit.valid());
  EXPECT_EQ(unit.id(), "opencl/generated/activation_f32");
  const compiler::BackendCapabilities capabilities(
      target, compiler::make_opencl_operation_support_policy(registry));
  const auto support = capabilities.query_operation({relu});
  ASSERT_TRUE(support.semantic_legal) << support.semantic_reason;
  EXPECT_EQ(support.preferred_route_kind, unit.route_kind());
  EXPECT_EQ(support.preferred_route, unit.id());
}

TEST(ActivationRouteContractTest,
     MetalDynamicBetaSwishUsesGeneratedActivationMslRuntimeScalarKernelUnit) {
  const auto input = param(ov::element::f32, ov::Shape{2, 3});
  const auto beta = param(ov::element::f32, ov::Shape{});
  const auto swish = std::make_shared<ov::op::v4::Swish>(input, beta);

  const auto target = compiler::BackendTarget::from_backend(GpuBackend::Metal);
  const compiler::BackendCapabilities capabilities(
      target, compiler::make_metal_operation_support_policy());
  const auto support = capabilities.query_operation({swish});
  EXPECT_TRUE(support.semantic_legal);
  EXPECT_EQ(support.preferred_route_kind, LoweringRouteKind::GeneratedKernel);
  EXPECT_EQ(support.preferred_route, "metal/generated/activation");

  const auto plan = make_activation_msl_kernel_source_plan(swish);
  ASSERT_TRUE(plan.valid());
  EXPECT_EQ(plan.source.entry_point, "activation_swish_runtime_beta_kernel");
  EXPECT_EQ(plan.source.signature.arg_count, 4u);
  EXPECT_EQ(plan.source.signature.output_arg_count, 1u);
  EXPECT_EQ(plan.binding.stage_manifest.stage_family,
            GfxKernelStageFamily::Activation);
  EXPECT_EQ(plan.binding.runtime_binding.scalar_args,
            (std::vector<int32_t>{6}));
  EXPECT_EQ(plan.binding.stage_manifest.custom_kernel.scalar_args,
            (std::vector<int32_t>{6}));

  const auto roles =
      plan.binding.stage_manifest.custom_kernel.external_buffer_abi.roles;
  ASSERT_EQ(roles.size(), 4u);
  EXPECT_EQ(roles[0], GfxKernelBufferRole::TensorInput);
  EXPECT_EQ(roles[1], GfxKernelBufferRole::TensorInput);
  EXPECT_EQ(roles[2], GfxKernelBufferRole::TensorOutput);
  EXPECT_EQ(roles[3], GfxKernelBufferRole::ScalarParam);
  EXPECT_NE(
      plan.source.msl_source.find("device const float* beta [[buffer(1)]]"),
      std::string::npos);
  EXPECT_NE(plan.source.msl_source.find("beta_value * x"), std::string::npos);
}

TEST(ActivationUnsupportedContractTest,
     OpenClI32AbsRejectsMissingActivationKernelUnit) {
  const auto input = param(ov::element::i32, ov::Shape{2, 3});
  const auto abs = std::make_shared<ov::op::v0::Abs>(input);

  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  const compiler::BackendCapabilities capabilities(
      target, compiler::make_opencl_operation_support_policy());
  const auto support = capabilities.query_operation({abs});
  EXPECT_FALSE(support.semantic_legal);
  EXPECT_EQ(support.semantic_reason, "missing_opencl_activation_kernel_unit");
}

TEST(ActivationUnsupportedContractTest,
     OpenClDynamicShapeBetaSwishRejectsMissingRuntimeScalarAbi) {
  const auto input = param(ov::element::f32, ov::Shape{2, 3});
  const auto beta = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f32, ov::PartialShape::dynamic());
  const auto swish = std::make_shared<ov::op::v4::Swish>(input, beta);

  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  const compiler::BackendCapabilities capabilities(
      target, compiler::make_opencl_operation_support_policy());
  const auto support = capabilities.query_operation({swish});
  EXPECT_FALSE(support.semantic_legal);
  EXPECT_EQ(support.semantic_reason, "missing_opencl_activation_kernel_unit");
}

} // namespace
} // namespace gfx_plugin
} // namespace ov
