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
#include "compiler/executable_bundle.hpp"
#include "compiler/kernel_registry.hpp"
#include "compiler/manifest.hpp"
#include "compiler/operation_legalizer.hpp"
#include "gfx_opencl_source_artifact_verifier.hpp"
#include "gfx_runtime_model_runner.hpp"
#include "gfx_runtime_scenario.hpp"
#include "kernel_ir/metal_kernels/softmax_kernels.hpp"
#include "kernel_ir/opencl_kernels/softmax_f16_dynamic_kernel.hpp"
#include "kernel_ir/opencl_kernels/softmax_f16_kernel.hpp"
#include "kernel_ir/opencl_kernels/softmax_f32_dynamic_kernel.hpp"
#include "kernel_ir/opencl_kernels/softmax_f32_kernel.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "backends/metal/compiler/apple_vendor_descriptors.hpp"
#include "mlir/gfx_backend_custom_kernel_adapter.hpp"
#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "mlir/mlir_passes.hpp"
#include "mlir/mlir_support.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_ops.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_softmax.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"

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
using test::OpenClSourceArtifactVerifier;

std::shared_ptr<ov::op::v0::Parameter> param(const ov::element::Type &type,
                                             ov::PartialShape shape) {
  return std::make_shared<ov::op::v0::Parameter>(type, std::move(shape));
}

std::shared_ptr<ov::Model>
model_from_node(const std::shared_ptr<ov::Node> &node,
                ov::ParameterVector params) {
  return std::make_shared<ov::Model>(
      ov::ResultVector{std::make_shared<ov::op::v0::Result>(node)},
      std::move(params));
}

std::vector<GfxOpenClSourceScalarArg> softmax_static_scalar_args() {
  std::vector<GfxOpenClSourceScalarArg> args = {
      GfxOpenClSourceScalarArg::ElementCount};
  args.insert(args.end(), 3, GfxOpenClSourceScalarArg::StaticU32);
  return args;
}

std::vector<GfxOpenClSourceScalarArg> softmax_dynamic_scalar_args() {
  std::vector<GfxOpenClSourceScalarArg> args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::StaticU32, GfxOpenClSourceScalarArg::StaticU32};
  for (uint32_t axis = 0; axis < 8; ++axis) {
    args.push_back(static_cast<GfxOpenClSourceScalarArg>(
        static_cast<uint32_t>(GfxOpenClSourceScalarArg::Input0Dim0) + axis));
  }
  return args;
}

struct SoftmaxMlirCase {
  std::string name;
  std::function<std::shared_ptr<ov::Node>()> make_node;
};

std::vector<SoftmaxMlirCase> softmax_mlir_cases() {
  return {
      {"SoftmaxV1Axis1F32",
       [] {
         const auto input = param(ov::element::f32, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v1::Softmax>(input, 1);
       }},
      {"SoftmaxV8NegativeAxisF16",
       [] {
         const auto input = param(ov::element::f16, ov::Shape{1, 2, 3, 5});
         return std::make_shared<ov::op::v8::Softmax>(input, -1);
       }},
      {"LogSoftmaxF32",
       [] {
         const auto input = param(ov::element::f32, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v5::LogSoftmax>(input, 2);
       }},
  };
}

class SoftmaxMlirContract final {
public:
  explicit SoftmaxMlirContract(SoftmaxMlirCase test_case)
      : m_case(std::move(test_case)) {}

  void verify() const {
    const auto node = m_case.make_node();
    auto &ctx = gfx_mlir_context();
    auto module = build_mlir_for_node(node, ctx);
    ASSERT_TRUE(module);
    ASSERT_TRUE(static_cast<bool>(
        module.lookupSymbol<mlir::func::FuncOp>("softmax_main")));
    EXPECT_TRUE(mlir_supports_node(node));
    ASSERT_NO_THROW(run_mlir_pipeline(module, /*use_alloca=*/true,
                                      /*use_parallel_loops=*/false));
  }

private:
  SoftmaxMlirCase m_case;
};

std::string
softmax_mlir_case_name(const ::testing::TestParamInfo<SoftmaxMlirCase> &info) {
  return info.param.name;
}

class SoftmaxMlirContractTest
    : public ::testing::TestWithParam<SoftmaxMlirCase> {};

TEST_P(SoftmaxMlirContractTest, BuildsFamilyOwnedModule) {
  SoftmaxMlirContract(GetParam()).verify();
}

INSTANTIATE_TEST_SUITE_P(Softmax, SoftmaxMlirContractTest,
                         ::testing::ValuesIn(softmax_mlir_cases()),
                         softmax_mlir_case_name);

struct SoftmaxOpenClArtifactCase {
  std::string name;
  std::function<std::shared_ptr<ov::Node>()> make_node;
  std::string expected_source_id;
  std::string expected_entry_point;
  uint32_t expected_arg_count = 0;
  std::vector<GfxOpenClSourceScalarArg> expected_scalar_args;
  std::vector<uint32_t> expected_static_u32_scalars;
  const GfxKernelSource *expected_source = nullptr;
};

std::vector<SoftmaxOpenClArtifactCase> softmax_opencl_artifact_cases() {
  return {
      {"SoftmaxF32StaticAxis",
       [] {
         const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v1::Softmax>(data, 1);
       },
       "opencl/generated/softmax_f32",
       "gfx_opencl_generated_softmax_f32",
       6u,
       softmax_static_scalar_args(),
       {2, 3, 4},
       &opencl_generated_softmax_f32_kernel_source()},
      {"SoftmaxF16StaticAxis",
       [] {
         const auto data = param(ov::element::f16, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v1::Softmax>(data, 1);
       },
       "opencl/generated/softmax_f16",
       "gfx_opencl_generated_softmax_f16",
       6u,
       softmax_static_scalar_args(),
       {2, 3, 4},
       &opencl_generated_softmax_f16_kernel_source()},
      {"SoftmaxF32Opset8NegativeAxis",
       [] {
         const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v8::Softmax>(data, -1);
       },
       "opencl/generated/softmax_f32",
       "gfx_opencl_generated_softmax_f32",
       6u,
       softmax_static_scalar_args(),
       {6, 4, 1},
       &opencl_generated_softmax_f32_kernel_source()},
      {"SoftmaxF16DynamicStaticRank",
       [] {
         const auto data =
             param(ov::element::f16,
                   ov::PartialShape{ov::Dimension::dynamic(), 3, 4});
         return std::make_shared<ov::op::v8::Softmax>(data, -2);
       },
       "opencl/generated/softmax_f16_dynamic_static_rank",
       "gfx_opencl_generated_softmax_dynamic_f16",
       13u,
       softmax_dynamic_scalar_args(),
       {3, 1},
       &opencl_generated_softmax_f16_dynamic_kernel_source()},
  };
}

class SoftmaxOpenClArtifactContract final {
public:
  explicit SoftmaxOpenClArtifactContract(SoftmaxOpenClArtifactCase test_case)
      : m_case(std::move(test_case)) {}

  void verify() const {
    const auto node = m_case.make_node();
    OpenClSourceArtifactVerifier(node)
        .expect_artifact(GfxKernelStageFamily::Softmax,
                         m_case.expected_source_id, m_case.expected_entry_point,
                         m_case.expected_arg_count, 1u,
                         m_case.expected_scalar_args, {0},
                         m_case.expected_static_u32_scalars)
        .uses_source(*m_case.expected_source)
        .excludes({"gfx_opencl_baseline_softmax",
                   "gfx_opencl_baseline_binary_f32", "__global long*",
                   "__global half"})
        .has_op(GfxOpenClArtifactOp::Softmax)
        .supports_opencl_compiler();
  }

private:
  SoftmaxOpenClArtifactCase m_case;
};

std::string softmax_opencl_case_name(
    const ::testing::TestParamInfo<SoftmaxOpenClArtifactCase> &info) {
  return info.param.name;
}

class SoftmaxOpenClArtifactContractTest
    : public ::testing::TestWithParam<SoftmaxOpenClArtifactCase> {};

TEST_P(SoftmaxOpenClArtifactContractTest, UsesGeneratedKernelUnitFile) {
  SoftmaxOpenClArtifactContract(GetParam()).verify();
}

INSTANTIATE_TEST_SUITE_P(Softmax, SoftmaxOpenClArtifactContractTest,
                         ::testing::ValuesIn(softmax_opencl_artifact_cases()),
                         softmax_opencl_case_name);

struct SoftmaxMslArtifactCase {
  std::string name;
  std::function<std::shared_ptr<ov::Node>()> make_node;
  std::string expected_kernel_id;
  std::string expected_entry_point;
  const GfxKernelSource *expected_source = nullptr;
};

std::vector<SoftmaxMslArtifactCase> softmax_msl_artifact_cases() {
  return {
      {"SoftmaxF32FileBackedFallback",
       [] {
         const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v1::Softmax>(data, 1);
       },
       "metal/generated/softmax_f32", "gfx_metal_generated_softmax_f32",
       &metal_generated_softmax_f32_kernel_source()},
      {"SoftmaxF16FileBackedFallback",
       [] {
         const auto data = param(ov::element::f16, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v1::Softmax>(data, 1);
       },
       "metal/generated/softmax_f16", "gfx_metal_generated_softmax_f16",
       &metal_generated_softmax_f16_kernel_source()},
      {"LogSoftmaxF32FileBackedFallback",
       [] {
         const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v5::LogSoftmax>(data, 2);
       },
       "metal/generated/logsoftmax_f32", "gfx_metal_generated_logsoftmax_f32",
       &metal_generated_logsoftmax_f32_kernel_source()},
  };
}

class SoftmaxMslArtifactContract final {
public:
  explicit SoftmaxMslArtifactContract(SoftmaxMslArtifactCase test_case)
      : m_case(std::move(test_case)) {}

  void verify() const {
    const auto node = m_case.make_node();
    const auto descriptor = softmax_msl_kernel_descriptor(node);
    ASSERT_TRUE(descriptor.has_value());
    EXPECT_EQ(descriptor->kernel_unit_id, m_case.expected_kernel_id);
    EXPECT_EQ(descriptor->entry_point, m_case.expected_entry_point);

    const auto plan = make_softmax_msl_kernel_source_plan(node);
    ASSERT_TRUE(plan.valid());
    EXPECT_EQ(plan.binding.stage_manifest.stage_family,
              GfxKernelStageFamily::Softmax);
    EXPECT_EQ(plan.source.entry_point, m_case.expected_entry_point);
    EXPECT_EQ(plan.source.signature.arg_count, 5u);
    EXPECT_EQ(plan.source.signature.output_arg_count, 1u);
    EXPECT_EQ(plan.binding.scalar_arg_count, 3u);
    ASSERT_NE(m_case.expected_source, nullptr);
    EXPECT_EQ(plan.source.msl_source,
              std::string(m_case.expected_source->source));
  }

private:
  SoftmaxMslArtifactCase m_case;
};

std::string softmax_msl_case_name(
    const ::testing::TestParamInfo<SoftmaxMslArtifactCase> &info) {
  return info.param.name;
}

class SoftmaxMslArtifactContractTest
    : public ::testing::TestWithParam<SoftmaxMslArtifactCase> {};

TEST_P(SoftmaxMslArtifactContractTest, UsesFamilyOwnedGeneratedKernelUnitFile) {
  SoftmaxMslArtifactContract(GetParam()).verify();
}

INSTANTIATE_TEST_SUITE_P(Softmax, SoftmaxMslArtifactContractTest,
                         ::testing::ValuesIn(softmax_msl_artifact_cases()),
                         softmax_msl_case_name);

TEST(SoftmaxRuntimeBindingContractTest,
     LegacySoftmaxKernelUsesManifestRuntimeParamsBuffer) {
  const auto binding =
      make_backend_custom_kernel_binding_plan(GfxKernelBackendDomain::AppleMsl, "Softmax", "softmax_kernel");
  ASSERT_TRUE(binding.valid);
  EXPECT_EQ(binding.stage_manifest.stage_family, GfxKernelStageFamily::Softmax);
  EXPECT_EQ(binding.scalar_arg_count, 0u);

  ASSERT_TRUE(binding.stage_manifest.custom_kernel.external_buffer_abi.valid);
  EXPECT_EQ(
      materialize_gfx_kernel_external_buffer_roles(
          binding.stage_manifest.custom_kernel.external_buffer_abi),
      std::vector<GfxKernelBufferRole>({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorOutput,
                                        GfxKernelBufferRole::RuntimeParams}));

  EXPECT_EQ(binding.runtime_binding.inputs, std::vector<size_t>({0}));
  EXPECT_EQ(binding.runtime_binding.input_arg_count, 2u);
  EXPECT_EQ(binding.runtime_binding.operand_kinds,
            std::vector<int32_t>({1, 1, 1}));
  EXPECT_EQ(binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 2, 1}));
}

TEST(SoftmaxRuntimeBindingContractTest,
     AppleMslSourcePlannerPreservesManifestRuntimeParamsBuffer) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto softmax = std::make_shared<ov::op::v1::Softmax>(data, 1);
  auto &ctx = gfx_mlir_context();
  auto module = build_mlir_for_node(softmax, ctx);
  ASSERT_TRUE(module);

  const auto binding = make_backend_custom_kernel_roles_binding_plan(
      "Softmax", "softmax_kernel",
      {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
       GfxKernelBufferRole::RuntimeParams},
      GfxKernelBackendDomain::AppleMsl);
  ASSERT_TRUE(binding.valid);
  ASSERT_TRUE(
      annotate_backend_custom_kernel_module_with_binding_plan(module, binding));

  KernelSource source;
  source.module = module;
  source.entry_point = "softmax_main";
  auto configured = make_apple_metal_softmax_kernel_source(
      std::move(source), softmax, ov::Shape{2, 3, 4});
  ASSERT_TRUE(configured.has_value());
  EXPECT_EQ(configured->entry_point, "softmax_kernel");
  EXPECT_EQ(configured->signature.arg_count, 3u);
  EXPECT_EQ(configured->signature.output_arg_count, 1u);
  ASSERT_TRUE(static_cast<bool>(configured->msl_generator));

  const std::string msl = configured->msl_generator(configured->module);
  EXPECT_NE(msl.find("kernel void softmax_kernel"), std::string::npos);
  EXPECT_NE(msl.find("SoftmaxParams"), std::string::npos);
  EXPECT_NE(msl.find("[[buffer(2)]]"), std::string::npos);
  EXPECT_EQ(msl.find("[[buffer(3)]]"), std::string::npos);
  EXPECT_EQ(msl.find("[[buffer(4)]]"), std::string::npos);

  GfxKernelStageManifest manifest{};
  ASSERT_TRUE(read_backend_custom_kernel_stage_manifest_from_module(
      configured->module, GfxKernelBackendDomain::AppleMsl, manifest));
  EXPECT_EQ(
      materialize_gfx_kernel_external_buffer_roles(
          manifest.custom_kernel.external_buffer_abi),
      std::vector<GfxKernelBufferRole>({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorOutput,
                                        GfxKernelBufferRole::RuntimeParams}));
}

struct SoftmaxRouteCase {
  std::string name;
  compiler::BackendTarget target;
  std::shared_ptr<const compiler::OperationSupportPolicy> policy;
  compiler::KernelRegistry kernel_registry;
  compiler::KernelArtifactPayloadResolver payload_resolver;
  LoweringRouteKind expected_route = LoweringRouteKind::GeneratedKernel;
  KernelArtifactOrigin expected_origin = KernelArtifactOrigin::Generated;
  KernelArtifactPayloadKind expected_payload = KernelArtifactPayloadKind::None;
  std::string expected_kernel_id;
  std::string expected_entry_point;
  uint32_t expected_abi_arg_count = 0;
  std::function<std::shared_ptr<ov::Model>()> make_model;
};

struct SoftmaxCompiledContract {
  LoweringPlan plan;
  ManifestBundle manifest;
  ExecutableBundle executable;
};

class SoftmaxRouteContract final {
public:
  explicit SoftmaxRouteContract(SoftmaxRouteCase route)
      : m_route(std::move(route)) {}

  SoftmaxCompiledContract compile() const {
    const compiler::BackendCapabilities capabilities(m_route.target,
                                                     m_route.policy);
    const compiler::OperationLegalizer legalizer(capabilities);
    const compiler::LoweringPlanner planner(m_route.target,
                                            m_route.kernel_registry);
    SoftmaxCompiledContract compiled;
    compiled.plan = planner.plan(m_route.make_model(), legalizer);
    compiled.manifest = compiler::ManifestBuilder{}.build(compiled.plan);
    compiled.executable =
        compiler::ExecutableBundleBuilder(m_route.payload_resolver)
            .build(compiled.manifest, compiled.plan);
    return compiled;
  }

  void verify(const SoftmaxCompiledContract &compiled) const {
    ASSERT_TRUE(compiled.plan.executable());
    EXPECT_EQ(compiled.plan.route_count(m_route.expected_route), 1u);
    ASSERT_TRUE(compiled.manifest.verify().valid());
    ASSERT_TRUE(compiled.executable.verify().valid());

    const auto &stage = find_softmax_stage(compiled.manifest);
    EXPECT_EQ(stage.execution_kind, m_route.expected_route);
    EXPECT_EQ(stage.backend_domain, m_route.target.backend_id());
    EXPECT_EQ(stage.kernel_unit_id, m_route.expected_kernel_id);
    EXPECT_FALSE(stage.memory.hidden_host_copy_allowed);

    const auto &artifact = find_softmax_artifact(compiled.executable);
    EXPECT_EQ(artifact.kernel.origin, m_route.expected_origin);
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
  find_softmax_stage(const ManifestBundle &manifest) const {
    const auto it = std::find_if(manifest.stages.begin(), manifest.stages.end(),
                                 [this](const compiler::StageRecord &stage) {
                                   return stage.kernel_unit_id ==
                                          m_route.expected_kernel_id;
                                 });
    OPENVINO_ASSERT(it != manifest.stages.end(),
                    "Softmax stage is missing from manifest");
    return *it;
  }

  const KernelArtifactDescriptor &
  find_softmax_artifact(const ExecutableBundle &executable) const {
    const auto it = std::find_if(
        executable.artifact_descriptors.begin(),
        executable.artifact_descriptors.end(),
        [this](const KernelArtifactDescriptor &artifact) {
          return artifact.kernel.kernel_id == m_route.expected_kernel_id;
        });
    OPENVINO_ASSERT(it != executable.artifact_descriptors.end(),
                    "Softmax artifact descriptor is missing");
    return *it;
  }

  void verify_payload(const compiler::KernelArtifactPayload *payload) const {
    if (m_route.expected_payload == KernelArtifactPayloadKind::OpenClSource) {
      const auto *source_payload =
          dynamic_cast<const GfxOpenClSourceArtifactPayload *>(payload);
      ASSERT_NE(source_payload, nullptr);
      EXPECT_EQ(source_payload->artifact().stage_manifest.stage_family,
                GfxKernelStageFamily::Softmax);
      return;
    }
    if (m_route.expected_payload ==
        KernelArtifactPayloadKind::VendorDescriptor) {
      const auto *vendor_payload = dynamic_cast<
          const compiler::GfxMetalVendorPrimitiveArtifactPayload *>(payload);
      ASSERT_NE(vendor_payload, nullptr);
      EXPECT_EQ(vendor_payload->contract().descriptor.kind,
                GfxAppleMpsVendorPrimitiveKind::Softmax);
      return;
    }
    const auto *source_payload =
        dynamic_cast<const GfxKernelSourcePayload *>(payload);
    ASSERT_NE(source_payload, nullptr);
    const std::string source(source_payload->source().source);
    EXPECT_NE(source.find("kernel void " + m_route.expected_entry_point),
              std::string::npos);
  }

  SoftmaxRouteCase m_route;
};

std::shared_ptr<ov::Model> softmax_f32_model() {
  auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  auto node = std::make_shared<ov::op::v1::Softmax>(data, 1);
  return model_from_node(node, {data});
}

std::shared_ptr<ov::Model> softmax_f32_last_axis_model() {
  auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  auto node = std::make_shared<ov::op::v1::Softmax>(data, 2);
  return model_from_node(node, {data});
}

std::shared_ptr<ov::Model> softmax_f16_dynamic_model() {
  auto data =
      param(ov::element::f16, ov::PartialShape{ov::Dimension::dynamic(), 3, 4});
  auto node = std::make_shared<ov::op::v8::Softmax>(data, -2);
  return model_from_node(node, {data});
}

std::shared_ptr<ov::Model> logsoftmax_f32_model() {
  auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  auto node = std::make_shared<ov::op::v5::LogSoftmax>(data, 2);
  return model_from_node(node, {data});
}

SoftmaxRouteCase opencl_softmax_static_case() {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  return {"OpenClGeneratedSoftmaxF32",
          target,
          compiler::make_opencl_operation_support_policy(),
          compiler::make_opencl_kernel_registry(target),
          compiler::make_opencl_kernel_artifact_payload_resolver(),
          LoweringRouteKind::GeneratedKernel,
          KernelArtifactOrigin::Generated,
          KernelArtifactPayloadKind::OpenClSource,
          "opencl/generated/softmax_f32",
          "gfx_opencl_generated_softmax_f32",
          6u,
          softmax_f32_model};
}

SoftmaxRouteCase opencl_softmax_dynamic_case() {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  return {"OpenClGeneratedSoftmaxF16DynamicStaticRank",
          target,
          compiler::make_opencl_operation_support_policy(),
          compiler::make_opencl_kernel_registry(target),
          compiler::make_opencl_kernel_artifact_payload_resolver(),
          LoweringRouteKind::GeneratedKernel,
          KernelArtifactOrigin::Generated,
          KernelArtifactPayloadKind::OpenClSource,
          "opencl/generated/softmax_f16_dynamic_static_rank",
          "gfx_opencl_generated_softmax_dynamic_f16",
          13u,
          softmax_f16_dynamic_model};
}

SoftmaxRouteCase metal_mps_softmax_case() {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::Metal);
  return {"MetalMpsSoftmaxFirst",
          target,
          compiler::make_metal_operation_support_policy(),
          compiler::make_metal_kernel_registry(target),
          compiler::make_metal_kernel_artifact_payload_resolver(),
          LoweringRouteKind::VendorPrimitive,
          KernelArtifactOrigin::VendorPrimitive,
          KernelArtifactPayloadKind::VendorDescriptor,
          "metal/vendor/mps_softmax",
          "mps_softmax",
          2u,
          softmax_f32_last_axis_model};
}

SoftmaxRouteCase metal_logsoftmax_msl_case() {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::Metal);
  return {"MetalGeneratedLogSoftmaxF32",
          target,
          compiler::make_metal_operation_support_policy(),
          compiler::make_metal_kernel_registry(target),
          compiler::make_metal_kernel_artifact_payload_resolver(),
          LoweringRouteKind::GeneratedKernel,
          KernelArtifactOrigin::Generated,
          KernelArtifactPayloadKind::MslSource,
          "metal/generated/logsoftmax_f32",
          "softmax_kernel",
          3u,
          logsoftmax_f32_model};
}

std::string softmax_route_case_name(
    const ::testing::TestParamInfo<SoftmaxRouteCase> &info) {
  return info.param.name;
}

class SoftmaxRouteContractTest
    : public ::testing::TestWithParam<SoftmaxRouteCase> {};

TEST_P(SoftmaxRouteContractTest, CompilesThroughExpectedKernelUnit) {
  const SoftmaxRouteContract contract(GetParam());
  contract.verify(contract.compile());
}

INSTANTIATE_TEST_SUITE_P(SoftmaxBackends, SoftmaxRouteContractTest,
                         ::testing::Values(opencl_softmax_static_case(),
                                           opencl_softmax_dynamic_case(),
                                           metal_mps_softmax_case(),
                                           metal_logsoftmax_msl_case()),
                         softmax_route_case_name);

using RuntimeScenarioPtr =
    std::shared_ptr<const ov::test::gfx::RuntimeScenario>;

ov::Tensor filled_f32(const ov::Shape &shape, int modulus, int shift,
                      float scale) {
  ov::Tensor tensor(ov::element::f32, shape);
  auto *data = tensor.data<float>();
  for (size_t i = 0; i < tensor.get_size(); ++i) {
    data[i] =
        static_cast<float>((static_cast<int>(i % modulus) - shift)) * scale;
  }
  return tensor;
}

RuntimeScenarioPtr scenario(std::string name,
                            ov::test::gfx::RuntimeModelBuilder model_builder,
                            ov::test::gfx::RuntimeInputBuilder input_builder,
                            int timeout = 15, float atol = 1e-5f) {
  return ov::test::gfx::runtime_scenario(
      std::move(name), std::move(model_builder), std::move(input_builder),
      timeout, atol);
}

std::vector<RuntimeScenarioPtr> softmax_runtime_scenarios() {
  return {
      scenario(
          "SoftmaxV1LastAxis3D",
          [] {
            auto input = std::make_shared<ov::op::v0::Parameter>(
                ov::element::f32, ov::Shape{1, 4, 4});
            auto softmax = std::make_shared<ov::op::v1::Softmax>(input, 2);
            return model_from_node(softmax, {input});
          },
          [] {
            return std::vector<ov::Tensor>{
                filled_f32({1, 4, 4}, 19, 9, 0.125f)};
          }),
      scenario(
          "SoftmaxV8NegativeAxis4D",
          [] {
            auto input = std::make_shared<ov::op::v0::Parameter>(
                ov::element::f32, ov::Shape{1, 2, 3, 5});
            auto softmax = std::make_shared<ov::op::v8::Softmax>(input, -1);
            return model_from_node(softmax, {input});
          },
          [] {
            return std::vector<ov::Tensor>{
                filled_f32({1, 2, 3, 5}, 29, 14, 0.0625f)};
          }),
  };
}

class SoftmaxRuntimeContractTest
    : public ::testing::TestWithParam<RuntimeScenarioPtr> {};

TEST_P(SoftmaxRuntimeContractTest, MatchesTemplate) {
  const auto &contract = *GetParam();
  ov::test::gfx::RuntimeModelRunner runner;
  runner.compare_model(contract.make_model(), contract.make_inputs(),
                       contract.timeout_seconds(), contract.atol(),
                       contract.rtol());
}

INSTANTIATE_TEST_SUITE_P(Softmax, SoftmaxRuntimeContractTest,
                         ::testing::ValuesIn(softmax_runtime_scenarios()),
                         [](const auto &info) { return info.param->name(); });

} // namespace
} // namespace gfx_plugin
} // namespace ov
