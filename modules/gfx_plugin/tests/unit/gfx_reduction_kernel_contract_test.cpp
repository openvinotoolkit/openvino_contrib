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
#include "kernel_ir/gfx_kernel_source.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "kernel_ir/metal_kernels/reduction_kernels.hpp"
#include "kernel_ir/opencl_kernels/reduction_f32_kernel.hpp"
#include "kernel_ir/opencl_kernels/reduction_logical_bool_kernel.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "mlir/mlir_passes.hpp"
#include "mlir/mlir_support.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_reduction.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/result.hpp"

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
                                             ov::Shape shape) {
  return std::make_shared<ov::op::v0::Parameter>(type, std::move(shape));
}

std::shared_ptr<ov::op::v0::Constant> i64_const(ov::Shape shape,
                                                std::vector<int64_t> values) {
  return ov::op::v0::Constant::create(ov::element::i64, std::move(shape),
                                      std::move(values));
}

std::shared_ptr<ov::Model>
model_from_node(const std::shared_ptr<ov::Node> &node,
                ov::ParameterVector params) {
  return std::make_shared<ov::Model>(
      ov::ResultVector{std::make_shared<ov::op::v0::Result>(node)},
      std::move(params));
}

std::vector<GfxOpenClSourceScalarArg> reduction_static_scalar_args() {
  std::vector<GfxOpenClSourceScalarArg> args = {
      GfxOpenClSourceScalarArg::ElementCount, GfxOpenClSourceScalarArg::OpCode};
  args.insert(args.end(), 15, GfxOpenClSourceScalarArg::StaticU32);
  return args;
}

std::vector<uint32_t> reduce_axis1_static_u32_scalars(bool keep_dims) {
  return keep_dims ? std::vector<uint32_t>{3, 3, 2, 3, 4, 1, 2, 1,
                                           4, 1, 2, 0, 4, 2, 4}
                   : std::vector<uint32_t>{3, 2, 2, 3, 4, 1, 2, 4,
                                           1, 1, 2, 0, 2, 4, 4};
}

std::vector<std::string> non_reduction_opencl_sources() {
  return {"float",
          "long",
          "gfx_opencl_generated_eltwise_logical_unary_bool",
          "gfx_opencl_generated_eltwise_logical_binary_bool",
          "gfx_opencl_generated_eltwise_logical_binary_broadcast_bool",
          "gfx_opencl_generated_eltwise_select_f32",
          "gfx_opencl_generated_eltwise_compare_f32"};
}

struct ReductionOpCase {
  std::string name;
  std::function<std::shared_ptr<ov::Node>()> make_node;
};

std::vector<ReductionOpCase> reduction_mlir_cases() {
  return {
      {"ReduceSumF32LastAxis",
       [] {
         const auto input = param(ov::element::f32, ov::Shape{1, 8400, 4, 16});
         return std::make_shared<ov::op::v1::ReduceSum>(
             input, i64_const(ov::Shape{1}, {3}), false);
       }},
      {"ReduceMeanF32Axis",
       [] {
         const auto input = param(ov::element::f32, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v1::ReduceMean>(
             input, i64_const(ov::Shape{1}, {1}), false);
       }},
      {"ReduceMaxF32Axis",
       [] {
         const auto input = param(ov::element::f32, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v1::ReduceMax>(
             input, i64_const(ov::Shape{1}, {1}), false);
       }},
      {"ReduceMinF32Axis",
       [] {
         const auto input = param(ov::element::f32, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v1::ReduceMin>(
             input, i64_const(ov::Shape{1}, {1}), false);
       }},
      {"ReduceProdF32Axis",
       [] {
         const auto input = param(ov::element::f32, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v1::ReduceProd>(
             input, i64_const(ov::Shape{1}, {1}), false);
       }},
      {"ReduceL1F32Axis",
       [] {
         const auto input = param(ov::element::f32, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v4::ReduceL1>(
             input, i64_const(ov::Shape{1}, {1}), false);
       }},
      {"ReduceL2F32Axis",
       [] {
         const auto input = param(ov::element::f32, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v4::ReduceL2>(
             input, i64_const(ov::Shape{1}, {1}), false);
       }},
      {"ReduceLogicalAndBoolAxis",
       [] {
         const auto input = param(ov::element::boolean, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v1::ReduceLogicalAnd>(
             input, i64_const(ov::Shape{1}, {1}), false);
       }},
      {"ReduceLogicalOrBoolKeepDims",
       [] {
         const auto input = param(ov::element::boolean, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v1::ReduceLogicalOr>(
             input, i64_const(ov::Shape{2}, {1, 2}), true);
       }},
  };
}

class ReductionMlirContract final {
public:
  explicit ReductionMlirContract(ReductionOpCase test_case)
      : m_case(std::move(test_case)) {}

  void verify() const {
    const auto node = m_case.make_node();
    ASSERT_TRUE(node);

    auto &ctx = gfx_mlir_context();
    auto module = build_mlir_for_node(node, ctx);
    ASSERT_TRUE(module);
    ASSERT_TRUE(static_cast<bool>(
        module.lookupSymbol<mlir::func::FuncOp>("reduce_main")));
    EXPECT_TRUE(mlir_supports_node(node));
    ASSERT_NO_THROW(run_mlir_pipeline(module, /*use_alloca=*/true,
                                      /*use_parallel_loops=*/false));
  }

private:
  ReductionOpCase m_case;
};

std::string
reduction_op_case_name(const ::testing::TestParamInfo<ReductionOpCase> &info) {
  return info.param.name;
}

class ReductionMlirContractTest
    : public ::testing::TestWithParam<ReductionOpCase> {};

TEST_P(ReductionMlirContractTest, BuildsFamilyOwnedModule) {
  ReductionMlirContract(GetParam()).verify();
}

INSTANTIATE_TEST_SUITE_P(Reduction, ReductionMlirContractTest,
                         ::testing::ValuesIn(reduction_mlir_cases()),
                         reduction_op_case_name);

struct ReductionOpenClArtifactCase {
  std::string name;
  std::function<std::shared_ptr<ov::Node>()> make_node;
  GfxOpenClArtifactOp op = GfxOpenClArtifactOp::ReduceSum;
  std::vector<uint32_t> static_u32_scalars;
};

std::vector<ReductionOpenClArtifactCase> reduction_opencl_artifact_cases() {
  return {
      {"ReduceSumF32",
       [] {
         const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v1::ReduceSum>(
             data, i64_const(ov::Shape{1}, {1}), false);
       },
       GfxOpenClArtifactOp::ReduceSum, reduce_axis1_static_u32_scalars(false)},
      {"ReduceMeanF32KeepDims",
       [] {
         const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v1::ReduceMean>(
             data, i64_const(ov::Shape{1}, {1}), true);
       },
       GfxOpenClArtifactOp::ReduceMean, reduce_axis1_static_u32_scalars(true)},
      {"ReduceMaxF32",
       [] {
         const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v1::ReduceMax>(
             data, i64_const(ov::Shape{1}, {1}), false);
       },
       GfxOpenClArtifactOp::ReduceMax, reduce_axis1_static_u32_scalars(false)},
      {"ReduceMinF32",
       [] {
         const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v1::ReduceMin>(
             data, i64_const(ov::Shape{1}, {1}), false);
       },
       GfxOpenClArtifactOp::ReduceMin, reduce_axis1_static_u32_scalars(false)},
      {"ReduceProdF32",
       [] {
         const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v1::ReduceProd>(
             data, i64_const(ov::Shape{1}, {1}), false);
       },
       GfxOpenClArtifactOp::ReduceProd, reduce_axis1_static_u32_scalars(false)},
      {"ReduceL1F32",
       [] {
         const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v4::ReduceL1>(
             data, i64_const(ov::Shape{1}, {1}), false);
       },
       GfxOpenClArtifactOp::ReduceL1, reduce_axis1_static_u32_scalars(false)},
      {"ReduceL2F32",
       [] {
         const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v4::ReduceL2>(
             data, i64_const(ov::Shape{1}, {1}), false);
       },
       GfxOpenClArtifactOp::ReduceL2, reduce_axis1_static_u32_scalars(false)},
  };
}

class ReductionNumericOpenClArtifactContract final {
public:
  explicit ReductionNumericOpenClArtifactContract(
      ReductionOpenClArtifactCase test_case)
      : m_case(std::move(test_case)) {}

  void verify() const {
    const auto node = m_case.make_node();
    OpenClSourceArtifactVerifier(node)
        .expect_artifact(
            GfxKernelStageFamily::Reduction, "opencl/generated/reduction_f32",
            "gfx_opencl_generated_reduction_f32", 19u, 1u,
            reduction_static_scalar_args(), {0}, m_case.static_u32_scalars)
        .uses_source(opencl_generated_reduction_f32_kernel_source())
        .excludes({"gfx_opencl_generated_reduction_bool",
                   "gfx_opencl_baseline_binary_f32"})
        .has_op(m_case.op)
        .supports_opencl_compiler();
  }

private:
  ReductionOpenClArtifactCase m_case;
};

std::string reduction_opencl_case_name(
    const ::testing::TestParamInfo<ReductionOpenClArtifactCase> &info) {
  return info.param.name;
}

class ReductionNumericOpenClArtifactContractTest
    : public ::testing::TestWithParam<ReductionOpenClArtifactCase> {};

TEST_P(ReductionNumericOpenClArtifactContractTest,
       UsesFamilyOwnedGeneratedKernelUnit) {
  ReductionNumericOpenClArtifactContract(GetParam()).verify();
}

INSTANTIATE_TEST_SUITE_P(Reduction, ReductionNumericOpenClArtifactContractTest,
                         ::testing::ValuesIn(reduction_opencl_artifact_cases()),
                         reduction_opencl_case_name);

TEST(ReductionOpenClArtifactContract,
     ReduceLogicalBoolArtifactsCarryStaticAxisMetadata) {
  const auto data = param(ov::element::boolean, ov::Shape{2, 3, 4});

  const auto reduce_and = std::make_shared<ov::op::v1::ReduceLogicalAnd>(
      data, i64_const(ov::Shape{1}, {1}), false);
  OpenClSourceArtifactVerifier(reduce_and)
      .expect_artifact(GfxKernelStageFamily::Reduction,
                       "opencl/generated/reduction_bool",
                       "gfx_opencl_generated_reduction_bool", 19u, 1u,
                       reduction_static_scalar_args(), {0},
                       reduce_axis1_static_u32_scalars(false))
      .uses_source(opencl_generated_reduction_bool_kernel_source())
      .excludes(non_reduction_opencl_sources())
      .has_op(GfxOpenClArtifactOp::ReduceLogicalAnd)
      .supports_opencl_compiler();

  const auto reduce_or = std::make_shared<ov::op::v1::ReduceLogicalOr>(
      data, i64_const(ov::Shape{2}, {1, 2}), true);
  const std::vector<uint32_t> or_static_u32_scalars = {3, 3, 2, 3, 4, 1, 2, 1,
                                                       1, 1, 6, 0, 4, 4, 4};
  OpenClSourceArtifactVerifier(reduce_or)
      .expect_artifact(
          GfxKernelStageFamily::Reduction, "opencl/generated/reduction_bool",
          "gfx_opencl_generated_reduction_bool", 19u, 1u,
          reduction_static_scalar_args(), {0}, or_static_u32_scalars)
      .uses_source(opencl_generated_reduction_bool_kernel_source())
      .excludes(non_reduction_opencl_sources())
      .has_op(GfxOpenClArtifactOp::ReduceLogicalOr)
      .supports_opencl_compiler();
}

struct ReductionMslArtifactCase {
  std::string name;
  std::function<std::shared_ptr<ov::Node>()> make_node;
  std::string expected_kernel_id;
  std::string expected_entry_point;
  const GfxKernelSource *expected_source = nullptr;
  uint32_t expected_op_code = 0;
};

std::vector<ReductionMslArtifactCase> reduction_msl_artifact_cases() {
  return {
      {"ReduceSumF32",
       [] {
         const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v1::ReduceSum>(
             data, i64_const(ov::Shape{1}, {1}), false);
       },
       "metal/generated/reduction_f32", "gfx_metal_generated_reduction_f32",
       &metal_generated_reduction_f32_kernel_source(),
       reduction_kernel_op_code(ReduceKind::Sum)},
      {"ReduceMeanF32KeepDims",
       [] {
         const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v1::ReduceMean>(
             data, i64_const(ov::Shape{1}, {1}), true);
       },
       "metal/generated/reduction_f32", "gfx_metal_generated_reduction_f32",
       &metal_generated_reduction_f32_kernel_source(),
       reduction_kernel_op_code(ReduceKind::Mean)},
      {"ReduceLogicalAndBool",
       [] {
         const auto data = param(ov::element::boolean, ov::Shape{2, 3, 4});
         return std::make_shared<ov::op::v1::ReduceLogicalAnd>(
             data, i64_const(ov::Shape{1}, {1}), false);
       },
       "metal/generated/reduction_logical_bool",
       "gfx_metal_generated_reduction_logical_bool",
       &metal_generated_reduction_logical_bool_kernel_source(),
       reduction_kernel_op_code(ReduceKind::LogicalAnd)},
  };
}

class ReductionMslArtifactContract final {
public:
  explicit ReductionMslArtifactContract(ReductionMslArtifactCase test_case)
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
    EXPECT_EQ(support.preferred_route, m_case.expected_kernel_id);

    const auto plan = make_reduction_msl_kernel_source_plan(node);
    ASSERT_TRUE(plan.valid());
    EXPECT_EQ(plan.binding.stage_manifest.stage_family,
              GfxKernelStageFamily::Reduction);
    EXPECT_EQ(plan.source.entry_point, m_case.expected_entry_point);
    EXPECT_EQ(plan.source.signature.arg_count, 10u);
    EXPECT_EQ(plan.source.signature.output_arg_count, 1u);
    EXPECT_EQ(plan.binding.scalar_arg_count, 3u);
    ASSERT_EQ(plan.binding.runtime_binding.scalar_args.size(), 3u);
    EXPECT_EQ(
        static_cast<uint32_t>(plan.binding.runtime_binding.scalar_args[2]),
        m_case.expected_op_code);
    ASSERT_NE(m_case.expected_source, nullptr);
    EXPECT_EQ(plan.source.msl_source,
              std::string(m_case.expected_source->source));
  }

private:
  ReductionMslArtifactCase m_case;
};

std::string reduction_msl_case_name(
    const ::testing::TestParamInfo<ReductionMslArtifactCase> &info) {
  return info.param.name;
}

class ReductionMslArtifactContractTest
    : public ::testing::TestWithParam<ReductionMslArtifactCase> {};

TEST_P(ReductionMslArtifactContractTest,
       UsesFamilyOwnedGeneratedKernelUnitFile) {
  ReductionMslArtifactContract(GetParam()).verify();
}

INSTANTIATE_TEST_SUITE_P(Reduction, ReductionMslArtifactContractTest,
                         ::testing::ValuesIn(reduction_msl_artifact_cases()),
                         reduction_msl_case_name);

struct ReductionRouteCase {
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

struct ReductionCompiledContract {
  LoweringPlan plan;
  ManifestBundle manifest;
  ExecutableBundle executable;
};

class ReductionRouteContract final {
public:
  explicit ReductionRouteContract(ReductionRouteCase route)
      : m_route(std::move(route)) {}

  ReductionCompiledContract compile() const {
    const compiler::BackendCapabilities capabilities(m_route.target,
                                                     m_route.policy);
    const compiler::OperationLegalizer legalizer(capabilities);
    const compiler::LoweringPlanner planner(m_route.target,
                                            m_route.kernel_registry);
    ReductionCompiledContract compiled;
    compiled.plan = planner.plan(m_route.make_model(), legalizer);
    compiled.manifest = compiler::ManifestBuilder{}.build(compiled.plan);
    compiled.executable =
        compiler::ExecutableBundleBuilder(m_route.payload_resolver)
            .build(compiled.manifest, compiled.plan);
    return compiled;
  }

  void verify(const ReductionCompiledContract &compiled) const {
    ASSERT_TRUE(compiled.plan.executable());
    EXPECT_EQ(compiled.plan.route_count(m_route.expected_route), 1u);
    ASSERT_TRUE(compiled.manifest.verify().valid());
    ASSERT_TRUE(compiled.executable.verify().valid());

    const auto &stage = find_reduction_stage(compiled.manifest);
    EXPECT_EQ(stage.execution_kind, m_route.expected_route);
    EXPECT_EQ(stage.backend_domain, m_route.target.backend_id());
    EXPECT_EQ(stage.kernel_unit_id, m_route.expected_kernel_id);
    EXPECT_FALSE(stage.memory.hidden_host_copy_allowed);

    const auto &artifact = find_reduction_artifact(compiled.executable);
    EXPECT_EQ(artifact.kernel.origin, m_route.expected_origin);
    EXPECT_EQ(artifact.payload_kind, m_route.expected_payload);
    EXPECT_EQ(artifact.kernel.kernel_id, m_route.expected_kernel_id);
    EXPECT_EQ(artifact.entry_point, m_route.expected_entry_point);
    EXPECT_EQ(artifact.abi_arg_count, m_route.expected_abi_arg_count);
    EXPECT_EQ(artifact.abi_output_arg_count, 1u);
    EXPECT_TRUE(artifact.kernel.exception_ticket.empty());
    EXPECT_TRUE(artifact.kernel.exception_reason.empty());
    EXPECT_TRUE(artifact.kernel.exception_removal_condition.empty());

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
  find_reduction_stage(const ManifestBundle &manifest) const {
    const auto it = std::find_if(manifest.stages.begin(), manifest.stages.end(),
                                 [this](const compiler::StageRecord &stage) {
                                   return stage.kernel_unit_id ==
                                          m_route.expected_kernel_id;
                                 });
    OPENVINO_ASSERT(it != manifest.stages.end(),
                    "Reduction stage is missing from manifest");
    return *it;
  }

  const KernelArtifactDescriptor &
  find_reduction_artifact(const ExecutableBundle &executable) const {
    const auto it = std::find_if(
        executable.artifact_descriptors.begin(),
        executable.artifact_descriptors.end(),
        [this](const KernelArtifactDescriptor &artifact) {
          return artifact.kernel.kernel_id == m_route.expected_kernel_id;
        });
    OPENVINO_ASSERT(it != executable.artifact_descriptors.end(),
                    "Reduction artifact descriptor is missing");
    return *it;
  }

  void verify_payload(const compiler::KernelArtifactPayload *payload) const {
    if (m_route.expected_payload == KernelArtifactPayloadKind::OpenClSource) {
      const auto *source_payload =
          dynamic_cast<const GfxOpenClSourceArtifactPayload *>(payload);
      ASSERT_NE(source_payload, nullptr);
      EXPECT_EQ(source_payload->artifact().stage_manifest.stage_family,
                GfxKernelStageFamily::Reduction);
      return;
    }
    const auto *source_payload =
        dynamic_cast<const GfxKernelSourcePayload *>(payload);
    ASSERT_NE(source_payload, nullptr);
    const std::string source(source_payload->source().source);
    EXPECT_NE(source.find("kernel void " + m_route.expected_entry_point),
              std::string::npos);
  }

  ReductionRouteCase m_route;
};

std::shared_ptr<ov::Model> reduce_sum_model() {
  auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  auto node = std::make_shared<ov::op::v1::ReduceSum>(
      data, i64_const(ov::Shape{1}, {1}), false);
  return model_from_node(node, {data});
}

std::shared_ptr<ov::Model> reduce_logical_and_model() {
  auto data = param(ov::element::boolean, ov::Shape{2, 3, 4});
  auto node = std::make_shared<ov::op::v1::ReduceLogicalAnd>(
      data, i64_const(ov::Shape{1}, {1}), false);
  return model_from_node(node, {data});
}

ReductionRouteCase opencl_reduction_f32_case() {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  return {"OpenClGeneratedReduceSumF32",
          target,
          compiler::make_opencl_operation_support_policy(),
          compiler::make_opencl_kernel_registry(target),
          compiler::make_opencl_kernel_artifact_payload_resolver(),
          LoweringRouteKind::GeneratedKernel,
          KernelArtifactOrigin::Generated,
          KernelArtifactPayloadKind::OpenClSource,
          "opencl/generated/reduction_f32",
          "gfx_opencl_generated_reduction_f32",
          19u,
          reduce_sum_model};
}

ReductionRouteCase opencl_reduction_logical_case() {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  return {"OpenClGeneratedReduceLogicalAndBool",
          target,
          compiler::make_opencl_operation_support_policy(),
          compiler::make_opencl_kernel_registry(target),
          compiler::make_opencl_kernel_artifact_payload_resolver(),
          LoweringRouteKind::GeneratedKernel,
          KernelArtifactOrigin::Generated,
          KernelArtifactPayloadKind::OpenClSource,
          "opencl/generated/reduction_bool",
          "gfx_opencl_generated_reduction_bool",
          19u,
          reduce_logical_and_model};
}

ReductionRouteCase metal_reduction_f32_case() {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::Metal);
  return {"MetalGeneratedReduceSumF32",
          target,
          compiler::make_metal_operation_support_policy(),
          compiler::make_metal_kernel_registry(target),
          compiler::make_metal_kernel_artifact_payload_resolver(),
          LoweringRouteKind::GeneratedKernel,
          KernelArtifactOrigin::Generated,
          KernelArtifactPayloadKind::MslSource,
          "metal/generated/reduction_f32",
          "gfx_metal_generated_reduction_f32",
          10u,
          reduce_sum_model};
}

ReductionRouteCase metal_reduction_logical_case() {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::Metal);
  return {"MetalGeneratedReduceLogicalAndBool",
          target,
          compiler::make_metal_operation_support_policy(),
          compiler::make_metal_kernel_registry(target),
          compiler::make_metal_kernel_artifact_payload_resolver(),
          LoweringRouteKind::GeneratedKernel,
          KernelArtifactOrigin::Generated,
          KernelArtifactPayloadKind::MslSource,
          "metal/generated/reduction_logical_bool",
          "gfx_metal_generated_reduction_logical_bool",
          10u,
          reduce_logical_and_model};
}

std::string reduction_route_case_name(
    const ::testing::TestParamInfo<ReductionRouteCase> &info) {
  return info.param.name;
}

class ReductionRouteContractTest
    : public ::testing::TestWithParam<ReductionRouteCase> {};

TEST_P(ReductionRouteContractTest, CompilesThroughExpectedKernelUnit) {
  const ReductionRouteContract contract(GetParam());
  contract.verify(contract.compile());
}

INSTANTIATE_TEST_SUITE_P(ReductionBackends, ReductionRouteContractTest,
                         ::testing::Values(opencl_reduction_f32_case(),
                                           opencl_reduction_logical_case(),
                                           metal_reduction_f32_case(),
                                           metal_reduction_logical_case()),
                         reduction_route_case_name);

} // namespace
} // namespace gfx_plugin
} // namespace ov
