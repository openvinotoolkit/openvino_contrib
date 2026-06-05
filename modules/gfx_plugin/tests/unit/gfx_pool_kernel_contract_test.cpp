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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "backends/metal/compiler/apple_vendor_descriptors.hpp"
#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "mlir/mlir_passes.hpp"
#include "mlir/mlir_support.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/parameter.hpp"
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

std::shared_ptr<ov::Node> maxpool_node(const ov::element::Type &type,
                                       ov::Shape shape) {
  auto data = param(type, std::move(shape));
  return std::make_shared<ov::op::v1::MaxPool>(
      data, ov::Strides{2, 2}, ov::Shape{0, 0}, ov::Shape{0, 0},
      ov::Shape{2, 2}, ov::op::RoundingType::FLOOR);
}

std::shared_ptr<ov::Node> avgpool_node(const ov::element::Type &type,
                                       ov::Shape shape) {
  auto data = param(type, std::move(shape));
  return std::make_shared<ov::op::v1::AvgPool>(
      data, ov::Strides{2, 2}, ov::Shape{0, 0}, ov::Shape{0, 0},
      ov::Shape{2, 2}, true, ov::op::RoundingType::FLOOR);
}

std::vector<GfxOpenClSourceScalarArg> pool2d_static_scalar_args() {
  std::vector<GfxOpenClSourceScalarArg> args = {
      GfxOpenClSourceScalarArg::ElementCount};
  args.insert(args.end(), 18, GfxOpenClSourceScalarArg::StaticU32);
  return args;
}

struct PoolMlirCase {
  std::string name;
  std::function<std::shared_ptr<ov::Node>()> make_node;
};

std::vector<PoolMlirCase> pool_mlir_cases() {
  return {
      {"MaxPoolF32",
       [] { return maxpool_node(ov::element::f32, ov::Shape{1, 4, 4, 4}); }},
      {"AvgPoolF16",
       [] { return avgpool_node(ov::element::f16, ov::Shape{1, 4, 4, 4}); }},
  };
}

class PoolMlirContract final {
public:
  explicit PoolMlirContract(PoolMlirCase test_case)
      : m_case(std::move(test_case)) {}

  void verify() const {
    const auto node = m_case.make_node();
    auto &ctx = gfx_mlir_context();
    auto module = build_mlir_for_node(node, ctx);
    ASSERT_TRUE(module);
    ASSERT_TRUE(static_cast<bool>(
        module.lookupSymbol<mlir::func::FuncOp>(
            node->get_type_name() == std::string("AvgPool") ? "avgpool_main"
                                                             : "maxpool_main")));
    EXPECT_TRUE(mlir_supports_node(node));
    ASSERT_NO_THROW(run_mlir_pipeline(module, /*use_alloca=*/true,
                                      /*use_parallel_loops=*/false));
  }

private:
  PoolMlirCase m_case;
};

std::string pool_mlir_case_name(
    const ::testing::TestParamInfo<PoolMlirCase> &info) {
  return info.param.name;
}

class PoolMlirContractTest : public ::testing::TestWithParam<PoolMlirCase> {};

TEST_P(PoolMlirContractTest, BuildsFamilyOwnedModule) {
  PoolMlirContract(GetParam()).verify();
}

INSTANTIATE_TEST_SUITE_P(Pooling, PoolMlirContractTest,
                         ::testing::ValuesIn(pool_mlir_cases()),
                         pool_mlir_case_name);

struct PoolOpenClArtifactCase {
  std::string name;
  std::function<std::shared_ptr<ov::Node>()> make_node;
  std::string expected_source_id;
  std::string expected_entry_point;
  std::vector<uint32_t> expected_static_u32_scalars;
  GfxOpenClArtifactOp expected_op = GfxOpenClArtifactOp::Identity;
};

std::vector<PoolOpenClArtifactCase> pool_opencl_artifact_cases() {
  return {
      {"MaxPoolF32",
       [] { return maxpool_node(ov::element::f32, ov::Shape{1, 3, 4, 4}); },
       "opencl/generated/pool2d_f32",
       "gfx_opencl_generated_pool2d_f32",
       {1, 3, 4, 4, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0, 2, 2, 0, 1},
       GfxOpenClArtifactOp::MaxPool},
      {"AvgPoolF16",
       [] { return avgpool_node(ov::element::f16, ov::Shape{1, 2, 4, 4}); },
       "opencl/generated/pool2d_f16",
       "gfx_opencl_generated_pool2d_f16",
       {1, 2, 4, 4, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0, 2, 2, 1, 1},
       GfxOpenClArtifactOp::AvgPool},
  };
}

class PoolOpenClArtifactContract final {
public:
  explicit PoolOpenClArtifactContract(PoolOpenClArtifactCase test_case)
      : m_case(std::move(test_case)) {}

  void verify() const {
    const auto node = m_case.make_node();
    OpenClSourceArtifactVerifier(node)
        .expect_artifact(GfxKernelStageFamily::Pooling,
                         m_case.expected_source_id, m_case.expected_entry_point,
                         21u, 1u, pool2d_static_scalar_args(), {0},
                         m_case.expected_static_u32_scalars)
        .has_op(m_case.expected_op)
        .supports_opencl_compiler();
  }

private:
  PoolOpenClArtifactCase m_case;
};

std::string pool_opencl_case_name(
    const ::testing::TestParamInfo<PoolOpenClArtifactCase> &info) {
  return info.param.name;
}

class PoolOpenClArtifactContractTest
    : public ::testing::TestWithParam<PoolOpenClArtifactCase> {};

TEST_P(PoolOpenClArtifactContractTest, UsesGeneratedKernelArtifactContract) {
  PoolOpenClArtifactContract(GetParam()).verify();
}

INSTANTIATE_TEST_SUITE_P(Pooling, PoolOpenClArtifactContractTest,
                         ::testing::ValuesIn(pool_opencl_artifact_cases()),
                         pool_opencl_case_name);

struct PoolRouteCase {
  std::string name;
  compiler::BackendTarget target;
  std::shared_ptr<const compiler::OperationSupportPolicy> policy;
  compiler::KernelRegistry kernel_registry;
  compiler::KernelArtifactPayloadResolver payload_resolver;
  LoweringRouteKind expected_route = LoweringRouteKind::Unsupported;
  KernelArtifactOrigin expected_origin = KernelArtifactOrigin::Unknown;
  KernelArtifactPayloadKind expected_payload = KernelArtifactPayloadKind::None;
  std::string expected_kernel_id;
  std::string expected_entry_point;
  uint32_t expected_abi_arg_count = 0;
  bool expected_avg = false;
  std::function<std::shared_ptr<ov::Model>()> make_model;
};

struct PoolCompiledContract {
  LoweringPlan plan;
  ManifestBundle manifest;
  ExecutableBundle executable;
};

class PoolRouteContract final {
public:
  explicit PoolRouteContract(PoolRouteCase route)
      : m_route(std::move(route)) {}

  PoolCompiledContract compile() const {
    const compiler::BackendCapabilities capabilities(m_route.target,
                                                     m_route.policy);
    const compiler::OperationLegalizer legalizer(capabilities);
    const compiler::LoweringPlanner planner(m_route.target,
                                            m_route.kernel_registry);
    PoolCompiledContract compiled;
    compiled.plan = planner.plan(m_route.make_model(), legalizer);
    compiled.manifest = compiler::ManifestBuilder{}.build(compiled.plan);
    compiled.executable =
        compiler::ExecutableBundleBuilder(m_route.payload_resolver)
            .build(compiled.manifest, compiled.plan);
    return compiled;
  }

  void verify(const PoolCompiledContract &compiled) const {
    ASSERT_TRUE(compiled.plan.executable());
    EXPECT_EQ(compiled.plan.route_count(m_route.expected_route), 1u);
    ASSERT_TRUE(compiled.manifest.verify().valid());
    ASSERT_TRUE(compiled.executable.verify().valid());

    const auto &stage = find_pool_stage(compiled.manifest);
    EXPECT_EQ(stage.execution_kind, m_route.expected_route);
    EXPECT_EQ(stage.backend_domain, m_route.target.backend_id());
    EXPECT_EQ(stage.kernel_unit_id, m_route.expected_kernel_id);
    EXPECT_FALSE(stage.memory.hidden_host_copy_allowed);

    const auto &artifact = find_pool_artifact(compiled.executable);
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
  find_pool_stage(const ManifestBundle &manifest) const {
    const auto it = std::find_if(
        manifest.stages.begin(), manifest.stages.end(),
        [](const compiler::StageRecord &stage) {
          return stage.normalized_op_family == "MaxPool" ||
                 stage.normalized_op_family == "AvgPool";
        });
    OPENVINO_ASSERT(it != manifest.stages.end(),
                    "Pooling stage is missing from manifest");
    return *it;
  }

  const KernelArtifactDescriptor &
  find_pool_artifact(const ExecutableBundle &executable) const {
    const auto it = std::find_if(
        executable.artifact_descriptors.begin(),
        executable.artifact_descriptors.end(),
        [this](const KernelArtifactDescriptor &artifact) {
          return artifact.kernel.kernel_id == m_route.expected_kernel_id;
        });
    OPENVINO_ASSERT(it != executable.artifact_descriptors.end(),
                    "Pooling artifact descriptor is missing");
    return *it;
  }

  void verify_payload(const compiler::KernelArtifactPayload *payload) const {
    if (m_route.expected_payload == KernelArtifactPayloadKind::OpenClSource) {
      const auto *source_payload =
          dynamic_cast<const GfxOpenClSourceArtifactPayload *>(payload);
      ASSERT_NE(source_payload, nullptr);
      EXPECT_EQ(source_payload->artifact().stage_manifest.stage_family,
                GfxKernelStageFamily::Pooling);
      EXPECT_EQ(source_payload->artifact().op,
                m_route.expected_avg ? GfxOpenClArtifactOp::AvgPool
                                     : GfxOpenClArtifactOp::MaxPool);
      return;
    }

    const auto *vendor_payload = dynamic_cast<
        const compiler::GfxMetalVendorPrimitiveArtifactPayload *>(payload);
    ASSERT_NE(vendor_payload, nullptr);
    const auto &contract = vendor_payload->contract();
    ASSERT_TRUE(contract.valid);
    EXPECT_EQ(contract.descriptor.kind, GfxAppleMpsVendorPrimitiveKind::Pool2D);
    EXPECT_EQ(contract.descriptor.pool2d.is_avg, m_route.expected_avg ? 1u : 0u);
    ASSERT_EQ(contract.input_descs.size(), 1u);
    ASSERT_EQ(contract.output_descs.size(), 1u);
    EXPECT_EQ(contract.input_descs.front().image_feature_channels, 1u);
    EXPECT_EQ(contract.output_descs.front().image_feature_channels, 1u);
    EXPECT_EQ(contract.external_buffer_abi.buffer_count,
              m_route.expected_abi_arg_count);
    EXPECT_EQ(contract.external_buffer_abi.output_buffer_count, 1u);
  }

  PoolRouteCase m_route;
};

std::shared_ptr<ov::Model> opencl_maxpool_model() {
  auto data = param(ov::element::f32, ov::Shape{1, 3, 4, 4});
  return model_from_node(
      std::make_shared<ov::op::v1::MaxPool>(
          data, ov::Strides{2, 2}, ov::Shape{0, 0}, ov::Shape{0, 0},
          ov::Shape{2, 2}, ov::op::RoundingType::FLOOR),
      {data});
}

std::shared_ptr<ov::Model> opencl_avgpool_model() {
  auto data = param(ov::element::f16, ov::Shape{1, 2, 4, 4});
  return model_from_node(
      std::make_shared<ov::op::v1::AvgPool>(
          data, ov::Strides{2, 2}, ov::Shape{0, 0}, ov::Shape{0, 0},
          ov::Shape{2, 2}, true, ov::op::RoundingType::FLOOR),
      {data});
}

std::shared_ptr<ov::Model> metal_maxpool_model() {
  auto data = param(ov::element::f32, ov::Shape{1, 1, 4, 4});
  return model_from_node(
      std::make_shared<ov::op::v1::MaxPool>(
          data, ov::Strides{2, 2}, ov::Shape{0, 0}, ov::Shape{0, 0},
          ov::Shape{2, 2}, ov::op::RoundingType::FLOOR),
      {data});
}

std::shared_ptr<ov::Model> metal_avgpool_model() {
  auto data = param(ov::element::f32, ov::Shape{1, 1, 4, 4});
  return model_from_node(
      std::make_shared<ov::op::v1::AvgPool>(
          data, ov::Strides{2, 2}, ov::Shape{0, 0}, ov::Shape{0, 0},
          ov::Shape{2, 2}, true, ov::op::RoundingType::FLOOR),
      {data});
}

std::shared_ptr<ov::Node> indexed_maxpool_node() {
  auto data = param(ov::element::f16, ov::Shape{1, 4, 16, 16});
  return std::make_shared<ov::op::v8::MaxPool>(
      data, ov::Strides{2, 2}, ov::Strides{1, 1}, ov::Shape{0, 0},
      ov::Shape{0, 0}, ov::Shape{2, 2}, ov::op::RoundingType::FLOOR,
      ov::op::PadType::EXPLICIT, ov::element::i64, 0);
}

PoolRouteCase opencl_maxpool_case() {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  return {"OpenClGeneratedMaxPoolF32",
          target,
          compiler::make_opencl_operation_support_policy(),
          compiler::make_opencl_kernel_registry(target),
          compiler::make_opencl_kernel_artifact_payload_resolver(),
          LoweringRouteKind::GeneratedKernel,
          KernelArtifactOrigin::Generated,
          KernelArtifactPayloadKind::OpenClSource,
          "opencl/generated/pool2d_f32",
          "gfx_opencl_generated_pool2d_f32",
          21u,
          false,
          opencl_maxpool_model};
}

PoolRouteCase opencl_avgpool_case() {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  return {"OpenClGeneratedAvgPoolF16",
          target,
          compiler::make_opencl_operation_support_policy(),
          compiler::make_opencl_kernel_registry(target),
          compiler::make_opencl_kernel_artifact_payload_resolver(),
          LoweringRouteKind::GeneratedKernel,
          KernelArtifactOrigin::Generated,
          KernelArtifactPayloadKind::OpenClSource,
          "opencl/generated/pool2d_f16",
          "gfx_opencl_generated_pool2d_f16",
          21u,
          true,
          opencl_avgpool_model};
}

PoolRouteCase metal_mps_maxpool_case() {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::Metal);
  return {"MetalMpsMaxPoolFirst",
          target,
          compiler::make_metal_operation_support_policy(),
          compiler::make_metal_kernel_registry(target),
          compiler::make_metal_kernel_artifact_payload_resolver(),
          LoweringRouteKind::VendorPrimitive,
          KernelArtifactOrigin::VendorPrimitive,
          KernelArtifactPayloadKind::VendorDescriptor,
          "metal/vendor/mps_pool2d",
          "mps_pool2d",
          3u,
          false,
          metal_maxpool_model};
}

PoolRouteCase metal_mps_avgpool_case() {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::Metal);
  return {"MetalMpsAvgPoolFirst",
          target,
          compiler::make_metal_operation_support_policy(),
          compiler::make_metal_kernel_registry(target),
          compiler::make_metal_kernel_artifact_payload_resolver(),
          LoweringRouteKind::VendorPrimitive,
          KernelArtifactOrigin::VendorPrimitive,
          KernelArtifactPayloadKind::VendorDescriptor,
          "metal/vendor/mps_pool2d",
          "mps_pool2d",
          3u,
          true,
          metal_avgpool_model};
}

std::string pool_route_case_name(
    const ::testing::TestParamInfo<PoolRouteCase> &info) {
  return info.param.name;
}

class PoolRouteContractTest : public ::testing::TestWithParam<PoolRouteCase> {};

TEST_P(PoolRouteContractTest, CompilesThroughExpectedKernelUnit) {
  const PoolRouteContract contract(GetParam());
  contract.verify(contract.compile());
}

INSTANTIATE_TEST_SUITE_P(PoolingBackends, PoolRouteContractTest,
                         ::testing::Values(opencl_maxpool_case(),
                                           opencl_avgpool_case(),
                                           metal_mps_maxpool_case(),
                                           metal_mps_avgpool_case()),
                         pool_route_case_name);

struct PoolUnsupportedRouteCase {
  std::string name;
  compiler::BackendTarget target;
  std::shared_ptr<const compiler::OperationSupportPolicy> policy;
  std::function<std::shared_ptr<ov::Node>()> make_node;
  std::string expected_reason;
};

class PoolUnsupportedRouteContract final {
public:
  explicit PoolUnsupportedRouteContract(PoolUnsupportedRouteCase route)
      : m_route(std::move(route)) {}

  void verify() const {
    const compiler::BackendCapabilities capabilities(m_route.target,
                                                     m_route.policy);
    const compiler::OperationLegalizer legalizer(capabilities);
    const auto support = legalizer.query(m_route.make_node());
    EXPECT_FALSE(support.semantic_legal);
    EXPECT_EQ(support.preferred_route_kind, LoweringRouteKind::Unsupported);
    EXPECT_EQ(support.semantic_reason, m_route.expected_reason);
  }

private:
  PoolUnsupportedRouteCase m_route;
};

PoolUnsupportedRouteCase opencl_indexed_maxpool_reject_case() {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  return {"OpenClRejectsIndexedMaxPoolWithoutKernelUnit",
          target,
          compiler::make_opencl_operation_support_policy(),
          indexed_maxpool_node,
          "missing_opencl_pooling_kernel_unit"};
}

PoolUnsupportedRouteCase metal_indexed_maxpool_reject_case() {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::Metal);
  return {"MetalRejectsIndexedMaxPoolWithoutMpsFamilyRoute",
          target,
          compiler::make_metal_operation_support_policy(),
          indexed_maxpool_node,
          "missing_apple_pooling_mps_family_route"};
}

std::string pool_unsupported_case_name(
    const ::testing::TestParamInfo<PoolUnsupportedRouteCase> &info) {
  return info.param.name;
}

class PoolUnsupportedRouteContractTest
    : public ::testing::TestWithParam<PoolUnsupportedRouteCase> {};

TEST_P(PoolUnsupportedRouteContractTest, RejectsMissingExplicitRoute) {
  PoolUnsupportedRouteContract(GetParam()).verify();
}

INSTANTIATE_TEST_SUITE_P(
    PoolingBackends, PoolUnsupportedRouteContractTest,
    ::testing::Values(opencl_indexed_maxpool_reject_case(),
                      metal_indexed_maxpool_reject_case()),
    pool_unsupported_case_name);

TEST(PoolLoweringPlannerContractTest,
     OpenClPoolSupportRequiresRegisteredKernelUnit) {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  const compiler::BackendCapabilities capabilities(
      target, compiler::make_opencl_operation_support_policy());
  const compiler::OperationLegalizer legalizer(capabilities);
  const compiler::LoweringPlanner planner(
      target, compiler::make_common_kernel_registry(target));

  const auto plan = planner.plan(opencl_maxpool_model(), legalizer);
  EXPECT_FALSE(plan.executable());

  bool saw_pool_type = false;
  for (const auto &entry : plan.unsupported.type_counts) {
    if (entry.first == "MaxPool") {
      saw_pool_type = true;
      EXPECT_EQ(entry.second, 1u);
    }
  }
  EXPECT_TRUE(saw_pool_type);

  bool saw_missing_kernel_unit = false;
  for (const auto &node : plan.unsupported.node_names) {
    if (node.find("missing_kernel_unit:opencl/generated/pool2d_f32") !=
        std::string::npos) {
      saw_missing_kernel_unit = true;
    }
  }
  EXPECT_TRUE(saw_missing_kernel_unit);
}

using RuntimeScenarioPtr =
    std::shared_ptr<const ov::test::gfx::RuntimeScenario>;

ov::Tensor filled_f32(const ov::Shape &shape, int modulus, int shift,
                      float scale) {
  ov::Tensor tensor(ov::element::f32, shape);
  auto *data = tensor.data<float>();
  for (size_t i = 0; i < tensor.get_size(); ++i) {
    data[i] =
        static_cast<float>(static_cast<int>(i % modulus) - shift) * scale;
  }
  return tensor;
}

ov::Tensor filled_f16(const ov::Shape &shape, int modulus, int shift,
                      float scale) {
  ov::Tensor tensor(ov::element::f16, shape);
  auto *data = tensor.data<ov::float16>();
  for (size_t i = 0; i < tensor.get_size(); ++i) {
    const float value =
        static_cast<float>(static_cast<int>(i % modulus) - shift) * scale;
    data[i] = ov::float16(value);
  }
  return tensor;
}

std::vector<RuntimeScenarioPtr> pool_runtime_scenarios() {
  return {
      ov::test::gfx::runtime_scenario(
          "MaxPool2D",
          [] {
            auto data = std::make_shared<ov::op::v0::Parameter>(
                ov::element::f32, ov::Shape{1, 4, 4, 4});
            auto pool = std::make_shared<ov::op::v1::MaxPool>(
                data, ov::Strides{2, 2}, ov::Shape{0, 0}, ov::Shape{0, 0},
                ov::Shape{2, 2}, ov::op::RoundingType::FLOOR);
            return model_from_node(pool, {data});
          },
          [] {
            return std::vector<ov::Tensor>{
                filled_f32({1, 4, 4, 4}, 17, 8, 0.25f)};
          }),
      ov::test::gfx::runtime_scenario(
          "AvgPool2D",
          [] {
            auto data = std::make_shared<ov::op::v0::Parameter>(
                ov::element::f32, ov::Shape{1, 4, 4, 4});
            auto pool = std::make_shared<ov::op::v1::AvgPool>(
                data, ov::Strides{2, 2}, ov::Shape{0, 0}, ov::Shape{0, 0},
                ov::Shape{2, 2}, true, ov::op::RoundingType::FLOOR);
            return model_from_node(pool, {data});
          },
          [] {
            return std::vector<ov::Tensor>{
                filled_f32({1, 4, 4, 4}, 23, 11, 0.125f)};
          }),
      ov::test::gfx::runtime_scenario(
          "MaxPool2DF16",
          [] {
            auto data = std::make_shared<ov::op::v0::Parameter>(
                ov::element::f16, ov::Shape{1, 4, 4, 4});
            auto pool = std::make_shared<ov::op::v1::MaxPool>(
                data, ov::Strides{2, 2}, ov::Shape{0, 0}, ov::Shape{0, 0},
                ov::Shape{2, 2}, ov::op::RoundingType::FLOOR);
            return model_from_node(pool, {data});
          },
          [] {
            return std::vector<ov::Tensor>{
                filled_f16({1, 4, 4, 4}, 17, 8, 0.25f)};
          },
          90,
          2e-3f,
          2e-3f),
      ov::test::gfx::runtime_scenario(
          "AvgPool2DF16",
          [] {
            auto data = std::make_shared<ov::op::v0::Parameter>(
                ov::element::f16, ov::Shape{1, 4, 4, 4});
            auto pool = std::make_shared<ov::op::v1::AvgPool>(
                data, ov::Strides{2, 2}, ov::Shape{0, 0}, ov::Shape{0, 0},
                ov::Shape{2, 2}, true, ov::op::RoundingType::FLOOR);
            return model_from_node(pool, {data});
          },
          [] {
            return std::vector<ov::Tensor>{
                filled_f16({1, 4, 4, 4}, 23, 11, 0.125f)};
          },
          90,
          2e-3f,
          2e-3f),
  };
}

class PoolRuntimeContractTest
    : public ::testing::TestWithParam<RuntimeScenarioPtr> {};

TEST_P(PoolRuntimeContractTest, MatchesTemplate) {
  const auto &contract = *GetParam();
  ov::test::gfx::RuntimeModelRunner runner;
  runner.compare_model(contract.make_model(), contract.make_inputs(),
                       contract.timeout_seconds(), contract.atol(),
                       contract.rtol());
}

INSTANTIATE_TEST_SUITE_P(Pooling, PoolRuntimeContractTest,
                         ::testing::ValuesIn(pool_runtime_scenarios()),
                         [](const auto &info) { return info.param->name(); });

} // namespace
} // namespace gfx_plugin
} // namespace ov
