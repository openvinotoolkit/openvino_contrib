// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backends/opencl/compiler/opencl_operation_support.hpp"
#include "compiler/lowering_planner.hpp"
#include "compiler/kernel_registry.hpp"
#include "compiler/manifest.hpp"
#include "compiler/operation_support.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "kernel_ir/opencl_kernels/activation_kernel.hpp"
#include "kernel_ir/opencl_kernels/interpolate_f16_kernel.hpp"
#include "kernel_ir/opencl_kernels/interpolate_f32_kernel.hpp"
#include "kernel_ir/opencl_kernels/matmul_f32_kernel.hpp"
#include "kernel_ir/opencl_kernels/softmax_f16_kernel.hpp"
#include "kernel_ir/opencl_kernels/softmax_f32_kernel.hpp"
#include "mlir/mlir_support.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/hsigmoid.hpp"
#include "openvino/op/hswish.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/softsign.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/tanh.hpp"
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
using ov::gfx_plugin::compiler::ManifestBuilder;
using ov::gfx_plugin::compiler::OperationLegalizer;
using ov::gfx_plugin::compiler::make_opencl_kernel_registry;
using ov::gfx_plugin::compiler::make_opencl_operation_support_policy;

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

std::shared_ptr<ov::op::v0::Constant> i32_const(ov::Shape shape,
                                                std::vector<int32_t> values) {
  return ov::op::v0::Constant::create(ov::element::i32, std::move(shape),
                                      std::move(values));
}

bool opencl_compiler_supports_node(const std::shared_ptr<const ov::Node>& node) {
  const auto target = BackendTarget::from_backend(GpuBackend::OpenCL);
  const BackendCapabilities capabilities(target,
                                         make_opencl_operation_support_policy());
  return capabilities.query_operation({node}).semantic_legal;
}

void expect_opencl_artifact(const std::shared_ptr<const ov::Node>& node,
                            GfxKernelStageFamily family,
                            const std::string& source_id,
                            const std::string& entry_point,
                            uint32_t arg_count,
                            uint32_t direct_input_count,
                            std::vector<GfxOpenClSourceScalarArg> scalar_args =
                                {GfxOpenClSourceScalarArg::ElementCount,
                                 GfxOpenClSourceScalarArg::OpCode},
                            std::vector<size_t> direct_input_indices = {},
                            std::vector<uint32_t> static_u32_scalars = {},
                            uint32_t direct_output_count = 1) {
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
  EXPECT_EQ(artifact->direct_input_indices, direct_input_indices);
  EXPECT_EQ(artifact->baseline_local_size, 64u);
  EXPECT_EQ(artifact->scalar_args, scalar_args);
  EXPECT_EQ(artifact->static_u32_scalars, static_u32_scalars);
  EXPECT_NE(artifact->source.find("__kernel void " + entry_point),
            std::string::npos);

  const auto roles = artifact->stage_manifest.custom_kernel
                         .external_buffer_abi.roles;
  ASSERT_EQ(roles.size(), arg_count);
  for (size_t i = 0; i < direct_input_count; ++i) {
    EXPECT_EQ(roles[i], GfxKernelBufferRole::TensorInput);
  }
  for (size_t i = 0; i < direct_output_count; ++i) {
    EXPECT_EQ(roles[direct_input_count + i],
              GfxKernelBufferRole::TensorOutput);
  }
  for (size_t i = direct_input_count + direct_output_count; i < roles.size();
       ++i) {
    EXPECT_EQ(roles[i], GfxKernelBufferRole::ScalarParam);
  }
}

void expect_opencl_source_excludes(const std::shared_ptr<const ov::Node>& node,
                                   const std::vector<std::string>& needles) {
  const auto artifact = resolve_gfx_opencl_source_artifact(node);
  ASSERT_TRUE(artifact.has_value());
  for (const auto& needle : needles) {
    EXPECT_EQ(artifact->source.find(needle), std::string::npos) << needle;
  }
}

}  // namespace

TEST(GfxOpenClSourceArtifactsTest, BackendTargetIsStableAndCapabilityDriven) {
  const auto target = BackendTarget::from_backend(GpuBackend::OpenCL);
  EXPECT_EQ(target.backend(), GpuBackend::OpenCL);
  EXPECT_NE(target.fingerprint().find("backend=opencl"), std::string::npos);
  EXPECT_TRUE(target.is_compatible_with_fingerprint(target.fingerprint()));

  BackendCapabilities capabilities(target, make_opencl_operation_support_policy());
  const auto kernel_registry = make_opencl_kernel_registry(target);
  const auto audit = kernel_registry.audit();
  ASSERT_TRUE(audit.valid());
  EXPECT_EQ(audit.handwritten_exception_count, 2u);
  EXPECT_EQ(kernel_registry.route_count(LoweringRouteKind::GeneratedKernel), 15u);
  EXPECT_EQ(kernel_registry.route_count(LoweringRouteKind::HandwrittenKernelException), 2u);
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
  for (const auto& op : plan.operations) {
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
  for (const auto& stage : manifest.stages) {
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
     ActivationArtifactsUseGeneratedOpenClFamilyManifest) {
  struct Case {
    std::string type;
    std::function<std::shared_ptr<ov::Node>()> make_node;
    GfxOpenClBaselineOp op;
    std::vector<float> static_f32_scalars{0.0f, 0.0f};
  };

  const std::vector<Case> cases = {
      {"Relu",
       [] {
         return std::make_shared<ov::op::v0::Relu>(
             param(ov::element::f32, ov::Shape{2, 3}));
       },
       GfxOpenClBaselineOp::Relu},
      {"Sigmoid",
       [] {
         return std::make_shared<ov::op::v0::Sigmoid>(
             param(ov::element::f32, ov::Shape{2, 3}));
       },
       GfxOpenClBaselineOp::Sigmoid},
      {"Tanh",
       [] {
         return std::make_shared<ov::op::v0::Tanh>(
             param(ov::element::f32, ov::Shape{2, 3}));
       },
       GfxOpenClBaselineOp::Tanh},
      {"Elu",
       [] {
         return std::make_shared<ov::op::v0::Elu>(
             param(ov::element::f32, ov::Shape{2, 3}), 0.5);
       },
       GfxOpenClBaselineOp::Elu,
       {0.5f, 0.0f}},
      {"Clamp",
       [] {
         return std::make_shared<ov::op::v0::Clamp>(
             param(ov::element::f32, ov::Shape{2, 3}), -0.25, 0.75);
       },
       GfxOpenClBaselineOp::Clamp,
       {-0.25f, 0.75f}},
      {"GeluTanh",
       [] {
         return std::make_shared<ov::op::v7::Gelu>(
             param(ov::element::f32, ov::Shape{2, 3}),
             ov::op::GeluApproximationMode::TANH);
       },
       GfxOpenClBaselineOp::GeluTanh},
      {"HSwish",
       [] {
         return std::make_shared<ov::op::v4::HSwish>(
             param(ov::element::f32, ov::Shape{2, 3}));
       },
       GfxOpenClBaselineOp::HSwish},
      {"HSigmoid",
       [] {
         return std::make_shared<ov::op::v5::HSigmoid>(
             param(ov::element::f32, ov::Shape{2, 3}));
       },
       GfxOpenClBaselineOp::HSigmoid},
      {"SoftPlus",
       [] {
         return std::make_shared<ov::op::v4::SoftPlus>(
             param(ov::element::f32, ov::Shape{2, 3}));
       },
       GfxOpenClBaselineOp::SoftPlus},
      {"SoftSign",
       [] {
         return std::make_shared<ov::op::v9::SoftSign>(
             param(ov::element::f32, ov::Shape{2, 3}));
       },
       GfxOpenClBaselineOp::SoftSign},
      {"Sign",
       [] {
         return std::make_shared<ov::op::v0::Sign>(
             param(ov::element::f32, ov::Shape{2, 3}));
       },
       GfxOpenClBaselineOp::Sign},
      {"RoundEven",
       [] {
         return std::make_shared<ov::op::v5::Round>(
             param(ov::element::f32, ov::Shape{2, 3}),
             ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
       },
       GfxOpenClBaselineOp::RoundEven},
  };

  for (const auto& c : cases) {
    SCOPED_TRACE(c.type);
    const auto node = c.make_node();
    expect_opencl_artifact(node, GfxKernelStageFamily::Activation,
                           "opencl/generated/activation_f32",
                           "gfx_opencl_generated_activation_f32",
                           /*arg_count=*/6,
                           /*direct_input_count=*/1,
                           {GfxOpenClSourceScalarArg::ElementCount,
                            GfxOpenClSourceScalarArg::OpCode,
                            GfxOpenClSourceScalarArg::StaticF32,
                            GfxOpenClSourceScalarArg::StaticF32});
    const auto artifact = resolve_gfx_opencl_source_artifact(node);
    ASSERT_TRUE(artifact.has_value());
    EXPECT_EQ(artifact->source,
              opencl_generated_activation_kernel_source().source);
    expect_opencl_source_excludes(
        node, {"long",
               "gfx_binary_f32",
               "gfx_compare_f32",
               "gfx_opencl_baseline_binary_f32",
               "gfx_opencl_baseline_binary_scalar_f32",
               "gfx_opencl_baseline_binary_const_f32",
               "gfx_opencl_baseline_compare_f32",
               "gfx_opencl_baseline_select_f32"});
    EXPECT_EQ(artifact->op, c.op);
    EXPECT_EQ(artifact->scalar_constant_f32, 0.0f);
    EXPECT_EQ(artifact->static_f32_scalars, c.static_f32_scalars);
    EXPECT_TRUE(opencl_compiler_supports_node(node));
  }
}

TEST(GfxOpenClSourceArtifactsTest,
     SameShapeBinaryArtifactsUseSharedOpenClManifest) {
  const auto lhs = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto rhs = param(ov::element::f32, ov::Shape{2, 3, 4});

  const std::vector<std::pair<std::string, std::shared_ptr<ov::Node>>> cases = {
      {"Multiply", std::make_shared<ov::op::v1::Multiply>(lhs, rhs)},
  };

  for (const auto& c : cases) {
    SCOPED_TRACE(c.first);
    expect_opencl_artifact(c.second, GfxKernelStageFamily::Eltwise,
                           "opencl/generated/eltwise_binary_f32",
                           "gfx_opencl_generated_eltwise_binary_f32",
                           /*arg_count=*/5,
                           /*direct_input_count=*/2);
    expect_opencl_source_excludes(
        c.second, {"long",
                   "gfx_opencl_baseline_binary_broadcast_f32",
                   "gfx_opencl_baseline_binary_scalar_f32",
                   "gfx_opencl_baseline_binary_const_f32",
                   "gfx_opencl_baseline_compare_f32",
                   "gfx_opencl_baseline_select_f32"});
    EXPECT_TRUE(opencl_compiler_supports_node(c.second));
  }
}

TEST(GfxOpenClSourceArtifactsTest,
     TypedBinaryArtifactsUseSharedOpenClManifest) {
  struct Case {
    std::string name;
    std::shared_ptr<ov::Node> node;
    std::string suffix;
    GfxOpenClBaselineOp op;
  };

  const auto f16_lhs = param(ov::element::f16, ov::Shape{2, 3, 4});
  const auto f16_rhs = param(ov::element::f16, ov::Shape{2, 3, 4});
  const auto i32_lhs = param(ov::element::i32, ov::Shape{2, 3, 4});
  const auto i32_rhs = param(ov::element::i32, ov::Shape{2, 3, 4});

  const std::vector<Case> cases = {
      {"f16 SquaredDifference",
       std::make_shared<ov::op::v0::SquaredDifference>(f16_lhs, f16_rhs),
       "f16", GfxOpenClBaselineOp::SquaredDifference},
      {"i32 Divide", std::make_shared<ov::op::v1::Divide>(i32_lhs, i32_rhs),
       "i32", GfxOpenClBaselineOp::Divide},
      {"i32 Mod", std::make_shared<ov::op::v1::Mod>(i32_lhs, i32_rhs),
       "i32", GfxOpenClBaselineOp::Mod},
      {"i32 FloorMod", std::make_shared<ov::op::v1::FloorMod>(i32_lhs, i32_rhs),
       "i32", GfxOpenClBaselineOp::FloorMod},
      {"i32 Power", std::make_shared<ov::op::v1::Power>(i32_lhs, i32_rhs),
       "i32", GfxOpenClBaselineOp::Power},
  };

  for (const auto& c : cases) {
    SCOPED_TRACE(c.name);
    expect_opencl_artifact(c.node, GfxKernelStageFamily::Eltwise,
                           "opencl/generated/eltwise_binary_" + c.suffix,
                           "gfx_opencl_generated_eltwise_binary_" + c.suffix,
                           /*arg_count=*/5,
                           /*direct_input_count=*/2);
    expect_opencl_source_excludes(
        c.node, {"long",
                 "__global const long*",
                 "gfx_opencl_baseline_binary_f32",
                 "gfx_opencl_baseline_binary_broadcast_f32",
                 "gfx_opencl_baseline_binary_scalar_f32",
                 "gfx_opencl_baseline_binary_const_f32"});
    const auto artifact = resolve_gfx_opencl_source_artifact(c.node);
    ASSERT_TRUE(artifact.has_value());
    EXPECT_EQ(artifact->op, c.op);
    if (c.name == "i32 Power") {
      EXPECT_NE(artifact->source.find("gfx_pow_i32_exact"), std::string::npos);
      EXPECT_EQ(artifact->source.find("(int)pow((float)lhs, (float)rhs)"),
                std::string::npos);
    }
    EXPECT_TRUE(opencl_compiler_supports_node(c.node));
  }
}

TEST(GfxOpenClSourceArtifactsTest,
     TypedScalarBinaryArtifactsKeepScalarInputsAsTensorSlots) {
  const auto f16_tensor = param(ov::element::f16, ov::Shape{2, 3, 4});
  const auto f16_scalar = f16_const(ov::Shape{}, {2.0f});
  const auto i32_tensor = param(ov::element::i32, ov::Shape{2, 3, 4});
  const auto i32_scalar = i32_const(ov::Shape{}, {5});

  const auto f16_multiply =
      std::make_shared<ov::op::v1::Multiply>(f16_tensor, f16_scalar);
  expect_opencl_artifact(f16_multiply, GfxKernelStageFamily::Eltwise,
                         "opencl/generated/eltwise_scalar_f16",
                         "gfx_opencl_generated_eltwise_scalar_f16",
                         /*arg_count=*/6,
                         /*direct_input_count=*/2,
                         {GfxOpenClSourceScalarArg::ElementCount,
                          GfxOpenClSourceScalarArg::OpCode,
                          GfxOpenClSourceScalarArg::InputMode},
                         {0, 1});
  expect_opencl_source_excludes(
      f16_multiply, {"long",
                     "__global const long*",
                     "gfx_opencl_baseline_binary_const_f32"});
  EXPECT_EQ(resolve_gfx_opencl_source_artifact(f16_multiply)->input_mode,
            GfxOpenClBaselineInputMode::RhsScalar);
  EXPECT_TRUE(opencl_compiler_supports_node(f16_multiply));

  const auto i32_floor_mod =
      std::make_shared<ov::op::v1::FloorMod>(i32_scalar, i32_tensor);
  expect_opencl_artifact(i32_floor_mod, GfxKernelStageFamily::Eltwise,
                         "opencl/generated/eltwise_scalar_i32",
                         "gfx_opencl_generated_eltwise_scalar_i32",
                         /*arg_count=*/6,
                         /*direct_input_count=*/2,
                         {GfxOpenClSourceScalarArg::ElementCount,
                          GfxOpenClSourceScalarArg::OpCode,
                          GfxOpenClSourceScalarArg::InputMode},
                         {0, 1});
  expect_opencl_source_excludes(
      i32_floor_mod, {"long",
                      "__global const long*",
                      "gfx_opencl_baseline_binary_const_f32"});
  EXPECT_EQ(resolve_gfx_opencl_source_artifact(i32_floor_mod)->input_mode,
            GfxOpenClBaselineInputMode::LhsScalar);
  EXPECT_EQ(resolve_gfx_opencl_source_artifact(i32_floor_mod)->op,
            GfxOpenClBaselineOp::FloorMod);
  EXPECT_TRUE(opencl_compiler_supports_node(i32_floor_mod));
}

TEST(GfxOpenClSourceArtifactsTest,
     TypedBroadcastBinaryArtifactsCarryAlignedStrideMetadata) {
  const auto i32_lhs = param(ov::element::i32, ov::Shape{2, 3, 4});
  const auto i32_rhs = param(ov::element::i32, ov::Shape{3, 1});
  const auto i32_mod = std::make_shared<ov::op::v1::Mod>(i32_lhs, i32_rhs);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::OpCode};
  scalar_args.insert(scalar_args.end(), 13,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      3,           // rank
      2, 3, 4, 1,  // output dims padded to rank 4
      12, 4, 1, 0, // lhs strides
      0, 1, 0, 0,  // rhs aligned broadcast strides
  };

  expect_opencl_artifact(i32_mod, GfxKernelStageFamily::Eltwise,
                         "opencl/generated/eltwise_broadcast_i32",
                         "gfx_opencl_generated_eltwise_broadcast_i32",
                         /*arg_count=*/18,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         static_u32_scalars);
  expect_opencl_source_excludes(
      i32_mod, {"long",
                "__global const long*",
                "gfx_opencl_baseline_binary_broadcast_f32"});
  EXPECT_EQ(resolve_gfx_opencl_source_artifact(i32_mod)->op,
            GfxOpenClBaselineOp::Mod);
  EXPECT_TRUE(opencl_compiler_supports_node(i32_mod));

  const auto f16_lhs = param(ov::element::f16, ov::Shape{3, 1});
  const auto f16_rhs = param(ov::element::f16, ov::Shape{2, 3, 4});
  const auto f16_sub = std::make_shared<ov::op::v1::Subtract>(f16_lhs, f16_rhs);
  const std::vector<uint32_t> f16_static_u32_scalars = {
      3,           // rank
      2, 3, 4, 1,  // output dims padded to rank 4
      0, 1, 0, 0,  // lhs aligned broadcast strides
      12, 4, 1, 0, // rhs strides
  };

  expect_opencl_artifact(f16_sub, GfxKernelStageFamily::Eltwise,
                         "opencl/generated/eltwise_broadcast_f16",
                         "gfx_opencl_generated_eltwise_broadcast_f16",
                         /*arg_count=*/18,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         f16_static_u32_scalars);
  expect_opencl_source_excludes(
      f16_sub, {"long",
                "__global const long*",
                "gfx_opencl_baseline_binary_broadcast_f32"});
  EXPECT_EQ(resolve_gfx_opencl_source_artifact(f16_sub)->op,
            GfxOpenClBaselineOp::Subtract);
  EXPECT_TRUE(opencl_compiler_supports_node(f16_sub));
}

TEST(GfxOpenClSourceArtifactsTest,
     ConstantVectorBinaryArtifactsKeepConstantInputsAsTensorSlots) {
  const auto tensor = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto same_shape_const =
      f32_const(ov::Shape{2, 3, 4}, std::vector<float>(24, 2.0f));
  const auto broadcast_const =
      f32_const(ov::Shape{3, 1}, std::vector<float>(3, 2.0f));
  const auto multiply_same_shape =
      std::make_shared<ov::op::v1::Multiply>(tensor, same_shape_const);
  const auto multiply_broadcast =
      std::make_shared<ov::op::v1::Multiply>(tensor, broadcast_const);

  expect_opencl_artifact(multiply_same_shape, GfxKernelStageFamily::Eltwise,
                         "opencl/generated/eltwise_binary_f32",
                         "gfx_opencl_generated_eltwise_binary_f32",
                         /*arg_count=*/5,
                         /*direct_input_count=*/2,
                         {GfxOpenClSourceScalarArg::ElementCount,
                          GfxOpenClSourceScalarArg::OpCode},
                         {0, 1});

  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::OpCode};
  scalar_args.insert(scalar_args.end(), 13,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      3,           // rank
      2, 3, 4, 1,  // output dims padded to rank 4
      12, 4, 1, 0, // lhs strides
      0, 1, 0, 0,  // rhs aligned broadcast strides
  };
  expect_opencl_artifact(multiply_broadcast, GfxKernelStageFamily::Eltwise,
                         "opencl/generated/eltwise_broadcast_f32",
                         "gfx_opencl_generated_eltwise_broadcast_f32",
                         /*arg_count=*/18,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         static_u32_scalars);

  EXPECT_TRUE(opencl_compiler_supports_node(multiply_same_shape));
  EXPECT_TRUE(opencl_compiler_supports_node(multiply_broadcast));
}

TEST(GfxOpenClSourceArtifactsTest,
     BroadcastBinaryArtifactsCarryAlignedStrideMetadata) {
  const auto lhs = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto rhs = param(ov::element::f32, ov::Shape{3, 1});
  const auto multiply = std::make_shared<ov::op::v1::Multiply>(lhs, rhs);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::OpCode};
  scalar_args.insert(scalar_args.end(), 13,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      3,           // rank
      2, 3, 4, 1,  // output dims padded to rank 4
      12, 4, 1, 0, // lhs strides
      0, 1, 0, 0,  // rhs aligned broadcast strides
  };

  expect_opencl_artifact(multiply, GfxKernelStageFamily::Eltwise,
                         "opencl/generated/eltwise_broadcast_f32",
                         "gfx_opencl_generated_eltwise_broadcast_f32",
                         /*arg_count=*/18,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         static_u32_scalars);
  const auto artifact = resolve_gfx_opencl_source_artifact(multiply);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->source.find("long"), std::string::npos);
  EXPECT_EQ(artifact->source.find("out_dim[4]"), std::string::npos);
  EXPECT_EQ(artifact->source.find("lhs_stride[4]"), std::string::npos);
  EXPECT_EQ(artifact->source.find("gfx_opencl_baseline_binary_scalar_f32"),
            std::string::npos);
  EXPECT_EQ(artifact->source.find("gfx_opencl_baseline_select_f32"),
            std::string::npos);
  EXPECT_TRUE(opencl_compiler_supports_node(multiply));
}

TEST(GfxOpenClSourceArtifactsTest,
     BroadcastBinaryArtifactsKeepInputSlotsForLhsBroadcast) {
  const auto lhs = param(ov::element::f32, ov::Shape{3, 1});
  const auto rhs = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto sub = std::make_shared<ov::op::v1::Subtract>(lhs, rhs);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::OpCode};
  scalar_args.insert(scalar_args.end(), 13,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      3,           // rank
      2, 3, 4, 1,  // output dims padded to rank 4
      0, 1, 0, 0,  // lhs aligned broadcast strides
      12, 4, 1, 0, // rhs strides
  };

  expect_opencl_artifact(sub, GfxKernelStageFamily::Eltwise,
                         "opencl/generated/eltwise_broadcast_f32",
                         "gfx_opencl_generated_eltwise_broadcast_f32",
                         /*arg_count=*/18,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         static_u32_scalars);
  EXPECT_EQ(resolve_gfx_opencl_source_artifact(sub)->op,
            GfxOpenClBaselineOp::Subtract);
  EXPECT_TRUE(opencl_compiler_supports_node(sub));
}

TEST(GfxOpenClSourceArtifactsTest,
     ScalarBinaryArtifactsUseManifestRolesAndInputSlotMetadata) {
  const auto tensor = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto scalar = param(ov::element::f32, ov::Shape{1});
  const auto rhs_scalar =
      std::make_shared<ov::op::v1::Subtract>(tensor, scalar);
  const auto lhs_scalar =
      std::make_shared<ov::op::v1::Subtract>(scalar, tensor);
  const auto rhs_const =
      std::make_shared<ov::op::v1::Multiply>(
          tensor, f32_const(ov::Shape{}, {2.0f}));
  const auto lhs_const =
      std::make_shared<ov::op::v1::Subtract>(
          f32_const(ov::Shape{}, {2.0f}), tensor);

  expect_opencl_artifact(rhs_scalar, GfxKernelStageFamily::Eltwise,
                         "opencl/generated/eltwise_scalar_f32",
                         "gfx_opencl_generated_eltwise_scalar_f32",
                         /*arg_count=*/6,
                         /*direct_input_count=*/2,
                         {GfxOpenClSourceScalarArg::ElementCount,
                          GfxOpenClSourceScalarArg::OpCode,
                          GfxOpenClSourceScalarArg::InputMode},
                         {0, 1});
  expect_opencl_source_excludes(
      rhs_scalar, {"long",
                   "gfx_opencl_baseline_binary_f32",
                   "gfx_opencl_baseline_binary_broadcast_f32",
                   "gfx_opencl_baseline_binary_const_f32",
                   "gfx_opencl_baseline_compare_f32",
                   "gfx_opencl_baseline_select_f32"});
  EXPECT_EQ(resolve_gfx_opencl_source_artifact(rhs_scalar)->input_mode,
            GfxOpenClBaselineInputMode::RhsScalar);

  expect_opencl_artifact(lhs_scalar, GfxKernelStageFamily::Eltwise,
                         "opencl/generated/eltwise_scalar_f32",
                         "gfx_opencl_generated_eltwise_scalar_f32",
                         /*arg_count=*/6,
                         /*direct_input_count=*/2,
                         {GfxOpenClSourceScalarArg::ElementCount,
                          GfxOpenClSourceScalarArg::OpCode,
                          GfxOpenClSourceScalarArg::InputMode},
                         {0, 1});
  EXPECT_EQ(resolve_gfx_opencl_source_artifact(lhs_scalar)->input_mode,
            GfxOpenClBaselineInputMode::LhsScalar);

  expect_opencl_artifact(rhs_const, GfxKernelStageFamily::Eltwise,
                         "opencl/generated/eltwise_const_f32",
                         "gfx_opencl_generated_eltwise_const_f32",
                         /*arg_count=*/6,
                         /*direct_input_count=*/1,
                         {GfxOpenClSourceScalarArg::ElementCount,
                          GfxOpenClSourceScalarArg::OpCode,
                          GfxOpenClSourceScalarArg::InputMode,
                          GfxOpenClSourceScalarArg::ScalarConstantF32},
                         {0});
  expect_opencl_source_excludes(
      rhs_const, {"long",
                  "gfx_opencl_baseline_binary_f32",
                  "gfx_opencl_baseline_binary_broadcast_f32",
                  "gfx_opencl_baseline_binary_scalar_f32",
                  "gfx_opencl_baseline_compare_f32",
                  "gfx_opencl_baseline_select_f32"});
  auto rhs_const_artifact = resolve_gfx_opencl_source_artifact(rhs_const);
  ASSERT_TRUE(rhs_const_artifact.has_value());
  EXPECT_EQ(rhs_const_artifact->input_mode,
            GfxOpenClBaselineInputMode::RhsScalarConstant);
  EXPECT_FLOAT_EQ(rhs_const_artifact->scalar_constant_f32, 2.0f);

  expect_opencl_artifact(lhs_const, GfxKernelStageFamily::Eltwise,
                         "opencl/generated/eltwise_const_f32",
                         "gfx_opencl_generated_eltwise_const_f32",
                         /*arg_count=*/6,
                         /*direct_input_count=*/1,
                         {GfxOpenClSourceScalarArg::ElementCount,
                          GfxOpenClSourceScalarArg::OpCode,
                          GfxOpenClSourceScalarArg::InputMode,
                          GfxOpenClSourceScalarArg::ScalarConstantF32},
                         {1});
  EXPECT_EQ(resolve_gfx_opencl_source_artifact(lhs_const)->input_mode,
            GfxOpenClBaselineInputMode::LhsScalarConstant);

  EXPECT_TRUE(opencl_compiler_supports_node(rhs_scalar));
  EXPECT_TRUE(opencl_compiler_supports_node(lhs_scalar));
  EXPECT_TRUE(opencl_compiler_supports_node(rhs_const));
  EXPECT_TRUE(opencl_compiler_supports_node(lhs_const));
}

TEST(GfxOpenClSourceArtifactsTest,
     CompareAndSelectArtifactsUseTheSameSourceManifestPath) {
  const auto lhs = param(ov::element::f32, ov::Shape{2, 3});
  const auto rhs = param(ov::element::f32, ov::Shape{2, 3});
  const auto condition = param(ov::element::boolean, ov::Shape{2, 3});
  const auto greater = std::make_shared<ov::op::v1::Greater>(lhs, rhs);
  const auto select =
      std::make_shared<ov::op::v1::Select>(condition, lhs, rhs);

  expect_opencl_artifact(greater, GfxKernelStageFamily::Eltwise,
                         "opencl/baseline/compare_f32",
                         "gfx_opencl_baseline_compare_f32",
                         /*arg_count=*/5,
                         /*direct_input_count=*/2);
  expect_opencl_source_excludes(
      greater, {"long",
                "gfx_opencl_baseline_binary_f32",
                "gfx_opencl_baseline_binary_broadcast_f32",
                "gfx_opencl_baseline_binary_scalar_f32",
                "gfx_opencl_baseline_binary_const_f32",
                "gfx_opencl_baseline_compare_broadcast_f32",
                "gfx_opencl_baseline_select_f32",
                "gfx_opencl_baseline_select_broadcast_f32"});
  EXPECT_EQ(resolve_gfx_opencl_source_artifact(greater)->op,
            GfxOpenClBaselineOp::Greater);

  expect_opencl_artifact(select, GfxKernelStageFamily::Eltwise,
                         "opencl/baseline/select_f32",
                         "gfx_opencl_baseline_select_f32",
                         /*arg_count=*/5,
                         /*direct_input_count=*/3,
                         {GfxOpenClSourceScalarArg::ElementCount},
                         {0, 1, 2});
  expect_opencl_source_excludes(
      select, {"long",
               "gfx_opencl_baseline_binary_f32",
               "gfx_opencl_baseline_binary_broadcast_f32",
               "gfx_opencl_baseline_binary_scalar_f32",
               "gfx_opencl_baseline_binary_const_f32",
               "gfx_opencl_baseline_compare_f32",
               "gfx_opencl_baseline_compare_broadcast_f32",
               "gfx_opencl_baseline_select_broadcast_f32"});

  EXPECT_TRUE(opencl_compiler_supports_node(greater));
  EXPECT_TRUE(opencl_compiler_supports_node(select));
}

TEST(GfxOpenClSourceArtifactsTest,
     CompareAndSelectBroadcastArtifactsCarryAlignedStrideMetadata) {
  const auto lhs = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto rhs = param(ov::element::f32, ov::Shape{3, 1});
  const auto greater = std::make_shared<ov::op::v1::Greater>(lhs, rhs);

  std::vector<GfxOpenClSourceScalarArg> compare_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::OpCode};
  compare_args.insert(compare_args.end(), 13,
                      GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> compare_static_u32_scalars = {
      3,           // rank
      2, 3, 4, 1,  // output dims padded to rank 4
      12, 4, 1, 0, // lhs strides
      0, 1, 0, 0,  // rhs aligned broadcast strides
  };

  expect_opencl_artifact(greater, GfxKernelStageFamily::Eltwise,
                         "opencl/baseline/compare_broadcast_f32",
                         "gfx_opencl_baseline_compare_broadcast_f32",
                         /*arg_count=*/18,
                         /*direct_input_count=*/2,
                         compare_args,
                         {0, 1},
                         compare_static_u32_scalars);
  expect_opencl_source_excludes(
      greater, {"long",
                "gfx_opencl_baseline_binary_f32",
                "gfx_opencl_baseline_binary_broadcast_f32",
                "gfx_opencl_baseline_binary_scalar_f32",
                "gfx_opencl_baseline_binary_const_f32",
                "gfx_opencl_baseline_compare_f32",
                "gfx_opencl_baseline_select_f32",
                "gfx_opencl_baseline_select_broadcast_f32"});
  EXPECT_EQ(resolve_gfx_opencl_source_artifact(greater)->op,
            GfxOpenClBaselineOp::Greater);
  EXPECT_TRUE(opencl_compiler_supports_node(greater));

  const auto cond = param(ov::element::boolean, ov::Shape{1, 3, 1});
  const auto then_data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto else_data = param(ov::element::f32, ov::Shape{1, 1, 4});
  const auto select =
      std::make_shared<ov::op::v1::Select>(cond, then_data, else_data);

  std::vector<GfxOpenClSourceScalarArg> select_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  select_args.insert(select_args.end(), 17,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> select_static_u32_scalars = {
      3,           // rank
      2, 3, 4, 1,  // output dims padded to rank 4
      0, 1, 0, 0,  // cond aligned broadcast strides
      12, 4, 1, 0, // then_data strides
      0, 0, 1, 0,  // else_data aligned broadcast strides
  };

  expect_opencl_artifact(select, GfxKernelStageFamily::Eltwise,
                         "opencl/baseline/select_broadcast_f32",
                         "gfx_opencl_baseline_select_broadcast_f32",
                         /*arg_count=*/22,
                         /*direct_input_count=*/3,
                         select_args,
                         {0, 1, 2},
                         select_static_u32_scalars);
  expect_opencl_source_excludes(
      select, {"long",
               "gfx_opencl_baseline_binary_f32",
               "gfx_opencl_baseline_binary_broadcast_f32",
               "gfx_opencl_baseline_binary_scalar_f32",
               "gfx_opencl_baseline_binary_const_f32",
               "gfx_opencl_baseline_compare_f32",
               "gfx_opencl_baseline_compare_broadcast_f32",
               "gfx_opencl_baseline_select_f32"});
  EXPECT_TRUE(opencl_compiler_supports_node(select));
}

TEST(GfxOpenClSourceArtifactsTest,
     LogicalBoolArtifactsUsePackedBooleanSourceBundles) {
  const auto lhs = param(ov::element::boolean, ov::Shape{2, 3});
  const auto rhs = param(ov::element::boolean, ov::Shape{2, 3});

  const auto logical_not = std::make_shared<ov::op::v1::LogicalNot>(lhs);
  expect_opencl_artifact(logical_not, GfxKernelStageFamily::Eltwise,
                         "opencl/baseline/logical_unary_bool",
                         "gfx_opencl_baseline_logical_unary_bool",
                         /*arg_count=*/4,
                         /*direct_input_count=*/1);
  expect_opencl_source_excludes(
      logical_not, {"float",
                    "long",
                    "gfx_opencl_baseline_compare_f32",
                    "gfx_opencl_baseline_compare_broadcast_f32",
                    "gfx_opencl_baseline_select_f32",
                    "gfx_opencl_baseline_select_broadcast_f32",
                    "gfx_opencl_baseline_logical_binary_bool",
                    "gfx_opencl_baseline_logical_binary_broadcast_bool"});
  EXPECT_EQ(resolve_gfx_opencl_source_artifact(logical_not)->op,
            GfxOpenClBaselineOp::LogicalNot);
  EXPECT_TRUE(opencl_compiler_supports_node(logical_not));

  struct BinaryCase {
    std::string type;
    std::shared_ptr<ov::Node> node;
    GfxOpenClBaselineOp op;
  };
  const std::vector<BinaryCase> binary_cases = {
      {"LogicalAnd",
       std::make_shared<ov::op::v1::LogicalAnd>(lhs, rhs),
       GfxOpenClBaselineOp::LogicalAnd},
      {"LogicalOr",
       std::make_shared<ov::op::v1::LogicalOr>(lhs, rhs),
       GfxOpenClBaselineOp::LogicalOr},
      {"LogicalXor",
       std::make_shared<ov::op::v1::LogicalXor>(lhs, rhs),
       GfxOpenClBaselineOp::LogicalXor},
  };

  for (const auto& c : binary_cases) {
    SCOPED_TRACE(c.type);
    expect_opencl_artifact(c.node, GfxKernelStageFamily::Eltwise,
                           "opencl/baseline/logical_binary_bool",
                           "gfx_opencl_baseline_logical_binary_bool",
                           /*arg_count=*/5,
                           /*direct_input_count=*/2);
    expect_opencl_source_excludes(
        c.node, {"float",
                 "long",
                 "gfx_opencl_baseline_compare_f32",
                 "gfx_opencl_baseline_compare_broadcast_f32",
                 "gfx_opencl_baseline_select_f32",
                 "gfx_opencl_baseline_select_broadcast_f32",
                 "gfx_opencl_baseline_logical_unary_bool",
                 "gfx_opencl_baseline_logical_binary_broadcast_bool"});
    EXPECT_EQ(resolve_gfx_opencl_source_artifact(c.node)->op, c.op);
    EXPECT_TRUE(opencl_compiler_supports_node(c.node));
  }
}

TEST(GfxOpenClSourceArtifactsTest,
     LogicalBoolBroadcastArtifactsCarryAlignedStrideMetadata) {
  const auto lhs = param(ov::element::boolean, ov::Shape{2, 3, 4});
  const auto rhs = param(ov::element::boolean, ov::Shape{3, 1});
  const auto logical_or = std::make_shared<ov::op::v1::LogicalOr>(lhs, rhs);

  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::OpCode};
  scalar_args.insert(scalar_args.end(), 13,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      3,           // rank
      2, 3, 4, 1,  // output dims padded to rank 4
      12, 4, 1, 0, // lhs strides
      0, 1, 0, 0,  // rhs aligned broadcast strides
  };

  expect_opencl_artifact(logical_or, GfxKernelStageFamily::Eltwise,
                         "opencl/baseline/logical_binary_broadcast_bool",
                         "gfx_opencl_baseline_logical_binary_broadcast_bool",
                         /*arg_count=*/18,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         static_u32_scalars);
  expect_opencl_source_excludes(
      logical_or, {"float",
                   "long",
                   "gfx_opencl_baseline_compare_f32",
                   "gfx_opencl_baseline_compare_broadcast_f32",
                   "gfx_opencl_baseline_select_f32",
                   "gfx_opencl_baseline_select_broadcast_f32",
                   "gfx_opencl_baseline_logical_unary_bool",
                   "gfx_opencl_baseline_logical_binary_bool"});
  EXPECT_EQ(resolve_gfx_opencl_source_artifact(logical_or)->op,
            GfxOpenClBaselineOp::LogicalOr);
  EXPECT_TRUE(opencl_compiler_supports_node(logical_or));
}

TEST(GfxOpenClSourceArtifactsTest,
     ReduceLogicalBoolArtifactsCarryStaticAxisMetadata) {
  const auto data = param(ov::element::boolean, ov::Shape{2, 3, 4});

  const auto reduce_and = std::make_shared<ov::op::v1::ReduceLogicalAnd>(
      data, i64_const(ov::Shape{1}, {1}), false);
  std::vector<GfxOpenClSourceScalarArg> and_scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::OpCode};
  and_scalar_args.insert(and_scalar_args.end(), 15,
                         GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> and_static_u32_scalars = {
      3,           // input rank
      2,           // output rank
      2, 3, 4, 1,  // input dims padded to rank 4
      2, 4, 1, 1,  // output dims padded to rank 4
      2,           // reduce axis mask: axis 1
      0, 2, 4, 4,  // output-axis to input-axis map
  };

  expect_opencl_artifact(reduce_and, GfxKernelStageFamily::Reduction,
                         "opencl/baseline/reduce_logical_bool",
                         "gfx_opencl_baseline_reduce_logical_bool",
                         /*arg_count=*/19,
                         /*direct_input_count=*/1,
                         and_scalar_args,
                         {0},
                         and_static_u32_scalars);
  expect_opencl_source_excludes(
      reduce_and, {"float",
                   "long",
                   "gfx_opencl_baseline_logical_unary_bool",
                   "gfx_opencl_baseline_logical_binary_bool",
                   "gfx_opencl_baseline_logical_binary_broadcast_bool",
                   "gfx_opencl_baseline_select_f32",
                   "gfx_opencl_baseline_compare_f32"});
  EXPECT_EQ(resolve_gfx_opencl_source_artifact(reduce_and)->op,
            GfxOpenClBaselineOp::ReduceLogicalAnd);
  EXPECT_TRUE(opencl_compiler_supports_node(reduce_and));

  const auto reduce_or = std::make_shared<ov::op::v1::ReduceLogicalOr>(
      data, i64_const(ov::Shape{2}, {1, 2}), true);
  std::vector<GfxOpenClSourceScalarArg> or_scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::OpCode};
  or_scalar_args.insert(or_scalar_args.end(), 15,
                        GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> or_static_u32_scalars = {
      3,           // input rank
      3,           // output rank
      2, 3, 4, 1,  // input dims padded to rank 4
      2, 1, 1, 1,  // output dims padded to rank 4
      6,           // reduce axis mask: axes 1 and 2
      0, 4, 4, 4,  // reduced keep-dims axes do not map to input coords
  };

  expect_opencl_artifact(reduce_or, GfxKernelStageFamily::Reduction,
                         "opencl/baseline/reduce_logical_bool",
                         "gfx_opencl_baseline_reduce_logical_bool",
                         /*arg_count=*/19,
                         /*direct_input_count=*/1,
                         or_scalar_args,
                         {0},
                         or_static_u32_scalars);
  EXPECT_EQ(resolve_gfx_opencl_source_artifact(reduce_or)->op,
            GfxOpenClBaselineOp::ReduceLogicalOr);
  EXPECT_TRUE(opencl_compiler_supports_node(reduce_or));
}

TEST(GfxOpenClSourceArtifactsTest,
     LayoutArtifactsIgnoreShapeOperandsAndUseDataInputOnly) {
  const auto data = param(ov::element::f32, ov::Shape{1, 2, 3});
  const auto reshape = std::make_shared<ov::op::v1::Reshape>(
      data, i64_const(ov::Shape{1}, {6}), false);
  const auto squeeze = std::make_shared<ov::op::v0::Squeeze>(
      data, i64_const(ov::Shape{1}, {0}));
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

  EXPECT_TRUE(opencl_compiler_supports_node(f32_to_f32));
  EXPECT_TRUE(opencl_compiler_supports_node(f32_to_i32));
  EXPECT_TRUE(opencl_compiler_supports_node(i64_to_f32));
  EXPECT_TRUE(opencl_compiler_supports_node(i32_to_i64));
}

TEST(GfxOpenClSourceArtifactsTest,
     MatMulArtifactsUseGeneratedF32KernelUnitMetadata) {
  const auto lhs = param(ov::element::f32, ov::Shape{2, 3});
  const auto rhs = param(ov::element::f32, ov::Shape{3, 4});
  const auto matmul =
      std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, false);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 9,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      2, 4, 3,  // M, N, K
      0, 0,     // lhs/rhs batch strides
      3, 1,     // lhs row/col strides
      4, 1,     // rhs row/col strides
  };

  expect_opencl_artifact(matmul, GfxKernelStageFamily::Gemm,
                         "opencl/generated/matmul_f32",
                         "gfx_opencl_generated_matmul_f32",
                         /*arg_count=*/13,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         static_u32_scalars);
  const auto artifact = resolve_gfx_opencl_source_artifact(matmul);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->source,
            opencl_generated_matmul_f32_kernel_source().source);
  EXPECT_EQ(artifact->source.find("__global long*"), std::string::npos);
  EXPECT_TRUE(opencl_compiler_supports_node(matmul));
}

TEST(GfxOpenClSourceArtifactsTest,
     MatMulArtifactsCarryTransposeAndBatchBroadcastMetadata) {
  const auto lhs = param(ov::element::f32, ov::Shape{2, 1, 3, 2});
  const auto rhs = param(ov::element::f32, ov::Shape{1, 3, 4});
  const auto matmul =
      std::make_shared<ov::op::v0::MatMul>(lhs, rhs, true, false);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 9,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      2, 4, 3,  // M, N, K
      6, 0,     // lhs batches advance, rhs broadcasts
      1, 2,     // transposed lhs row/col strides
      4, 1,     // rhs row/col strides
  };

  expect_opencl_artifact(matmul, GfxKernelStageFamily::Gemm,
                         "opencl/generated/matmul_f32",
                         "gfx_opencl_generated_matmul_f32",
                         /*arg_count=*/13,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         static_u32_scalars);
  EXPECT_TRUE(opencl_compiler_supports_node(matmul));
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
     SoftmaxArtifactsUseStaticAxisMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto softmax = std::make_shared<ov::op::v1::Softmax>(data, 1);
  const auto f16_data = param(ov::element::f16, ov::Shape{2, 3, 4});
  const auto f16_softmax = std::make_shared<ov::op::v1::Softmax>(f16_data, 1);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 3,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      2,  // outer
      3,  // axis extent
      4,  // inner contiguous block
  };

  expect_opencl_artifact(softmax, GfxKernelStageFamily::Softmax,
                         "opencl/baseline/softmax_f32",
                         "gfx_opencl_baseline_softmax_f32",
                         /*arg_count=*/6,
                         /*direct_input_count=*/1,
                         scalar_args,
                         {0},
                         static_u32_scalars);
  const auto artifact = resolve_gfx_opencl_source_artifact(softmax);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->source,
            opencl_baseline_softmax_f32_kernel_source().source);
  EXPECT_EQ(artifact->source.find("__global long*"), std::string::npos);
  EXPECT_TRUE(opencl_compiler_supports_node(softmax));

  expect_opencl_artifact(f16_softmax, GfxKernelStageFamily::Softmax,
                         "opencl/baseline/softmax_f16",
                         "gfx_opencl_baseline_softmax_f16",
                         /*arg_count=*/6,
                         /*direct_input_count=*/1,
                         scalar_args,
                         {0},
                         static_u32_scalars);
  const auto f16_artifact = resolve_gfx_opencl_source_artifact(f16_softmax);
  ASSERT_TRUE(f16_artifact.has_value());
  EXPECT_EQ(f16_artifact->source,
            opencl_baseline_softmax_f16_kernel_source().source);
  EXPECT_NE(f16_artifact->source.find("gfx_f16_bits_to_f32"),
            std::string::npos);
  EXPECT_EQ(f16_artifact->source.find("__global half"), std::string::npos);
  EXPECT_EQ(f16_artifact->source.find("__global long*"), std::string::npos);
  EXPECT_TRUE(opencl_compiler_supports_node(f16_softmax));
}

TEST(GfxOpenClSourceArtifactsTest,
     SoftmaxArtifactsNormalizeOpset8NegativeAxes) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto softmax = std::make_shared<ov::op::v8::Softmax>(data, -1);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 3,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      6,  // outer
      4,  // axis extent
      1,  // inner contiguous block
  };

  expect_opencl_artifact(softmax, GfxKernelStageFamily::Softmax,
                         "opencl/baseline/softmax_f32",
                         "gfx_opencl_baseline_softmax_f32",
                         /*arg_count=*/6,
                         /*direct_input_count=*/1,
                         scalar_args,
                         {0},
                         static_u32_scalars);
  EXPECT_TRUE(opencl_compiler_supports_node(softmax));
}

TEST(GfxOpenClSourceArtifactsTest,
     SoftmaxDynamicStaticRankArtifactsUseRuntimeShapeMetadata) {
  const auto data = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{ov::Dimension::dynamic(), 3, 4});
  const auto softmax = std::make_shared<ov::op::v8::Softmax>(data, -2);

  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::StaticU32,
      GfxOpenClSourceScalarArg::StaticU32};
  for (uint32_t axis = 0; axis < 8; ++axis) {
    scalar_args.push_back(static_cast<GfxOpenClSourceScalarArg>(
        static_cast<uint32_t>(GfxOpenClSourceScalarArg::Input0Dim0) + axis));
  }
  const std::vector<uint32_t> static_u32_scalars = {
      3,  // rank
      1,  // normalized axis
  };

  expect_opencl_artifact(
      softmax, GfxKernelStageFamily::Softmax,
      "opencl/baseline/softmax_f16_dynamic_static_rank",
      "gfx_opencl_baseline_softmax_dynamic_f16",
      /*arg_count=*/13,
      /*direct_input_count=*/1,
      scalar_args,
      {0},
      static_u32_scalars);
  const auto artifact = resolve_gfx_opencl_source_artifact(softmax);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->source,
            opencl_baseline_softmax_f16_kernel_source().source);
  EXPECT_NE(artifact->source.find("gfx_opencl_baseline_softmax_dynamic_f16"),
            std::string::npos);
  EXPECT_EQ(artifact->source.find("__global half"), std::string::npos);
  EXPECT_TRUE(opencl_compiler_supports_node(softmax));
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
      0,  // nearest
      0,  // align_corners
      1,  // use_half_pixel
      0,  // nearest_mode
  };

  expect_opencl_artifact(interpolate, GfxKernelStageFamily::Layout,
                         "opencl/generated/interpolate_f32",
                         "gfx_opencl_generated_interpolate_f32",
                         /*arg_count=*/13,
                         /*direct_input_count=*/1,
                         scalar_args,
                         {0},
                         bilinear_half_pixel_scalars);
  const auto artifact = resolve_gfx_opencl_source_artifact(interpolate);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->source,
            opencl_generated_interpolate_f32_kernel_source().source);
  EXPECT_TRUE(opencl_compiler_supports_node(interpolate));

  const auto f16_data = param(ov::element::f16, ov::Shape{1, 4, 16, 16});
  const auto f16_output_shape = i64_const(ov::Shape{2}, {32, 32});
  attrs.mode = "nearest";
  const auto f16_interpolate =
      std::make_shared<ov::op::v0::Interpolate>(f16_data,
                                                f16_output_shape,
                                                attrs);
  const std::vector<uint32_t> nearest_half_pixel_scalars = {
      1,  // nearest
      0,  // align_corners
      1,  // use_half_pixel
      0,  // nearest_mode
  };
  expect_opencl_artifact(f16_interpolate, GfxKernelStageFamily::Layout,
                         "opencl/generated/interpolate_f16",
                         "gfx_opencl_generated_interpolate_f16",
                         /*arg_count=*/13,
                         /*direct_input_count=*/1,
                         scalar_args,
                         {0},
                         nearest_half_pixel_scalars);
  const auto f16_artifact =
      resolve_gfx_opencl_source_artifact(f16_interpolate);
  ASSERT_TRUE(f16_artifact.has_value());
  EXPECT_EQ(f16_artifact->source,
            opencl_generated_interpolate_f16_kernel_source().source);
  EXPECT_NE(f16_artifact->source.find("gfx_f16_bits_to_f32"),
            std::string::npos);
  EXPECT_EQ(f16_artifact->source.find("__global half"), std::string::npos);
  EXPECT_TRUE(opencl_compiler_supports_node(f16_interpolate));
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
  const auto interpolate =
      std::make_shared<ov::op::v4::Interpolate>(data,
                                                output_shape,
                                                scales,
                                                axes,
                                                attrs);

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
      3, 4, 2, 1,  // output dims padded to rank 4
      12, 4, 1, 1, // input strides padded to rank 4
      1, 2, 0, 0,  // permutation padded to rank 4
  };

  expect_opencl_artifact(transpose, GfxKernelStageFamily::Transpose,
                         "opencl/baseline/transpose_f32",
                         "gfx_opencl_baseline_transpose_f32",
                         /*arg_count=*/16,
                         /*direct_input_count=*/1,
                         scalar_args,
                         {0},
                         static_u32_scalars);
  expect_opencl_source_excludes(
      transpose, {"long", "__global long*", "gfx_opencl_baseline_range_i64"});
  EXPECT_TRUE(opencl_compiler_supports_node(transpose));
}

TEST(GfxOpenClSourceArtifactsTest,
     SliceArtifactsCarryShapeStrideBeginAndStepMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto slice = std::make_shared<ov::op::v8::Slice>(
      data,
      i64_const(ov::Shape{3}, {0, 1, 0}),
      i64_const(ov::Shape{3}, {2, 3, 4}),
      i64_const(ov::Shape{3}, {1, 1, 2}),
      i64_const(ov::Shape{3}, {0, 1, 2}));
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 17,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      3,           // rank
      2, 2, 2, 1,  // output dims padded to rank 4
      12, 4, 1, 1, // input strides padded to rank 4
      0, 1, 0, 0,  // begin coordinate padded to rank 4
      1, 1, 2, 1,  // step coordinate padded to rank 4
  };

  expect_opencl_artifact(slice, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/slice_f32",
                         "gfx_opencl_baseline_slice_f32",
                         /*arg_count=*/20,
                         /*direct_input_count=*/1,
                         scalar_args,
                         {0},
                         static_u32_scalars);
  expect_opencl_source_excludes(
      slice, {"long", "__global long*", "gfx_opencl_baseline_range_i64"});
  EXPECT_TRUE(opencl_compiler_supports_node(slice));
}

TEST(GfxOpenClSourceArtifactsTest,
     StridedSliceArtifactsReuseSliceKernelAndStaticMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto slice = std::make_shared<ov::op::v1::StridedSlice>(
      data,
      i64_const(ov::Shape{3}, {0, 1, 0}),
      i64_const(ov::Shape{3}, {2, 3, 4}),
      i64_const(ov::Shape{3}, {1, 1, 2}),
      std::vector<int64_t>{0, 0, 0},
      std::vector<int64_t>{0, 0, 0});
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 17,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      3,           // rank
      2, 2, 2, 1,  // output dims padded to rank 4
      12, 4, 1, 1, // input strides padded to rank 4
      0, 1, 0, 0,  // begin coordinate padded to rank 4
      1, 1, 2, 1,  // step coordinate padded to rank 4
  };

  expect_opencl_artifact(slice, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/slice_f32",
                         "gfx_opencl_baseline_slice_f32",
                         /*arg_count=*/20,
                         /*direct_input_count=*/1,
                         scalar_args,
                         {0},
                         static_u32_scalars);
  expect_opencl_source_excludes(
      slice, {"long", "__global long*", "gfx_opencl_baseline_range_i64"});
  EXPECT_TRUE(opencl_compiler_supports_node(slice));
}

TEST(GfxOpenClSourceArtifactsTest,
     RangeF32ArtifactsUseDirectScalarInputsAndElementCountOnly) {
  const auto range = std::make_shared<ov::op::v4::Range>(
      f32_const(ov::Shape{}, {1.5f}),
      f32_const(ov::Shape{}, {6.5f}),
      f32_const(ov::Shape{}, {1.0f}),
      ov::element::f32);

  expect_opencl_artifact(range, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/range_f32",
                         "gfx_opencl_baseline_range_f32",
                         /*arg_count=*/5,
                         /*direct_input_count=*/3,
                         {GfxOpenClSourceScalarArg::ElementCount},
                         {0, 1, 2});
  expect_opencl_source_excludes(
      range, {"long", "__global long*", "gfx_opencl_baseline_range_i64"});
  EXPECT_TRUE(opencl_compiler_supports_node(range));
}

TEST(GfxOpenClSourceArtifactsTest,
     RangeF16ArtifactsUsePackedF16KernelWithDirectScalarInputs) {
  const auto range = std::make_shared<ov::op::v4::Range>(
      f16_const(ov::Shape{}, {1.0f}),
      f16_const(ov::Shape{}, {4.0f}),
      f16_const(ov::Shape{}, {0.5f}),
      ov::element::f16);

  expect_opencl_artifact(range, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/range_f16",
                         "gfx_opencl_baseline_range_f16",
                         /*arg_count=*/5,
                         /*direct_input_count=*/3,
                         {GfxOpenClSourceScalarArg::ElementCount},
                         {0, 1, 2});
  EXPECT_TRUE(opencl_compiler_supports_node(range));
}

TEST(GfxOpenClSourceArtifactsTest,
     RangeI64ArtifactsUseTheSameManifestAbiWithI64OutputKernel) {
  const auto range = std::make_shared<ov::op::v4::Range>(
      i64_const(ov::Shape{}, {0}),
      i64_const(ov::Shape{}, {10}),
      i64_const(ov::Shape{}, {2}),
      ov::element::i64);

  expect_opencl_artifact(range, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/range_i64",
                         "gfx_opencl_baseline_range_i64",
                         /*arg_count=*/5,
                         /*direct_input_count=*/3,
                         {GfxOpenClSourceScalarArg::ElementCount},
                         {0, 1, 2});
  EXPECT_TRUE(opencl_compiler_supports_node(range));
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
      3,            // rank
      2, 4, 6, 1,   // output dims padded to rank 4
      2, 1, 3, 1,   // input dims padded to rank 4
      24, 6, 1, 1,  // output strides padded to rank 4
      3, 3, 1, 1,   // input strides padded to rank 4
  };

  expect_opencl_artifact(tile, GfxKernelStageFamily::Layout,
                         "opencl/baseline/tile_f32",
                         "gfx_opencl_baseline_tile_f32",
                         /*arg_count=*/20,
                         /*direct_input_count=*/1,
                         scalar_args,
                         {0},
                         static_u32_scalars);
  expect_opencl_source_excludes(
      tile, {"long", "__global long*", "gfx_opencl_baseline_range_i64"});
  EXPECT_TRUE(opencl_compiler_supports_node(tile));
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
      3,            // rank
      2, 4, 6, 1,   // output dims padded to rank 4
      2, 1, 3, 1,   // input dims padded to rank 4
      24, 6, 1, 1,  // output strides padded to rank 4
      3, 3, 1, 1,   // input strides padded to rank 4
  };

  expect_opencl_artifact(tile, GfxKernelStageFamily::Layout,
                         "opencl/baseline/tile_f16",
                         "gfx_opencl_baseline_tile_f16",
                         /*arg_count=*/20,
                         /*direct_input_count=*/1,
                         scalar_args,
                         {0},
                         static_u32_scalars);
  EXPECT_TRUE(opencl_compiler_supports_node(tile));
}

TEST(GfxOpenClSourceArtifactsTest,
     DynamicF16TileUsesRuntimeInputAndOutputDimsUnderSourceManifest) {
  const auto data =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                              ov::PartialShape{1, -1, 3});
  const auto repeats =
      std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
                                              ov::PartialShape{3});
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
                         "opencl/baseline/tile_dynamic_f16",
                         "gfx_opencl_baseline_tile_dynamic_f16",
                         /*arg_count=*/12,
                         /*direct_input_count=*/1,
                         scalar_args,
                         {0},
                         {3});
  EXPECT_TRUE(opencl_compiler_supports_node(tile));
}

TEST(GfxOpenClSourceArtifactsTest,
     DynamicF32TileUsesRuntimeInputAndOutputDimsUnderSourceManifest) {
  const auto data =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                              ov::PartialShape{1, -1, 3});
  const auto repeats =
      std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
                                              ov::PartialShape{3});
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
                         "opencl/baseline/tile_dynamic_f32",
                         "gfx_opencl_baseline_tile_dynamic_f32",
                         /*arg_count=*/12,
                         /*direct_input_count=*/1,
                         scalar_args,
                         {0},
                         {3});
  expect_opencl_source_excludes(
      tile, {"long", "__global long*", "gfx_opencl_baseline_range_i64"});
  EXPECT_TRUE(opencl_compiler_supports_node(tile));
}

TEST(GfxOpenClSourceArtifactsTest,
     GatherI64ArtifactsCarryLinearDimsMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto indices = param(ov::element::i64, ov::Shape{2});
  const auto gather = std::make_shared<ov::op::v8::Gather>(
      data, indices, i64_const(ov::Shape{}, {1}));
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 4,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      2,  // outer product before axis
      4,  // inner product after axis
      3,  // gathered axis extent
      2,  // flattened indices count
  };

  expect_opencl_artifact(gather, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/gather_i64_f32",
                         "gfx_opencl_baseline_gather_i64_f32",
                         /*arg_count=*/8,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         static_u32_scalars);
  EXPECT_TRUE(opencl_compiler_supports_node(gather));
}

TEST(GfxOpenClSourceArtifactsTest,
     GatherI32ArtifactsNormalizeNegativeAxisInMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto indices = param(ov::element::i32, ov::Shape{2});
  const auto gather = std::make_shared<ov::op::v8::Gather>(
      data, indices, i64_const(ov::Shape{}, {-1}));
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 4,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      6,  // outer product before axis
      1,  // inner product after axis
      4,  // gathered axis extent
      2,  // flattened indices count
  };

  expect_opencl_artifact(gather, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/gather_i32_f32",
                         "gfx_opencl_baseline_gather_i32_f32",
                         /*arg_count=*/8,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         static_u32_scalars);
  expect_opencl_source_excludes(
      gather, {"long", "__global const long*",
               "gfx_opencl_baseline_gather_i64_f32",
               "gfx_opencl_baseline_gather_elements_i64_f32",
               "gfx_opencl_baseline_gather_nd_i64_f32"});
  EXPECT_TRUE(opencl_compiler_supports_node(gather));
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
      2, 2, 4, 1,  // output dims padded to rank 4
      8, 4, 1, 1,  // output strides padded to rank 4
      2, 3, 4, 1,  // data dims padded to rank 4
      12, 4, 1, 1, // data strides padded to rank 4
  };

  expect_opencl_artifact(gather, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/gather_elements_i64_f32",
                         "gfx_opencl_baseline_gather_elements_i64_f32",
                         /*arg_count=*/22,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         static_u32_scalars);
  EXPECT_TRUE(opencl_compiler_supports_node(gather));
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
      2, 3, 2, 1,  // output dims padded to rank 4
      6, 2, 1, 1,  // output strides padded to rank 4
      2, 3, 4, 1,  // data dims padded to rank 4
      12, 4, 1, 1, // data strides padded to rank 4
  };

  expect_opencl_artifact(gather, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/gather_elements_i32_f32",
                         "gfx_opencl_baseline_gather_elements_i32_f32",
                         /*arg_count=*/22,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         static_u32_scalars);
  expect_opencl_source_excludes(
      gather, {"long", "__global const long*",
               "gfx_opencl_baseline_gather_i64_f32",
               "gfx_opencl_baseline_gather_elements_i64_f32",
               "gfx_opencl_baseline_gather_nd_i64_f32"});
  EXPECT_TRUE(opencl_compiler_supports_node(gather));
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
      2, 3, 4, 1,  // data dims padded to rank 4
      12, 4, 1, 1, // data strides padded to rank 4
  };

  expect_opencl_artifact(gather, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/gather_nd_i64_f32",
                         "gfx_opencl_baseline_gather_nd_i64_f32",
                         /*arg_count=*/15,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         static_u32_scalars);
  EXPECT_TRUE(opencl_compiler_supports_node(gather));
}

TEST(GfxOpenClSourceArtifactsTest,
     GatherNDI32ArtifactsCarryFullSliceMetadata) {
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
      2, 3, 4, 1,  // data dims padded to rank 4
      12, 4, 1, 1, // data strides padded to rank 4
  };

  expect_opencl_artifact(gather, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/gather_nd_i32_f32",
                         "gfx_opencl_baseline_gather_nd_i32_f32",
                         /*arg_count=*/15,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         static_u32_scalars);
  expect_opencl_source_excludes(
      gather, {"long", "__global const long*",
               "gfx_opencl_baseline_gather_i64_f32",
               "gfx_opencl_baseline_gather_elements_i64_f32",
               "gfx_opencl_baseline_gather_nd_i64_f32"});
  EXPECT_TRUE(opencl_compiler_supports_node(gather));
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
      3,              // data rank
      1,              // indices rank
      3,              // updates rank
      1,              // axis
      2,              // flattened indices count
      2, 3, 4, 1,     // data dims padded to rank 4
      12, 4, 1, 1,    // data strides padded to rank 4
      2, 1, 1, 1,     // indices dims padded to rank 4
      1, 1, 1, 1,     // indices strides padded to rank 4
      8, 4, 1, 1, 1, 1, 1, // update strides padded to rank 7
  };

  expect_opencl_artifact(scatter, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/scatter_update_i64_f32",
                         "gfx_opencl_baseline_scatter_update_i64_f32",
                         /*arg_count=*/33,
                         /*direct_input_count=*/3,
                         scalar_args,
                         {0, 1, 2},
                         static_u32_scalars);
  EXPECT_TRUE(opencl_compiler_supports_node(scatter));
}

TEST(GfxOpenClSourceArtifactsTest,
     ScatterElementsI32ArtifactsCarryAxisShapeAndStrideMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto indices = param(ov::element::i32, ov::Shape{2, 2, 4});
  const auto updates = param(ov::element::f32, ov::Shape{2, 2, 4});
  const auto scatter =
      std::make_shared<ov::op::v3::ScatterElementsUpdate>(
          data, indices, updates, i64_const(ov::Shape{}, {1}));
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 19,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      3,           // rank
      1,           // axis
      16,          // flattened update count
      2, 2, 4, 1,  // update dims padded to rank 4
      8, 4, 1, 1,  // update strides padded to rank 4
      2, 3, 4, 1,  // data dims padded to rank 4
      12, 4, 1, 1, // data strides padded to rank 4
  };

  expect_opencl_artifact(scatter, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/scatter_elements_i32_f32",
                         "gfx_opencl_baseline_scatter_elements_i32_f32",
                         /*arg_count=*/24,
                         /*direct_input_count=*/3,
                         scalar_args,
                         {0, 1, 2},
                         static_u32_scalars);
  EXPECT_TRUE(opencl_compiler_supports_node(scatter));
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
      2, 3, 4, 1,  // data dims padded to rank 4
      12, 4, 1, 1, // data strides padded to rank 4
  };

  expect_opencl_artifact(scatter, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/scatter_nd_i64_f32",
                         "gfx_opencl_baseline_scatter_nd_i64_f32",
                         /*arg_count=*/16,
                         /*direct_input_count=*/3,
                         scalar_args,
                         {0, 1, 2},
                         static_u32_scalars);
  EXPECT_TRUE(opencl_compiler_supports_node(scatter));
}

TEST(GfxOpenClSourceArtifactsTest,
     ShapeOfI32ArtifactsUseRuntimeShapeMetadata) {
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
                         "opencl/baseline/shapeof_i32",
                         "gfx_opencl_baseline_shapeof_i32",
                         /*arg_count=*/11,
                         /*direct_input_count=*/1,
                         scalar_args,
                         {0});
  EXPECT_TRUE(opencl_compiler_supports_node(shape_of));
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
                         "opencl/baseline/shapeof_i64",
                         "gfx_opencl_baseline_shapeof_i64",
                         /*arg_count=*/11,
                         /*direct_input_count=*/1,
                         scalar_args,
                         {0});
  EXPECT_TRUE(opencl_compiler_supports_node(shape_of));
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
  EXPECT_EQ(artifact->artifact_ref.source_id, "opencl/baseline/shapeof_i64");
  EXPECT_EQ(artifact->source.find("__global long*"), std::string::npos);
  EXPECT_TRUE(opencl_compiler_supports_node(shape_of));
}

TEST(GfxOpenClSourceArtifactsTest,
     DynamicF16ConcatUsesRuntimeAxisLengthsUnderSourceManifest) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  const auto concat = std::make_shared<ov::op::v0::Concat>(
      ov::OutputVector{lhs, rhs}, 1);
  const std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::StaticU32,
      GfxOpenClSourceScalarArg::Input0Dim1,
      GfxOpenClSourceScalarArg::Input1Dim1};

  expect_opencl_artifact(concat, GfxKernelStageFamily::ConcatSplit,
                         "opencl/baseline/concat2_f16_dynamic",
                         "gfx_opencl_baseline_concat2_f16",
                         /*arg_count=*/7,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         {4});
  EXPECT_TRUE(opencl_compiler_supports_node(concat));
}

TEST(GfxOpenClSourceArtifactsTest,
     DynamicF16ThreeInputConcatUsesRuntimeAxisLengthsUnderSourceManifest) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  auto mid = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  const auto concat = std::make_shared<ov::op::v0::Concat>(
      ov::OutputVector{lhs, mid, rhs}, 1);
  const std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::StaticU32,
      GfxOpenClSourceScalarArg::Input0Dim1,
      GfxOpenClSourceScalarArg::Input1Dim1,
      GfxOpenClSourceScalarArg::Input2Dim1};

  expect_opencl_artifact(concat, GfxKernelStageFamily::ConcatSplit,
                         "opencl/baseline/concat3_f16_dynamic",
                         "gfx_opencl_baseline_concat3_f16",
                         /*arg_count=*/9,
                         /*direct_input_count=*/3,
                         scalar_args,
                         {0, 1, 2},
                         {4});
  EXPECT_TRUE(opencl_compiler_supports_node(concat));
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
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         {3, 3});
  EXPECT_EQ(resolve_gfx_opencl_source_artifact(broadcast)->source.find("__global long*"),
            std::string::npos);
  EXPECT_TRUE(opencl_compiler_supports_node(broadcast));
}

TEST(GfxOpenClSourceArtifactsTest,
     DynamicF16SelectUsesSameShapeRuntimeElementCount) {
  auto cond = std::make_shared<ov::op::v0::Parameter>(
      ov::element::boolean, ov::PartialShape{1, -1, 4});
  auto then_data = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  auto else_data = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  const auto select = std::make_shared<ov::op::v1::Select>(
      cond, then_data, else_data);

  expect_opencl_artifact(select, GfxKernelStageFamily::Eltwise,
                         "opencl/baseline/select_f16_dynamic",
                         "gfx_opencl_baseline_select_f16",
                         /*arg_count=*/5,
                         /*direct_input_count=*/3,
                         {GfxOpenClSourceScalarArg::ElementCount},
                         {0, 1, 2});
  EXPECT_TRUE(opencl_compiler_supports_node(select));
}

TEST(GfxOpenClSourceArtifactsTest,
     DynamicF16StridedSliceUsesRuntimeInputAndOutputDims) {
  auto data = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  auto end = std::make_shared<ov::op::v0::Parameter>(
      ov::element::i64, ov::PartialShape{3});
  const auto begin = i64_const(ov::Shape{3}, {0, 0, 0});
  const auto strides = i64_const(ov::Shape{3}, {1, 1, 1});
  const auto slice = std::make_shared<ov::op::v1::StridedSlice>(
      data,
      begin,
      end,
      strides,
      std::vector<int64_t>{0, 0, 0},
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
  scalar_args.insert(scalar_args.end(), 8,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      3,
      0, 0, 0, 0,
      1, 1, 1, 1};

  expect_opencl_artifact(slice, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/slice_f16_dynamic",
                         "gfx_opencl_baseline_slice_f16",
                         /*arg_count=*/21,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 2},
                         static_u32_scalars);
  EXPECT_TRUE(opencl_compiler_supports_node(slice));
}

TEST(GfxOpenClSourceArtifactsTest,
     DynamicF16StridedSliceUsesRuntimeBeginEndStepsAndDims) {
  auto data = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  auto begin = std::make_shared<ov::op::v0::Parameter>(
      ov::element::i64, ov::PartialShape{3});
  auto end = std::make_shared<ov::op::v0::Parameter>(
      ov::element::i64, ov::PartialShape{3});
  auto strides = std::make_shared<ov::op::v0::Parameter>(
      ov::element::i64, ov::PartialShape{3});
  const auto slice = std::make_shared<ov::op::v1::StridedSlice>(
      data,
      begin,
      end,
      strides,
      std::vector<int64_t>{0, 0, 0},
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
                         /*direct_input_count=*/4,
                         scalar_args,
                         {0, 1, 2, 3},
                         {3});
  EXPECT_TRUE(opencl_compiler_supports_node(slice));
}

TEST(GfxOpenClSourceArtifactsTest,
     DynamicF16SliceUsesRuntimeStartsEndsStepsAndDims) {
  auto data = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  auto starts = std::make_shared<ov::op::v0::Parameter>(
      ov::element::i64, ov::PartialShape{3});
  auto ends = std::make_shared<ov::op::v0::Parameter>(
      ov::element::i64, ov::PartialShape{3});
  auto steps = std::make_shared<ov::op::v0::Parameter>(
      ov::element::i64, ov::PartialShape{3});
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
                         /*direct_input_count=*/4,
                         scalar_args,
                         {0, 1, 2, 3},
                         {3});
  EXPECT_EQ(resolve_gfx_opencl_source_artifact(slice)->source.find("__global long*"),
            std::string::npos);
  EXPECT_TRUE(opencl_compiler_supports_node(slice));
}

TEST(GfxOpenClSourceArtifactsTest,
     DynamicI64UnitRangeAvoidsOpenClLongInSourceArtifact) {
  auto stop = std::make_shared<ov::op::v0::Parameter>(
      ov::element::i64, ov::PartialShape{});
  const auto start = i64_const(ov::Shape{}, {0});
  const auto step = i64_const(ov::Shape{}, {1});
  const auto range = std::make_shared<ov::op::v4::Range>(
      start, stop, step, ov::element::i64);

  expect_opencl_artifact(range, GfxKernelStageFamily::GatherScatter,
                         "opencl/baseline/range_i64_unit_dynamic",
                         "gfx_opencl_baseline_range_i64_unit",
                         /*arg_count=*/3,
                         /*direct_input_count=*/1,
                         {GfxOpenClSourceScalarArg::ElementCount},
                         {1});
  expect_opencl_source_excludes(range, {"__global long*", "(long)"});
  EXPECT_TRUE(opencl_compiler_supports_node(range));
}

TEST(GfxOpenClSourceArtifactsTest,
     BinaryConcatArtifactsUseStaticAxisMetadata) {
  const auto lhs = param(ov::element::f32, ov::Shape{1, 2, 3});
  const auto rhs = param(ov::element::f32, ov::Shape{1, 4, 3});
  const auto concat = std::make_shared<ov::op::v0::Concat>(
      ov::OutputVector{lhs, rhs}, 1);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  const std::vector<uint32_t> static_u32_scalars = {
      6,     // output axis extent
      3,     // inner contiguous block
      0, 2,  // input 0 offset/axis extent
      2, 4,  // input 1 offset/axis extent
  };

  expect_opencl_artifact(concat, GfxKernelStageFamily::ConcatSplit,
                         "opencl/baseline/concat2_f32",
                         "gfx_opencl_baseline_concat2_f32",
                         /*arg_count=*/4,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         {});
  auto artifact = resolve_gfx_opencl_source_artifact(concat);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->source_static_u32_scalars, static_u32_scalars);
  EXPECT_NE(artifact->source.find("const uint axis_total = 6u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint inner = 3u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("chunk_axis_idx >= 2u && chunk_axis_idx < 6u"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("outer_idx * 4u + src_axis_idx"),
            std::string::npos);
  expect_opencl_source_excludes(
      concat, {"long", "__global long*", "gfx_opencl_baseline_shapeof_i64"});
  EXPECT_TRUE(opencl_compiler_supports_node(concat));
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
      7,     // output axis extent
      3,     // inner contiguous block
      0, 2,  // input 0 offset/axis extent
      2, 4,  // input 1 offset/axis extent
      6, 1,  // input 2 offset/axis extent
  };

  expect_opencl_artifact(concat, GfxKernelStageFamily::ConcatSplit,
                         "opencl/baseline/concat3_f32",
                         "gfx_opencl_baseline_concat3_f32",
                         /*arg_count=*/5,
                         /*direct_input_count=*/3,
                         scalar_args,
                         {0, 1, 2},
                         {});
  auto artifact = resolve_gfx_opencl_source_artifact(concat);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->source_static_u32_scalars, static_u32_scalars);
  EXPECT_NE(artifact->source.find("const uint axis_total = 7u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint inner = 3u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("chunk_axis_idx >= 6u && chunk_axis_idx < 7u"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("outer_idx * 1u + src_axis_idx"),
            std::string::npos);
  expect_opencl_source_excludes(
      concat, {"long", "__global long*", "gfx_opencl_baseline_shapeof_i64"});
  EXPECT_TRUE(opencl_compiler_supports_node(concat));
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
                         "opencl/baseline/concat30_f32",
                         "gfx_opencl_baseline_concat30_f32",
                         /*arg_count=*/32,
                         /*direct_input_count=*/30,
                         scalar_args);
  auto artifact = resolve_gfx_opencl_source_artifact(concat);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_NE(artifact->source.find("__global const float* src29"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint axis_total = 30u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint inner = 2u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("chunk_axis_idx >= 29u && chunk_axis_idx < 30u"),
            std::string::npos);
  expect_opencl_source_excludes(
      concat, {"__global long*", "gfx_opencl_baseline_shapeof_i64",
               "gfx_opencl_baseline_concat4_f32",
               "__global const float* src30"});
  EXPECT_TRUE(opencl_compiler_supports_node(concat));
}

TEST(GfxOpenClSourceArtifactsTest,
     StaticConcatArtifactsCanBuildFourInputChunksForLargeConcat) {
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

  auto chunk0 = make_gfx_opencl_concat_chunk_source_artifact(*base, 0, 4);
  ASSERT_TRUE(chunk0.has_value());
  EXPECT_EQ(chunk0->artifact_ref.entry_point, "gfx_opencl_baseline_concat4_f32");
  EXPECT_EQ(chunk0->arg_count, 6u);
  EXPECT_EQ(chunk0->direct_input_count, 4u);
  EXPECT_EQ(chunk0->direct_output_count, 1u);
  EXPECT_EQ(chunk0->scalar_args,
            std::vector<GfxOpenClSourceScalarArg>{
                GfxOpenClSourceScalarArg::ElementCount});
  EXPECT_EQ(chunk0->direct_input_indices, std::vector<size_t>({0, 1, 2, 3}));
  EXPECT_TRUE(chunk0->static_u32_scalars.empty());
  EXPECT_NE(chunk0->source.find("__global const float* src3"),
            std::string::npos);
  EXPECT_NE(chunk0->source.find("chunk_axis_idx >= 3u && chunk_axis_idx < 4u"),
            std::string::npos);
  EXPECT_EQ(chunk0->source.find("__global const float* src4"),
            std::string::npos);

  auto chunk_tail = make_gfx_opencl_concat_chunk_source_artifact(*base, 28, 2);
  ASSERT_TRUE(chunk_tail.has_value());
  EXPECT_EQ(chunk_tail->artifact_ref.entry_point,
            "gfx_opencl_baseline_concat2_f32");
  EXPECT_EQ(chunk_tail->arg_count, 4u);
  EXPECT_EQ(chunk_tail->direct_input_count, 2u);
  EXPECT_EQ(chunk_tail->direct_input_indices, std::vector<size_t>({28, 29}));
  EXPECT_NE(chunk_tail->source.find("chunk_axis_idx >= 0u && chunk_axis_idx < 1u"),
            std::string::npos);
  EXPECT_NE(chunk_tail->source.find("chunk_axis_idx >= 1u && chunk_axis_idx < 2u"),
            std::string::npos);
  EXPECT_EQ(chunk_tail->source.find("__global const float* src2"),
            std::string::npos);

  auto single_tail = make_gfx_opencl_concat_chunk_source_artifact(*base, 29, 1);
  ASSERT_TRUE(single_tail.has_value());
  EXPECT_EQ(single_tail->artifact_ref.entry_point,
            "gfx_opencl_baseline_concat1_f32");
  EXPECT_EQ(single_tail->arg_count, 3u);
  EXPECT_EQ(single_tail->direct_input_count, 1u);
  EXPECT_EQ(single_tail->direct_input_indices, std::vector<size_t>({29}));
  EXPECT_NE(single_tail->source.find("chunk_axis_idx >= 0u && chunk_axis_idx < 1u"),
            std::string::npos);
  EXPECT_EQ(single_tail->source.find("__global const float* src1"),
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
                         "opencl/baseline/concat5_f16",
                         "gfx_opencl_baseline_concat5_f16",
                         /*arg_count=*/7,
                         /*direct_input_count=*/5,
                         scalar_args);
  auto artifact = resolve_gfx_opencl_source_artifact(concat);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_NE(artifact->source.find("__global const uint* src4"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint axis_total = 15u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint inner = 2u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("chunk_axis_idx0 >= 10u && chunk_axis_idx0 < 15u"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("GFX_STORE_F16_PAIR(dst, dst_elem >> 1u, lo, hi);"),
            std::string::npos);
  EXPECT_EQ(artifact->source.find("chunk_axis_idx1"), std::string::npos);
  expect_opencl_source_excludes(
      concat, {"__global long*", "(long)", "__global half",
               "gfx_opencl_baseline_concat4_f16",
               "__global const uint* src5"});
  EXPECT_TRUE(opencl_compiler_supports_node(concat));
}

TEST(GfxOpenClSourceArtifactsTest,
     EqualSplitArtifactsUseMultiOutputStaticAxisMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{1, 6, 2});
  const auto split = std::make_shared<ov::op::v1::Split>(
      data, i64_const(ov::Shape{}, {1}), 3);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  expect_opencl_artifact(split, GfxKernelStageFamily::ConcatSplit,
                         "opencl/baseline/split3_f32",
                         "gfx_opencl_baseline_split3_f32",
                         /*arg_count=*/5,
                         /*direct_input_count=*/1,
                         scalar_args,
                         {0},
                         {},
                         /*direct_output_count=*/3);
  auto artifact = resolve_gfx_opencl_source_artifact(split);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->element_count_source,
            GfxOpenClSourceElementCountSource::Input0);
  EXPECT_NE(artifact->source.find("const uint axis_total = 6u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint inner = 2u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find(", 4u, 2u);"), std::string::npos);
  expect_opencl_source_excludes(
      split, {"long", "__global long*", "gfx_opencl_baseline_shapeof_i64"});
  EXPECT_TRUE(opencl_compiler_supports_node(split));
}

TEST(GfxOpenClSourceArtifactsTest,
     F16EqualSplitArtifactsReuseMultiOutputStaticAxisMetadata) {
  const auto data = param(ov::element::f16, ov::Shape{1, 6, 2});
  const auto split = std::make_shared<ov::op::v1::Split>(
      data, i64_const(ov::Shape{}, {1}), 3);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  expect_opencl_artifact(split, GfxKernelStageFamily::ConcatSplit,
                         "opencl/baseline/split3_f16",
                         "gfx_opencl_baseline_split3_f16",
                         /*arg_count=*/5,
                         /*direct_input_count=*/1,
                         scalar_args,
                         {0},
                         {},
                         /*direct_output_count=*/3);
  auto artifact = resolve_gfx_opencl_source_artifact(split);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->element_count_source,
            GfxOpenClSourceElementCountSource::Input0);
  EXPECT_NE(artifact->source.find("const uint axis_total = 6u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint inner = 2u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find(", 4u, 2u);"), std::string::npos);
  expect_opencl_source_excludes(
      split, {"__global long*", "(long)", "__global half"});
  EXPECT_TRUE(opencl_compiler_supports_node(split));
}

TEST(GfxOpenClSourceArtifactsTest,
     VariadicSplitArtifactsReuseMultiOutputStaticAxisMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{1, 7, 2});
  const auto split = std::make_shared<ov::op::v1::VariadicSplit>(
      data, i64_const(ov::Shape{}, {1}), i64_const(ov::Shape{3}, {2, 3, 2}));
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  expect_opencl_artifact(split, GfxKernelStageFamily::ConcatSplit,
                         "opencl/baseline/split3_f32",
                         "gfx_opencl_baseline_split3_f32",
                         /*arg_count=*/5,
                         /*direct_input_count=*/1,
                         scalar_args,
                         {0},
                         {},
                         /*direct_output_count=*/3);
  auto artifact = resolve_gfx_opencl_source_artifact(split);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->element_count_source,
            GfxOpenClSourceElementCountSource::Input0);
  EXPECT_NE(artifact->source.find("const uint axis_total = 7u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint inner = 2u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find(", 5u, 2u);"), std::string::npos);
  expect_opencl_source_excludes(
      split, {"long", "__global long*", "gfx_opencl_baseline_shapeof_i64"});
  EXPECT_TRUE(opencl_compiler_supports_node(split));
}

TEST(GfxOpenClSourceArtifactsTest,
     F16VariadicSplitArtifactsReuseMultiOutputStaticAxisMetadata) {
  const auto data = param(ov::element::f16, ov::Shape{1, 7, 2});
  const auto split = std::make_shared<ov::op::v1::VariadicSplit>(
      data, i64_const(ov::Shape{}, {1}), i64_const(ov::Shape{3}, {2, 3, 2}));
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  expect_opencl_artifact(split, GfxKernelStageFamily::ConcatSplit,
                         "opencl/baseline/split3_f16",
                         "gfx_opencl_baseline_split3_f16",
                         /*arg_count=*/5,
                         /*direct_input_count=*/1,
                         scalar_args,
                         {0},
                         {},
                         /*direct_output_count=*/3);
  auto artifact = resolve_gfx_opencl_source_artifact(split);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->element_count_source,
            GfxOpenClSourceElementCountSource::Input0);
  EXPECT_NE(artifact->source.find("const uint axis_total = 7u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint inner = 2u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find(", 5u, 2u);"), std::string::npos);
  expect_opencl_source_excludes(
      split, {"__global long*", "(long)", "__global half"});
  EXPECT_TRUE(opencl_compiler_supports_node(split));
}

TEST(GfxOpenClSourceArtifactsTest,
     EqualSplitArtifactsGenerateSharedLayerThirtyOutputSource) {
  const auto data = param(ov::element::f32, ov::Shape{30, 30, 30, 30});
  const auto split = std::make_shared<ov::op::v1::Split>(
      data, i64_const(ov::Shape{}, {0}), 30);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};

  expect_opencl_artifact(split, GfxKernelStageFamily::ConcatSplit,
                         "opencl/baseline/split30_f32",
                         "gfx_opencl_baseline_split30_f32",
                         /*arg_count=*/32,
                         /*direct_input_count=*/1,
                         scalar_args,
                         {0},
                         {},
                         /*direct_output_count=*/30);
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
      split, {"__global long*", "gfx_opencl_baseline_shapeof_i64",
              "gfx_opencl_baseline_split4_f32", "__global float* dst30"});
  EXPECT_TRUE(opencl_compiler_supports_node(split));
}

TEST(GfxOpenClSourceArtifactsTest,
     EqualSplitArtifactsCanBuildFourOutputChunksForLargeSplit) {
  const auto data = param(ov::element::f32, ov::Shape{30, 30, 30, 30});
  const auto split = std::make_shared<ov::op::v1::Split>(
      data, i64_const(ov::Shape{}, {0}), 30);
  auto base = resolve_gfx_opencl_source_artifact(split);
  ASSERT_TRUE(base.has_value());

  auto chunk0 = make_gfx_opencl_split_chunk_source_artifact(*base, 0, 4);
  ASSERT_TRUE(chunk0.has_value());
  EXPECT_EQ(chunk0->artifact_ref.entry_point, "gfx_opencl_baseline_split4_f32");
  EXPECT_EQ(chunk0->arg_count, 6u);
  EXPECT_EQ(chunk0->direct_output_count, 4u);
  EXPECT_EQ(chunk0->scalar_args,
            std::vector<GfxOpenClSourceScalarArg>{
                GfxOpenClSourceScalarArg::ElementCount});
  EXPECT_TRUE(chunk0->static_u32_scalars.empty());
  EXPECT_NE(chunk0->source.find("__global float* dst3"), std::string::npos);
  EXPECT_NE(chunk0->source.find(", 3u, 1u);"), std::string::npos);
  EXPECT_EQ(chunk0->source.find("__global float* dst4"), std::string::npos);

  auto chunk_tail = make_gfx_opencl_split_chunk_source_artifact(*base, 28, 2);
  ASSERT_TRUE(chunk_tail.has_value());
  EXPECT_EQ(chunk_tail->artifact_ref.entry_point,
            "gfx_opencl_baseline_split2_f32");
  EXPECT_EQ(chunk_tail->arg_count, 4u);
  EXPECT_EQ(chunk_tail->direct_output_count, 2u);
  EXPECT_NE(chunk_tail->source.find(", 28u, 1u);"), std::string::npos);
  EXPECT_NE(chunk_tail->source.find(", 29u, 1u);"), std::string::npos);
  EXPECT_EQ(chunk_tail->source.find("__global float* dst2"), std::string::npos);
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
                         "opencl/baseline/split5_f16",
                         "gfx_opencl_baseline_split5_f16",
                         /*arg_count=*/7,
                         /*direct_input_count=*/1,
                         scalar_args,
                         {0},
                         {},
                         /*direct_output_count=*/5);
  auto artifact = resolve_gfx_opencl_source_artifact(split);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->element_count_source,
            GfxOpenClSourceElementCountSource::Input0);
  EXPECT_NE(artifact->source.find("__global uint* dst4"), std::string::npos);
  EXPECT_NE(artifact->source.find("const uint axis_total = 15u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find("const uint inner = 2u;"),
            std::string::npos);
  EXPECT_NE(artifact->source.find(", 10u, 5u);"), std::string::npos);
  expect_opencl_source_excludes(
      split, {"__global long*", "(long)", "__global half",
              "gfx_opencl_baseline_split4_f16", "__global uint* dst5"});
  EXPECT_TRUE(opencl_compiler_supports_node(split));
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
  EXPECT_FALSE(resolve_gfx_opencl_source_artifact(high_rank_broadcast_add).has_value());
  EXPECT_FALSE(resolve_gfx_opencl_source_artifact(convert_to_f16).has_value());

  EXPECT_TRUE(mlir_supports_node(i32_abs));
  EXPECT_TRUE(mlir_supports_node(high_rank_broadcast_add));
  EXPECT_TRUE(mlir_supports_node(convert_to_f16));
  EXPECT_FALSE(opencl_compiler_supports_node(i32_abs));
  EXPECT_FALSE(opencl_compiler_supports_node(high_rank_broadcast_add));
  EXPECT_FALSE(opencl_compiler_supports_node(convert_to_f16));
}
