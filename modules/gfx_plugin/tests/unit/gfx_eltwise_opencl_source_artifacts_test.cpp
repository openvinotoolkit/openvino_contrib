// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backends/opencl/compiler/opencl_operation_support.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/subtract.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

using compiler::BackendCapabilities;
using compiler::BackendTarget;
using compiler::make_opencl_operation_support_policy;

std::shared_ptr<ov::op::v0::Parameter> param(const ov::element::Type &type,
                                             ov::Shape shape) {
  return std::make_shared<ov::op::v0::Parameter>(type, std::move(shape));
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

bool opencl_compiler_supports_node(
    const std::shared_ptr<const ov::Node> &node) {
  const auto target = BackendTarget::from_backend(GpuBackend::OpenCL);
  const BackendCapabilities capabilities(
      target, make_opencl_operation_support_policy());
  return capabilities.query_operation({node}).semantic_legal;
}

class OpenClSourceArtifactVerifier final {
public:
  explicit OpenClSourceArtifactVerifier(std::shared_ptr<const ov::Node> node)
      : m_node(std::move(node)) {}

  OpenClSourceArtifactVerifier &
  expect_artifact(GfxKernelStageFamily family, const std::string &source_id,
                  const std::string &entry_point, uint32_t arg_count,
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

    auto artifact = resolve_gfx_opencl_source_artifact(m_node);
    if (!artifact.has_value()) {
      ADD_FAILURE() << "missing OpenCL source artifact";
      return *this;
    }
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

    const auto roles =
        artifact->stage_manifest.custom_kernel.external_buffer_abi.roles;
    if (roles.size() != arg_count) {
      ADD_FAILURE() << "unexpected ABI role count";
      return *this;
    }
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

    return *this;
  }

  OpenClSourceArtifactVerifier &
  excludes(const std::vector<std::string> &needles) {
    const auto artifact = resolve_gfx_opencl_source_artifact(m_node);
    if (!artifact.has_value()) {
      ADD_FAILURE() << "missing OpenCL source artifact";
      return *this;
    }
    for (const auto &needle : needles) {
      EXPECT_EQ(artifact->source.find(needle), std::string::npos) << needle;
    }
    return *this;
  }

  OpenClSourceArtifactVerifier &has_op(GfxOpenClBaselineOp op) {
    const auto artifact = resolve_gfx_opencl_source_artifact(m_node);
    if (!artifact.has_value()) {
      ADD_FAILURE() << "missing OpenCL source artifact";
      return *this;
    }
    EXPECT_EQ(artifact->op, op);
    return *this;
  }

  OpenClSourceArtifactVerifier &
  has_input_mode(GfxOpenClBaselineInputMode mode) {
    const auto artifact = resolve_gfx_opencl_source_artifact(m_node);
    if (!artifact.has_value()) {
      ADD_FAILURE() << "missing OpenCL source artifact";
      return *this;
    }
    EXPECT_EQ(artifact->input_mode, mode);
    return *this;
  }

  OpenClSourceArtifactVerifier &has_scalar_constant(float expected) {
    const auto artifact = resolve_gfx_opencl_source_artifact(m_node);
    if (!artifact.has_value()) {
      ADD_FAILURE() << "missing OpenCL source artifact";
      return *this;
    }
    EXPECT_FLOAT_EQ(artifact->scalar_constant_f32, expected);
    return *this;
  }

  OpenClSourceArtifactVerifier &supports_opencl_compiler() {
    EXPECT_TRUE(opencl_compiler_supports_node(m_node));
    return *this;
  }

  OpenClSourceArtifactVerifier &contains_source(const std::string &needle) {
    const auto artifact = resolve_gfx_opencl_source_artifact(m_node);
    if (!artifact.has_value()) {
      ADD_FAILURE() << "missing OpenCL source artifact";
      return *this;
    }
    EXPECT_NE(artifact->source.find(needle), std::string::npos);
    return *this;
  }

private:
  std::shared_ptr<const ov::Node> m_node;
};

std::vector<GfxOpenClSourceScalarArg> op_and_broadcast_scalar_args() {
  std::vector<GfxOpenClSourceScalarArg> args = {
      GfxOpenClSourceScalarArg::ElementCount, GfxOpenClSourceScalarArg::OpCode};
  args.insert(args.end(), 13, GfxOpenClSourceScalarArg::StaticU32);
  return args;
}

std::vector<uint32_t> rhs_31_broadcast_to_234_strides() {
  return {
      3, 2, 3, 4, 1, 12, 4, 1, 0, 0, 1, 0, 0,
  };
}

std::vector<uint32_t> lhs_31_broadcast_to_234_strides() {
  return {
      3, 2, 3, 4, 1, 0, 1, 0, 0, 12, 4, 1, 0,
  };
}

std::vector<GfxOpenClSourceScalarArg> scalar_input_args() {
  return {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::OpCode,
      GfxOpenClSourceScalarArg::InputMode,
  };
}

std::vector<GfxOpenClSourceScalarArg> scalar_constant_args() {
  return {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::OpCode,
      GfxOpenClSourceScalarArg::InputMode,
      GfxOpenClSourceScalarArg::ScalarConstantF32,
  };
}

} // namespace

TEST(EltwiseOpenClSourceArtifactsTest,
     SameShapeBinaryArtifactsUseSharedOpenClManifest) {
  const auto lhs = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto rhs = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto multiply = std::make_shared<ov::op::v1::Multiply>(lhs, rhs);

  OpenClSourceArtifactVerifier(multiply)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_binary_f32",
                       "gfx_opencl_generated_eltwise_binary_f32", 5u, 2u)
      .excludes({"long", "gfx_opencl_baseline_binary_broadcast_f32",
                 "gfx_opencl_baseline_binary_scalar_f32",
                 "gfx_opencl_baseline_binary_const_f32",
                 "gfx_opencl_baseline_compare_f32",
                 "gfx_opencl_baseline_select_f32"})
      .supports_opencl_compiler();
}

TEST(EltwiseOpenClSourceArtifactsTest,
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
       std::make_shared<ov::op::v0::SquaredDifference>(f16_lhs, f16_rhs), "f16",
       GfxOpenClBaselineOp::SquaredDifference},
      {"i32 Divide", std::make_shared<ov::op::v1::Divide>(i32_lhs, i32_rhs),
       "i32", GfxOpenClBaselineOp::Divide},
      {"i32 Mod", std::make_shared<ov::op::v1::Mod>(i32_lhs, i32_rhs), "i32",
       GfxOpenClBaselineOp::Mod},
      {"i32 FloorMod", std::make_shared<ov::op::v1::FloorMod>(i32_lhs, i32_rhs),
       "i32", GfxOpenClBaselineOp::FloorMod},
      {"i32 Power", std::make_shared<ov::op::v1::Power>(i32_lhs, i32_rhs),
       "i32", GfxOpenClBaselineOp::Power},
  };

  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    auto verifier =
        OpenClSourceArtifactVerifier(test_case.node)
            .expect_artifact(
                GfxKernelStageFamily::Eltwise,
                "opencl/generated/eltwise_binary_" + test_case.suffix,
                "gfx_opencl_generated_eltwise_binary_" + test_case.suffix, 5u,
                2u)
            .excludes({"long", "__global const long*",
                       "gfx_opencl_baseline_binary_f32",
                       "gfx_opencl_baseline_binary_broadcast_f32",
                       "gfx_opencl_baseline_binary_scalar_f32",
                       "gfx_opencl_baseline_binary_const_f32"})
            .has_op(test_case.op)
            .supports_opencl_compiler();
    if (test_case.name == "i32 Power") {
      verifier.contains_source("gfx_pow_i32_exact")
          .excludes({"(int)pow((float)lhs, (float)rhs)"});
    }
  }
}

TEST(EltwiseOpenClSourceArtifactsTest,
     TypedScalarBinaryArtifactsKeepScalarInputsAsTensorSlots) {
  const auto f16_tensor = param(ov::element::f16, ov::Shape{2, 3, 4});
  const auto f16_scalar = f16_const(ov::Shape{}, {2.0f});
  const auto i32_tensor = param(ov::element::i32, ov::Shape{2, 3, 4});
  const auto i32_scalar = i32_const(ov::Shape{}, {5});

  const auto f16_multiply =
      std::make_shared<ov::op::v1::Multiply>(f16_tensor, f16_scalar);
  OpenClSourceArtifactVerifier(f16_multiply)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_scalar_f16",
                       "gfx_opencl_generated_eltwise_scalar_f16", 6u, 2u,
                       scalar_input_args(), {0, 1})
      .excludes({"long", "__global const long*",
                 "gfx_opencl_baseline_binary_const_f32"})
      .has_input_mode(GfxOpenClBaselineInputMode::RhsScalar)
      .supports_opencl_compiler();

  const auto i32_floor_mod =
      std::make_shared<ov::op::v1::FloorMod>(i32_scalar, i32_tensor);
  OpenClSourceArtifactVerifier(i32_floor_mod)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_scalar_i32",
                       "gfx_opencl_generated_eltwise_scalar_i32", 6u, 2u,
                       scalar_input_args(), {0, 1})
      .excludes({"long", "__global const long*",
                 "gfx_opencl_baseline_binary_const_f32"})
      .has_input_mode(GfxOpenClBaselineInputMode::LhsScalar)
      .has_op(GfxOpenClBaselineOp::FloorMod)
      .supports_opencl_compiler();
}

TEST(EltwiseOpenClSourceArtifactsTest,
     TypedBroadcastBinaryArtifactsCarryAlignedStrideMetadata) {
  const auto i32_lhs = param(ov::element::i32, ov::Shape{2, 3, 4});
  const auto i32_rhs = param(ov::element::i32, ov::Shape{3, 1});
  const auto i32_mod = std::make_shared<ov::op::v1::Mod>(i32_lhs, i32_rhs);
  OpenClSourceArtifactVerifier(i32_mod)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_broadcast_i32",
                       "gfx_opencl_generated_eltwise_broadcast_i32", 18u, 2u,
                       op_and_broadcast_scalar_args(), {0, 1},
                       rhs_31_broadcast_to_234_strides())
      .excludes({"long", "__global const long*",
                 "gfx_opencl_baseline_binary_broadcast_f32"})
      .has_op(GfxOpenClBaselineOp::Mod)
      .supports_opencl_compiler();

  const auto f16_lhs = param(ov::element::f16, ov::Shape{3, 1});
  const auto f16_rhs = param(ov::element::f16, ov::Shape{2, 3, 4});
  const auto f16_sub = std::make_shared<ov::op::v1::Subtract>(f16_lhs, f16_rhs);
  OpenClSourceArtifactVerifier(f16_sub)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_broadcast_f16",
                       "gfx_opencl_generated_eltwise_broadcast_f16", 18u, 2u,
                       op_and_broadcast_scalar_args(), {0, 1},
                       lhs_31_broadcast_to_234_strides())
      .excludes({"long", "__global const long*",
                 "gfx_opencl_baseline_binary_broadcast_f32"})
      .has_op(GfxOpenClBaselineOp::Subtract)
      .supports_opencl_compiler();
}

TEST(EltwiseOpenClSourceArtifactsTest,
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

  OpenClSourceArtifactVerifier(multiply_same_shape)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_binary_f32",
                       "gfx_opencl_generated_eltwise_binary_f32", 5u, 2u,
                       {GfxOpenClSourceScalarArg::ElementCount,
                        GfxOpenClSourceScalarArg::OpCode},
                       {0, 1})
      .supports_opencl_compiler();

  OpenClSourceArtifactVerifier(multiply_broadcast)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_broadcast_f32",
                       "gfx_opencl_generated_eltwise_broadcast_f32", 18u, 2u,
                       op_and_broadcast_scalar_args(), {0, 1},
                       rhs_31_broadcast_to_234_strides())
      .supports_opencl_compiler();
}

TEST(EltwiseOpenClSourceArtifactsTest,
     BroadcastBinaryArtifactsCarryAlignedStrideMetadata) {
  const auto lhs = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto rhs = param(ov::element::f32, ov::Shape{3, 1});
  const auto multiply = std::make_shared<ov::op::v1::Multiply>(lhs, rhs);

  OpenClSourceArtifactVerifier(multiply)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_broadcast_f32",
                       "gfx_opencl_generated_eltwise_broadcast_f32", 18u, 2u,
                       op_and_broadcast_scalar_args(), {0, 1},
                       rhs_31_broadcast_to_234_strides())
      .excludes({"long", "out_dim[4]", "lhs_stride[4]",
                 "gfx_opencl_baseline_binary_scalar_f32",
                 "gfx_opencl_baseline_select_f32"})
      .supports_opencl_compiler();
}

TEST(EltwiseOpenClSourceArtifactsTest,
     BroadcastBinaryArtifactsKeepInputSlotsForLhsBroadcast) {
  const auto lhs = param(ov::element::f32, ov::Shape{3, 1});
  const auto rhs = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto sub = std::make_shared<ov::op::v1::Subtract>(lhs, rhs);

  OpenClSourceArtifactVerifier(sub)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_broadcast_f32",
                       "gfx_opencl_generated_eltwise_broadcast_f32", 18u, 2u,
                       op_and_broadcast_scalar_args(), {0, 1},
                       lhs_31_broadcast_to_234_strides())
      .has_op(GfxOpenClBaselineOp::Subtract)
      .supports_opencl_compiler();
}

TEST(EltwiseOpenClSourceArtifactsTest,
     ScalarBinaryArtifactsUseManifestRolesAndInputSlotMetadata) {
  const auto tensor = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto scalar = param(ov::element::f32, ov::Shape{1});
  const auto rhs_scalar =
      std::make_shared<ov::op::v1::Subtract>(tensor, scalar);
  const auto lhs_scalar =
      std::make_shared<ov::op::v1::Subtract>(scalar, tensor);
  const auto rhs_const = std::make_shared<ov::op::v1::Multiply>(
      tensor, f32_const(ov::Shape{}, {2.0f}));
  const auto lhs_const = std::make_shared<ov::op::v1::Subtract>(
      f32_const(ov::Shape{}, {2.0f}), tensor);

  OpenClSourceArtifactVerifier(rhs_scalar)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_scalar_f32",
                       "gfx_opencl_generated_eltwise_scalar_f32", 6u, 2u,
                       scalar_input_args(), {0, 1})
      .excludes({"long", "gfx_opencl_baseline_binary_f32",
                 "gfx_opencl_baseline_binary_broadcast_f32",
                 "gfx_opencl_baseline_binary_const_f32",
                 "gfx_opencl_baseline_compare_f32",
                 "gfx_opencl_baseline_select_f32"})
      .has_input_mode(GfxOpenClBaselineInputMode::RhsScalar)
      .supports_opencl_compiler();

  OpenClSourceArtifactVerifier(lhs_scalar)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_scalar_f32",
                       "gfx_opencl_generated_eltwise_scalar_f32", 6u, 2u,
                       scalar_input_args(), {0, 1})
      .has_input_mode(GfxOpenClBaselineInputMode::LhsScalar)
      .supports_opencl_compiler();

  OpenClSourceArtifactVerifier(rhs_const)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_const_f32",
                       "gfx_opencl_generated_eltwise_const_f32", 6u, 1u,
                       scalar_constant_args(), {0})
      .excludes({"long", "gfx_opencl_baseline_binary_f32",
                 "gfx_opencl_baseline_binary_broadcast_f32",
                 "gfx_opencl_baseline_binary_scalar_f32",
                 "gfx_opencl_baseline_compare_f32",
                 "gfx_opencl_baseline_select_f32"})
      .has_input_mode(GfxOpenClBaselineInputMode::RhsScalarConstant)
      .has_scalar_constant(2.0f)
      .supports_opencl_compiler();

  OpenClSourceArtifactVerifier(lhs_const)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_const_f32",
                       "gfx_opencl_generated_eltwise_const_f32", 6u, 1u,
                       scalar_constant_args(), {1})
      .has_input_mode(GfxOpenClBaselineInputMode::LhsScalarConstant)
      .supports_opencl_compiler();
}

TEST(EltwiseOpenClSourceArtifactsTest,
     CompareAndSelectArtifactsUseTheSameSourceManifestPath) {
  const auto lhs = param(ov::element::f32, ov::Shape{2, 3});
  const auto rhs = param(ov::element::f32, ov::Shape{2, 3});
  const auto condition = param(ov::element::boolean, ov::Shape{2, 3});
  const auto greater = std::make_shared<ov::op::v1::Greater>(lhs, rhs);
  const auto select = std::make_shared<ov::op::v1::Select>(condition, lhs, rhs);

  OpenClSourceArtifactVerifier(greater)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/baseline/compare_f32",
                       "gfx_opencl_baseline_compare_f32", 5u, 2u)
      .excludes({"long", "gfx_opencl_baseline_binary_f32",
                 "gfx_opencl_baseline_binary_broadcast_f32",
                 "gfx_opencl_baseline_binary_scalar_f32",
                 "gfx_opencl_baseline_binary_const_f32",
                 "gfx_opencl_baseline_compare_broadcast_f32",
                 "gfx_opencl_baseline_select_f32",
                 "gfx_opencl_baseline_select_broadcast_f32"})
      .has_op(GfxOpenClBaselineOp::Greater)
      .supports_opencl_compiler();

  OpenClSourceArtifactVerifier(select)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/baseline/select_f32",
                       "gfx_opencl_baseline_select_f32", 5u, 3u,
                       {GfxOpenClSourceScalarArg::ElementCount}, {0, 1, 2})
      .excludes({"long", "gfx_opencl_baseline_binary_f32",
                 "gfx_opencl_baseline_binary_broadcast_f32",
                 "gfx_opencl_baseline_binary_scalar_f32",
                 "gfx_opencl_baseline_binary_const_f32",
                 "gfx_opencl_baseline_compare_f32",
                 "gfx_opencl_baseline_compare_broadcast_f32",
                 "gfx_opencl_baseline_select_broadcast_f32"})
      .supports_opencl_compiler();
}

TEST(EltwiseOpenClSourceArtifactsTest,
     CompareAndSelectBroadcastArtifactsCarryAlignedStrideMetadata) {
  const auto lhs = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto rhs = param(ov::element::f32, ov::Shape{3, 1});
  const auto greater = std::make_shared<ov::op::v1::Greater>(lhs, rhs);

  std::vector<GfxOpenClSourceScalarArg> compare_args = {
      GfxOpenClSourceScalarArg::ElementCount, GfxOpenClSourceScalarArg::OpCode};
  compare_args.insert(compare_args.end(), 13,
                      GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> compare_static_u32_scalars = {
      3, 2, 3, 4, 1, 12, 4, 1, 0, 0, 1, 0, 0,
  };

  OpenClSourceArtifactVerifier(greater)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/baseline/compare_broadcast_f32",
                       "gfx_opencl_baseline_compare_broadcast_f32", 18u, 2u,
                       compare_args, {0, 1}, compare_static_u32_scalars)
      .excludes({"long", "gfx_opencl_baseline_binary_f32",
                 "gfx_opencl_baseline_binary_broadcast_f32",
                 "gfx_opencl_baseline_binary_scalar_f32",
                 "gfx_opencl_baseline_binary_const_f32",
                 "gfx_opencl_baseline_compare_f32",
                 "gfx_opencl_baseline_select_f32",
                 "gfx_opencl_baseline_select_broadcast_f32"})
      .has_op(GfxOpenClBaselineOp::Greater)
      .supports_opencl_compiler();

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
      3, 2, 3, 4, 1, 0, 1, 0, 0, 12, 4, 1, 0, 0, 0, 1, 0,
  };

  OpenClSourceArtifactVerifier(select)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/baseline/select_broadcast_f32",
                       "gfx_opencl_baseline_select_broadcast_f32", 22u, 3u,
                       select_args, {0, 1, 2}, select_static_u32_scalars)
      .excludes({"long", "gfx_opencl_baseline_binary_f32",
                 "gfx_opencl_baseline_binary_broadcast_f32",
                 "gfx_opencl_baseline_binary_scalar_f32",
                 "gfx_opencl_baseline_binary_const_f32",
                 "gfx_opencl_baseline_compare_f32",
                 "gfx_opencl_baseline_compare_broadcast_f32",
                 "gfx_opencl_baseline_select_f32"})
      .supports_opencl_compiler();
}

TEST(EltwiseOpenClSourceArtifactsTest,
     LogicalBoolArtifactsUsePackedBooleanSourceBundles) {
  const auto lhs = param(ov::element::boolean, ov::Shape{2, 3});
  const auto rhs = param(ov::element::boolean, ov::Shape{2, 3});

  const auto logical_not = std::make_shared<ov::op::v1::LogicalNot>(lhs);
  OpenClSourceArtifactVerifier(logical_not)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/baseline/logical_unary_bool",
                       "gfx_opencl_baseline_logical_unary_bool", 4u, 1u)
      .excludes({"float", "long", "gfx_opencl_baseline_compare_f32",
                 "gfx_opencl_baseline_compare_broadcast_f32",
                 "gfx_opencl_baseline_select_f32",
                 "gfx_opencl_baseline_select_broadcast_f32",
                 "gfx_opencl_baseline_logical_binary_bool",
                 "gfx_opencl_baseline_logical_binary_broadcast_bool"})
      .has_op(GfxOpenClBaselineOp::LogicalNot)
      .supports_opencl_compiler();

  struct BinaryCase {
    std::string name;
    std::shared_ptr<ov::Node> node;
    GfxOpenClBaselineOp op;
  };
  const std::vector<BinaryCase> binary_cases = {
      {"LogicalAnd", std::make_shared<ov::op::v1::LogicalAnd>(lhs, rhs),
       GfxOpenClBaselineOp::LogicalAnd},
      {"LogicalOr", std::make_shared<ov::op::v1::LogicalOr>(lhs, rhs),
       GfxOpenClBaselineOp::LogicalOr},
      {"LogicalXor", std::make_shared<ov::op::v1::LogicalXor>(lhs, rhs),
       GfxOpenClBaselineOp::LogicalXor},
  };

  for (const auto &test_case : binary_cases) {
    SCOPED_TRACE(test_case.name);
    OpenClSourceArtifactVerifier(test_case.node)
        .expect_artifact(GfxKernelStageFamily::Eltwise,
                         "opencl/baseline/logical_binary_bool",
                         "gfx_opencl_baseline_logical_binary_bool", 5u, 2u)
        .excludes({"float", "long", "gfx_opencl_baseline_compare_f32",
                   "gfx_opencl_baseline_compare_broadcast_f32",
                   "gfx_opencl_baseline_select_f32",
                   "gfx_opencl_baseline_select_broadcast_f32",
                   "gfx_opencl_baseline_logical_unary_bool",
                   "gfx_opencl_baseline_logical_binary_broadcast_bool"})
        .has_op(test_case.op)
        .supports_opencl_compiler();
  }
}

TEST(EltwiseOpenClSourceArtifactsTest,
     LogicalBoolBroadcastArtifactsCarryAlignedStrideMetadata) {
  const auto lhs = param(ov::element::boolean, ov::Shape{2, 3, 4});
  const auto rhs = param(ov::element::boolean, ov::Shape{3, 1});
  const auto logical_or = std::make_shared<ov::op::v1::LogicalOr>(lhs, rhs);

  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount, GfxOpenClSourceScalarArg::OpCode};
  scalar_args.insert(scalar_args.end(), 13,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      3, 2, 3, 4, 1, 12, 4, 1, 0, 0, 1, 0, 0,
  };

  OpenClSourceArtifactVerifier(logical_or)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/baseline/logical_binary_broadcast_bool",
                       "gfx_opencl_baseline_logical_binary_broadcast_bool", 18u,
                       2u, scalar_args, {0, 1}, static_u32_scalars)
      .excludes({"float", "long", "gfx_opencl_baseline_compare_f32",
                 "gfx_opencl_baseline_compare_broadcast_f32",
                 "gfx_opencl_baseline_select_f32",
                 "gfx_opencl_baseline_select_broadcast_f32",
                 "gfx_opencl_baseline_logical_unary_bool",
                 "gfx_opencl_baseline_logical_binary_bool"})
      .has_op(GfxOpenClBaselineOp::LogicalOr)
      .supports_opencl_compiler();
}

TEST(EltwiseOpenClSourceArtifactsTest,
     DynamicF16SelectUsesSameShapeRuntimeElementCount) {
  const auto cond = std::make_shared<ov::op::v0::Parameter>(
      ov::element::boolean, ov::PartialShape{1, -1, 4});
  const auto then_data = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  const auto else_data = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  const auto select =
      std::make_shared<ov::op::v1::Select>(cond, then_data, else_data);

  OpenClSourceArtifactVerifier(select)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/baseline/select_f16_dynamic",
                       "gfx_opencl_baseline_select_f16", 5u, 3u,
                       {GfxOpenClSourceScalarArg::ElementCount}, {0, 1, 2})
      .supports_opencl_compiler();
}

} // namespace gfx_plugin
} // namespace ov
