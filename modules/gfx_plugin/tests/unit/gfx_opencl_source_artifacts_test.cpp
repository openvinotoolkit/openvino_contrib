// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "plugin/gfx_op_support.hpp"
#include "runtime/gpu_buffer.hpp"

using namespace ov::gfx_plugin;

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

TEST(GfxOpenClSourceArtifactsTest,
     UnaryElementwiseArtifactsUseSharedOpenClManifest) {
  struct Case {
    std::string type;
    std::function<std::shared_ptr<ov::Node>()> make_node;
    GfxOpenClBaselineOp op;
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
  };

  for (const auto& c : cases) {
    SCOPED_TRACE(c.type);
    const auto node = c.make_node();
    expect_opencl_artifact(node, GfxKernelStageFamily::Eltwise,
                           "opencl/baseline/eltwise_unary_f32",
                           "gfx_opencl_baseline_unary_f32",
                           /*arg_count=*/4,
                           /*direct_input_count=*/1);
    expect_opencl_source_excludes(
        node, {"long",
               "gfx_binary_f32",
               "gfx_compare_f32",
               "gfx_opencl_baseline_binary_f32",
               "gfx_opencl_baseline_binary_scalar_f32",
               "gfx_opencl_baseline_binary_const_f32",
               "gfx_opencl_baseline_compare_f32",
               "gfx_opencl_baseline_select_f32"});
    EXPECT_EQ(resolve_gfx_opencl_source_artifact(node)->op, c.op);
    EXPECT_TRUE(is_supported_node(node, GpuBackend::OpenCL));
  }
}

TEST(GfxOpenClSourceArtifactsTest,
     SameShapeBinaryArtifactsUseSharedOpenClManifest) {
  const auto lhs = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto rhs = param(ov::element::f32, ov::Shape{2, 3, 4});

  const std::vector<std::pair<std::string, std::shared_ptr<ov::Node>>> cases = {
      {"Add", std::make_shared<ov::op::v1::Add>(lhs, rhs)},
      {"Multiply", std::make_shared<ov::op::v1::Multiply>(lhs, rhs)},
  };

  for (const auto& c : cases) {
    SCOPED_TRACE(c.first);
    expect_opencl_artifact(c.second, GfxKernelStageFamily::Eltwise,
                           "opencl/baseline/eltwise_binary_f32",
                           "gfx_opencl_baseline_binary_f32",
                           /*arg_count=*/5,
                           /*direct_input_count=*/2);
    expect_opencl_source_excludes(
        c.second, {"long",
                   "gfx_opencl_baseline_binary_broadcast_f32",
                   "gfx_opencl_baseline_binary_scalar_f32",
                   "gfx_opencl_baseline_binary_const_f32",
                   "gfx_opencl_baseline_compare_f32",
                   "gfx_opencl_baseline_select_f32"});
    EXPECT_TRUE(is_supported_node(c.second, GpuBackend::OpenCL));
  }
}

TEST(GfxOpenClSourceArtifactsTest,
     ConstantVectorBinaryArtifactsKeepConstantInputsAsTensorSlots) {
  const auto tensor = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto same_shape_const =
      f32_const(ov::Shape{2, 3, 4}, std::vector<float>(24, 2.0f));
  const auto broadcast_const =
      f32_const(ov::Shape{3, 1}, std::vector<float>(3, 2.0f));
  const auto add_same_shape =
      std::make_shared<ov::op::v1::Add>(tensor, same_shape_const);
  const auto add_broadcast =
      std::make_shared<ov::op::v1::Add>(tensor, broadcast_const);

  expect_opencl_artifact(add_same_shape, GfxKernelStageFamily::Eltwise,
                         "opencl/baseline/eltwise_binary_f32",
                         "gfx_opencl_baseline_binary_f32",
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
  expect_opencl_artifact(add_broadcast, GfxKernelStageFamily::Eltwise,
                         "opencl/baseline/eltwise_binary_broadcast_f32",
                         "gfx_opencl_baseline_binary_broadcast_f32",
                         /*arg_count=*/18,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         static_u32_scalars);

  EXPECT_TRUE(is_supported_node(add_same_shape, GpuBackend::OpenCL));
  EXPECT_TRUE(is_supported_node(add_broadcast, GpuBackend::OpenCL));
}

TEST(GfxOpenClSourceArtifactsTest,
     BroadcastBinaryArtifactsCarryAlignedStrideMetadata) {
  const auto lhs = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto rhs = param(ov::element::f32, ov::Shape{3, 1});
  const auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);
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

  expect_opencl_artifact(add, GfxKernelStageFamily::Eltwise,
                         "opencl/baseline/eltwise_binary_broadcast_f32",
                         "gfx_opencl_baseline_binary_broadcast_f32",
                         /*arg_count=*/18,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         static_u32_scalars);
  const auto artifact = resolve_gfx_opencl_source_artifact(add);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->source.find("long"), std::string::npos);
  EXPECT_EQ(artifact->source.find("out_dim[4]"), std::string::npos);
  EXPECT_EQ(artifact->source.find("lhs_stride[4]"), std::string::npos);
  EXPECT_EQ(artifact->source.find("gfx_opencl_baseline_binary_scalar_f32"),
            std::string::npos);
  EXPECT_EQ(artifact->source.find("gfx_opencl_baseline_select_f32"),
            std::string::npos);
  EXPECT_TRUE(is_supported_node(add, GpuBackend::OpenCL));
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
                         "opencl/baseline/eltwise_binary_broadcast_f32",
                         "gfx_opencl_baseline_binary_broadcast_f32",
                         /*arg_count=*/18,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         static_u32_scalars);
  EXPECT_EQ(resolve_gfx_opencl_source_artifact(sub)->op,
            GfxOpenClBaselineOp::Subtract);
  EXPECT_TRUE(is_supported_node(sub, GpuBackend::OpenCL));
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
                         "opencl/baseline/eltwise_binary_scalar_f32",
                         "gfx_opencl_baseline_binary_scalar_f32",
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
                         "opencl/baseline/eltwise_binary_scalar_f32",
                         "gfx_opencl_baseline_binary_scalar_f32",
                         /*arg_count=*/6,
                         /*direct_input_count=*/2,
                         {GfxOpenClSourceScalarArg::ElementCount,
                          GfxOpenClSourceScalarArg::OpCode,
                          GfxOpenClSourceScalarArg::InputMode},
                         {0, 1});
  EXPECT_EQ(resolve_gfx_opencl_source_artifact(lhs_scalar)->input_mode,
            GfxOpenClBaselineInputMode::LhsScalar);

  expect_opencl_artifact(rhs_const, GfxKernelStageFamily::Eltwise,
                         "opencl/baseline/eltwise_binary_const_f32",
                         "gfx_opencl_baseline_binary_const_f32",
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
                         "opencl/baseline/eltwise_binary_const_f32",
                         "gfx_opencl_baseline_binary_const_f32",
                         /*arg_count=*/6,
                         /*direct_input_count=*/1,
                         {GfxOpenClSourceScalarArg::ElementCount,
                          GfxOpenClSourceScalarArg::OpCode,
                          GfxOpenClSourceScalarArg::InputMode,
                          GfxOpenClSourceScalarArg::ScalarConstantF32},
                         {1});
  EXPECT_EQ(resolve_gfx_opencl_source_artifact(lhs_const)->input_mode,
            GfxOpenClBaselineInputMode::LhsScalarConstant);

  EXPECT_TRUE(is_supported_node(rhs_scalar, GpuBackend::OpenCL));
  EXPECT_TRUE(is_supported_node(lhs_scalar, GpuBackend::OpenCL));
  EXPECT_TRUE(is_supported_node(rhs_const, GpuBackend::OpenCL));
  EXPECT_TRUE(is_supported_node(lhs_const, GpuBackend::OpenCL));
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

  EXPECT_TRUE(is_supported_node(greater, GpuBackend::OpenCL));
  EXPECT_TRUE(is_supported_node(select, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(greater, GpuBackend::OpenCL));

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
  EXPECT_TRUE(is_supported_node(select, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(logical_not, GpuBackend::OpenCL));

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
    EXPECT_TRUE(is_supported_node(c.node, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(logical_or, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(reduce_and, GpuBackend::OpenCL));

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
  EXPECT_TRUE(is_supported_node(reduce_or, GpuBackend::OpenCL));
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

  EXPECT_TRUE(is_supported_node(reshape, GpuBackend::OpenCL));
  EXPECT_TRUE(is_supported_node(squeeze, GpuBackend::OpenCL));
  EXPECT_TRUE(is_supported_node(unsqueeze, GpuBackend::OpenCL));
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

  EXPECT_TRUE(is_supported_node(f32_to_f32, GpuBackend::OpenCL));
  EXPECT_TRUE(is_supported_node(f32_to_i32, GpuBackend::OpenCL));
  EXPECT_TRUE(is_supported_node(i64_to_f32, GpuBackend::OpenCL));
  EXPECT_TRUE(is_supported_node(i32_to_i64, GpuBackend::OpenCL));
}

TEST(GfxOpenClSourceArtifactsTest,
     MatMulArtifactsUseStaticF32SourceKernelMetadata) {
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
                         "opencl/baseline/matmul_f32",
                         "gfx_opencl_baseline_matmul_f32",
                         /*arg_count=*/13,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         static_u32_scalars);
  const auto artifact = resolve_gfx_opencl_source_artifact(matmul);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->source.find("__global long*"), std::string::npos);
  EXPECT_TRUE(is_supported_node(matmul, GpuBackend::OpenCL));
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
                         "opencl/baseline/matmul_f32",
                         "gfx_opencl_baseline_matmul_f32",
                         /*arg_count=*/13,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         static_u32_scalars);
  EXPECT_TRUE(is_supported_node(matmul, GpuBackend::OpenCL));
}

TEST(GfxOpenClSourceArtifactsTest,
     SoftmaxArtifactsUseStaticAxisMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto softmax = std::make_shared<ov::op::v1::Softmax>(data, 1);
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
  EXPECT_EQ(artifact->source.find("__global long*"), std::string::npos);
  EXPECT_TRUE(is_supported_node(softmax, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(softmax, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(transpose, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(slice, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(slice, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(range, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(range, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(tile, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(gather, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(gather, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(gather, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(gather, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(gather, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(gather, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(scatter, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(scatter, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(scatter, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(shape_of, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(shape_of, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(shape_of, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(concat, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(concat, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(broadcast, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(select, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(slice, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(slice, GpuBackend::OpenCL));
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
  EXPECT_TRUE(is_supported_node(range, GpuBackend::OpenCL));
}

TEST(GfxOpenClSourceArtifactsTest,
     BinaryConcatArtifactsUseStaticAxisMetadata) {
  const auto lhs = param(ov::element::f32, ov::Shape{1, 2, 3});
  const auto rhs = param(ov::element::f32, ov::Shape{1, 4, 3});
  const auto concat = std::make_shared<ov::op::v0::Concat>(
      ov::OutputVector{lhs, rhs}, 1);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 6,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      6,     // output axis extent
      3,     // inner contiguous block
      0, 2,  // input 0 offset/axis extent
      2, 4,  // input 1 offset/axis extent
  };

  expect_opencl_artifact(concat, GfxKernelStageFamily::ConcatSplit,
                         "opencl/baseline/concat2_f32",
                         "gfx_opencl_baseline_concat2_f32",
                         /*arg_count=*/10,
                         /*direct_input_count=*/2,
                         scalar_args,
                         {0, 1},
                         static_u32_scalars);
  EXPECT_TRUE(is_supported_node(concat, GpuBackend::OpenCL));
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
  scalar_args.insert(scalar_args.end(), 8,
                     GfxOpenClSourceScalarArg::StaticU32);
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
                         /*arg_count=*/13,
                         /*direct_input_count=*/3,
                         scalar_args,
                         {0, 1, 2},
                         static_u32_scalars);
  EXPECT_TRUE(is_supported_node(concat, GpuBackend::OpenCL));
}

TEST(GfxOpenClSourceArtifactsTest,
     EqualSplitArtifactsUseMultiOutputStaticAxisMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{1, 6, 2});
  const auto split = std::make_shared<ov::op::v1::Split>(
      data, i64_const(ov::Shape{}, {1}), 3);
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 8,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      6,     // input axis extent
      2,     // inner contiguous block
      0, 2,  // output 0 offset/axis extent
      2, 2,  // output 1 offset/axis extent
      4, 2,  // output 2 offset/axis extent
  };

  expect_opencl_artifact(split, GfxKernelStageFamily::ConcatSplit,
                         "opencl/baseline/split3_f32",
                         "gfx_opencl_baseline_split3_f32",
                         /*arg_count=*/13,
                         /*direct_input_count=*/1,
                         scalar_args,
                         {0},
                         static_u32_scalars,
                         /*direct_output_count=*/3);
  auto artifact = resolve_gfx_opencl_source_artifact(split);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->element_count_source,
            GfxOpenClSourceElementCountSource::Input0);
  EXPECT_TRUE(is_supported_node(split, GpuBackend::OpenCL));
}

TEST(GfxOpenClSourceArtifactsTest,
     VariadicSplitArtifactsReuseMultiOutputStaticAxisMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{1, 7, 2});
  const auto split = std::make_shared<ov::op::v1::VariadicSplit>(
      data, i64_const(ov::Shape{}, {1}), i64_const(ov::Shape{3}, {2, 3, 2}));
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 8,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      7,     // input axis extent
      2,     // inner contiguous block
      0, 2,  // output 0 offset/axis extent
      2, 3,  // output 1 offset/axis extent
      5, 2,  // output 2 offset/axis extent
  };

  expect_opencl_artifact(split, GfxKernelStageFamily::ConcatSplit,
                         "opencl/baseline/split3_f32",
                         "gfx_opencl_baseline_split3_f32",
                         /*arg_count=*/13,
                         /*direct_input_count=*/1,
                         scalar_args,
                         {0},
                         static_u32_scalars,
                         /*direct_output_count=*/3);
  auto artifact = resolve_gfx_opencl_source_artifact(split);
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->element_count_source,
            GfxOpenClSourceElementCountSource::Input0);
  EXPECT_TRUE(is_supported_node(split, GpuBackend::OpenCL));
}

TEST(GfxOpenClSourceArtifactsTest,
     UnsupportedCasesStayRejectedUntilBaselineArtifactExists) {
  const auto f32 = param(ov::element::f32, ov::Shape{2, 3});
  const auto f16 = param(ov::element::f16, ov::Shape{2, 3});
  const auto high_rank_lhs = param(ov::element::f32, ov::Shape{1, 1, 1, 1, 3});
  const auto high_rank_rhs = param(ov::element::f32, ov::Shape{3});

  const auto f16_softmax = std::make_shared<ov::op::v1::Softmax>(f16, 1);
  const auto f16_relu = std::make_shared<ov::op::v0::Relu>(f16);
  const auto high_rank_broadcast_add =
      std::make_shared<ov::op::v1::Add>(high_rank_lhs, high_rank_rhs);
  const auto convert_to_f16 =
      std::make_shared<ov::op::v0::Convert>(f32, ov::element::f16);

  EXPECT_FALSE(resolve_gfx_opencl_source_artifact(f16_softmax).has_value());
  EXPECT_FALSE(resolve_gfx_opencl_source_artifact(f16_relu).has_value());
  EXPECT_FALSE(resolve_gfx_opencl_source_artifact(high_rank_broadcast_add).has_value());
  EXPECT_FALSE(resolve_gfx_opencl_source_artifact(convert_to_f16).has_value());

  EXPECT_FALSE(is_supported_node(f16_softmax, GpuBackend::OpenCL));
  EXPECT_FALSE(is_supported_node(f16_relu, GpuBackend::OpenCL));
  EXPECT_FALSE(is_supported_node(high_rank_broadcast_add, GpuBackend::OpenCL));
  EXPECT_FALSE(is_supported_node(convert_to_f16, GpuBackend::OpenCL));
}
