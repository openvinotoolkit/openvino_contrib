// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/hsigmoid.hpp"
#include "openvino/op/hswish.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/openvino.hpp"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/codegen_common.hpp"
#include "mlir/gfx_apple_stage_pipeline.hpp"
#include "mlir/gfx_backend_custom_kernel_adapter.hpp"
#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "mlir/gfx_mlir_kernel_metadata.hpp"
#include "mlir/gfx_mpsrt_conv_metadata.hpp"
#include "mlir/gfx_mpsrt_matmul_metadata.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "mlir/gfx_mpsrt_ops.hpp"
#include "mlir/gfx_stage_kernel_binding.hpp"
#include "mlir/mlir_builder.hpp"
#include "mlir/mlir_kernel_plan_utils.hpp"
#include "mlir/mlir_passes.hpp"
#include "mlir/mlir_support.hpp"
#include "mlir/msl_codegen.hpp"
#include "mlir/msl_codegen_apple_mps.hpp"
#include "mlir/msl_codegen_apple_msl.hpp"
#include "mlir/msl_codegen_apple_msl_dispatch.hpp"
#include "mlir/msl_codegen_apple_msl_shape.hpp"
#include "mlir/msl_codegen_apple_msl_slice_static.hpp"
#include "mlir/msl_codegen_apple_msl_split.hpp"
#include "mlir/msl_codegen_attention.hpp"
#include "mlir/msl_codegen_matmul_metal.hpp"
#include "mlir/msl_codegen_matmul_mpsrt.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sigmoid.hpp"
#include "backends/metal/compiler/metal_stage_placement.hpp"
#include "backends/opencl/compiler/opencl_stage_placement.hpp"
#include "compiler/operation_support.hpp"
#include "runtime/gfx_mpsrt_builder_plan.hpp"
#include "runtime/gfx_mpsrt_kernel_manifest_adapter.hpp"
#include "runtime/gfx_stage_policy.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transforms/fusion_pass.hpp"
#include "transforms/pipeline.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
namespace {

ov::gfx_plugin::GfxStageCompilerPolicy
test_stage_compiler_policy(ov::gfx_plugin::GpuBackend backend) {
  static const auto opencl_stage_placement =
      ov::gfx_plugin::compiler::make_opencl_stage_placement_policy();
  static const auto metal_stage_placement =
      ov::gfx_plugin::compiler::make_metal_stage_placement_policy();
  static const auto opencl_post_ops =
      ov::gfx_plugin::compiler::make_post_op_fusion_capabilities(
          ov::gfx_plugin::GpuBackend::OpenCL);
  static const auto metal_post_ops =
      ov::gfx_plugin::compiler::make_post_op_fusion_capabilities(
          ov::gfx_plugin::GpuBackend::Metal);

  ov::gfx_plugin::GfxStageCompilerPolicy policy{};
  switch (backend) {
  case ov::gfx_plugin::GpuBackend::OpenCL:
    policy.placement = opencl_stage_placement.get();
    policy.post_ops = &opencl_post_ops;
    break;
  case ov::gfx_plugin::GpuBackend::Metal:
    policy.placement = metal_stage_placement.get();
    policy.post_ops = &metal_post_ops;
    break;
  case ov::gfx_plugin::GpuBackend::Unknown:
  default:
    break;
  }
  return policy;
}

ov::gfx_plugin::GfxStageOptimizationPlan select_test_stage_optimization_plan(
    const ov::gfx_plugin::GpuBufferManager *buffer_manager,
    ov::gfx_plugin::GpuBackend backend, const std::string &stage_type,
    const std::shared_ptr<const ov::Node> &node,
    const ov::element::Type &element_type, bool has_bias, bool has_activation,
    bool has_batchnorm, const ov::gfx_plugin::GfxStageRuntimeTraits &traits) {
  const auto policy = test_stage_compiler_policy(backend);
  return ov::gfx_plugin::select_stage_optimization_plan(
      buffer_manager, backend, stage_type, node, element_type, has_bias,
      has_activation, has_batchnorm, traits, &policy);
}

ov::gfx_plugin::GfxMpsrtCustomKernelDispatchSpec
dispatch_spec_from_stage(const ov::gfx_plugin::GfxMpsrtStageDesc &stage) {
  return ov::gfx_plugin::gfx_mpsrt_custom_dispatch_spec_from_kernel_manifest(
      stage.stage_manifest.custom_kernel);
}

ov::gfx_plugin::transforms::PipelineOptions
ranking_pipeline_options() {
  ov::gfx_plugin::transforms::PipelineOptions options;
  options.canonicalize_sigmoid_before_ranking = true;
  return options;
}

ov::gfx_plugin::GfxStageOptimizationPlan make_opencl_contract_stage_plan(
    const std::string &stage_type, const std::shared_ptr<const ov::Node> &node,
    const ov::element::Type &element_type) {
  ov::gfx_plugin::compiler::StagePlacementQuery query{};
  query.backend = ov::gfx_plugin::GpuBackend::OpenCL;
  query.stage_type = stage_type;
  query.node = node;
  query.element_type = element_type;

  ov::gfx_plugin::GfxStageOptimizationPlan plan{};
  plan.placement = ov::gfx_plugin::compiler::make_opencl_stage_placement_policy()
                       ->select_placement(query);
  return plan;
}

} // namespace

TEST(GfxTransforms, MlirFusionConvReluPlan) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 3, 4, 4});
  auto weights = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{2, 3, 3, 3},
      std::vector<float>(2 * 3 * 3 * 3, 1.f));
  auto conv = std::make_shared<ov::op::v1::Convolution>(
      param, weights, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});
  auto relu = std::make_shared<ov::op::v0::Relu>(conv);
  auto res = std::make_shared<ov::op::v0::Result>(relu);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{param}, "conv_relu");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "ConvActivation" || group.node_indices.size() != 2) {
      continue;
    }
    const auto conv_idx = group.node_indices[0];
    const auto act_idx = group.node_indices[1];
    ASSERT_LT(conv_idx, ordered.size());
    ASSERT_LT(act_idx, ordered.size());
    const auto &conv_node = ordered[conv_idx];
    const auto &act_node = ordered[act_idx];
    if (ov::as_type_ptr<const ov::op::v1::Convolution>(conv_node) &&
        ov::as_type_ptr<const ov::op::v0::Relu>(act_node) &&
        group.activation.has_value() &&
        group.activation.value() == ov::gfx_plugin::ActivationKind::Relu) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionConvGeluPlan) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 3, 4, 4});
  auto weights = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{2, 3, 3, 3},
      std::vector<float>(2 * 3 * 3 * 3, 1.f));
  auto conv = std::make_shared<ov::op::v1::Convolution>(
      param, weights, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});
  auto gelu = std::make_shared<ov::op::v7::Gelu>(
      conv, ov::op::GeluApproximationMode::TANH);
  auto res = std::make_shared<ov::op::v0::Result>(gelu);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{param}, "conv_gelu");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "ConvActivation" || group.node_indices.size() != 2) {
      continue;
    }
    const auto conv_idx = group.node_indices[0];
    const auto act_idx = group.node_indices[1];
    ASSERT_LT(conv_idx, ordered.size());
    ASSERT_LT(act_idx, ordered.size());
    const auto &conv_node = ordered[conv_idx];
    const auto &act_node = ordered[act_idx];
    if (ov::as_type_ptr<const ov::op::v1::Convolution>(conv_node) &&
        ov::as_type_ptr<const ov::op::v7::Gelu>(act_node) &&
        group.activation.has_value() &&
        group.activation.value() == ov::gfx_plugin::ActivationKind::Gelu) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionConvHSwishPlan) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 3, 4, 4});
  auto weights = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{2, 3, 3, 3},
      std::vector<float>(2 * 3 * 3 * 3, 1.f));
  auto conv = std::make_shared<ov::op::v1::Convolution>(
      param, weights, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});
  auto hswish = std::make_shared<ov::op::v4::HSwish>(conv);
  auto res = std::make_shared<ov::op::v0::Result>(hswish);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{param}, "conv_hswish");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "ConvActivation" || group.node_indices.size() != 2) {
      continue;
    }
    const auto conv_idx = group.node_indices[0];
    const auto act_idx = group.node_indices[1];
    ASSERT_LT(conv_idx, ordered.size());
    ASSERT_LT(act_idx, ordered.size());
    const auto &conv_node = ordered[conv_idx];
    const auto &act_node = ordered[act_idx];
    if (ov::as_type_ptr<const ov::op::v1::Convolution>(conv_node) &&
        ov::as_type_ptr<const ov::op::v4::HSwish>(act_node) &&
        group.activation.has_value() &&
        group.activation.value() == ov::gfx_plugin::ActivationKind::HSwish) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionAddReluPlan) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 4, 4});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 4, 4});
  auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);
  auto relu = std::make_shared<ov::op::v0::Relu>(add);
  auto res = std::make_shared<ov::op::v0::Result>(relu);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{lhs, rhs}, "add_relu");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "EltwiseActivation" || group.node_indices.size() != 2) {
      continue;
    }
    const auto elt_idx = group.node_indices[0];
    const auto act_idx = group.node_indices[1];
    ASSERT_LT(elt_idx, ordered.size());
    ASSERT_LT(act_idx, ordered.size());
    const auto &elt_node = ordered[elt_idx];
    const auto &act_node = ordered[act_idx];
    if (ov::as_type_ptr<const ov::op::v1::Add>(elt_node) &&
        ov::as_type_ptr<const ov::op::v0::Relu>(act_node) &&
        group.activation.has_value() &&
        group.activation.value() == ov::gfx_plugin::ActivationKind::Relu) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionAddGeluPlan) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 4, 4});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 4, 4});
  auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);
  auto gelu = std::make_shared<ov::op::v7::Gelu>(
      add, ov::op::GeluApproximationMode::TANH);
  auto res = std::make_shared<ov::op::v0::Result>(gelu);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{lhs, rhs}, "add_gelu");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "EltwiseActivation" || group.node_indices.size() != 2) {
      continue;
    }
    const auto elt_idx = group.node_indices[0];
    const auto act_idx = group.node_indices[1];
    ASSERT_LT(elt_idx, ordered.size());
    ASSERT_LT(act_idx, ordered.size());
    const auto &elt_node = ordered[elt_idx];
    const auto &act_node = ordered[act_idx];
    if (ov::as_type_ptr<const ov::op::v1::Add>(elt_node) &&
        ov::as_type_ptr<const ov::op::v7::Gelu>(act_node) &&
        group.activation.has_value() &&
        group.activation.value() == ov::gfx_plugin::ActivationKind::Gelu) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionAddHSigmoidPlan) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 4, 4});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 4, 4});
  auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);
  auto hsigmoid = std::make_shared<ov::op::v5::HSigmoid>(add);
  auto res = std::make_shared<ov::op::v0::Result>(hsigmoid);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{lhs, rhs}, "add_hsigmoid");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "EltwiseActivation" || group.node_indices.size() != 2) {
      continue;
    }
    const auto add_idx = group.node_indices[0];
    const auto act_idx = group.node_indices[1];
    ASSERT_LT(add_idx, ordered.size());
    ASSERT_LT(act_idx, ordered.size());
    const auto &add_node = ordered[add_idx];
    const auto &act_node = ordered[act_idx];
    if (ov::as_type_ptr<const ov::op::v1::Add>(add_node) &&
        ov::as_type_ptr<const ov::op::v5::HSigmoid>(act_node) &&
        group.activation.has_value() &&
        group.activation.value() == ov::gfx_plugin::ActivationKind::HSigmoid) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionMulReluPlan) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 4, 4});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 4, 4});
  auto mul = std::make_shared<ov::op::v1::Multiply>(lhs, rhs);
  auto relu = std::make_shared<ov::op::v0::Relu>(mul);
  auto res = std::make_shared<ov::op::v0::Result>(relu);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{lhs, rhs}, "mul_relu");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "EltwiseActivation" || group.node_indices.size() != 2) {
      continue;
    }
    const auto elt_idx = group.node_indices[0];
    const auto act_idx = group.node_indices[1];
    ASSERT_LT(elt_idx, ordered.size());
    ASSERT_LT(act_idx, ordered.size());
    const auto &elt_node = ordered[elt_idx];
    const auto &act_node = ordered[act_idx];
    if (ov::as_type_ptr<const ov::op::v1::Multiply>(elt_node) &&
        ov::as_type_ptr<const ov::op::v0::Relu>(act_node) &&
        group.activation.has_value() &&
        group.activation.value() == ov::gfx_plugin::ActivationKind::Relu) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionMaxReluPlan) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 4, 4});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 4, 4});
  auto max = std::make_shared<ov::op::v1::Maximum>(lhs, rhs);
  auto relu = std::make_shared<ov::op::v0::Relu>(max);
  auto res = std::make_shared<ov::op::v0::Result>(relu);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{lhs, rhs}, "max_relu");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "EltwiseActivation" || group.node_indices.size() != 2) {
      continue;
    }
    const auto elt_idx = group.node_indices[0];
    const auto act_idx = group.node_indices[1];
    ASSERT_LT(elt_idx, ordered.size());
    ASSERT_LT(act_idx, ordered.size());
    const auto &elt_node = ordered[elt_idx];
    const auto &act_node = ordered[act_idx];
    if (ov::as_type_ptr<const ov::op::v1::Maximum>(elt_node) &&
        ov::as_type_ptr<const ov::op::v0::Relu>(act_node) &&
        group.activation.has_value() &&
        group.activation.value() == ov::gfx_plugin::ActivationKind::Relu) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionMulBiasReluPlan) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 4, 4});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 4, 4});
  auto bias = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{1, 4, 1, 1}, std::vector<float>(4, 0.25f));
  auto mul = std::make_shared<ov::op::v1::Multiply>(lhs, rhs);
  auto add = std::make_shared<ov::op::v1::Add>(mul, bias);
  auto relu = std::make_shared<ov::op::v0::Relu>(add);
  auto res = std::make_shared<ov::op::v0::Result>(relu);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{lhs, rhs}, "mul_bias_relu");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "EltwiseBiasActivation" ||
        group.node_indices.size() != 3) {
      continue;
    }
    const auto elt_idx = group.node_indices[0];
    const auto add_idx = group.node_indices[1];
    const auto act_idx = group.node_indices[2];
    ASSERT_LT(elt_idx, ordered.size());
    ASSERT_LT(add_idx, ordered.size());
    ASSERT_LT(act_idx, ordered.size());
    const auto &elt_node = ordered[elt_idx];
    const auto &add_node = ordered[add_idx];
    const auto &act_node = ordered[act_idx];
    if (ov::as_type_ptr<const ov::op::v1::Multiply>(elt_node) &&
        ov::as_type_ptr<const ov::op::v1::Add>(add_node) &&
        ov::as_type_ptr<const ov::op::v0::Relu>(act_node) &&
        group.activation.has_value() &&
        group.activation.value() == ov::gfx_plugin::ActivationKind::Relu) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionMaxBiasPlan) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 4, 4});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 4, 4});
  auto bias = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{1, 4, 1, 1}, std::vector<float>(4, -0.5f));
  auto max = std::make_shared<ov::op::v1::Maximum>(lhs, rhs);
  auto add = std::make_shared<ov::op::v1::Add>(max, bias);
  auto res = std::make_shared<ov::op::v0::Result>(add);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{lhs, rhs}, "max_bias");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "EltwiseBias" || group.node_indices.size() != 2) {
      continue;
    }
    const auto elt_idx = group.node_indices[0];
    const auto add_idx = group.node_indices[1];
    ASSERT_LT(elt_idx, ordered.size());
    ASSERT_LT(add_idx, ordered.size());
    const auto &elt_node = ordered[elt_idx];
    const auto &add_node = ordered[add_idx];
    if (ov::as_type_ptr<const ov::op::v1::Maximum>(elt_node) &&
        ov::as_type_ptr<const ov::op::v1::Add>(add_node)) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionAttentionPlan) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 4});
  auto w1 = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{4, 4}, std::vector<float>(16, 0.5f));
  auto w2 = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{4, 4}, std::vector<float>(16, 0.25f));
  auto mm1 = std::make_shared<ov::op::v0::MatMul>(param, w1, false, false);
  auto sm = std::make_shared<ov::op::v1::Softmax>(mm1, 1);
  auto mm2 = std::make_shared<ov::op::v0::MatMul>(sm, w2, false, false);
  auto res = std::make_shared<ov::op::v0::Result>(mm2);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{param}, "attn_plan");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "Attention" || group.node_indices.size() != 3) {
      continue;
    }
    const auto mm1_idx = group.node_indices[0];
    const auto sm_idx = group.node_indices[1];
    const auto mm2_idx = group.node_indices[2];
    ASSERT_LT(mm1_idx, ordered.size());
    ASSERT_LT(sm_idx, ordered.size());
    ASSERT_LT(mm2_idx, ordered.size());
    const auto &mm1_node = ordered[mm1_idx];
    const auto &sm_node = ordered[sm_idx];
    const auto &mm2_node = ordered[mm2_idx];
    if (ov::as_type_ptr<const ov::op::v0::MatMul>(mm1_node) &&
        ov::as_type_ptr<const ov::op::v1::Softmax>(sm_node) &&
        ov::as_type_ptr<const ov::op::v0::MatMul>(mm2_node)) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionAttentionCanBeDisabledForVendorFirstPlacement) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 4});
  auto w1 = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{4, 4}, std::vector<float>(16, 0.5f));
  auto w2 = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{4, 4}, std::vector<float>(16, 0.25f));
  auto mm1 = std::make_shared<ov::op::v0::MatMul>(param, w1, false, false);
  auto sm = std::make_shared<ov::op::v1::Softmax>(mm1, 1);
  auto mm2 = std::make_shared<ov::op::v0::MatMul>(sm, w2, false, false);
  auto res = std::make_shared<ov::op::v0::Result>(mm2);
  auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                           ov::ParameterVector{param},
                                           "attn_plan_vendor_first");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  cfg.enable_attention_fusion = false;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);

  for (const auto &group : plan.groups) {
    EXPECT_NE(group.kind, "Attention");
    EXPECT_NE(group.kind, "AttentionScale");
    EXPECT_NE(group.kind, "AttentionScaleMask");
    EXPECT_NE(group.kind, "NativeSDPA");
  }
}

TEST(GfxTransforms, MlirFusionVendorAttentionPlan) {
  auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                   ov::Shape{1, 2, 3, 5});
  auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                   ov::Shape{1, 2, 3, 7});
  auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                   ov::Shape{1, 2, 4, 7});
  auto scale = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{1}, std::vector<float>{0.57735026f});
  auto scores = std::make_shared<ov::op::v0::MatMul>(q, k, true, false);
  auto scaled = std::make_shared<ov::op::v1::Multiply>(scores, scale);
  auto softmax = std::make_shared<ov::op::v8::Softmax>(scaled, -1);
  auto out = std::make_shared<ov::op::v0::MatMul>(v, softmax, false, true);
  auto res = std::make_shared<ov::op::v0::Result>(out);
  auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                           ov::ParameterVector{q, k, v},
                                           "vendor_attention_plan");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  cfg.enable_attention_fusion = false;
  cfg.enable_vendor_attention_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);

  bool found = false;
  for (const auto &group : plan.groups) {
    if (group.kind == "VendorAttention" && group.node_indices.size() == 4) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionVendorAttentionPreScaledKeyPlan) {
  auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                   ov::Shape{1, 2, 3, 5});
  auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                   ov::Shape{1, 2, 3, 7});
  auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                   ov::Shape{1, 2, 4, 7});
  auto scale = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{1, 1, 1, 1}, std::vector<float>{0.57735026f});
  auto scaled_k = std::make_shared<ov::op::v1::Multiply>(k, scale);
  auto scores = std::make_shared<ov::op::v0::MatMul>(q, scaled_k, true, false);
  auto softmax = std::make_shared<ov::op::v8::Softmax>(scores, -1);
  auto out = std::make_shared<ov::op::v0::MatMul>(v, softmax, false, true);
  auto res = std::make_shared<ov::op::v0::Result>(out);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{q, k, v},
      "vendor_attention_prescaled_key_plan");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  cfg.enable_attention_fusion = false;
  cfg.enable_vendor_attention_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);

  bool found = false;
  for (const auto &group : plan.groups) {
    if (group.kind == "VendorAttention" && group.node_indices.size() == 4) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionAttentionScaleMaskPlan) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 4});
  auto w1 = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{4, 4}, std::vector<float>(16, 0.5f));
  auto w2 = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{4, 4}, std::vector<float>(16, 0.25f));
  auto scale = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{1}, std::vector<float>{0.5f});
  auto mask = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{1, 4}, std::vector<float>(4, -1.0f));
  auto mm1 = std::make_shared<ov::op::v0::MatMul>(param, w1, false, false);
  auto scaled = std::make_shared<ov::op::v1::Multiply>(mm1, scale);
  auto add = std::make_shared<ov::op::v1::Add>(scaled, mask);
  auto sm = std::make_shared<ov::op::v1::Softmax>(add, 1);
  auto mm2 = std::make_shared<ov::op::v0::MatMul>(sm, w2, false, false);
  auto res = std::make_shared<ov::op::v0::Result>(mm2);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{param}, "attn_scale_mask");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "AttentionScaleMask" || group.node_indices.size() != 5) {
      continue;
    }
    const auto mm1_idx = group.node_indices[0];
    const auto scale_idx = group.node_indices[1];
    const auto add_idx = group.node_indices[2];
    const auto sm_idx = group.node_indices[3];
    const auto mm2_idx = group.node_indices[4];
    ASSERT_LT(mm1_idx, ordered.size());
    ASSERT_LT(scale_idx, ordered.size());
    ASSERT_LT(add_idx, ordered.size());
    ASSERT_LT(sm_idx, ordered.size());
    ASSERT_LT(mm2_idx, ordered.size());
    const auto &mm1_node = ordered[mm1_idx];
    const auto &scale_node = ordered[scale_idx];
    const auto &add_node = ordered[add_idx];
    const auto &sm_node = ordered[sm_idx];
    const auto &mm2_node = ordered[mm2_idx];
    if (ov::as_type_ptr<const ov::op::v0::MatMul>(mm1_node) &&
        ov::as_type_ptr<const ov::op::v1::Multiply>(scale_node) &&
        ov::as_type_ptr<const ov::op::v1::Add>(add_node) &&
        ov::as_type_ptr<const ov::op::v1::Softmax>(sm_node) &&
        ov::as_type_ptr<const ov::op::v0::MatMul>(mm2_node)) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionAttentionScalePlanWithConvertedConstant) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 4});
  auto w1 = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{4, 4}, std::vector<float>(16, 0.5f));
  auto w2 = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{4, 4}, std::vector<float>(16, 0.25f));
  auto scale_const = std::make_shared<ov::op::v0::Constant>(
      ov::element::f16, ov::Shape{1},
      std::vector<ov::float16>{ov::float16(0.5f)});
  auto scale =
      std::make_shared<ov::op::v0::Convert>(scale_const, ov::element::f32);
  auto mm1 = std::make_shared<ov::op::v0::MatMul>(param, w1, false, false);
  auto scaled = std::make_shared<ov::op::v1::Multiply>(mm1, scale);
  auto sm = std::make_shared<ov::op::v1::Softmax>(scaled, 1);
  auto mm2 = std::make_shared<ov::op::v0::MatMul>(sm, w2, false, false);
  auto res = std::make_shared<ov::op::v0::Result>(mm2);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{param}, "attn_scale");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "AttentionScale" && group.kind != "AttentionScaleMask") {
      continue;
    }
    bool has_convert = false;
    for (const auto idx : group.node_indices) {
      ASSERT_LT(idx, ordered.size());
      has_convert =
          has_convert ||
          static_cast<bool>(
              ov::as_type_ptr<const ov::op::v0::Convert>(ordered[idx]));
    }
    if (has_convert) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionAttentionScaleMaskPlanWithConvertedConstants) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 4});
  auto w1 = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{4, 4}, std::vector<float>(16, 0.5f));
  auto w2 = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{4, 4}, std::vector<float>(16, 0.25f));
  auto scale_const = std::make_shared<ov::op::v0::Constant>(
      ov::element::f16, ov::Shape{1},
      std::vector<ov::float16>{ov::float16(0.5f)});
  auto mask_const = std::make_shared<ov::op::v0::Constant>(
      ov::element::f16, ov::Shape{1, 4},
      std::vector<ov::float16>(4, ov::float16(-1.0f)));
  auto scale =
      std::make_shared<ov::op::v0::Convert>(scale_const, ov::element::f32);
  auto mask =
      std::make_shared<ov::op::v0::Convert>(mask_const, ov::element::f32);
  auto mm1 = std::make_shared<ov::op::v0::MatMul>(param, w1, false, false);
  auto scaled = std::make_shared<ov::op::v1::Multiply>(mm1, scale);
  auto add = std::make_shared<ov::op::v1::Add>(scaled, mask);
  auto sm = std::make_shared<ov::op::v1::Softmax>(add, 1);
  auto mm2 = std::make_shared<ov::op::v0::MatMul>(sm, w2, false, false);
  auto res = std::make_shared<ov::op::v0::Result>(mm2);
  auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                           ov::ParameterVector{param},
                                           "attn_scale_mask_convert");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "AttentionScaleMask") {
      continue;
    }
    size_t convert_count = 0;
    for (const auto idx : group.node_indices) {
      ASSERT_LT(idx, ordered.size());
      if (ov::as_type_ptr<const ov::op::v0::Convert>(ordered[idx])) {
        ++convert_count;
      }
    }
    if (convert_count >= 2) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionAttentionScalePlanExpandsLayoutWindow) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 3, 4, 4});
  auto perm = std::make_shared<ov::op::v0::Constant>(
      ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 1, 3, 2});
  auto transposed = std::make_shared<ov::op::v1::Transpose>(param, perm);
  auto axis = std::make_shared<ov::op::v0::Constant>(
      ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
  auto split = std::make_shared<ov::op::v1::Split>(transposed, axis, 3);
  auto scale = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{1}, std::vector<float>{0.5f});
  auto qk = std::make_shared<ov::op::v0::MatMul>(split->output(0),
                                                 split->output(1), false, true);
  auto scaled = std::make_shared<ov::op::v1::Multiply>(qk, scale);
  auto softmax = std::make_shared<ov::op::v1::Softmax>(scaled, 3);
  auto attn = std::make_shared<ov::op::v0::MatMul>(softmax, split->output(2),
                                                   false, false);
  auto post_transpose = std::make_shared<ov::op::v1::Transpose>(attn, perm);
  auto shape = std::make_shared<ov::op::v0::Constant>(
      ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, 4, 4});
  auto reshaped =
      std::make_shared<ov::op::v1::Reshape>(post_transpose, shape, false);
  auto res = std::make_shared<ov::op::v0::Result>(reshaped);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{param}, "attn_layout_window");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if ((group.kind != "AttentionScale" &&
         group.kind != "AttentionScaleMask") ||
        group.node_indices.size() < 7) {
      continue;
    }
    bool has_split = false;
    bool has_pre_transpose = false;
    bool has_post_reshape = false;
    for (const auto idx : group.node_indices) {
      ASSERT_LT(idx, ordered.size());
      const auto &node = ordered[idx];
      has_split =
          has_split ||
          static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Split>(node));
      has_pre_transpose =
          has_pre_transpose ||
          static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Transpose>(node));
      has_post_reshape =
          has_post_reshape ||
          static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Reshape>(node));
    }
    if (has_split && has_pre_transpose && has_post_reshape) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionAttentionPreScalePlanExpandsLayoutWindow) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 3, 4, 4});
  auto perm = std::make_shared<ov::op::v0::Constant>(
      ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 1, 3, 2});
  auto transposed = std::make_shared<ov::op::v1::Transpose>(param, perm);
  auto axis = std::make_shared<ov::op::v0::Constant>(
      ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
  auto split = std::make_shared<ov::op::v1::Split>(transposed, axis, 3);
  auto scale = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{1}, std::vector<float>{0.5f});
  auto pre_scaled_q =
      std::make_shared<ov::op::v1::Multiply>(split->output(1), scale);
  auto qk = std::make_shared<ov::op::v0::MatMul>(split->output(0), pre_scaled_q,
                                                 false, true);
  auto softmax = std::make_shared<ov::op::v1::Softmax>(qk, 3);
  auto attn = std::make_shared<ov::op::v0::MatMul>(split->output(2), softmax,
                                                   false, false);
  auto post_transpose = std::make_shared<ov::op::v1::Transpose>(attn, perm);
  auto shape = std::make_shared<ov::op::v0::Constant>(
      ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, 4, 4});
  auto reshaped =
      std::make_shared<ov::op::v1::Reshape>(post_transpose, shape, false);
  auto res = std::make_shared<ov::op::v0::Result>(reshaped);
  auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                           ov::ParameterVector{param},
                                           "attn_prescale_layout_window");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "AttentionScale" || group.node_indices.size() < 8) {
      continue;
    }
    bool has_split = false;
    bool has_pre_scale = false;
    bool has_matmul = false;
    bool has_post_reshape = false;
    for (const auto idx : group.node_indices) {
      ASSERT_LT(idx, ordered.size());
      const auto &node = ordered[idx];
      has_split =
          has_split ||
          static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Split>(node));
      has_pre_scale =
          has_pre_scale ||
          static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Multiply>(node));
      has_matmul =
          has_matmul ||
          static_cast<bool>(ov::as_type_ptr<const ov::op::v0::MatMul>(node));
      has_post_reshape =
          has_post_reshape ||
          static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Reshape>(node));
    }
    if (has_split && has_pre_scale && has_matmul && has_post_reshape) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxMlir, MatMulBuilderProducesModule) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 2});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 2});
  auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, true);

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(matmul, ctx);
  ASSERT_TRUE(module);
  auto func = module.lookupSymbol<mlir::func::FuncOp>("matmul_main");
  ASSERT_TRUE(static_cast<bool>(func));
  const auto func_type = func.getFunctionType();
  ASSERT_EQ(func_type.getNumInputs(), 2u);
  ASSERT_EQ(func_type.getNumResults(), 1u);
}

TEST(GfxMlir, MatMulMpsrtMetadataAnnotatesPlacementAndTensorDescriptors) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 4, 2});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 4, 2});
  auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, true);

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(matmul, ctx);
  ASSERT_TRUE(module);

  ASSERT_FALSE(static_cast<bool>(
      module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops")));
  const auto to_i64_dims = [](const ov::Shape &shape) {
    std::vector<int64_t> dims;
    dims.reserve(shape.size());
    for (const auto dim : shape) {
      dims.push_back(static_cast<int64_t>(dim));
    }
    return dims;
  };
  ov::gfx_plugin::GfxMpsrtGemmAbiDesc gemm_desc{};
  gemm_desc.transpose_rhs = 1;
  gemm_desc.alpha = 1.0f;
  ov::gfx_plugin::GfxAppleMpsVendorPrimitiveContract contract{};
  ASSERT_TRUE(ov::gfx_plugin::gfx_apple_make_mps_gemm_contract(
      gemm_desc,
      ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
          to_i64_dims(lhs->get_shape()), ov::element::f16,
          ov::gfx_plugin::GfxStageStorageKind::Matrix,
          ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo),
      ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
          to_i64_dims(rhs->get_shape()), ov::element::f16,
          ov::gfx_plugin::GfxStageStorageKind::Matrix,
          ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo),
      ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
          to_i64_dims(matmul->get_output_shape(0)), ov::element::f16,
          ov::gfx_plugin::GfxStageStorageKind::Matrix,
          ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo),
      contract));
  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "MatMul", matmul,
      ov::element::f16, false, false, false, {});
  const auto materialized =
      ov::gfx_plugin::materialize_apple_mps_vendor_contract_program(
          module, plan, "MatMul", contract);
  ASSERT_TRUE(materialized.valid);
  ASSERT_TRUE(materialized.typed_program_materialized);
  ASSERT_TRUE(static_cast<bool>(
      module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops")));
  ASSERT_FALSE(module->hasAttr("gfx.apple.pipeline.program.kind"));
  ASSERT_FALSE(module->hasAttr("gfx.apple.pipeline.program.record_key"));
  ASSERT_FALSE(module->hasAttr("gfx.apple.pipeline.program.stage_count"));
  const auto apple_passes =
      ov::gfx_plugin::gfx_apple_stage_pipeline_pass_boundaries(
          /*materialize_typed_program=*/true);
  ASSERT_EQ(ov::gfx_plugin::gfx_apple_stage_pipeline_pass_name(apple_passes[4]),
            std::string("gfx-apple-vendor-descriptor"));

  ASSERT_FALSE(module->hasAttr("gfx.backend"));
  ASSERT_FALSE(module->hasAttr("gfx.storage"));
  ASSERT_FALSE(module->hasAttr("gfx.stage_type"));
  ASSERT_FALSE(module->hasAttr("gfx.uses_vendor_primitive"));
  ASSERT_FALSE(module->hasAttr("gfx.uses_custom_kernel"));
  ASSERT_FALSE(module->hasAttr("gfx.specialization_key"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage_desc.kind"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage_record_key"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.gemm.transpose_rhs"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.input_count"));

  ov::gfx_plugin::GfxMpsrtModuleStagePlan extracted;
  ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_stage_plan(module, extracted));
  ASSERT_TRUE(extracted.valid);
  ASSERT_EQ(extracted.stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
  ASSERT_EQ(extracted.stage.domain,
            ov::gfx_plugin::GfxStageBackendDomain::AppleMps);
  ASSERT_EQ(extracted.stage.kernel_name, "mps_gemm");
  ASSERT_STREQ(ov::gfx_plugin::gfx_mpsrt_stage_builder_symbol(extracted.stage),
               "ovgfx_mpsrt_encode_gemm");
  ASSERT_EQ(extracted.stage.stage_manifest.stage_family,
            ov::gfx_plugin::GfxKernelStageFamily::Gemm);
  ASSERT_EQ(extracted.stage.stage_manifest.backend_domain,
            ov::gfx_plugin::GfxKernelBackendDomain::AppleMps);
  ASSERT_EQ(extracted.stage.stage_manifest.execution_kind,
            ov::gfx_plugin::GfxKernelExecutionKind::VendorPrimitive);
  ASSERT_EQ(extracted.stage.stage_manifest.storage,
            ov::gfx_plugin::GfxKernelStorageKind::Matrix);
  ASSERT_EQ(extracted.stage.stage_manifest.specialization_key,
            "apple_mps:matrix:MatMul");
  ASSERT_EQ(ov::gfx_plugin::gfx_mpsrt_stage_plan_record_key(extracted),
            "mps_gemm|apple_mps|matrix|matrix|row_major|MatMul|apple_mps:"
            "matrix:MatMul|"
            "gemm:ta0:tb1:alpha1.000000:beta0.000000");
  ASSERT_EQ(extracted.stage.gemm_desc.transpose_lhs, 0u);
  ASSERT_EQ(extracted.stage.gemm_desc.transpose_rhs, 1u);
  ASSERT_EQ(extracted.stage.gemm_desc.alpha, 1.0f);

  ov::gfx_plugin::GfxMpsrtProgram program;
  ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_program(module, program));
  ASSERT_TRUE(program.valid);
  ASSERT_FALSE(program.multi_stage);
  ASSERT_EQ(program.record_key,
            ov::gfx_plugin::gfx_mpsrt_stage_plan_record_key(extracted));
  ASSERT_EQ(program.external_buffer_abi.buffer_roles,
            std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput}));
  ASSERT_EQ(extracted.stage.gemm_desc.beta, 0.0f);
  ASSERT_EQ(extracted.inputs.size(), 2u);
  ASSERT_EQ(extracted.outputs.size(), 1u);
  ASSERT_EQ(extracted.inputs[0].storage,
            ov::gfx_plugin::GfxMpsrtStorage::Matrix);
  ASSERT_EQ(extracted.inputs[0].layout,
            ov::gfx_plugin::GfxMpsrtLayout::RowMajor);
  ASSERT_EQ(extracted.inputs[0].dtype, ov::gfx_plugin::GfxMpsrtDType::F16);
  ASSERT_EQ(extracted.inputs[0].matrix_rows, 4u);
  ASSERT_EQ(extracted.inputs[0].matrix_columns, 2u);
  ASSERT_EQ(extracted.inputs[0].matrix_row_bytes, 4u);
  ASSERT_TRUE(extracted.stage.stage_manifest.valid);
  ASSERT_EQ(extracted.stage.stage_manifest.stage_family,
            ov::gfx_plugin::GfxKernelStageFamily::Gemm);
  ASSERT_EQ(extracted.stage.stage_manifest.backend_domain,
            ov::gfx_plugin::GfxKernelBackendDomain::AppleMps);
  ASSERT_EQ(extracted.stage.stage_manifest.execution_kind,
            ov::gfx_plugin::GfxKernelExecutionKind::VendorPrimitive);
  ASSERT_EQ(extracted.stage.stage_manifest.storage,
            ov::gfx_plugin::GfxKernelStorageKind::Matrix);
  ASSERT_FALSE(extracted.stage.stage_manifest.custom_kernel.valid);
  ASSERT_EQ(extracted.stage.input_storage,
            ov::gfx_plugin::GfxMpsrtStorage::Matrix);
  ASSERT_EQ(extracted.stage.output_storage,
            ov::gfx_plugin::GfxMpsrtStorage::Matrix);
  ASSERT_EQ(extracted.stage.layout, ov::gfx_plugin::GfxMpsrtLayout::RowMajor);
  ASSERT_EQ(ov::gfx_plugin::gfx_mpsrt_stage_plan_record_key(extracted),
            "mps_gemm|apple_mps|matrix|matrix|row_major|MatMul|apple_mps:"
            "matrix:MatMul|"
            "gemm:ta0:tb1:alpha1.000000:beta0.000000");
  ASSERT_EQ(extracted.inputs.size(), 2u);
  ASSERT_EQ(extracted.outputs.size(), 1u);
  ASSERT_EQ(extracted.inputs[0].dtype, ov::gfx_plugin::GfxMpsrtDType::F16);
  ASSERT_EQ(extracted.inputs[0].matrix_rows, 4u);
  ASSERT_EQ(extracted.inputs[0].matrix_columns, 2u);
  ASSERT_EQ(extracted.outputs[0].byte_length, 1u * 4u * 4u * 2u);

  const auto module_builder_plan =
      ov::gfx_plugin::build_module_mpsrt_builder_plan(module);
  ASSERT_TRUE(module_builder_plan.valid);
  const auto &builder_plan = module_builder_plan.builder_plan;
  ASSERT_TRUE(builder_plan.valid);
  ASSERT_EQ(builder_plan.records.size(), 5u);
  ASSERT_EQ(builder_plan.input_values.size(), 2u);
  ASSERT_EQ(builder_plan.output_values.size(), 1u);
  ASSERT_EQ(builder_plan.storage_bridges.size(), 3u);
  ASSERT_EQ(builder_plan.storage_bridges[0].direction,
            ov::gfx_plugin::GfxMpsrtStorageBridgeDirection::BufferToMatrix);
  ASSERT_EQ(builder_plan.storage_bridges[1].direction,
            ov::gfx_plugin::GfxMpsrtStorageBridgeDirection::BufferToMatrix);
  ASSERT_EQ(builder_plan.storage_bridges[2].direction,
            ov::gfx_plugin::GfxMpsrtStorageBridgeDirection::MatrixToBuffer);
  ASSERT_EQ(builder_plan.records[0].kind,
            ov::gfx_plugin::GfxMpsrtBuilderRecordKind::ModelBegin);
  ASSERT_EQ(builder_plan.records[0].symbol, "ovgfx_mpsrt_model_begin");
  ASSERT_EQ(builder_plan.records[1].kind,
            ov::gfx_plugin::GfxMpsrtBuilderRecordKind::AddTensor);
  ASSERT_EQ(builder_plan.records[1].symbol, "ovgfx_mpsrt_add_tensor");
  ASSERT_EQ(builder_plan.records[1].value, 0u);
  ASSERT_EQ(builder_plan.records[1].tensor_descs[0].storage,
            static_cast<uint32_t>(ov::gfx_plugin::GfxMpsrtStorage::Matrix));
  ASSERT_EQ(builder_plan.records[3].kind,
            ov::gfx_plugin::GfxMpsrtBuilderRecordKind::EncodeStage);
  ASSERT_EQ(builder_plan.records[3].symbol, "ovgfx_mpsrt_encode_gemm");
  ASSERT_EQ(builder_plan.records[3].inputs.size(), 2u);
  ASSERT_EQ(builder_plan.records[3].outputs.size(), 1u);
  ASSERT_EQ(builder_plan.records[3].outputs[0], 2u);
  ASSERT_EQ(builder_plan.records[3].stage_desc.gemm_desc.transpose_lhs, 0u);
  ASSERT_EQ(builder_plan.records[3].stage_desc.gemm_desc.transpose_rhs, 1u);
  ASSERT_EQ(builder_plan.records[3].tensor_descs[0].byte_length,
            1u * 4u * 4u * 2u);
  ASSERT_EQ(builder_plan.records[4].kind,
            ov::gfx_plugin::GfxMpsrtBuilderRecordKind::ModelEnd);
  ASSERT_EQ(builder_plan.records[4].symbol, "ovgfx_mpsrt_model_end");
}

TEST(GfxMlir, ConvMpsrtMetadataAnnotatesVendorDescriptorFromOpenVINONode) {
  auto input = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::Shape{1, 16, 32, 32});
  auto weights =
      ov::op::v0::Constant::create(ov::element::f16, ov::Shape{8, 16, 3, 3},
                                   std::vector<float>(8 * 16 * 3 * 3, 1.f));
  auto conv = std::make_shared<ov::op::v1::Convolution>(
      input, weights, ov::Strides{2, 1}, ov::CoordinateDiff{1, 2},
      ov::CoordinateDiff{3, 4}, ov::Strides{1, 2});

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(conv, ctx);
  ASSERT_TRUE(module);

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "Convolution", conv,
      ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  const auto lowering_kind =
      ov::gfx_plugin::annotate_module_with_conv_mpsrt_plan(module, plan, conv,
                                                           "Convolution");
  ASSERT_EQ(lowering_kind, ov::gfx_plugin::GfxConvMpsrtLoweringKind::MpsConv2D);

  ASSERT_FALSE(module->hasAttr("gfx.backend"));
  ASSERT_FALSE(module->hasAttr("gfx.storage"));
  ASSERT_FALSE(module->hasAttr("gfx.stage_type"));
  ASSERT_FALSE(module->hasAttr("gfx.uses_vendor_primitive"));
  ASSERT_FALSE(module->hasAttr("gfx.uses_custom_kernel"));
  ASSERT_FALSE(module->hasAttr("gfx.specialization_key"));
  ASSERT_EQ(
      module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.stage_family")
          .str(),
      "convolution");
  ASSERT_EQ(
      module
          ->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.backend_domain")
          .str(),
      "apple_mps");
  ASSERT_EQ(
      module
          ->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.execution_kind")
          .str(),
      "vendor_primitive");
  ASSERT_EQ(
      module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.storage")
          .str(),
      "image");
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage_desc.kind"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage_record_key"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.conv2d.groups"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.input1.storage"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.storage_bridge_count"));

  ov::gfx_plugin::GfxMpsrtModuleStagePlan extracted;
  ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_stage_plan(module, extracted));
  ASSERT_TRUE(extracted.valid);
  ASSERT_EQ(extracted.stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MPSConv2D);
  ASSERT_STREQ(ov::gfx_plugin::gfx_mpsrt_stage_builder_symbol(extracted.stage),
               "ovgfx_mpsrt_encode_conv2d");
  ASSERT_EQ(ov::gfx_plugin::gfx_mpsrt_stage_plan_record_key(extracted),
            "mps_conv2d|apple_mps|image|image|nhwc4|Convolution|apple_mps:"
            "image:Convolution|"
            "conv2d:g1:s2x1:d1x2:p1,2,3,4");
  ASSERT_EQ(extracted.stage.stage_manifest.stage_family,
            ov::gfx_plugin::GfxKernelStageFamily::Convolution);
  ASSERT_EQ(extracted.stage.stage_manifest.backend_domain,
            ov::gfx_plugin::GfxKernelBackendDomain::AppleMps);
  ASSERT_EQ(extracted.stage.stage_manifest.execution_kind,
            ov::gfx_plugin::GfxKernelExecutionKind::VendorPrimitive);
  ASSERT_EQ(extracted.stage.stage_manifest.semantic_input_roles,
            std::vector<ov::gfx_plugin::GfxKernelBufferRole>(
                {ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
                 ov::gfx_plugin::GfxKernelBufferRole::ConstTensor}));
  ASSERT_EQ(extracted.stage.stage_manifest.semantic_output_roles,
            std::vector<ov::gfx_plugin::GfxKernelBufferRole>(
                {ov::gfx_plugin::GfxKernelBufferRole::TensorOutput}));
  ASSERT_EQ(extracted.stage.conv2d_desc.strides[0], 2u);
  ASSERT_EQ(extracted.stage.conv2d_desc.strides[1], 1u);
  ASSERT_EQ(extracted.stage.conv2d_desc.dilations[0], 1u);
  ASSERT_EQ(extracted.stage.conv2d_desc.dilations[1], 2u);
  ASSERT_EQ(extracted.stage.conv2d_desc.pads[0], 1u);
  ASSERT_EQ(extracted.stage.conv2d_desc.pads[1], 2u);
  ASSERT_EQ(extracted.stage.conv2d_desc.pads[2], 3u);
  ASSERT_EQ(extracted.stage.conv2d_desc.pads[3], 4u);
  ASSERT_EQ(extracted.stage.conv2d_desc.groups, 1u);
  ASSERT_EQ(extracted.inputs[1].storage,
            ov::gfx_plugin::GfxMpsrtStorage::Buffer);
  ASSERT_EQ(extracted.inputs[1].flags, ov::gfx_plugin::GfxMpsrtTensorFlagConst);

  const auto module_builder_plan =
      ov::gfx_plugin::build_module_mpsrt_builder_plan(module);
  ASSERT_TRUE(module_builder_plan.valid);
  ASSERT_EQ(module_builder_plan.builder_plan.storage_bridges.size(), 2u);
  ASSERT_EQ(module_builder_plan.builder_plan.storage_bridges[0].value, 0u);
  ASSERT_EQ(module_builder_plan.builder_plan.storage_bridges[0].direction,
            ov::gfx_plugin::GfxMpsrtStorageBridgeDirection::BufferToImage);
  ASSERT_EQ(module_builder_plan.builder_plan.storage_bridges[0].source_storage,
            ov::gfx_plugin::GfxMpsrtStorage::Buffer);
  ASSERT_EQ(module_builder_plan.builder_plan.storage_bridges[0].target_storage,
            ov::gfx_plugin::GfxMpsrtStorage::Image);
  ASSERT_EQ(module_builder_plan.builder_plan.storage_bridges[0].tensor.storage,
            static_cast<uint32_t>(ov::gfx_plugin::GfxMpsrtStorage::Image));
  ASSERT_EQ(module_builder_plan.builder_plan.storage_bridges[1].value, 2u);
  ASSERT_EQ(module_builder_plan.builder_plan.storage_bridges[1].direction,
            ov::gfx_plugin::GfxMpsrtStorageBridgeDirection::ImageToBuffer);

  ASSERT_EQ(module_builder_plan.builder_plan.records.size(), 5u);
  ASSERT_EQ(module_builder_plan.builder_plan.records[3].stage_desc.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MPSConv2D);
  ASSERT_EQ(module_builder_plan.builder_plan.records[3]
                .stage_desc.conv2d_desc.strides[0],
            2u);
  ASSERT_EQ(module_builder_plan.builder_plan.records[3]
                .stage_desc.conv2d_desc.dilations[1],
            2u);
  ASSERT_EQ(module_builder_plan.builder_plan.records[3]
                .stage_desc.conv2d_desc.pads[3],
            4u);
  ASSERT_EQ(module_builder_plan.stage_plan.inputs.size(), 2u);
  ASSERT_EQ(module_builder_plan.builder_plan.storage_bridges.size(), 2u);
  ASSERT_EQ(module_builder_plan.builder_plan.storage_bridges[0].value, 0u);
  ASSERT_EQ(module_builder_plan.builder_plan.storage_bridges[0].direction,
            ov::gfx_plugin::GfxMpsrtStorageBridgeDirection::BufferToImage);
  ASSERT_EQ(module_builder_plan.builder_plan.storage_bridges[1].value, 2u);
  ASSERT_EQ(module_builder_plan.builder_plan.storage_bridges[1].direction,
            ov::gfx_plugin::GfxMpsrtStorageBridgeDirection::ImageToBuffer);
  ASSERT_EQ(module_builder_plan.stage_plan.inputs[1].storage,
            ov::gfx_plugin::GfxMpsrtStorage::Buffer);
  ASSERT_EQ(module_builder_plan.stage_plan.inputs[1].flags,
            ov::gfx_plugin::GfxMpsrtTensorFlagConst);
  ASSERT_EQ(module_builder_plan.builder_plan.records[2].tensor_descs[0].storage,
            static_cast<uint32_t>(ov::gfx_plugin::GfxMpsrtStorage::Buffer));
  ASSERT_EQ(module_builder_plan.builder_plan.records[2].tensor_descs[0].flags,
            ov::gfx_plugin::GfxMpsrtTensorFlagConst);

  ASSERT_TRUE(ov::gfx_plugin::materialize_module_mpsrt_ops_from_stage_plan(
      module, extracted));
  auto ops_func = module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops");
  ASSERT_TRUE(static_cast<bool>(ops_func));

  size_t to_image_count = 0;
  size_t conv_count = 0;
  size_t to_buffer_count = 0;
  mlir::Operation *to_image_op = nullptr;
  mlir::Operation *to_buffer_op = nullptr;
  ops_func.walk([&](mlir::Operation *op) {
    const auto op_name = op->getName().getStringRef();
    if (op_name == "gfx.mpsrt.to_image") {
      ++to_image_count;
      to_image_op = op;
    } else if (op_name == "gfx.mpsrt.conv2d") {
      ++conv_count;
    } else if (op_name == "gfx.mpsrt.to_buffer") {
      ++to_buffer_count;
      to_buffer_op = op;
    }
  });
  ASSERT_EQ(to_image_count, 1u);
  ASSERT_EQ(conv_count, 1u);
  ASSERT_EQ(to_buffer_count, 1u);
  ASSERT_TRUE(to_image_op);
  ASSERT_TRUE(to_buffer_op);
  ASSERT_TRUE(
      to_image_op
          ->getAttrOfType<mlir::BoolAttr>("gfx.mpsrt.storage_bridge.generated")
          .getValue());
  ASSERT_EQ(to_image_op
                ->getAttrOfType<mlir::StringAttr>(
                    "gfx.mpsrt.storage_bridge.direction")
                .str(),
            "buffer_to_image");
  ASSERT_EQ(to_image_op
                ->getAttrOfType<mlir::StringAttr>(
                    "gfx.mpsrt.storage_bridge.target_storage")
                .str(),
            "image");
  ASSERT_EQ(to_buffer_op
                ->getAttrOfType<mlir::StringAttr>(
                    "gfx.mpsrt.storage_bridge.direction")
                .str(),
            "image_to_buffer");
  ASSERT_EQ(to_buffer_op
                ->getAttrOfType<mlir::StringAttr>(
                    "gfx.mpsrt.storage_bridge.target_storage")
                .str(),
            "buffer");
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));

  ov::gfx_plugin::GfxMpsrtProgram typed_program;
  ASSERT_TRUE(
      ov::gfx_plugin::read_module_mpsrt_ops_program(module, typed_program));
  ASSERT_TRUE(typed_program.has_storage_bridges);
  ASSERT_EQ(typed_program.storage_bridges.size(), 2u);
}

TEST(GfxMlir, GroupConvMpsrtMetadataDerivesStageTypeFromManifest) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                       ov::Shape{1, 4, 16, 16});
  auto weights =
      ov::op::v0::Constant::create(ov::element::f16, ov::Shape{4, 1, 1, 3, 3},
                                   std::vector<float>(4 * 1 * 1 * 3 * 3, 1.f));
  auto group_conv = std::make_shared<ov::op::v1::GroupConvolution>(
      input, weights, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(group_conv, ctx);
  ASSERT_TRUE(module);

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "GroupConvolution",
      group_conv, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  const auto lowering_kind =
      ov::gfx_plugin::annotate_module_with_conv_mpsrt_plan(
          module, plan, group_conv, "GroupConv2D");
  ASSERT_EQ(lowering_kind,
            ov::gfx_plugin::GfxConvMpsrtLoweringKind::MpsGroupConv2D);
  ASSERT_FALSE(module->hasAttr("gfx.stage_type"));
  ASSERT_EQ(
      module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.stage_family")
          .str(),
      "group_convolution");
  ASSERT_EQ(
      module
          ->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.backend_domain")
          .str(),
      "apple_mps");
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage_desc.kind"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.conv2d.groups"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.input1.storage"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage_record_key"));

  ov::gfx_plugin::GfxMpsrtModuleStagePlan extracted;
  ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_stage_plan(module, extracted));
  ASSERT_TRUE(extracted.valid);
  ASSERT_EQ(extracted.stage.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MPSGroupConv2D);
  ASSERT_EQ(ov::gfx_plugin::gfx_mpsrt_stage_type(extracted.stage),
            "GroupConvolution");
  ASSERT_EQ(extracted.stage.conv2d_desc.groups, 4u);
  ASSERT_EQ(extracted.inputs[1].storage,
            ov::gfx_plugin::GfxMpsrtStorage::Buffer);
  ASSERT_EQ(extracted.inputs[1].flags, ov::gfx_plugin::GfxMpsrtTensorFlagConst);
  ASSERT_EQ(ov::gfx_plugin::gfx_mpsrt_stage_plan_record_key(extracted),
            "mps_group_conv2d|apple_mps|image|image|nhwc4|GroupConvolution|"
            "apple_mps:image:GroupConvolution|conv2d:g4:s1x1:d1x1:p1,1,1,1");
  ASSERT_EQ(extracted.stage.stage_manifest.stage_family,
            ov::gfx_plugin::GfxKernelStageFamily::GroupConvolution);
  ASSERT_EQ(extracted.stage.stage_manifest.semantic_input_roles,
            std::vector<ov::gfx_plugin::GfxKernelBufferRole>(
                {ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
                 ov::gfx_plugin::GfxKernelBufferRole::ConstTensor}));

  const auto module_builder_plan =
      ov::gfx_plugin::build_module_mpsrt_builder_plan(module);
  ASSERT_TRUE(module_builder_plan.valid);
  ASSERT_EQ(module_builder_plan.builder_plan.records[3].stage_desc.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MPSGroupConv2D);
  ASSERT_EQ(
      module_builder_plan.builder_plan.records[3].stage_desc.conv2d_desc.groups,
      4u);
  ASSERT_EQ(module_builder_plan.stage_plan.inputs.size(), 2u);
  ASSERT_EQ(module_builder_plan.stage_plan.inputs[1].storage,
            ov::gfx_plugin::GfxMpsrtStorage::Buffer);
  ASSERT_EQ(module_builder_plan.stage_plan.inputs[1].flags,
            ov::gfx_plugin::GfxMpsrtTensorFlagConst);
  ASSERT_EQ(module_builder_plan.builder_plan.records[2].tensor_descs[0].storage,
            static_cast<uint32_t>(ov::gfx_plugin::GfxMpsrtStorage::Buffer));
  ASSERT_EQ(module_builder_plan.builder_plan.records[2].tensor_descs[0].flags,
            ov::gfx_plugin::GfxMpsrtTensorFlagConst);
}

TEST(GfxMlir, MpsrtTypedProgramMaterializesMatrixStorageConversionOps) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  auto lhs = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 8, 16}, ov::element::f16, ov::gfx_plugin::GfxStageStorageKind::Matrix,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  auto rhs = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 16, 4}, ov::element::f16, ov::gfx_plugin::GfxStageStorageKind::Matrix,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  auto output = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 8, 4}, ov::element::f16, ov::gfx_plugin::GfxStageStorageKind::Matrix,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "MatMul", nullptr,
      ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  auto stage = ov::gfx_plugin::gfx_mpsrt_make_stage_desc(plan, "MatMul");
  ASSERT_EQ(stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);

  ov::gfx_plugin::GfxMpsrtProgram program{};
  program.valid = true;
  program.record_key = "matrix_storage_bridge_model";
  program.inputs = {lhs, rhs};
  program.output_values = {2u};
  program.stages.push_back({stage, {0u, 1u}, {2u}, {output}});
  program.has_storage_bridges = true;

  ov::gfx_plugin::GfxMpsrtStorageBridgeDesc bridge{};
  ASSERT_TRUE(ov::gfx_plugin::gfx_mpsrt_make_storage_bridge_desc(
      0u, ov::gfx_plugin::gfx_mpsrt_to_abi_desc(lhs),
      ov::gfx_plugin::GfxMpsrtStorageBridgeDirection::BufferToMatrix, bridge));
  program.storage_bridges.push_back(bridge);
  ASSERT_TRUE(ov::gfx_plugin::gfx_mpsrt_make_storage_bridge_desc(
      1u, ov::gfx_plugin::gfx_mpsrt_to_abi_desc(rhs),
      ov::gfx_plugin::GfxMpsrtStorageBridgeDirection::BufferToMatrix, bridge));
  program.storage_bridges.push_back(bridge);
  ASSERT_TRUE(ov::gfx_plugin::gfx_mpsrt_make_storage_bridge_desc(
      2u, ov::gfx_plugin::gfx_mpsrt_to_abi_desc(output),
      ov::gfx_plugin::GfxMpsrtStorageBridgeDirection::MatrixToBuffer, bridge));
  program.storage_bridges.push_back(bridge);

  ASSERT_TRUE(ov::gfx_plugin::materialize_module_mpsrt_ops(module, program));
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));
  auto ops_func = module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops");
  ASSERT_TRUE(static_cast<bool>(ops_func));

  std::vector<std::string> mpsrt_ops;
  for (auto &op : ops_func.getBody().front().getOperations()) {
    const auto name = op.getName().getStringRef();
    if (name.starts_with("gfx.mpsrt.")) {
      mpsrt_ops.push_back(name.str());
    }
  }
  ASSERT_EQ(mpsrt_ops,
            std::vector<std::string>(
                {"gfx.mpsrt.to_matrix", "gfx.mpsrt.to_matrix", "gfx.mpsrt.gemm",
                 "gfx.mpsrt.to_buffer", "gfx.mpsrt.return"}));

  ov::gfx_plugin::GfxMpsrtProgram roundtrip;
  ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_ops_program(module, roundtrip));
  ASSERT_TRUE(roundtrip.has_storage_bridges);
  ASSERT_EQ(roundtrip.storage_bridges.size(), 3u);
  ASSERT_EQ(roundtrip.storage_bridges[0].direction,
            ov::gfx_plugin::GfxMpsrtStorageBridgeDirection::BufferToMatrix);
  ASSERT_EQ(roundtrip.storage_bridges[2].direction,
            ov::gfx_plugin::GfxMpsrtStorageBridgeDirection::MatrixToBuffer);
}

TEST(GfxMlir, MpsrtModuleMetadataRoundTripsMultiStageMpsGemmPlusMslDispatch) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 128, 256});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 256, 64});
  auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, false);

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(matmul, ctx);
  ASSERT_TRUE(module);

  ov::gfx_plugin::MatMulCodegenDesc desc{};
  desc.element_type = ov::element::f16;
  desc.input_a_type = ov::element::f16;
  desc.input_b_type = ov::element::f16;
  desc.output_type = ov::element::f16;
  desc.batch = 1;
  desc.batch_a = 1;
  desc.batch_b = 1;
  desc.M = 128;
  desc.N = 64;
  desc.K = 256;
  desc.has_activation = true;
  desc.activation = ov::gfx_plugin::ActivationKind::Relu;

  ov::gfx_plugin::annotate_module_with_matmul_mpsrt_epilogue_plan(
      module, desc, lhs->get_shape(), rhs->get_shape());

  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.model_stage_count"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage0.backend"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage1.backend"));
  ASSERT_FALSE(
      module->hasAttr("gfx.mpsrt.stage1.stage_manifest.kernel.entry_point"));
  ASSERT_EQ(ov::gfx_plugin::gfx_apple_stage_pipeline_pass_boundaries(
                /*materialize_typed_program=*/true)
                .size(),
            7u);
  ASSERT_FALSE(module->hasAttr("gfx.apple.pipeline.program.kind"));
  ASSERT_FALSE(module->hasAttr("gfx.apple.pipeline.program.stage_count"));
  ASSERT_FALSE(
      module->hasAttr("gfx.apple.pipeline.program.stage0.backend_domain"));
  ASSERT_FALSE(
      module->hasAttr("gfx.apple.pipeline.program.stage1.execution_kind"));

  ov::gfx_plugin::GfxMpsrtProgram extracted;
  ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_program(module, extracted));
  ASSERT_TRUE(extracted.valid);
  ASSERT_TRUE(extracted.multi_stage);
  ASSERT_EQ(extracted.record_key, "mps_gemm_plus_msl_epilogue_model|MatMul");
  ASSERT_EQ(extracted.inputs.size(), 2u);
  ASSERT_EQ(extracted.stages.size(), 2u);
  ASSERT_EQ(extracted.stages[0].stage.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
  ASSERT_EQ(extracted.stages[0].stage.domain,
            ov::gfx_plugin::GfxStageBackendDomain::AppleMps);
  ASSERT_EQ(extracted.stages[0].stage.gemm_desc.alpha, 1.0f);
  ASSERT_EQ(extracted.stages[1].stage.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
  ASSERT_EQ(extracted.stages[1].stage.domain,
            ov::gfx_plugin::GfxStageBackendDomain::AppleMsl);
  const auto extracted_dispatch =
      dispatch_spec_from_stage(extracted.stages[1].stage);
  ASSERT_TRUE(extracted_dispatch.valid);
  ASSERT_EQ(extracted_dispatch.entry_point, "eltwise_fused_buffer");
  ASSERT_EQ(extracted_dispatch.threads_per_threadgroup, 256u);
  ASSERT_TRUE(extracted_dispatch.precompiled_binary_required);
  ASSERT_EQ(extracted.stages[1].inputs,
            std::vector<ov::gfx_plugin::GfxMpsrtValue>({2u}));
  ASSERT_EQ(extracted.stages[1].outputs,
            std::vector<ov::gfx_plugin::GfxMpsrtValue>({3u}));

  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.program.symbol"));
  ASSERT_FALSE(static_cast<bool>(
      module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_program")));
  auto generated_ops = module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops");
  ASSERT_TRUE(static_cast<bool>(generated_ops));
  ASSERT_TRUE(
      generated_ops->getAttrOfType<mlir::BoolAttr>("gfx.mpsrt.ops.generated")
          .getValue());
  ASSERT_EQ(generated_ops->getAttrOfType<mlir::StringAttr>("gfx.mpsrt.ops.kind")
                .str(),
            "multi_stage");
  ASSERT_EQ(generated_ops
                ->getAttrOfType<mlir::IntegerAttr>("gfx.mpsrt.ops.stage_count")
                .getInt(),
            2);

  mlir::Builder stale_builder(module.getContext());
  module->setAttr("gfx.mpsrt.model_record_key",
                  stale_builder.getStringAttr("legacy_attrs_are_not_primary"));
  ov::gfx_plugin::GfxMpsrtProgram program;
  ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_program(module, program));
  ASSERT_TRUE(program.valid);
  ASSERT_TRUE(program.multi_stage);
  ASSERT_EQ(program.record_key, "mps_gemm_plus_msl_epilogue_model|MatMul");
  ASSERT_EQ(program.inputs.size(), 2u);
  ASSERT_EQ(program.stages.size(), 2u);
  ASSERT_EQ(program.stages[0].stage.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
  ASSERT_EQ(program.stages[1].stage.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
  ASSERT_TRUE(program.external_buffer_abi.valid);
  ASSERT_EQ(program.external_buffer_abi.buffer_roles,
            std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput}));

  ov::gfx_plugin::GfxMpsrtBuilderPlan program_builder_plan;
  ASSERT_TRUE(ov::gfx_plugin::gfx_mpsrt_build_builder_plan_from_program(
      program, program_builder_plan));
  ASSERT_TRUE(program_builder_plan.valid);
  ASSERT_EQ(program_builder_plan.records.size(), 6u);
  ASSERT_EQ(program_builder_plan.records[3].stage_desc.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
  ASSERT_EQ(program_builder_plan.records[4].stage_desc.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);

  ASSERT_TRUE(ov::gfx_plugin::materialize_module_mpsrt_ops(module, program));
  auto ops_func = module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops");
  ASSERT_TRUE(static_cast<bool>(ops_func));
  ASSERT_TRUE(ops_func->getAttrOfType<mlir::BoolAttr>("gfx.mpsrt.ops.generated")
                  .getValue());
  std::vector<std::string> mpsrt_ops;
  ops_func.walk([&](mlir::Operation *op) {
    if (op->getAttrOfType<mlir::BoolAttr>("gfx.mpsrt.op.generated")) {
      mpsrt_ops.push_back(op->getName().getStringRef().str());
    }
  });
  ASSERT_EQ(mpsrt_ops,
            std::vector<std::string>(
                {"gfx.mpsrt.gemm", "gfx.mpsrt.dispatch", "gfx.mpsrt.return"}));
  mlir::Operation *gemm_op = nullptr;
  mlir::Operation *dispatch_op = nullptr;
  ops_func.walk([&](mlir::Operation *op) {
    if (!op->getAttrOfType<mlir::BoolAttr>("gfx.mpsrt.op.generated")) {
      return;
    }
    const auto name = op->getName().getStringRef();
    if (name == "gfx.mpsrt.gemm") {
      gemm_op = op;
    } else if (name == "gfx.mpsrt.dispatch") {
      dispatch_op = op;
    }
  });
  ASSERT_NE(gemm_op, nullptr);
  ASSERT_NE(dispatch_op, nullptr);
  ASSERT_FALSE(static_cast<bool>(
      module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_op_gemm")));
  ASSERT_FALSE(static_cast<bool>(
      module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_op_dispatch")));
  ASSERT_EQ(
      gemm_op
          ->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.backend_domain")
          .str(),
      "apple_mps");
  ASSERT_EQ(
      gemm_op
          ->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.execution_kind")
          .str(),
      "vendor_primitive");
  ASSERT_EQ(
      gemm_op->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.storage")
          .str(),
      "matrix");
  ASSERT_EQ(
      dispatch_op
          ->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.backend_domain")
          .str(),
      "apple_msl");
  ASSERT_EQ(
      dispatch_op
          ->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.execution_kind")
          .str(),
      "custom_kernel");
  ASSERT_EQ(
      dispatch_op->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.storage")
          .str(),
      "buffer");
  ASSERT_EQ(dispatch_op
                ->getAttrOfType<mlir::StringAttr>(
                    "gfx.stage_manifest.kernel.entry_point")
                .str(),
            "eltwise_fused_buffer");
  ASSERT_FALSE(
      dispatch_op->hasAttr("gfx.mpsrt.op.stage.stage_manifest.backend_domain"));
  ASSERT_FALSE(
      dispatch_op->hasAttr("gfx.mpsrt.op.stage.stage_manifest.execution_kind"));
  ASSERT_FALSE(dispatch_op->hasAttr("gfx.mpsrt.op.stage.backend"));
  ASSERT_FALSE(dispatch_op->hasAttr("gfx.mpsrt.op.stage.kind"));
  ASSERT_FALSE(dispatch_op->hasAttr("gfx.mpsrt.op.stage.stage_record_key"));
  ASSERT_FALSE(gemm_op->hasAttr("gfx.mpsrt.op.stage.input_storage"));
  ASSERT_FALSE(gemm_op->hasAttr("gfx.mpsrt.op.stage.output_storage"));
  ASSERT_FALSE(gemm_op->hasAttr("gfx.mpsrt.op.stage.layout"));
  ASSERT_FALSE(dispatch_op->hasAttr("gfx.mpsrt.op.stage.input_storage"));
  ASSERT_FALSE(dispatch_op->hasAttr("gfx.mpsrt.op.stage.output_storage"));
  ASSERT_FALSE(dispatch_op->hasAttr("gfx.mpsrt.op.stage.layout"));
  ASSERT_TRUE(mlir::succeeded(mlir::verify(ops_func)));

  mlir::Builder op_builder(module.getContext());
  const auto dispatch_entry_point =
      dispatch_op->getAttrOfType<mlir::StringAttr>(
          "gfx.stage_manifest.kernel.entry_point");
  auto verify_ops_func_with_expected_failure = [&]() {
    mlir::ScopedDiagnosticHandler diag(
        module.getContext(),
        [](mlir::Diagnostic &) { return mlir::success(); });
    return mlir::verify(ops_func);
  };
  dispatch_op->removeAttr("gfx.stage_manifest.kernel.entry_point");
  ASSERT_TRUE(mlir::failed(verify_ops_func_with_expected_failure()));
  dispatch_op->setAttr("gfx.stage_manifest.kernel.entry_point",
                       dispatch_entry_point);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(ops_func)));

  const auto dispatch_backend_domain =
      dispatch_op->getAttrOfType<mlir::StringAttr>(
          "gfx.stage_manifest.backend_domain");
  dispatch_op->setAttr("gfx.stage_manifest.backend_domain",
                       op_builder.getStringAttr("apple_mps"));
  ASSERT_TRUE(mlir::failed(verify_ops_func_with_expected_failure()));
  dispatch_op->setAttr("gfx.stage_manifest.backend_domain",
                       dispatch_backend_domain);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(ops_func)));

  dispatch_op->setAttr("gfx.mpsrt.op.stage.stage_manifest.backend_domain",
                       op_builder.getStringAttr("apple_mps"));
  dispatch_op->setAttr("gfx.mpsrt.op.stage.stage_manifest.execution_kind",
                       op_builder.getStringAttr("vendor_primitive"));
  ov::gfx_plugin::GfxMpsrtProgram ops_program;
  ASSERT_TRUE(
      ov::gfx_plugin::read_module_mpsrt_ops_program(module, ops_program));
  ASSERT_TRUE(ops_program.valid);
  ASSERT_TRUE(ops_program.multi_stage);
  ASSERT_EQ(ops_program.record_key, "mps_gemm_plus_msl_epilogue_model|MatMul");
  ASSERT_EQ(ops_program.stages.size(), 2u);
  ASSERT_EQ(ops_program.stages[0].stage.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
  ASSERT_EQ(ops_program.stages[1].stage.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
  ASSERT_EQ(ops_program.stages[1].stage.domain,
            ov::gfx_plugin::GfxStageBackendDomain::AppleMsl);
  ASSERT_TRUE(ov::gfx_plugin::gfx_mpsrt_stage_uses_custom_kernel(
      ops_program.stages[1].stage));
  dispatch_op->setAttr("gfx.mpsrt.op.stage_index",
                       op_builder.getI32IntegerAttr(0));
  ov::gfx_plugin::GfxMpsrtProgram invalid_ops_program;
  ASSERT_FALSE(ov::gfx_plugin::read_module_mpsrt_ops_program(
      module, invalid_ops_program));
  dispatch_op->setAttr("gfx.mpsrt.op.stage_index",
                       op_builder.getI32IntegerAttr(1));

  const auto module_builder_plan =
      ov::gfx_plugin::build_module_mpsrt_builder_plan(module);
  ASSERT_TRUE(module_builder_plan.valid);
  ASSERT_TRUE(module_builder_plan.multi_stage);
  ASSERT_TRUE(module_builder_plan.program.valid);
  ASSERT_TRUE(module_builder_plan.program.multi_stage);
  ASSERT_EQ(module_builder_plan.program.stages.size(), 2u);
  ASSERT_TRUE(module_builder_plan.builder_plan.valid);
  ASSERT_EQ(module_builder_plan.builder_plan.records.size(), 6u);
  ASSERT_EQ(module_builder_plan.builder_plan.input_values,
            std::vector<ov::gfx_plugin::GfxMpsrtValue>({0u, 1u}));
  ASSERT_EQ(module_builder_plan.builder_plan.output_values,
            std::vector<ov::gfx_plugin::GfxMpsrtValue>({3u}));
  ASSERT_EQ(module_builder_plan.builder_plan.records[3].stage_desc.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
  ASSERT_EQ(module_builder_plan.builder_plan.records[4].stage_desc.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
  ASSERT_EQ(module_builder_plan.builder_plan.records[4].kernel_buffer_order,
            std::vector<ov::gfx_plugin::GfxMpsrtValue>({2u, 3u}));
  ASSERT_TRUE(module_builder_plan.builder_plan.external_buffer_abi_valid);
  ASSERT_EQ(module_builder_plan.builder_plan.external_buffer_roles,
            std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput}));
}

TEST(GfxMlir, AppleMpsrtProgramPlanMaterializesExplicitMixedValueEdges) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  auto lhs = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 8, 16}, ov::element::f16, ov::gfx_plugin::GfxStageStorageKind::Matrix,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  auto rhs = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 16, 4}, ov::element::f16, ov::gfx_plugin::GfxStageStorageKind::Matrix,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  auto gemm_output = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 8, 4}, ov::element::f16, ov::gfx_plugin::GfxStageStorageKind::Matrix);
  auto output = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 8, 4}, ov::element::f16, ov::gfx_plugin::GfxStageStorageKind::Buffer,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "MatMul", nullptr,
      ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  auto gemm_stage = ov::gfx_plugin::gfx_mpsrt_make_stage_desc(plan, "MatMul");
  gemm_stage.gemm_desc.alpha = 1.0f;

  ov::gfx_plugin::GfxMpsrtStageDesc epilogue_stage{};
  epilogue_stage.kind = ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch;
  epilogue_stage.domain = ov::gfx_plugin::GfxStageBackendDomain::AppleMsl;
  epilogue_stage.input_storage = ov::gfx_plugin::GfxMpsrtStorage::Buffer;
  epilogue_stage.output_storage = ov::gfx_plugin::GfxMpsrtStorage::Buffer;
  epilogue_stage.layout = ov::gfx_plugin::GfxMpsrtLayout::Linear;
  const auto epilogue_binding =
      ov::gfx_plugin::make_backend_custom_kernel_roles_binding_plan(
          "MatMulEpilogue", "eltwise_fused_buffer",
          {ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
           ov::gfx_plugin::GfxKernelBufferRole::TensorOutput});
  ASSERT_TRUE(epilogue_binding.valid);
  ASSERT_TRUE(epilogue_binding.stage_manifest.valid);
  epilogue_stage.stage_manifest = epilogue_binding.stage_manifest;

  ov::gfx_plugin::GfxAppleMpsrtProgramPlan program_plan{};
  program_plan.record_key = "explicit_apple_program_plan";
  program_plan.inputs = {lhs, rhs};
  program_plan.output_values = {3u};
  program_plan.stages.push_back({gemm_stage, {0u, 1u}, {2u}, {gemm_output}});
  program_plan.stages.push_back({epilogue_stage, {2u}, {3u}, {output}});

  auto missing_abi_module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
  const auto missing_abi_rejected =
      ov::gfx_plugin::materialize_apple_mpsrt_program_plan(missing_abi_module,
                                                           program_plan);
  ASSERT_FALSE(missing_abi_rejected.valid);
  ASSERT_FALSE(static_cast<bool>(
      missing_abi_module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops")));

  program_plan.external_buffer_abi =
      ov::gfx_plugin::gfx_mpsrt_make_external_buffer_abi_from_roles(
          {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
           ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
           ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput});

  const auto materialized =
      ov::gfx_plugin::materialize_apple_mpsrt_program_plan(module,
                                                           program_plan);
  ASSERT_TRUE(materialized.valid);
  ASSERT_TRUE(materialized.typed_program_materialized);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));

  ov::gfx_plugin::GfxMpsrtProgram roundtrip;
  ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_ops_program(module, roundtrip));
  ASSERT_EQ(roundtrip.record_key, "explicit_apple_program_plan");
  ASSERT_TRUE(roundtrip.multi_stage);
  ASSERT_EQ(roundtrip.inputs.size(), 2u);
  ASSERT_EQ(roundtrip.output_values,
            std::vector<ov::gfx_plugin::GfxMpsrtValue>({3u}));
  ASSERT_TRUE(roundtrip.external_buffer_abi.valid);
  ASSERT_EQ(roundtrip.external_buffer_abi.buffer_roles,
            std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput}));
  ASSERT_EQ(roundtrip.stages[0].inputs,
            std::vector<ov::gfx_plugin::GfxMpsrtValue>({0u, 1u}));
  ASSERT_EQ(roundtrip.stages[1].inputs,
            std::vector<ov::gfx_plugin::GfxMpsrtValue>({2u}));
  ASSERT_EQ(roundtrip.stages[1].stage.input_storage,
            ov::gfx_plugin::GfxMpsrtStorage::Matrix);
  ASSERT_EQ(roundtrip.stages[1].stage.output_storage,
            ov::gfx_plugin::GfxMpsrtStorage::Buffer);
  ASSERT_EQ(roundtrip.stages[1].stage.layout,
            ov::gfx_plugin::GfxMpsrtLayout::Linear);
  const auto roundtrip_dispatch =
      dispatch_spec_from_stage(roundtrip.stages[1].stage);
  ASSERT_TRUE(roundtrip_dispatch.valid);
  ASSERT_EQ(roundtrip_dispatch.entry_point, "eltwise_fused_buffer");
  ASSERT_EQ(roundtrip_dispatch.threads_per_threadgroup, 256u);

  auto invalid_plan = program_plan;
  invalid_plan.stages[0].stage.domain =
      ov::gfx_plugin::GfxStageBackendDomain::OpenCl;
  auto invalid_module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
  const auto rejected = ov::gfx_plugin::materialize_apple_mpsrt_program_plan(
      invalid_module, invalid_plan);
  ASSERT_FALSE(rejected.valid);
  ASSERT_FALSE(static_cast<bool>(
      invalid_module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops")));
}

TEST(GfxMlir, MatMulMpsrtLoweringEntryPointSelectsSingleAndMultiStagePlans) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 128, 256});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 256, 64});
  auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, false);

  ov::gfx_plugin::MatMulCodegenDesc desc{};
  desc.element_type = ov::element::f16;
  desc.input_a_type = ov::element::f16;
  desc.input_b_type = ov::element::f16;
  desc.output_type = ov::element::f16;
  desc.batch = 1;
  desc.batch_a = 1;
  desc.batch_b = 1;
  desc.M = 128;
  desc.N = 64;
  desc.K = 256;

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "MatMul", matmul,
      ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto gemm_module = ov::gfx_plugin::build_mlir_for_node(matmul, ctx);
  ASSERT_TRUE(gemm_module);
  const auto gemm_lowering =
      ov::gfx_plugin::annotate_module_with_matmul_mpsrt_plan(
          gemm_module, plan, desc, lhs->get_shape(), rhs->get_shape());
  ASSERT_EQ(gemm_lowering, ov::gfx_plugin::GfxMatMulMpsrtLoweringKind::MpsGemm);

  ov::gfx_plugin::GfxMpsrtModuleStagePlan stage_plan;
  ASSERT_TRUE(
      ov::gfx_plugin::read_module_mpsrt_stage_plan(gemm_module, stage_plan));
  ASSERT_EQ(stage_plan.stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
  ov::gfx_plugin::GfxMpsrtProgram gemm_program;
  ASSERT_TRUE(
      ov::gfx_plugin::read_module_mpsrt_program(gemm_module, gemm_program));
  ASSERT_FALSE(gemm_program.multi_stage);
  ASSERT_EQ(gemm_program.record_key,
            ov::gfx_plugin::gfx_mpsrt_stage_plan_record_key(stage_plan));
  ASSERT_FALSE(gemm_module->hasAttr("gfx.apple.pipeline.program.kind"));
  ASSERT_FALSE(gemm_module->hasAttr("gfx.apple.pipeline.program.stage0.kind"));
  const auto generic_gemm_source_plan =
      ov::gfx_plugin::make_mpsrt_kernel_source_plan_from_module(gemm_module);
  ASSERT_TRUE(generic_gemm_source_plan.valid());
  ASSERT_EQ(generic_gemm_source_plan.kind,
            ov::gfx_plugin::GfxMpsrtKernelSourcePlanKind::SingleStage);
  ASSERT_EQ(generic_gemm_source_plan.first_stage_kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
  ASSERT_EQ(generic_gemm_source_plan.last_stage_kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
  ASSERT_EQ(generic_gemm_source_plan.source.entry_point, "mps_gemm");
  ASSERT_EQ(generic_gemm_source_plan.source.signature.arg_count, 3u);
  ASSERT_EQ(generic_gemm_source_plan.source.signature.output_arg_count, 1u);

  auto epilogue_module = ov::gfx_plugin::build_mlir_for_node(matmul, ctx);
  ASSERT_TRUE(epilogue_module);
  auto epilogue_desc = desc;
  epilogue_desc.has_bias = true;
  epilogue_desc.bias_type = ov::element::f16;
  epilogue_desc.bias_dims = {1, 1, 64};
  epilogue_desc.has_activation = true;
  epilogue_desc.activation = ov::gfx_plugin::ActivationKind::Swish;
  const auto epilogue_lowering =
      ov::gfx_plugin::annotate_module_with_matmul_mpsrt_plan(
          epilogue_module, plan, epilogue_desc, lhs->get_shape(),
          rhs->get_shape());
  ASSERT_EQ(epilogue_lowering,
            ov::gfx_plugin::GfxMatMulMpsrtLoweringKind::MpsGemmWithMslEpilogue);

  ov::gfx_plugin::GfxMpsrtProgram multi_stage;
  ASSERT_TRUE(
      ov::gfx_plugin::read_module_mpsrt_program(epilogue_module, multi_stage));
  ASSERT_TRUE(multi_stage.multi_stage);
  ASSERT_EQ(multi_stage.stages.size(), 2u);
  ASSERT_EQ(multi_stage.stages[0].stage.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
  ASSERT_EQ(multi_stage.stages[1].stage.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
  ASSERT_EQ(multi_stage.stages[1].stage.stage_manifest.execution_kind,
            ov::gfx_plugin::GfxKernelExecutionKind::CustomKernel);

  auto source_module = ov::gfx_plugin::build_mlir_for_node(matmul, ctx);
  ASSERT_TRUE(source_module);
  const auto source_plan = ov::gfx_plugin::lower_matmul_module_to_mpsrt_plan(
      source_module, plan, epilogue_desc, lhs->get_shape(), rhs->get_shape());
  ASSERT_TRUE(source_plan.valid());
  ASSERT_TRUE(source_plan.mpsrt_plan.requires_mpsrt_model);
  ASSERT_EQ(source_plan.lowering,
            ov::gfx_plugin::GfxMatMulMpsrtLoweringKind::MpsGemmWithMslEpilogue);
  ASSERT_TRUE(source_plan.mpsrt_plan.valid());
  ASSERT_EQ(source_plan.mpsrt_plan.kind,
            ov::gfx_plugin::GfxMpsrtKernelSourcePlanKind::MultiStage);
  ASSERT_EQ(source_plan.mpsrt_plan.first_stage_kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
  ASSERT_EQ(source_plan.mpsrt_plan.last_stage_kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
  const auto &epilogue_source = source_plan.mpsrt_plan.source;
  ASSERT_EQ(epilogue_source.entry_point, "eltwise_fused_buffer");
  ASSERT_FALSE(epilogue_source.msl_source.empty());
  ASSERT_EQ(epilogue_source.signature.arg_count, 3u);
  ASSERT_EQ(epilogue_source.signature.output_arg_count, 1u);
  const auto epilogue_metadata =
      ov::gfx_plugin::extract_kernel_runtime_metadata(
          epilogue_source.module, epilogue_source.signature.output_arg_count,
          /*fallback_input_arg_count=*/999, epilogue_source.entry_point);
  ASSERT_TRUE(epilogue_metadata.valid);
  ASSERT_EQ(epilogue_metadata.kernel_input_arg_count, 2u);
  EXPECT_EQ(epilogue_metadata.operands.operand_kinds,
            std::vector<int32_t>({1, 1, 1}));
  EXPECT_EQ(epilogue_metadata.operands.operand_arg_indices,
            std::vector<int32_t>({0, 1, 2}));

  auto fallback_desc = desc;
  fallback_desc.batch = 3;
  fallback_desc.batch_a = 2;
  fallback_desc.batch_b = 3;
  EXPECT_THROW(
      (void)ov::gfx_plugin::make_apple_metal_runtime_matmul_kernel_source_plan(
          ctx, nullptr, matmul, fallback_desc, lhs->get_shape(),
          rhs->get_shape(), "matmul_mpsrt_reject_test"),
      ov::Exception);
}

TEST(GfxMlir, AppleMpsrtTypedProgramMaterializesMultiStageBuilderRecords) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 128, 256});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 256, 64});
  auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, false);

  ov::gfx_plugin::MatMulCodegenDesc desc{};
  desc.element_type = ov::element::f16;
  desc.input_a_type = ov::element::f16;
  desc.input_b_type = ov::element::f16;
  desc.output_type = ov::element::f16;
  desc.bias_type = ov::element::f16;
  desc.M = 128;
  desc.N = 64;
  desc.K = 256;
  desc.batch = 1;
  desc.batch_a = 1;
  desc.batch_b = 1;
  desc.has_bias = true;
  desc.bias_dims = {1, 1, 64};
  desc.has_activation = true;
  desc.activation = ov::gfx_plugin::ActivationKind::Swish;

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "MatMul", matmul,
      ov::element::f16,
      /*has_bias=*/true,
      /*has_activation=*/true,
      /*has_batchnorm=*/false, {});
  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(matmul, ctx);
  ASSERT_TRUE(module);
  const auto lowering = ov::gfx_plugin::annotate_module_with_matmul_mpsrt_plan(
      module, plan, desc, lhs->get_shape(), rhs->get_shape());
  ASSERT_EQ(lowering,
            ov::gfx_plugin::GfxMatMulMpsrtLoweringKind::MpsGemmWithMslEpilogue);
  ASSERT_FALSE(static_cast<bool>(
      module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_program")));
  ASSERT_TRUE(static_cast<bool>(
      module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops")));
  ASSERT_FALSE(module->hasAttr("gfx.apple.pipeline.program.kind"));
  ASSERT_FALSE(module->hasAttr("gfx.apple.pipeline.program.stage0.kind"));
  ASSERT_FALSE(module->hasAttr("gfx.apple.pipeline.program.stage1.kind"));
  ov::gfx_plugin::GfxMpsrtProgram program;
  ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_program(module, program));
  ASSERT_EQ(program.stages.size(), 2u);
  const auto &epilogue_abi =
      program.stages[1].stage.stage_manifest.custom_kernel.external_buffer_abi;
  ASSERT_TRUE(epilogue_abi.valid);
  ASSERT_EQ(epilogue_abi.roles,
            std::vector<ov::gfx_plugin::GfxKernelBufferRole>(
                {ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
                 ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
                 ov::gfx_plugin::GfxKernelBufferRole::TensorOutput}));
  module->removeAttr("gfx.mpsrt.stage_desc.kind");
  module->removeAttr("gfx.mpsrt.model_record_key");
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage_desc.kind"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.model_record_key"));

  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.runtime_abi.call_plan_symbol"));
  ASSERT_FALSE(
      module->hasAttr("gfx.apple.pipeline.runtime_abi.call_plan_materialized"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.runtime_abi.valid"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.runtime_abi.kind"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.runtime_abi.record_key"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.runtime_abi.record_count"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.runtime_abi.external_buffer_count"));
  ASSERT_FALSE(
      module->hasAttr("gfx.mpsrt.runtime_abi.external_output_buffer_count"));
  ASSERT_FALSE(
      module->hasAttr("gfx.mpsrt.runtime_abi.record4.stage_desc.kind"));
  ASSERT_FALSE(
      module->hasAttr("gfx.mpsrt.runtime_abi.record5.stage_desc.kind"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.runtime_abi.record5.kernel_name"));
  ASSERT_FALSE(
      module->hasAttr("gfx.mpsrt.runtime_abi.record5.kernel_buffer_order"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.model_stage_count"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.input_count"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.storage_bridge_count"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage0.backend"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage1.dispatch_entry_point"));
  ASSERT_FALSE(static_cast<bool>(
      module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_program")));
  ASSERT_TRUE(static_cast<bool>(
      module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops")));

  ov::gfx_plugin::GfxMpsrtProgram cleaned_multi_stage;
  ASSERT_TRUE(
      ov::gfx_plugin::read_module_mpsrt_program(module, cleaned_multi_stage));
  ASSERT_TRUE(cleaned_multi_stage.valid);
  ASSERT_TRUE(cleaned_multi_stage.multi_stage);
  ASSERT_EQ(cleaned_multi_stage.record_key,
            "mps_gemm_plus_msl_epilogue_model|MatMul");
  ASSERT_EQ(cleaned_multi_stage.stages.size(), 2u);
  ASSERT_EQ(cleaned_multi_stage.stages[0].stage.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
  ASSERT_EQ(cleaned_multi_stage.stages[1].stage.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);

  ASSERT_FALSE(static_cast<bool>(
      module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_runtime_abi_plan")));

  ov::gfx_plugin::GfxMpsrtBuilderPlan call_builder_plan;
  ASSERT_TRUE(ov::gfx_plugin::gfx_mpsrt_build_builder_plan_from_program(
      cleaned_multi_stage, call_builder_plan));
  ASSERT_TRUE(call_builder_plan.valid);
  ASSERT_EQ(call_builder_plan.model_record_key,
            "mps_gemm_plus_msl_epilogue_model|MatMul");
  ASSERT_EQ(call_builder_plan.records.size(), 7u);
  ASSERT_EQ(call_builder_plan.input_values.size(), 3u);
  ASSERT_EQ(call_builder_plan.output_values.size(), 1u);
  ASSERT_TRUE(call_builder_plan.external_buffer_abi_valid);
  ASSERT_EQ(call_builder_plan.external_buffer_count, 4u);
  ASSERT_EQ(call_builder_plan.external_output_buffer_count, 1u);
  ASSERT_EQ(call_builder_plan.records[1].kind,
            ov::gfx_plugin::GfxMpsrtBuilderRecordKind::AddTensor);
  ASSERT_EQ(call_builder_plan.records[1].value, 0u);
  ASSERT_EQ(call_builder_plan.records[1].tensor_descs.size(), 1u);
  ASSERT_EQ(call_builder_plan.records[1].tensor_descs[0].storage,
            static_cast<uint32_t>(ov::gfx_plugin::GfxMpsrtStorage::Matrix));
  ASSERT_EQ(call_builder_plan.records[4].stage_desc.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
  ASSERT_EQ(call_builder_plan.records[4].tensor_descs.size(), 1u);
  ASSERT_EQ(call_builder_plan.records[4].stage_desc.gemm_desc.transpose_lhs,
            0u);
  ASSERT_EQ(call_builder_plan.records[4].stage_desc.gemm_desc.transpose_rhs,
            0u);
  ASSERT_EQ(call_builder_plan.records[5].stage_desc.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
  ASSERT_EQ(call_builder_plan.records[5].stage_desc.kernel_name,
            "eltwise_fused_buffer");
  ASSERT_EQ(call_builder_plan.records[5].tensor_descs.size(), 1u);
  const auto call_dispatch_desc =
      ov::gfx_plugin::gfx_mpsrt_make_msl_dispatch_desc(
          call_builder_plan.records[5].stage_desc,
          static_cast<uint32_t>(call_builder_plan.records[5].inputs.size()),
          static_cast<uint32_t>(call_builder_plan.records[5].outputs.size()));
  ASSERT_EQ(call_dispatch_desc.kernel_family,
            static_cast<uint32_t>(
                ov::gfx_plugin::GfxKernelFamily::EltwiseFusedBuffer));
  ASSERT_EQ(call_dispatch_desc.input_count, 2u);
  ASSERT_EQ(call_dispatch_desc.output_count, 1u);
  ASSERT_EQ(call_builder_plan.records[5].kernel_buffer_order.size(), 3u);
}

TEST(GfxMlir, MatMulMpsrtEpilogueMslSourceUsesCanonicalCustomKernelAbi) {
  ov::gfx_plugin::MatMulCodegenDesc desc{};
  desc.element_type = ov::element::f16;
  desc.output_type = ov::element::f16;
  desc.bias_type = ov::element::f32;
  desc.batch = 2;
  desc.M = 4;
  desc.N = 8;
  desc.has_bias = true;
  desc.bias_dims = {1, 1, 8};
  desc.has_activation = true;
  desc.activation = ov::gfx_plugin::ActivationKind::Swish;

  const auto msl = ov::gfx_plugin::generate_msl_for_matmul_mpsrt_epilogue(desc);
  ASSERT_NE(msl.find("kernel void eltwise_fused_buffer"), std::string::npos);
  ASSERT_NE(msl.find("device const half* gemm [[buffer(0)]]"),
            std::string::npos);
  ASSERT_NE(msl.find("device const float* bias [[buffer(1)]]"),
            std::string::npos);
  ASSERT_NE(msl.find("device half* output [[buffer(2)]]"), std::string::npos);
  ASSERT_NE(msl.find("constant uint BATCH = 2;"), std::string::npos);
  ASSERT_NE(msl.find("constant uint M = 4;"), std::string::npos);
  ASSERT_NE(msl.find("constant uint N = 8;"), std::string::npos);
  ASSERT_NE(msl.find("x = x / (1.0f + precise::exp(-x));"), std::string::npos);
}

TEST(GfxMlir, SigmoidMslCodegenUsesPreciseExpForScoreStability) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  ov::gfx_plugin::UnaryCodegenDesc unary{};
  unary.element_type = ov::element::f32;
  unary.activation = ov::gfx_plugin::ActivationKind::Sigmoid;
  EXPECT_NE(ov::gfx_plugin::generate_msl_for_unary(unary, module)
                .find("precise::exp(-x)"),
            std::string::npos);

  ov::gfx_plugin::EltwiseCodegenDesc eltwise{};
  eltwise.element_type = ov::element::f32;
  eltwise.input0_type = ov::element::f32;
  eltwise.input1_type = ov::element::f32;
  eltwise.output_type = ov::element::f32;
  eltwise.eltwise_kind = ov::gfx_plugin::EltwiseKind::Add;
  eltwise.num_elements = 16;
  eltwise.has_activation = true;
  eltwise.activation = ov::gfx_plugin::ActivationKind::Sigmoid;
  EXPECT_NE(ov::gfx_plugin::generate_msl_for_eltwise(eltwise, module)
                .find("precise::exp(-x)"),
            std::string::npos);

  ov::gfx_plugin::MatMulCodegenDesc matmul{};
  matmul.element_type = ov::element::f32;
  matmul.output_type = ov::element::f32;
  matmul.M = 2;
  matmul.N = 3;
  matmul.K = 4;
  matmul.has_activation = true;
  matmul.activation = ov::gfx_plugin::ActivationKind::Sigmoid;
  EXPECT_NE(ov::gfx_plugin::generate_msl_for_matmul(matmul, module)
                .find("precise::exp(-x)"),
            std::string::npos);

  ov::gfx_plugin::Conv2DCodegenDesc conv{};
  conv.element_type = ov::element::f32;
  conv.input_type = ov::element::f32;
  conv.weight_type = ov::element::f32;
  conv.output_type = ov::element::f32;
  conv.N = 1;
  conv.C_in = 3;
  conv.H = 4;
  conv.W = 4;
  conv.C_out = 2;
  conv.C_in_pg = 3;
  conv.C_out_pg = 2;
  conv.kH = 1;
  conv.kW = 1;
  conv.outH = 4;
  conv.outW = 4;
  conv.has_activation = true;
  conv.activation = ov::gfx_plugin::ActivationKind::Sigmoid;
  EXPECT_NE(ov::gfx_plugin::generate_msl_for_conv2d(conv, module)
                .find("precise::exp(-acc)"),
            std::string::npos);
}

TEST(GfxMlir, AddMslMetadataUsesRequiredMpsrtKernelFamily) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 64, 80, 80});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 64, 80, 80});
  auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(add, ctx);
  ASSERT_TRUE(module);
  mlir::Builder builder(module.getContext());

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "Add", add, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Add");

  ASSERT_FALSE(module->hasAttr("gfx.backend"));
  ASSERT_FALSE(module->hasAttr("gfx.storage"));
  ASSERT_FALSE(module->hasAttr("gfx.stage_type"));
  ASSERT_FALSE(module->hasAttr("gfx.uses_vendor_primitive"));
  ASSERT_FALSE(module->hasAttr("gfx.uses_custom_kernel"));
  ASSERT_FALSE(module->hasAttr("gfx.specialization_key"));
  ASSERT_FALSE(module->hasAttr("gfx.msl.kernel_family"));
  ASSERT_FALSE(module->hasAttr("gfx.msl.required_entry_point"));
  ASSERT_FALSE(module->hasAttr("gfx.msl.precompiled_metallib_required"));
  ASSERT_FALSE(module->hasAttr("gfx.msl.threads_per_threadgroup"));
  ASSERT_EQ(
      module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.stage_family")
          .str(),
      "eltwise");
  ASSERT_EQ(
      module
          ->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.backend_domain")
          .str(),
      "apple_msl");
  ASSERT_EQ(
      module
          ->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.execution_kind")
          .str(),
      "custom_kernel");
  ASSERT_EQ(
      module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.storage")
          .str(),
      "buffer");
  ASSERT_EQ(
      module
          ->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.kernel.family")
          .str(),
      "eltwise_fused_buffer");
  ASSERT_EQ(module
                ->getAttrOfType<mlir::StringAttr>(
                    "gfx.stage_manifest.kernel.entry_point")
                .str(),
            "eltwise_fused_buffer");
  ASSERT_FALSE(module->hasAttr(
      "gfx.stage_manifest.kernel.external_buffer_abi.tail_outputs"));
  const auto add_role_values =
      ov::gfx_plugin::detail::gfx_mpsrt_read_u32_vector_attr(
          module, "gfx.stage_manifest.kernel.external_buffer_abi.roles");
  ASSERT_EQ(add_role_values.size(), 8u);
  EXPECT_EQ(
      add_role_values[3],
      static_cast<uint32_t>(ov::gfx_plugin::GfxKernelBufferRole::ScalarParam));
  EXPECT_EQ(
      add_role_values[4],
      static_cast<uint32_t>(ov::gfx_plugin::GfxKernelBufferRole::ScalarParam));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage_desc.kind"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.dispatch_kernel_family"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.dispatch_entry_point"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.dispatch_kernel_family_id"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.dispatch_flags"));
  ASSERT_FALSE(
      module->hasAttr("gfx.mpsrt.dispatch_precompiled_kernel_required"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.dispatch_threads_per_threadgroup"));
  ASSERT_FALSE(
      module->hasAttr("gfx.stage_manifest.kernel.precompiled_binary_required"));
  ASSERT_FALSE(
      module->hasAttr("gfx.stage_manifest.kernel.threads_per_threadgroup"));
  ASSERT_EQ(module
                ->getAttrOfType<mlir::StringAttr>(
                    "gfx.stage_manifest.kernel.dispatch_policy.grid")
                .str(),
            "linear_1d");
  ASSERT_EQ(module
                ->getAttrOfType<mlir::IntegerAttr>(
                    "gfx.stage_manifest.kernel.dispatch_policy.threads_per_"
                    "threadgroup")
                .getInt(),
            256);
  ASSERT_TRUE(
      module
          ->getAttrOfType<mlir::BoolAttr>("gfx.stage_manifest.kernel.dispatch_"
                                          "policy.precompiled_binary_required")
          .getValue());

  ov::gfx_plugin::GfxMpsrtModuleStagePlan extracted_stage;
  ASSERT_TRUE(
      ov::gfx_plugin::read_module_mpsrt_stage_plan(module, extracted_stage));
  ASSERT_EQ(extracted_stage.stage.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
  const auto extracted_dispatch =
      dispatch_spec_from_stage(extracted_stage.stage);
  ASSERT_TRUE(extracted_dispatch.valid);
  ASSERT_EQ(extracted_dispatch.kernel_family, "eltwise_fused_buffer");
  ASSERT_EQ(extracted_dispatch.entry_point, "eltwise_fused_buffer");
  ASSERT_EQ(extracted_dispatch.kernel_family_id,
            static_cast<uint32_t>(
                ov::gfx_plugin::GfxKernelFamily::EltwiseFusedBuffer));
  ASSERT_EQ(extracted_dispatch.flags,
            ov::gfx_plugin::GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
  ASSERT_TRUE(extracted_dispatch.precompiled_binary_required);
  ASSERT_EQ(extracted_dispatch.threads_per_threadgroup, 256u);

  const auto custom_kernel_plan =
      ov::gfx_plugin::make_gfx_custom_kernel_stage_plan("Add",
                                                        "eltwise_kernel");
  ASSERT_TRUE(custom_kernel_plan.valid);
  ASSERT_EQ(custom_kernel_plan.family,
            ov::gfx_plugin::GfxKernelFamily::EltwiseFusedBuffer);
  ASSERT_EQ(custom_kernel_plan.stage_manifest.custom_kernel.entry_point,
            "eltwise_fused_buffer");

  ov::gfx_plugin::KernelSource source;
  source.module = module;
  source.entry_point = "eltwise_kernel";
  source.msl_source =
      "#include <metal_stdlib>\n"
      "using namespace metal;\n"
      "kernel void eltwise_kernel(device const half* A [[buffer(0)]]) {}\n";
  auto source_plan =
      ov::gfx_plugin::make_msl_generated_custom_kernel_source_plan(
          std::move(source), "Add");
  ASSERT_TRUE(source_plan.valid());
  source = std::move(source_plan.source);
  ASSERT_EQ(source.entry_point, "eltwise_fused_buffer");
  ASSERT_NE(source.msl_source.find("kernel void eltwise_fused_buffer"),
            std::string::npos);
  ASSERT_EQ(source.msl_source.find("kernel void eltwise_kernel"),
            std::string::npos);

  const auto module_builder_plan =
      ov::gfx_plugin::build_module_mpsrt_builder_plan(module);
  ASSERT_TRUE(module_builder_plan.valid);
  ASSERT_TRUE(module_builder_plan.stage_plan.stage.stage_manifest.valid);
  ASSERT_EQ(module_builder_plan.stage_plan.stage.stage_manifest.execution_kind,
            ov::gfx_plugin::GfxKernelExecutionKind::CustomKernel);
  ASSERT_TRUE(
      module_builder_plan.stage_plan.stage.stage_manifest.custom_kernel.valid);
  ASSERT_EQ(module_builder_plan.stage_plan.stage.stage_manifest.custom_kernel
                .kernel_family,
            "eltwise_fused_buffer");
  ASSERT_TRUE(module_builder_plan.external_buffer_abi.valid);
  ASSERT_TRUE(module_builder_plan.external_buffer_abi.has_buffer_count);
  ASSERT_TRUE(module_builder_plan.external_buffer_abi.has_output_buffer_count);
  ASSERT_EQ(module_builder_plan.external_buffer_abi.buffer_count, 6u);
  ASSERT_EQ(module_builder_plan.external_buffer_abi.output_buffer_count, 1u);
  ASSERT_TRUE(module_builder_plan.external_buffer_abi.has_buffer_roles);
  ASSERT_EQ(module_builder_plan.external_buffer_abi.buffer_roles,
            std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams}));
  ASSERT_TRUE(module_builder_plan.builder_plan.external_buffer_abi_valid);
  ASSERT_EQ(module_builder_plan.builder_plan.external_buffer_count, 6u);
  ASSERT_EQ(module_builder_plan.builder_plan.external_output_buffer_count, 1u);
  ASSERT_EQ(module_builder_plan.builder_plan.external_buffer_roles,
            module_builder_plan.external_buffer_abi.buffer_roles);
  ASSERT_EQ(module_builder_plan.builder_plan.records[3].kind,
            ov::gfx_plugin::GfxMpsrtBuilderRecordKind::EncodeStage);
  ASSERT_EQ(module_builder_plan.builder_plan.records[3].symbol,
            "ovgfx_mpsrt_encode_dispatch");
  ASSERT_EQ(module_builder_plan.builder_plan.records[3].stage_desc.kernel_name,
            "eltwise_fused_buffer");
  const auto dispatch =
      ov::gfx_plugin::gfx_mpsrt_custom_dispatch_spec_from_kernel_manifest(
          module_builder_plan.builder_plan.records[3]
              .stage_desc.stage_manifest.custom_kernel);
  ASSERT_TRUE(dispatch.valid);
  ASSERT_EQ(dispatch.kernel_family, "eltwise_fused_buffer");
  ASSERT_EQ(dispatch.entry_point, "eltwise_fused_buffer");
  ASSERT_EQ(dispatch.kernel_family_id,
            static_cast<uint32_t>(
                ov::gfx_plugin::GfxKernelFamily::EltwiseFusedBuffer));
  ASSERT_EQ(dispatch.flags,
            ov::gfx_plugin::GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
  ASSERT_EQ(dispatch.threads_per_threadgroup, 256u);
  ASSERT_TRUE(dispatch.precompiled_binary_required);
  const auto dispatch_desc = ov::gfx_plugin::gfx_mpsrt_make_msl_dispatch_desc(
      module_builder_plan.builder_plan.records[3].stage_desc,
      static_cast<uint32_t>(
          module_builder_plan.builder_plan.records[3].inputs.size()),
      static_cast<uint32_t>(
          module_builder_plan.builder_plan.records[3].outputs.size()));
  ASSERT_EQ(dispatch_desc.kernel_family,
            static_cast<uint32_t>(
                ov::gfx_plugin::GfxKernelFamily::EltwiseFusedBuffer));
  ASSERT_EQ(dispatch_desc.storage,
            static_cast<uint32_t>(ov::gfx_plugin::GfxMpsrtStorage::Buffer));
  ASSERT_EQ(dispatch_desc.layout,
            static_cast<uint32_t>(ov::gfx_plugin::GfxMpsrtLayout::Linear));
  ASSERT_EQ(dispatch_desc.threads_per_threadgroup, 256u);
  ASSERT_EQ(dispatch_desc.input_count, 2u);
  ASSERT_EQ(dispatch_desc.output_count, 1u);
  ASSERT_EQ(dispatch_desc.flags,
            ov::gfx_plugin::GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
}

TEST(GfxMlir, AppleMslLoweringMaterializesStageManifestBeforeTypedProgram) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 8});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 8});
  auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(add, ctx);
  ASSERT_TRUE(module);

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "Add", add, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  auto lowering_plan = ov::gfx_plugin::materialize_apple_msl_stage_manifest(
      module, plan, "Add", "eltwise_kernel");
  ASSERT_TRUE(lowering_plan.valid);
  ASSERT_EQ(lowering_plan.stage_plan.stage.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
  const auto apple_manifest_passes =
      ov::gfx_plugin::gfx_apple_stage_pipeline_pass_boundaries(
          /*materialize_typed_program=*/false);
  ASSERT_EQ(apple_manifest_passes.size(), 6u);
  ASSERT_EQ(ov::gfx_plugin::gfx_apple_stage_pipeline_pass_name(
                apple_manifest_passes[0]),
            std::string("gfx-core-canonicalize"));
  ASSERT_EQ(ov::gfx_plugin::gfx_apple_stage_pipeline_pass_name(
                apple_manifest_passes[4]),
            std::string("gfx-apple-vendor-descriptor"));
  ASSERT_EQ(ov::gfx_plugin::gfx_apple_stage_pipeline_pass_name(
                apple_manifest_passes[5]),
            std::string("gfx-apple-stage-manifest"));
  ASSERT_FALSE(module->hasAttr("gfx.apple.pipeline.pass_boundary_count"));
  ASSERT_FALSE(module->hasAttr("gfx.apple.pipeline.pass6.name"));
  ASSERT_FALSE(module->hasAttr("gfx.apple.pipeline.vendor_descriptor.kind"));
  ASSERT_EQ(
      module
          ->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.kernel.family")
          .str(),
      "eltwise_fused_buffer");
  ASSERT_FALSE(static_cast<bool>(
      module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops")));

  ov::gfx_plugin::GfxMpsrtModuleStagePlan extracted_stage;
  ASSERT_FALSE(
      ov::gfx_plugin::read_module_mpsrt_stage_plan(module, extracted_stage));
  const auto lowering_dispatch =
      dispatch_spec_from_stage(lowering_plan.stage_plan.stage);
  ASSERT_TRUE(lowering_dispatch.valid);
  ASSERT_EQ(lowering_dispatch.entry_point, "eltwise_fused_buffer");

  ov::gfx_plugin::GfxMpsrtExternalBufferAbiPlan external_buffer_abi{};
  ASSERT_TRUE(
      ov::gfx_plugin::gfx_mpsrt_external_buffer_abi_from_kernel_manifest(
          module, external_buffer_abi));
  ASSERT_TRUE(ov::gfx_plugin::materialize_apple_msl_typed_program(
      module, lowering_plan, external_buffer_abi));
  ASSERT_TRUE(static_cast<bool>(
      module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops")));
  ASSERT_FALSE(module->hasAttr("gfx.msl.kernel_family"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.dispatch_kernel_family"));
}

TEST(GfxMlir, AppleStagePipelineRunsNamedPassBoundariesBeforeTypedProgram) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 8});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 8});
  auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(add, ctx);
  ASSERT_TRUE(module);

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "Add", add, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  const auto result = ov::gfx_plugin::run_gfx_apple_stage_pipeline(
      module, plan, "Add", "eltwise_kernel");
  ASSERT_TRUE(result.valid);
  ASSERT_TRUE(result.typed_program_materialized);
  ASSERT_TRUE(static_cast<bool>(
      module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops")));
  const auto passes = ov::gfx_plugin::gfx_apple_stage_pipeline_pass_boundaries(
      /*materialize_typed_program=*/true);
  ASSERT_EQ(passes.size(), 7u);
  ASSERT_EQ(ov::gfx_plugin::gfx_apple_stage_pipeline_pass_name(passes[0]),
            std::string("gfx-core-canonicalize"));
  ASSERT_EQ(ov::gfx_plugin::gfx_apple_stage_pipeline_pass_name(passes[5]),
            std::string("gfx-apple-stage-manifest"));
  ASSERT_EQ(ov::gfx_plugin::gfx_apple_stage_pipeline_pass_name(passes[6]),
            std::string("gfx-apple-runtime-abi"));
  ASSERT_FALSE(module->hasAttr("gfx.apple.pipeline.pass_boundary_count"));
  ASSERT_FALSE(module->hasAttr("gfx.apple.pipeline.pass5.name"));
  ASSERT_FALSE(module->hasAttr("gfx.apple.pipeline.pass6.name"));
  ASSERT_FALSE(module->hasAttr("gfx.apple.pipeline.vendor_descriptor.kind"));
  ASSERT_FALSE(module->hasAttr("gfx.apple.pipeline.placement.backend_domain"));
  ASSERT_FALSE(module->hasAttr("gfx.apple.pipeline.placement.execution_kind"));
  ASSERT_FALSE(module->hasAttr("gfx.apple.pipeline.storage.contract"));
  ov::gfx_plugin::GfxMpsrtProgram typed_program;
  ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_program(module, typed_program));
  ASSERT_TRUE(typed_program.has_storage_bridges);
  ASSERT_TRUE(typed_program.storage_bridges.empty());
  ASSERT_FALSE(module->hasAttr("gfx.apple.pipeline.fusion.activation"));
  ASSERT_EQ(
      module
          ->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.backend_domain")
          .str(),
      "apple_msl");
  ASSERT_EQ(result.stage_plan.stage.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
  ASSERT_FALSE(module->hasAttr(
      "gfx.apple.pipeline.runtime_abi.typed_program_materialized"));
  ASSERT_FALSE(
      module->hasAttr("gfx.apple.pipeline.runtime_abi.call_plan_materialized"));
  ASSERT_FALSE(static_cast<bool>(
      module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_runtime_abi_plan")));
}

TEST(GfxMlir, OpenClStageManifestDoesNotMaterializeAppleMpsrtOps) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 2});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 1, 1});
  auto multiply = std::make_shared<ov::op::v1::Multiply>(lhs, rhs);

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(multiply, ctx);
  ASSERT_TRUE(module);

  const auto plan =
      make_opencl_contract_stage_plan("Multiply", multiply, ov::element::f32);
  ASSERT_EQ(plan.placement.domain,
            ov::gfx_plugin::GfxStageBackendDomain::OpenCl);

  const auto result = ov::gfx_plugin::run_gfx_apple_stage_pipeline(
      module, plan, "Multiply", "eltwise_main",
      /*materialize_typed_program=*/false);
  ASSERT_TRUE(result.valid);
  ASSERT_FALSE(result.typed_program_materialized);
  ASSERT_FALSE(module->hasAttr("gfx.apple.pipeline.placement.backend_domain"));
  ASSERT_EQ(result.stage_plan.stage.domain,
            ov::gfx_plugin::GfxStageBackendDomain::OpenCl);
  ASSERT_EQ(result.stage_plan.stage.stage_manifest.backend_domain,
            ov::gfx_plugin::GfxKernelBackendDomain::OpenCl);
  ASSERT_FALSE(static_cast<bool>(
      module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops")));
}

TEST(GfxMlir, AppleStagePipelineOwnsImageStorageBridgeAssignment) {
  auto input = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::Shape{1, 16, 32, 32});
  auto weights =
      ov::op::v0::Constant::create(ov::element::f16, ov::Shape{8, 16, 3, 3},
                                   std::vector<float>(8 * 16 * 3 * 3, 1.f));
  auto conv = std::make_shared<ov::op::v1::Convolution>(
      input, weights, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(conv, ctx);
  ASSERT_TRUE(module);

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "Convolution", conv,
      ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  ov::gfx_plugin::GfxAppleStagePipelineOptions options{};
  options.plan = plan;
  options.stage_type = "Convolution";
  options.semantic_input_roles = {
      ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
      ov::gfx_plugin::GfxKernelBufferRole::ConstTensor};
  const auto result =
      ov::gfx_plugin::run_gfx_apple_stage_pipeline(module, options);
  ASSERT_TRUE(result.valid);
  ASSERT_TRUE(result.typed_program_materialized);
  ov::gfx_plugin::GfxMpsrtProgram program;
  ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_ops_program(module, program));
  ASSERT_TRUE(program.has_storage_bridges);
  ASSERT_EQ(program.storage_bridges.size(), 2u);
  ASSERT_EQ(program.storage_bridges[0].direction,
            ov::gfx_plugin::GfxMpsrtStorageBridgeDirection::BufferToImage);
  ASSERT_EQ(program.storage_bridges[1].direction,
            ov::gfx_plugin::GfxMpsrtStorageBridgeDirection::ImageToBuffer);
  ASSERT_EQ(program.inputs[1].flags, ov::gfx_plugin::GfxMpsrtTensorFlagConst);
  ASSERT_EQ(program.inputs[1].storage, ov::gfx_plugin::GfxMpsrtStorage::Buffer);
  ASSERT_EQ(program.stages.front().stage.stage_manifest.semantic_input_roles,
            options.semantic_input_roles);
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.storage_bridge_count"));
}

TEST(GfxMlir, AppleMpsPrimitiveMaterializersOwnDescriptorEnrichment) {
  auto pool_input = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::Shape{1, 4, 16, 16});
  auto max_pool = std::make_shared<ov::op::v1::MaxPool>(
      pool_input, ov::Strides{2, 2}, ov::Shape{0, 0}, ov::Shape{0, 0},
      ov::Shape{2, 2}, ov::op::RoundingType::FLOOR);
  auto softmax_input = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::Shape{2, 8, 16});
  auto softmax = std::make_shared<ov::op::v1::Softmax>(softmax_input, 2);
  auto topk_input = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::Shape{2, 8, 16});
  auto k = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{},
                                        std::vector<int32_t>{4});
  auto topk = std::make_shared<ov::op::v11::TopK>(
      topk_input, k, 2, ov::op::TopKMode::MAX,
      ov::op::TopKSortType::SORT_VALUES, ov::element::i32);

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto pool_module = ov::gfx_plugin::build_mlir_for_node(max_pool, ctx);
  auto softmax_module = ov::gfx_plugin::build_mlir_for_node(softmax, ctx);
  auto topk_module = ov::gfx_plugin::build_mlir_for_node(topk, ctx);
  ASSERT_TRUE(pool_module);
  ASSERT_TRUE(softmax_module);
  ASSERT_TRUE(topk_module);

  const auto pool_plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "MaxPool", max_pool,
      ov::element::f16, false, false, false, {});
  ov::gfx_plugin::GfxMpsrtPool2DAbiDesc pool_desc{};
  pool_desc.kernel[0] = 2;
  pool_desc.kernel[1] = 2;
  pool_desc.strides[0] = 2;
  pool_desc.strides[1] = 2;
  pool_desc.dilations[0] = 1;
  pool_desc.dilations[1] = 1;
  ov::gfx_plugin::GfxAppleMpsVendorPrimitiveContract pool_contract{};
  ASSERT_TRUE(ov::gfx_plugin::gfx_apple_make_mps_pool2d_contract(
      max_pool, pool_desc, pool_contract));
  const auto pool_materialized =
      ov::gfx_plugin::materialize_apple_mps_vendor_contract_program(
          pool_module, pool_plan, "MaxPool", pool_contract);
  ASSERT_TRUE(pool_materialized.valid);
  ASSERT_TRUE(pool_materialized.typed_program_materialized);

  const auto softmax_plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "Softmax", softmax,
      ov::element::f16, false, false, false, {});
  ov::gfx_plugin::GfxMpsrtSoftmaxAbiDesc softmax_desc{};
  softmax_desc.axis = 2;
  ov::gfx_plugin::GfxAppleMpsVendorPrimitiveContract softmax_contract{};
  ASSERT_TRUE(ov::gfx_plugin::gfx_apple_make_mps_softmax_contract(
      softmax, softmax_desc, softmax_contract));
  const auto softmax_materialized =
      ov::gfx_plugin::materialize_apple_mps_vendor_contract_program(
          softmax_module, softmax_plan, "Softmax", softmax_contract);
  ASSERT_TRUE(softmax_materialized.valid);
  ASSERT_TRUE(softmax_materialized.typed_program_materialized);

  const auto topk_plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "TopK", topk,
      ov::element::f16, false, false, false, {});
  ov::gfx_plugin::GfxMpsrtTopKAbiDesc topk_desc{};
  topk_desc.axis = 2;
  topk_desc.k = 4;
  topk_desc.mode_max = 1;
  topk_desc.sort_type =
      static_cast<uint32_t>(ov::gfx_plugin::TopKSortType::SortValues);
  ov::gfx_plugin::GfxAppleMpsVendorPrimitiveContract topk_contract{};
  ASSERT_TRUE(ov::gfx_plugin::gfx_apple_make_mps_topk_contract(topk, topk_desc,
                                                               topk_contract));
  const auto topk_materialized =
      ov::gfx_plugin::materialize_apple_mps_vendor_contract_program(
          topk_module, topk_plan, "TopK", topk_contract);
  ASSERT_TRUE(topk_materialized.valid);
  ASSERT_TRUE(topk_materialized.typed_program_materialized);

  ov::gfx_plugin::GfxMpsrtProgram pool_program;
  ov::gfx_plugin::GfxMpsrtProgram softmax_program;
  ov::gfx_plugin::GfxMpsrtProgram topk_program;
  ASSERT_TRUE(
      ov::gfx_plugin::read_module_mpsrt_ops_program(pool_module, pool_program));
  ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_ops_program(softmax_module,
                                                            softmax_program));
  ASSERT_TRUE(
      ov::gfx_plugin::read_module_mpsrt_ops_program(topk_module, topk_program));
  ASSERT_FALSE(
      pool_module->hasAttr("gfx.apple.pipeline.vendor_descriptor.kind"));
  ASSERT_FALSE(
      softmax_module->hasAttr("gfx.apple.pipeline.vendor_descriptor.kind"));
  ASSERT_FALSE(
      topk_module->hasAttr("gfx.apple.pipeline.vendor_descriptor.kind"));

  ASSERT_EQ(pool_program.stages.front().stage.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MPSPool2D);
  ASSERT_EQ(pool_program.external_buffer_abi.buffer_roles,
            std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput}));
  ASSERT_EQ(pool_program.stages.front().stage.pool2d_desc.kernel[0], 2u);
  ASSERT_EQ(pool_program.stages.front().stage.pool2d_desc.strides[1], 2u);
  ASSERT_EQ(
      pool_program.stages.front().stage.stage_manifest.semantic_input_roles,
      std::vector<ov::gfx_plugin::GfxKernelBufferRole>(
          {ov::gfx_plugin::GfxKernelBufferRole::TensorInput}));

  ASSERT_EQ(softmax_program.stages.front().stage.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MPSSoftmax);
  ASSERT_EQ(softmax_program.external_buffer_abi.buffer_roles,
            std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput}));
  ASSERT_EQ(softmax_program.stages.front().stage.softmax_desc.axis, 2u);
  ASSERT_EQ(
      softmax_program.stages.front().stage.stage_manifest.semantic_output_roles,
      std::vector<ov::gfx_plugin::GfxKernelBufferRole>(
          {ov::gfx_plugin::GfxKernelBufferRole::TensorOutput}));

  ASSERT_EQ(topk_program.stages.front().stage.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MPSTopK);
  ASSERT_EQ(topk_program.external_buffer_abi.buffer_roles,
            std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput}));
  ASSERT_EQ(topk_program.stages.front().stage.topk_desc.k, 4u);
  ASSERT_EQ(topk_program.stages.front().stage.topk_desc.sort_type,
            static_cast<uint32_t>(ov::gfx_plugin::TopKSortType::SortValues));
  ASSERT_EQ(
      topk_program.stages.front().stage.stage_manifest.semantic_output_roles,
      std::vector<ov::gfx_plugin::GfxKernelBufferRole>(
          {ov::gfx_plugin::GfxKernelBufferRole::TensorOutput,
           ov::gfx_plugin::GfxKernelBufferRole::TensorOutput}));
  ASSERT_FALSE(pool_module->hasAttr("gfx.mpsrt.pool2d.kernel.0"));
  ASSERT_FALSE(softmax_module->hasAttr("gfx.mpsrt.softmax.axis"));
  ASSERT_FALSE(topk_module->hasAttr("gfx.mpsrt.topk.k"));
}

TEST(GfxMlir, AppleMpsPool2DContractRejectsIndexedMaxPoolOutputs) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                       ov::Shape{1, 4, 16, 16});
  auto indexed_pool = std::make_shared<ov::op::v8::MaxPool>(
      input, ov::Strides{2, 2}, ov::Strides{1, 1}, ov::Shape{0, 0},
      ov::Shape{0, 0}, ov::Shape{2, 2}, ov::op::RoundingType::FLOOR,
      ov::op::PadType::EXPLICIT, ov::element::i64, 0);
  ASSERT_EQ(indexed_pool->get_output_size(), 2u);

  ov::gfx_plugin::GfxMpsrtPool2DAbiDesc desc{};
  ASSERT_TRUE(
      ov::gfx_plugin::gfx_apple_make_mps_pool2d_desc(indexed_pool, desc));

  ov::gfx_plugin::GfxAppleMpsVendorPrimitiveContract contract{};
  EXPECT_FALSE(ov::gfx_plugin::gfx_apple_make_mps_pool2d_contract(
      indexed_pool, desc, contract));
  EXPECT_FALSE(contract.valid);
  EXPECT_TRUE(contract.input_descs.empty());
  EXPECT_TRUE(contract.output_descs.empty());
}

TEST(GfxMlir,
     StageManifestSuppliesMslDispatchMetadataWhenLegacyStageAttrsAreMissing) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 64, 80, 80});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 64, 80, 80});
  auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(add, ctx);
  ASSERT_TRUE(module);
  mlir::Builder builder(module.getContext());

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "Add", add, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Add");

  module->removeAttr("gfx.backend");
  module->removeAttr("gfx.specialization_key");
  module->removeAttr("gfx.uses_custom_kernel");
  module->removeAttr("gfx.uses_vendor_primitive");
  module->removeAttr("gfx.mpsrt.stage_desc.kind");
  module->removeAttr("gfx.mpsrt.kernel_name");
  module->removeAttr("gfx.mpsrt.builder_symbol");
  module->removeAttr("gfx.mpsrt.stage_record_key");
  module->removeAttr("gfx.mpsrt.dispatch_kernel_family");
  module->removeAttr("gfx.mpsrt.dispatch_entry_point");
  module->removeAttr("gfx.mpsrt.dispatch_kernel_family_id");
  module->removeAttr("gfx.mpsrt.dispatch_flags");
  module->removeAttr("gfx.mpsrt.dispatch_threads_per_threadgroup");
  module->removeAttr("gfx.mpsrt.dispatch_precompiled_kernel_required");

  ov::gfx_plugin::GfxMpsrtModuleStagePlan extracted;
  ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_stage_plan(module, extracted));
  ASSERT_TRUE(extracted.valid);
  ASSERT_EQ(extracted.stage.domain,
            ov::gfx_plugin::GfxStageBackendDomain::AppleMsl);
  ASSERT_TRUE(
      ov::gfx_plugin::gfx_mpsrt_stage_uses_custom_kernel(extracted.stage));
  ASSERT_FALSE(
      ov::gfx_plugin::gfx_mpsrt_stage_uses_vendor_primitive(extracted.stage));
  ASSERT_EQ(ov::gfx_plugin::gfx_mpsrt_stage_specialization_key(extracted.stage),
            "apple_msl:buffer:Add");
  const auto extracted_dispatch = dispatch_spec_from_stage(extracted.stage);
  ASSERT_TRUE(extracted_dispatch.valid);
  ASSERT_EQ(extracted_dispatch.kernel_family, "eltwise_fused_buffer");
  ASSERT_EQ(extracted_dispatch.entry_point, "eltwise_fused_buffer");
  ASSERT_EQ(extracted_dispatch.kernel_family_id,
            static_cast<uint32_t>(
                ov::gfx_plugin::GfxKernelFamily::EltwiseFusedBuffer));
  ASSERT_EQ(extracted_dispatch.flags,
            ov::gfx_plugin::GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
  ASSERT_EQ(extracted_dispatch.threads_per_threadgroup, 256u);
  ASSERT_TRUE(extracted_dispatch.precompiled_binary_required);
  ASSERT_FALSE(
      ov::gfx_plugin::gfx_mpsrt_stage_plan_record_key(extracted).empty());

  const auto module_builder_plan =
      ov::gfx_plugin::build_module_mpsrt_builder_plan(module);
  ASSERT_TRUE(module_builder_plan.valid);
  const auto builder_dispatch =
      ov::gfx_plugin::gfx_mpsrt_custom_dispatch_spec_from_kernel_manifest(
          module_builder_plan.builder_plan.records[3]
              .stage_desc.stage_manifest.custom_kernel);
  ASSERT_TRUE(builder_dispatch.valid);
  ASSERT_EQ(builder_dispatch.kernel_family, "eltwise_fused_buffer");
  ASSERT_EQ(builder_dispatch.entry_point, "eltwise_fused_buffer");
  ASSERT_EQ(builder_dispatch.kernel_family_id,
            static_cast<uint32_t>(
                ov::gfx_plugin::GfxKernelFamily::EltwiseFusedBuffer));
  ASSERT_EQ(builder_dispatch.flags,
            ov::gfx_plugin::GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
  ASSERT_EQ(builder_dispatch.threads_per_threadgroup, 256u);
  ASSERT_TRUE(builder_dispatch.precompiled_binary_required);
  const auto builder_dispatch_desc =
      ov::gfx_plugin::gfx_mpsrt_make_msl_dispatch_desc(
          module_builder_plan.builder_plan.records[3].stage_desc,
          static_cast<uint32_t>(
              module_builder_plan.builder_plan.records[3].inputs.size()),
          static_cast<uint32_t>(
              module_builder_plan.builder_plan.records[3].outputs.size()));
  ASSERT_EQ(builder_dispatch_desc.kernel_family,
            static_cast<uint32_t>(
                ov::gfx_plugin::GfxKernelFamily::EltwiseFusedBuffer));
  ASSERT_EQ(builder_dispatch_desc.threads_per_threadgroup, 256u);
  ASSERT_EQ(builder_dispatch_desc.flags,
            ov::gfx_plugin::GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
}

TEST(GfxMlir, StageManifestAloneDoesNotMaterializeMpsrtStageOrProgramReader) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 64, 80, 80});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 64, 80, 80});
  auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(add, ctx);
  ASSERT_TRUE(module);

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "Add", add, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Add");
  ASSERT_TRUE(static_cast<bool>(
      module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops")));

  ov::gfx_plugin::erase_module_mpsrt_ops(module);
  ASSERT_FALSE(static_cast<bool>(
      module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops")));

  ov::gfx_plugin::GfxMpsrtModuleStagePlan stage_plan;
  ASSERT_FALSE(
      ov::gfx_plugin::read_module_mpsrt_stage_plan(module, stage_plan));

  ov::gfx_plugin::GfxMpsrtProgram program;
  ASSERT_FALSE(ov::gfx_plugin::read_module_mpsrt_program(module, program));
  const auto module_builder_plan =
      ov::gfx_plugin::build_module_mpsrt_builder_plan(module);
  ASSERT_FALSE(module_builder_plan.valid);
}

TEST(GfxMlir, LegacyMpsrtStageAttrsWithoutManifestAreRejected) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 64, 80, 80});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 64, 80, 80});
  auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(add, ctx);
  ASSERT_TRUE(module);

  mlir::Builder builder(module.getContext());
  module->setAttr("gfx.backend", builder.getStringAttr("apple_msl"));
  module->setAttr("gfx.stage_type", builder.getStringAttr("Add"));
  module->setAttr("gfx.specialization_key",
                  builder.getStringAttr("apple_msl:buffer:Add"));
  module->setAttr("gfx.uses_custom_kernel", builder.getBoolAttr(true));
  module->setAttr("gfx.mpsrt.stage_desc.kind",
                  builder.getStringAttr("msl_dispatch"));
  module->setAttr("gfx.mpsrt.stage_record_key",
                  builder.getStringAttr("legacy_record"));
  module->setAttr("gfx.mpsrt.kernel_name",
                  builder.getStringAttr("eltwise_fused_buffer"));
  module->setAttr("gfx.mpsrt.builder_symbol",
                  builder.getStringAttr("ovgfx_mpsrt_encode_dispatch"));
  module->setAttr("gfx.mpsrt.dispatch_kernel_family",
                  builder.getStringAttr("eltwise_fused_buffer"));
  module->setAttr("gfx.mpsrt.dispatch_entry_point",
                  builder.getStringAttr("eltwise_fused_buffer"));
  ASSERT_FALSE(module->hasAttr("gfx.stage_manifest.stage_family"));

  ov::gfx_plugin::GfxMpsrtModuleStagePlan stage_plan;
  ASSERT_FALSE(
      ov::gfx_plugin::read_module_mpsrt_stage_plan(module, stage_plan));

  ov::gfx_plugin::GfxMpsrtProgram program;
  ASSERT_FALSE(ov::gfx_plugin::read_module_mpsrt_program(module, program));

  const auto module_builder_plan =
      ov::gfx_plugin::build_module_mpsrt_builder_plan(module);
  ASSERT_FALSE(module_builder_plan.valid);
}

TEST(GfxMlir, StageManifestOverridesConflictingGeneratedStageAttrs) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 64, 80, 80});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 64, 80, 80});
  auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(add, ctx);
  ASSERT_TRUE(module);
  mlir::Builder builder(module.getContext());

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "Add", add, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Add");

  auto generated_ops = module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops");
  ASSERT_TRUE(static_cast<bool>(generated_ops));
  mlir::Operation *dispatch_op = nullptr;
  generated_ops.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "gfx.mpsrt.dispatch") {
      dispatch_op = op;
    }
  });
  ASSERT_NE(dispatch_op, nullptr);

  ASSERT_FALSE(dispatch_op->hasAttr("gfx.mpsrt.op.stage.stage_type"));
  ASSERT_FALSE(dispatch_op->hasAttr("gfx.mpsrt.op.stage.backend"));
  ASSERT_FALSE(dispatch_op->hasAttr("gfx.mpsrt.op.stage.kind"));
  ASSERT_FALSE(dispatch_op->hasAttr("gfx.mpsrt.op.stage.stage_record_key"));
  ASSERT_FALSE(dispatch_op->hasAttr("gfx.mpsrt.op.stage.kernel_name"));
  ASSERT_FALSE(dispatch_op->hasAttr("gfx.mpsrt.op.stage.builder_symbol"));
  ASSERT_FALSE(dispatch_op->hasAttr("gfx.mpsrt.op.stage.specialization_key"));
  ASSERT_FALSE(
      dispatch_op->hasAttr("gfx.mpsrt.op.stage.dispatch_kernel_family"));
  ASSERT_FALSE(dispatch_op->hasAttr("gfx.mpsrt.op.stage.dispatch_entry_point"));
  ASSERT_FALSE(
      dispatch_op->hasAttr("gfx.mpsrt.op.stage.dispatch_kernel_family_id"));
  ASSERT_FALSE(dispatch_op->hasAttr(
      "gfx.mpsrt.op.stage.dispatch_threads_per_threadgroup"));
  ASSERT_FALSE(dispatch_op->hasAttr(
      "gfx.mpsrt.op.stage.dispatch_precompiled_kernel_required"));
  ASSERT_FALSE(
      dispatch_op->hasAttr("gfx.mpsrt.op.stage.uses_vendor_primitive"));
  ASSERT_FALSE(dispatch_op->hasAttr("gfx.mpsrt.op.stage.uses_custom_kernel"));

  dispatch_op->setAttr("gfx.mpsrt.op.stage.backend",
                       builder.getStringAttr("apple_mps"));
  dispatch_op->setAttr("gfx.mpsrt.op.stage.kind",
                       builder.getStringAttr("mps_gemm"));
  dispatch_op->setAttr("gfx.mpsrt.op.stage.stage_record_key",
                       builder.getStringAttr("legacy_wrong_key"));
  dispatch_op->setAttr("gfx.mpsrt.op.stage.kernel_name",
                       builder.getStringAttr("legacy_wrong_kernel"));
  dispatch_op->setAttr("gfx.mpsrt.op.stage.builder_symbol",
                       builder.getStringAttr("ovgfx_mpsrt_encode_gemm"));

  ov::gfx_plugin::GfxMpsrtProgram program;
  ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_program(module, program));
  ASSERT_TRUE(program.valid);
  ASSERT_FALSE(program.multi_stage);
  ASSERT_EQ(program.stages.size(), 1u);
  const auto &stage = program.stages.front();
  ASSERT_EQ(stage.stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
  ASSERT_EQ(stage.stage.domain,
            ov::gfx_plugin::GfxStageBackendDomain::AppleMsl);
  ASSERT_TRUE(ov::gfx_plugin::gfx_mpsrt_stage_uses_custom_kernel(stage.stage));
  ASSERT_FALSE(
      ov::gfx_plugin::gfx_mpsrt_stage_uses_vendor_primitive(stage.stage));
  ASSERT_EQ(stage.stage.kernel_name, "eltwise_fused_buffer");
  ASSERT_STREQ(ov::gfx_plugin::gfx_mpsrt_stage_builder_symbol(stage.stage),
               "ovgfx_mpsrt_encode_dispatch");
  ASSERT_EQ(
      ov::gfx_plugin::gfx_mpsrt_stage_record_key(stage.stage),
      "msl_dispatch|apple_msl|buffer|buffer|linear|Add|apple_msl:buffer:Add|"
      "dispatch:eltwise_fused_buffer:eltwise_fused_buffer:linear_1d:tg256:"
      "metallib");
}

TEST(GfxMlir, StageManifestSuppliesRoleAbiWithoutModuleMpsrtAttrs) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 64, 80, 80});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 64, 80, 80});
  auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(add, ctx);
  ASSERT_TRUE(module);

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "Add", add, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Add");
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_count"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_output_buffer_count"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_roles"));

  const auto abi =
      ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
  ASSERT_TRUE(abi.valid);
  ASSERT_TRUE(abi.has_buffer_count);
  ASSERT_TRUE(abi.has_output_buffer_count);
  ASSERT_TRUE(abi.has_buffer_roles);
  ASSERT_EQ(abi.buffer_count, 6u);
  ASSERT_EQ(abi.output_buffer_count, 1u);
  ASSERT_EQ(abi.buffer_roles,
            std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams}));

  const auto module_builder_plan =
      ov::gfx_plugin::build_module_mpsrt_builder_plan(module);
  ASSERT_TRUE(module_builder_plan.valid);
  ASSERT_TRUE(module_builder_plan.builder_plan.external_buffer_abi_valid);
  ASSERT_EQ(module_builder_plan.builder_plan.external_buffer_count, 6u);
  ASSERT_EQ(module_builder_plan.builder_plan.external_output_buffer_count, 1u);
  ASSERT_EQ(module_builder_plan.builder_plan.external_buffer_roles,
            abi.buffer_roles);
}

TEST(GfxMlir, TypedProgramExternalBufferAbiWinsOverStaleModuleManifest) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 64, 80, 80});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 64, 80, 80});
  auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(add, ctx);
  ASSERT_TRUE(module);

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "Add", add, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Add");
  ASSERT_TRUE(static_cast<bool>(
      module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops")));

  const auto stale_module_binding =
      ov::gfx_plugin::make_backend_custom_kernel_roles_binding_plan(
          "Softmax", "softmax_kernel",
          {ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
           ov::gfx_plugin::GfxKernelBufferRole::TensorOutput,
           ov::gfx_plugin::GfxKernelBufferRole::RuntimeParams});
  ASSERT_TRUE(stale_module_binding.valid);
  ASSERT_TRUE(
      ov::gfx_plugin::annotate_backend_custom_kernel_module_with_binding_plan(
          module, stale_module_binding));

  ov::gfx_plugin::GfxMpsrtModuleStagePlan stale_stage_plan;
  EXPECT_FALSE(ov::gfx_plugin::build_mpsrt_stage_plan_from_manifest(
      module, stale_stage_plan));
  ov::gfx_plugin::GfxMpsrtExternalBufferAbiPlan stale_abi;
  EXPECT_FALSE(
      ov::gfx_plugin::gfx_mpsrt_external_buffer_abi_from_kernel_manifest(
          module, stale_abi));

  const auto abi =
      ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
  ASSERT_TRUE(abi.valid);
  ASSERT_TRUE(abi.has_buffer_roles);
  ASSERT_EQ(abi.buffer_count, 6u);
  ASSERT_EQ(abi.output_buffer_count, 1u);
  ASSERT_EQ(abi.buffer_roles,
            std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams}));
}

TEST(GfxMlir,
     InvalidTypedProgramExternalBufferAbiDoesNotFallBackToStaleModuleManifest) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 64, 80, 80});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 64, 80, 80});
  auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(add, ctx);
  ASSERT_TRUE(module);

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "Add", add, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Add");
  ASSERT_TRUE(ov::gfx_plugin::module_has_mpsrt_ops_program(module));

  const auto stale_module_binding =
      ov::gfx_plugin::make_backend_custom_kernel_roles_binding_plan(
          "Softmax", "softmax_kernel",
          {ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
           ov::gfx_plugin::GfxKernelBufferRole::TensorOutput,
           ov::gfx_plugin::GfxKernelBufferRole::RuntimeParams});
  ASSERT_TRUE(stale_module_binding.valid);
  ASSERT_TRUE(
      ov::gfx_plugin::annotate_backend_custom_kernel_module_with_binding_plan(
          module, stale_module_binding));

  auto generated_ops = module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops");
  ASSERT_TRUE(static_cast<bool>(generated_ops));
  generated_ops->removeAttr("gfx.mpsrt.ops.stage_count");

  ov::gfx_plugin::GfxMpsrtProgram program;
  ASSERT_FALSE(ov::gfx_plugin::read_module_mpsrt_ops_program(module, program));

  const auto abi =
      ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
  ASSERT_FALSE(abi.valid);
  ASSERT_FALSE(abi.has_buffer_count);
  ASSERT_FALSE(abi.has_output_buffer_count);
  ASSERT_FALSE(abi.has_buffer_roles);
}

TEST(GfxMlir,
     StageManifestSuppliesKernelRuntimeMetadataAheadOfLegacyOperandAttrs) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "Add", nullptr,
      ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Add");

  mlir::Builder builder(module.getContext());
  module->setAttr("gfx.kernel_operand_kinds",
                  builder.getArrayAttr({builder.getI32IntegerAttr(1)}));
  module->setAttr("gfx.kernel_operand_arg_indices",
                  builder.getArrayAttr({builder.getI32IntegerAttr(42)}));

  const auto metadata = ov::gfx_plugin::extract_kernel_runtime_metadata(
      module,
      /*output_arg_count=*/1,
      /*fallback_input_arg_count=*/999, "eltwise_fused_buffer");
  ASSERT_TRUE(metadata.valid);
  ASSERT_EQ(metadata.kernel_input_arg_count, 5u);
  ASSERT_EQ(metadata.operands.operand_kinds,
            std::vector<int32_t>({1, 1, 1, 0, 0, 1, 1, 1}));
  ASSERT_EQ(metadata.operands.operand_arg_indices,
            std::vector<int32_t>({0, 1, 5, -1, -1, 2, 3, 4}));

  module->setAttr("gfx.fixed_arg_count", builder.getI32IntegerAttr(3));
  ASSERT_EQ(ov::gfx_plugin::infer_kernel_arg_count_from_module(
                module,
                /*fallback=*/999, "eltwise_fused_buffer"),
            8u);
}

TEST(GfxMlir, OpenClMetadataReaderIgnoresAppleMslStageManifest) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "Add", nullptr,
      ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Add");

  size_t manifest_arg_count = 0;
  EXPECT_TRUE(ov::gfx_plugin::infer_kernel_arg_count_from_stage_manifest(
      module, manifest_arg_count, "eltwise_fused_buffer"));
  EXPECT_FALSE(ov::gfx_plugin::infer_kernel_arg_count_from_stage_manifest(
      module, manifest_arg_count, "eltwise_fused_buffer",
      ov::gfx_plugin::GfxKernelBackendDomain::OpenCl));
  EXPECT_EQ(ov::gfx_plugin::infer_kernel_arg_count_from_module(
                module,
                /*fallback=*/3, "eltwise_fused_buffer",
                ov::gfx_plugin::GfxKernelBackendDomain::OpenCl),
            3u);

  const auto metadata = ov::gfx_plugin::extract_kernel_runtime_metadata(
      module,
      /*output_arg_count=*/1,
      /*fallback_input_arg_count=*/2, "eltwise_fused_buffer",
      ov::gfx_plugin::GfxKernelBackendDomain::OpenCl);
  ASSERT_TRUE(metadata.valid);
  EXPECT_EQ(metadata.kernel_input_arg_count, 2u);
  EXPECT_TRUE(metadata.operands.operand_kinds.empty());
  EXPECT_TRUE(metadata.operands.operand_arg_indices.empty());
}

TEST(GfxMlir, RequiredOpenClCustomKernelBindingAnnotatesStageManifest) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  const auto plan =
      ov::gfx_plugin::annotate_required_backend_custom_kernel_binding(
          module,
          /*is_opencl_backend=*/true, "SquaredDifference", "eltwise_kernel",
          std::vector<int32_t>{7, 11}, "sqdiff_test");

  ASSERT_TRUE(plan.valid);
  ASSERT_TRUE(module->hasAttr("gfx.stage_manifest.backend_domain"));
  EXPECT_EQ(
      module
          ->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.backend_domain")
          .str(),
      "opencl");
  ASSERT_TRUE(
      module->hasAttr("gfx.stage_manifest.kernel.external_buffer_abi.roles"));
  ASSERT_FALSE(module->hasAttr("gfx.kernel_operand_kinds"));
  ASSERT_FALSE(module->hasAttr("gfx.kernel_operand_arg_indices"));

  const auto metadata = ov::gfx_plugin::extract_kernel_runtime_metadata(
      module,
      /*output_arg_count=*/1,
      /*fallback_input_arg_count=*/999, "eltwise_fused_buffer",
      ov::gfx_plugin::GfxKernelBackendDomain::OpenCl);
  ASSERT_TRUE(metadata.valid);
  EXPECT_EQ(metadata.kernel_input_arg_count,
            plan.runtime_binding.input_arg_count);
  EXPECT_EQ(metadata.operands.operand_kinds,
            plan.runtime_binding.operand_kinds);
  EXPECT_EQ(metadata.operands.operand_arg_indices,
            plan.runtime_binding.operand_arg_indices);
  EXPECT_EQ(metadata.operands.scalar_args, std::vector<int32_t>({7, 11}));
}

TEST(GfxMlir, OpenClLinearBinaryCustomKernelUsesRuntimeParamManifest) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  const auto plan =
      ov::gfx_plugin::make_backend_custom_kernel_roles_binding_plan(
          "Add", "linear_binary",
          {ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
           ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
           ov::gfx_plugin::GfxKernelBufferRole::RuntimeParams,
           ov::gfx_plugin::GfxKernelBufferRole::TensorOutput},
          ov::gfx_plugin::GfxKernelBackendDomain::OpenCl,
          ov::gfx_plugin::GfxKernelStorageKind::Buffer, "opencl:buffer:");
  ASSERT_TRUE(plan.valid);
  ASSERT_TRUE(
      ov::gfx_plugin::annotate_backend_custom_kernel_module_with_binding_plan(
          module, plan));

  EXPECT_EQ(plan.runtime_binding.input_arg_count, 3u);
  EXPECT_EQ(plan.runtime_binding.operand_kinds,
            std::vector<int32_t>({1, 1, 1, 1}));
  EXPECT_EQ(plan.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 1, 2, 3}));

  const auto metadata = ov::gfx_plugin::extract_kernel_runtime_metadata(
      module,
      /*output_arg_count=*/1,
      /*fallback_input_arg_count=*/999, "linear_binary",
      ov::gfx_plugin::GfxKernelBackendDomain::OpenCl);
  ASSERT_TRUE(metadata.valid);
  EXPECT_EQ(metadata.kernel_input_arg_count, 3u);
  EXPECT_EQ(metadata.operands.operand_kinds,
            std::vector<int32_t>({1, 1, 1, 1}));
  EXPECT_EQ(metadata.operands.operand_arg_indices,
            std::vector<int32_t>({0, 1, 2, 3}));
  EXPECT_EQ(ov::gfx_plugin::infer_kernel_arg_count_from_module(
                module,
                /*fallback=*/999, "linear_binary",
                ov::gfx_plugin::GfxKernelBackendDomain::OpenCl),
            4u);
}

TEST(GfxMlir, OpenClCustomKernelArgCountIncludesScalarRoles) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  auto plan = ov::gfx_plugin::make_backend_custom_kernel_binding_plan(
      "Add", "eltwise_kernel", ov::gfx_plugin::GfxKernelBackendDomain::OpenCl,
      ov::gfx_plugin::GfxKernelStorageKind::Buffer, "opencl:buffer:");
  ASSERT_TRUE(plan.valid);
  ASSERT_EQ(plan.scalar_arg_count, 2u);
  ASSERT_TRUE(
      ov::gfx_plugin::annotate_backend_custom_kernel_module_with_binding_plan(
          module, plan));

  size_t manifest_arg_count = 0;
  ASSERT_TRUE(ov::gfx_plugin::infer_kernel_arg_count_from_stage_manifest(
      module, manifest_arg_count, plan.stage_manifest.custom_kernel.entry_point,
      ov::gfx_plugin::GfxKernelBackendDomain::OpenCl));
  EXPECT_EQ(manifest_arg_count, plan.runtime_binding.operand_kinds.size());

  ov::gfx_plugin::KernelSource source;
  source.module = module;
  source.entry_point = "gfx_kernel";
  ASSERT_TRUE(
      ov::gfx_plugin::
          configure_backend_custom_kernel_source_signature_from_module(source));
  EXPECT_EQ(source.signature.arg_count,
            plan.runtime_binding.operand_kinds.size());
}

TEST(GfxMlir, AppleMslRuntimeMetadataRejectsLegacyOperandAttrsWithoutManifest) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  mlir::Builder builder(module.getContext());
  module->setAttr("gfx.kernel_operand_kinds",
                  builder.getArrayAttr({builder.getI32IntegerAttr(1),
                                        builder.getI32IntegerAttr(1),
                                        builder.getI32IntegerAttr(1)}));
  module->setAttr("gfx.kernel_operand_arg_indices",
                  builder.getArrayAttr({builder.getI32IntegerAttr(0),
                                        builder.getI32IntegerAttr(1),
                                        builder.getI32IntegerAttr(2)}));
  module->setAttr("gfx.fixed_arg_count", builder.getI32IntegerAttr(3));
  module->setAttr("gfx.kernel_scalar_args",
                  builder.getArrayAttr({builder.getI32IntegerAttr(9)}));

  const auto msl_metadata = ov::gfx_plugin::extract_kernel_runtime_metadata(
      module,
      /*output_arg_count=*/1,
      /*fallback_input_arg_count=*/999, "eltwise_fused_buffer");
  ASSERT_FALSE(msl_metadata.valid);
  ASSERT_TRUE(msl_metadata.operands.operand_kinds.empty());
  ASSERT_TRUE(msl_metadata.operands.operand_arg_indices.empty());
  ASSERT_EQ(ov::gfx_plugin::infer_kernel_arg_count_from_module(
                module,
                /*fallback=*/999, "eltwise_fused_buffer"),
            999u);
}

TEST(
    GfxMlir,
    RuntimeMetadataRejectsLegacyAttrsButKeepsSignatureFallbackWhenNoManifestExists) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  const auto metadata = ov::gfx_plugin::extract_kernel_runtime_metadata(
      module,
      /*output_arg_count=*/1,
      /*fallback_input_arg_count=*/2, "gfx_kernel");
  ASSERT_TRUE(metadata.valid);
  EXPECT_EQ(metadata.kernel_input_arg_count, 2u);
  EXPECT_TRUE(metadata.operands.operand_kinds.empty());
  EXPECT_TRUE(metadata.operands.operand_arg_indices.empty());
  EXPECT_TRUE(metadata.operands.scalar_args.empty());
  EXPECT_EQ(ov::gfx_plugin::infer_kernel_arg_count_from_module(module,
                                                               /*fallback=*/3,
                                                               "gfx_kernel"),
            3u);
}

TEST(GfxMlir, KernelPlanRejectsLegacyArgCountAttrs) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  mlir::Builder builder(module.getContext());
  module->setAttr("gfx.fixed_arg_count", builder.getI32IntegerAttr(3));
  module->setAttr("gfx.kernel_scalar_args",
                  builder.getArrayAttr({builder.getI32IntegerAttr(9)}));

  ov::gfx_plugin::KernelPlan plan(module, "eltwise_fused_buffer",
                                  /*arg_count=*/0);
  ASSERT_EQ(plan.to_source().signature.arg_count, 0u);
}

TEST(GfxMlir, FixedArgRuntimeMetadataIsRejectedWithoutManifest) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  mlir::Builder builder(module.getContext());
  module->setAttr("gfx.fixed_arg_count", builder.getI32IntegerAttr(3));
  module->setAttr("gfx.kernel_scalar_values",
                  builder.getArrayAttr({builder.getI32IntegerAttr(7),
                                        builder.getI32IntegerAttr(11)}));
  module->setAttr("gfx.kernel_scalar_args",
                  builder.getArrayAttr({builder.getI32IntegerAttr(7),
                                        builder.getI32IntegerAttr(11)}));

  const auto metadata = ov::gfx_plugin::extract_kernel_runtime_metadata(
      module,
      /*output_arg_count=*/1,
      /*fallback_input_arg_count=*/999, "gfx_kernel");

  ASSERT_FALSE(metadata.valid);
  EXPECT_TRUE(metadata.operands.operand_kinds.empty());
  EXPECT_TRUE(metadata.operands.operand_arg_indices.empty());
  EXPECT_TRUE(metadata.operands.scalar_args.empty());
  EXPECT_EQ(ov::gfx_plugin::infer_kernel_arg_count_from_module(module,
                                                               /*fallback=*/3,
                                                               "gfx_kernel"),
            3u);
}

TEST(GfxMlir,
     OpenClStageManifestSuppliesKernelRuntimeMetadataWithoutLegacyAttrs) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  const auto plan =
      ov::gfx_plugin::make_backend_custom_kernel_roles_binding_plan(
          "Select", "select_kernel",
          {ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
           ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
           ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
           ov::gfx_plugin::GfxKernelBufferRole::RuntimeParams,
           ov::gfx_plugin::GfxKernelBufferRole::TensorOutput},
          ov::gfx_plugin::GfxKernelBackendDomain::OpenCl,
          ov::gfx_plugin::GfxKernelStorageKind::Buffer, "opencl:buffer:");
  ASSERT_TRUE(plan.valid);
  ASSERT_TRUE(
      ov::gfx_plugin::annotate_backend_custom_kernel_module_with_binding_plan(
          module, plan));
  ASSERT_FALSE(module->hasAttr("gfx.fixed_arg_count"));

  const auto metadata = ov::gfx_plugin::extract_kernel_runtime_metadata(
      module,
      /*output_arg_count=*/1,
      /*fallback_input_arg_count=*/999, "select_kernel");
  ASSERT_TRUE(metadata.valid);
  EXPECT_EQ(metadata.kernel_input_arg_count, 4u);
  EXPECT_EQ(metadata.operands.operand_kinds,
            std::vector<int32_t>({1, 1, 1, 1, 1}));
  EXPECT_EQ(metadata.operands.operand_arg_indices,
            std::vector<int32_t>({0, 1, 2, 3, 4}));
  EXPECT_EQ(ov::gfx_plugin::infer_kernel_arg_count_from_module(module,
                                                               /*fallback=*/999,
                                                               "select_kernel"),
            5u);
}

TEST(GfxMlir, AppleMslArgCountCanUseStageManifestWithoutMlirEntryMatch) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  const auto binding = ov::gfx_plugin::make_backend_custom_kernel_binding_plan(
      "Slice", "slice_kernel");
  ASSERT_TRUE(binding.valid);
  ASSERT_TRUE(
      ov::gfx_plugin::annotate_backend_custom_kernel_module_with_binding_plan(
          module, binding));

  ASSERT_EQ(ov::gfx_plugin::infer_kernel_arg_count_from_module(module,
                                                               /*fallback=*/2,
                                                               "slice_main"),
            2u);
  ASSERT_EQ(
      ov::gfx_plugin::infer_kernel_arg_count_from_module(module,
                                                         /*fallback=*/2,
                                                         /*entry_point=*/{}),
      8u);

  const auto metadata = ov::gfx_plugin::extract_kernel_runtime_metadata(
      module,
      /*output_arg_count=*/1,
      /*fallback_input_arg_count=*/1,
      /*entry_point=*/{});
  ASSERT_TRUE(metadata.valid);
  ASSERT_EQ(metadata.kernel_input_arg_count, 7u);
  ASSERT_EQ(metadata.operands.operand_arg_indices,
            std::vector<int32_t>({0, 7, 1, 2, 3, 4, 5, 6}));
}

TEST(GfxMlir, MpsrtOpsExternalAbiSuppliesKernelRuntimeMetadata) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  auto lhs = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 8, 16}, ov::element::f16, ov::gfx_plugin::GfxStageStorageKind::Matrix,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  auto rhs = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 16, 4}, ov::element::f16, ov::gfx_plugin::GfxStageStorageKind::Matrix,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  auto output = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 8, 4}, ov::element::f16, ov::gfx_plugin::GfxStageStorageKind::Matrix,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "MatMul", nullptr,
      ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  auto stage = ov::gfx_plugin::gfx_mpsrt_make_stage_desc(plan, "MatMul");
  ASSERT_EQ(stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);

  ov::gfx_plugin::GfxMpsrtProgram program{};
  program.valid = true;
  program.record_key = "mpsrt_external_abi_metadata";
  program.inputs = {lhs, rhs};
  program.output_values = {2u};
  program.stages.push_back({stage, {0u, 1u}, {2u}, {output}});
  program.external_buffer_abi.valid = true;
  program.external_buffer_abi.has_buffer_count = true;
  program.external_buffer_abi.has_output_buffer_count = true;
  program.external_buffer_abi.has_buffer_roles = true;
  program.external_buffer_abi.buffer_count = 4u;
  program.external_buffer_abi.output_buffer_count = 1u;
  program.external_buffer_abi.buffer_roles = {
      ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
      ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
      ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams,
      ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput};

  ASSERT_TRUE(ov::gfx_plugin::materialize_module_mpsrt_ops(module, program));
  const auto metadata = ov::gfx_plugin::extract_kernel_runtime_metadata(
      module,
      /*output_arg_count=*/1,
      /*fallback_input_arg_count=*/999);
  ASSERT_TRUE(metadata.valid);
  ASSERT_EQ(metadata.kernel_input_arg_count, 3u);
  EXPECT_EQ(metadata.operands.operand_kinds,
            std::vector<int32_t>({1, 1, 1, 1}));
  EXPECT_EQ(metadata.operands.operand_arg_indices,
            std::vector<int32_t>({0, 1, 2, 3}));
}

TEST(GfxMlir, TypedVendorProgramDerivesExternalIoAbiFromTypedStagePlan) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  auto lhs = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 8, 16}, ov::element::f16, ov::gfx_plugin::GfxStageStorageKind::Matrix,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  auto rhs = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 16, 4}, ov::element::f16, ov::gfx_plugin::GfxStageStorageKind::Matrix,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  auto output = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 8, 4}, ov::element::f16, ov::gfx_plugin::GfxStageStorageKind::Matrix,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "MatMul", nullptr,
      ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  ov::gfx_plugin::GfxMpsrtModuleStagePlan stage_plan{};
  stage_plan.stage = ov::gfx_plugin::gfx_mpsrt_make_stage_desc(plan, "MatMul");
  ASSERT_EQ(stage_plan.stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
  stage_plan.inputs = {lhs, rhs};
  stage_plan.outputs = {output};
  ASSERT_TRUE(ov::gfx_plugin::finalize_mpsrt_module_stage_plan(stage_plan));

  ASSERT_TRUE(ov::gfx_plugin::materialize_module_mpsrt_ops_from_stage_plan(
      module, stage_plan));

  const auto abi =
      ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
  ASSERT_TRUE(abi.valid);
  ASSERT_TRUE(abi.has_buffer_roles);
  EXPECT_EQ(abi.buffer_count, 3u);
  EXPECT_EQ(abi.output_buffer_count, 1u);
  EXPECT_EQ(abi.buffer_roles,
            std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput}));

  const auto module_builder_plan =
      ov::gfx_plugin::build_module_mpsrt_builder_plan(module);
  ASSERT_TRUE(module_builder_plan.valid);
  ASSERT_TRUE(module_builder_plan.builder_plan.external_buffer_abi_valid);
  EXPECT_EQ(module_builder_plan.builder_plan.external_buffer_count, 3u);
  EXPECT_EQ(module_builder_plan.builder_plan.external_output_buffer_count, 1u);
  EXPECT_EQ(module_builder_plan.builder_plan.external_buffer_roles,
            abi.buffer_roles);
}

TEST(GfxMlir, TileBuilderUsesFlatMemrefCustomKernelAbiWithoutTensorReshapes) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
                                                       ov::Shape{1, 300, 1});
  auto repeats =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {1, 1, 80});
  auto tile = std::make_shared<ov::op::v0::Tile>(input, repeats);
  auto result = std::make_shared<ov::op::v0::Result>(tile);
  auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                           ov::ParameterVector{input});

  mlir::MLIRContext ctx;
  auto module = ov::gfx_plugin::build_mlir_tile_from_model(model, ctx);
  ASSERT_TRUE(module);
  auto func = module.lookupSymbol<mlir::func::FuncOp>("tile_main");
  ASSERT_TRUE(func);
  EXPECT_EQ(func.getFunctionType().getNumInputs(), 8u);
  EXPECT_EQ(func.getFunctionType().getNumResults(), 0u);
  EXPECT_TRUE(
      module->getAttrOfType<mlir::BoolAttr>("gfx.prefer_parallel").getValue());
  EXPECT_TRUE(module->getAttrOfType<mlir::BoolAttr>("gfx.i64_storage_i32_lanes")
                  .getValue());

  std::string text;
  llvm::raw_string_ostream os(text);
  module.print(os);
  EXPECT_EQ(text.find("tensor.empty"), std::string::npos);
  EXPECT_EQ(text.find("tensor.collapse_shape"), std::string::npos);
  EXPECT_EQ(text.find("tensor.expand_shape"), std::string::npos);
  EXPECT_EQ(text.find("memref.alloc"), std::string::npos);
  EXPECT_EQ(text.find("memref<?xi64>"), std::string::npos);
  EXPECT_NE(text.find("memref<?xi32>"), std::string::npos);
}

TEST(GfxMlir, TileBuilderAcceptsStaticRankRuntimeRepeatsForMslKernelAbi) {
  auto input = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 3});
  auto repeats = std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
                                                         ov::PartialShape{3});
  auto tile = std::make_shared<ov::op::v0::Tile>(input, repeats);
  auto result = std::make_shared<ov::op::v0::Result>(tile);
  auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                           ov::ParameterVector{input, repeats});

  mlir::MLIRContext ctx;
  auto module = ov::gfx_plugin::build_mlir_tile_from_model(model, ctx);
  ASSERT_TRUE(module);
  auto func = module.lookupSymbol<mlir::func::FuncOp>("tile_main");
  ASSERT_TRUE(func);
  EXPECT_EQ(func.getFunctionType().getNumInputs(), 8u);
  EXPECT_EQ(func.getFunctionType().getNumResults(), 0u);
  EXPECT_TRUE(
      module->getAttrOfType<mlir::BoolAttr>("gfx.prefer_parallel").getValue());

  const auto binding =
      ov::gfx_plugin::annotate_required_backend_custom_kernel_binding(
          module, /*is_opencl_backend=*/false, "Tile", "tile_kernel", {0, 3},
          "dynamic_tile_stage");
  ASSERT_TRUE(binding.valid);
  EXPECT_EQ(binding.runtime_binding.operand_kinds,
            std::vector<int32_t>({1, 1, 0, 0, 1, 1, 1, 1}));
  EXPECT_EQ(binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 5, -1, -1, 1, 2, 3, 4}));
}

TEST(GfxMlir, AppleShapeRangeTileAndConcatSourcePlansOwnManifestAbi) {
  mlir::MLIRContext ctx;

  auto concat_lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                            ov::Shape{1, 2, 4});
  auto concat_rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                            ov::Shape{1, 3, 4});
  auto concat = std::make_shared<ov::op::v0::Concat>(
      ov::OutputVector{concat_lhs, concat_rhs}, 1);
  auto concat_result = std::make_shared<ov::op::v0::Result>(concat);
  auto concat_model =
      std::make_shared<ov::Model>(ov::ResultVector{concat_result},
                                  ov::ParameterVector{concat_lhs, concat_rhs});
  auto concat_module =
      ov::gfx_plugin::build_mlir_concat_from_model(concat_model, ctx);
  ASSERT_TRUE(concat_module);

  auto concat_source_plan =
      ov::gfx_plugin::make_concat_msl_kernel_source_plan(concat, concat_module);
  ASSERT_TRUE(concat_source_plan.valid());
  EXPECT_FALSE(static_cast<bool>(concat_source_plan.source.module));
  EXPECT_EQ(concat_source_plan.source.entry_point, "concat_kernel");
  EXPECT_EQ(concat_source_plan.source.signature.arg_count, 3u);
  EXPECT_EQ(concat_source_plan.source.signature.output_arg_count, 1u);
  EXPECT_EQ(concat_source_plan.binding.runtime_binding.inputs,
            std::vector<size_t>({0, 1}));
  EXPECT_EQ(concat_source_plan.binding.runtime_binding.input_arg_count, 2u);
  EXPECT_EQ(concat_source_plan.binding.runtime_binding.operand_kinds,
            std::vector<int32_t>({1, 1, 1}));
  EXPECT_EQ(concat_source_plan.binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 1, 2}));
  EXPECT_NE(
      concat_source_plan.source.msl_source.find("kernel void concat_kernel"),
      std::string::npos);

  auto shape_input = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::Shape{1, 2, 64});
  auto shape_of =
      std::make_shared<ov::op::v3::ShapeOf>(shape_input, ov::element::i64);
  auto shape_result = std::make_shared<ov::op::v0::Result>(shape_of);
  auto shape_model = std::make_shared<ov::Model>(
      ov::ResultVector{shape_result}, ov::ParameterVector{shape_input});
  auto shape_module =
      ov::gfx_plugin::build_mlir_shapeof_from_model(shape_model, ctx);
  ASSERT_TRUE(shape_module);

  auto shape_source_plan = ov::gfx_plugin::make_shapeof_msl_kernel_source_plan(
      shape_of, shape_module);
  ASSERT_TRUE(shape_source_plan.valid());
  EXPECT_FALSE(static_cast<bool>(shape_source_plan.source.module));
  EXPECT_EQ(shape_source_plan.source.entry_point, "shapeof_kernel");
  EXPECT_EQ(shape_source_plan.source.signature.arg_count, 4u);
  EXPECT_EQ(shape_source_plan.source.signature.output_arg_count, 1u);
  EXPECT_EQ(shape_source_plan.binding.runtime_binding.scalar_args,
            std::vector<int32_t>({3}));
  EXPECT_EQ(shape_source_plan.binding.runtime_binding.operand_kinds,
            std::vector<int32_t>({1, 1, 0, 1}));
  EXPECT_EQ(shape_source_plan.binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 2, -1, 1}));
  EXPECT_NE(
      shape_source_plan.source.msl_source.find("kernel void shapeof_kernel"),
      std::string::npos);

  auto tile_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                            ov::Shape{2, 3});
  auto repeats =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 2});
  auto tile = std::make_shared<ov::op::v0::Tile>(tile_input, repeats);
  auto tile_result = std::make_shared<ov::op::v0::Result>(tile);
  auto tile_model = std::make_shared<ov::Model>(
      ov::ResultVector{tile_result}, ov::ParameterVector{tile_input});
  auto tile_module =
      ov::gfx_plugin::build_mlir_tile_from_model(tile_model, ctx);
  ASSERT_TRUE(tile_module);

  auto tile_source_plan =
      ov::gfx_plugin::make_tile_msl_kernel_source_plan(tile, tile_module);
  ASSERT_TRUE(tile_source_plan.valid());
  EXPECT_FALSE(static_cast<bool>(tile_source_plan.source.module));
  EXPECT_EQ(tile_source_plan.source.entry_point, "tile_kernel");
  EXPECT_EQ(tile_source_plan.source.signature.arg_count, 8u);
  EXPECT_EQ(tile_source_plan.source.signature.output_arg_count, 1u);
  EXPECT_EQ(tile_source_plan.binding.runtime_binding.scalar_args,
            std::vector<int32_t>({12, 2}));
  EXPECT_EQ(tile_source_plan.binding.runtime_binding.operand_kinds,
            std::vector<int32_t>({1, 1, 0, 0, 1, 1, 1, 1}));
  EXPECT_EQ(tile_source_plan.binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 5, -1, -1, 1, 2, 3, 4}));
  EXPECT_NE(tile_source_plan.source.msl_source.find("constant uint& NUM_ELEMS"),
            std::string::npos);

  auto start =
      ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
  auto stop =
      ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {10.0f});
  auto step =
      ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {2.0f});
  auto range =
      std::make_shared<ov::op::v4::Range>(start, stop, step, ov::element::f32);
  auto range_result = std::make_shared<ov::op::v0::Result>(range);
  auto range_model = std::make_shared<ov::Model>(ov::ResultVector{range_result},
                                                 ov::ParameterVector{});
  auto range_module =
      ov::gfx_plugin::build_mlir_range_from_model(range_model, ctx);
  ASSERT_TRUE(range_module);

  auto range_source_plan =
      ov::gfx_plugin::make_range_msl_kernel_source_plan(range, range_module);
  ASSERT_TRUE(range_source_plan.valid());
  EXPECT_FALSE(static_cast<bool>(range_source_plan.source.module));
  EXPECT_EQ(range_source_plan.source.entry_point, "range_kernel");
  EXPECT_EQ(range_source_plan.source.signature.arg_count, 5u);
  EXPECT_EQ(range_source_plan.source.signature.output_arg_count, 1u);
  EXPECT_EQ(range_source_plan.binding.runtime_binding.scalar_args,
            std::vector<int32_t>({5}));
  EXPECT_EQ(range_source_plan.binding.runtime_binding.operand_kinds,
            std::vector<int32_t>({1, 1, 1, 1, 0}));
  EXPECT_EQ(range_source_plan.binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 1, 2, 3, -1}));
  EXPECT_NE(
      range_source_plan.source.msl_source.find("kernel void range_kernel"),
      std::string::npos);
}

TEST(GfxMlir, TypedMslStageManifestSuppliesRuntimeMetadataOverExternalAbi) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  const auto binding = ov::gfx_plugin::make_backend_custom_kernel_binding_plan(
      "Tile", "tile_kernel", {16, 4});
  ASSERT_TRUE(binding.valid);
  ASSERT_EQ(binding.runtime_binding.operand_kinds,
            std::vector<int32_t>({1, 1, 0, 0, 1, 1, 1, 1}));
  ASSERT_EQ(binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 5, -1, -1, 1, 2, 3, 4}));

  auto input = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 4}, ov::element::f32, ov::gfx_plugin::GfxStageStorageKind::Buffer,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  auto output = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 4}, ov::element::f32, ov::gfx_plugin::GfxStageStorageKind::Buffer,
      ov::gfx_plugin::GfxMpsrtTensorFlagTransient);

  ov::gfx_plugin::GfxMpsrtStageDesc stage{};
  stage.kind = ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch;
  stage.domain = ov::gfx_plugin::GfxStageBackendDomain::AppleMsl;
  stage.input_storage = ov::gfx_plugin::GfxMpsrtStorage::Buffer;
  stage.output_storage = ov::gfx_plugin::GfxMpsrtStorage::Buffer;
  stage.layout = ov::gfx_plugin::GfxMpsrtLayout::Linear;
  stage.kernel_name = binding.stage_manifest.custom_kernel.entry_point;
  stage.stage_manifest = binding.stage_manifest;

  const auto external_buffer_abi =
      ov::gfx_plugin::gfx_mpsrt_make_external_buffer_abi_from_roles(
          {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
           ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput,
           ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams,
           ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams,
           ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams,
           ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams});

  ov::gfx_plugin::GfxMpsrtProgram program{};
  program.valid = true;
  program.record_key = "typed_msl_stage_manifest_runtime_metadata";
  program.inputs = {input};
  program.output_values = {1u};
  program.stages.push_back({stage, {0u}, {1u}, {output}});
  program.external_buffer_abi = external_buffer_abi;
  ASSERT_TRUE(ov::gfx_plugin::materialize_module_mpsrt_ops(module, program));

  const auto metadata = ov::gfx_plugin::extract_kernel_runtime_metadata(
      module,
      /*output_arg_count=*/1,
      /*fallback_input_arg_count=*/999,
      binding.stage_manifest.custom_kernel.entry_point);
  ASSERT_TRUE(metadata.valid);
  EXPECT_EQ(metadata.kernel_input_arg_count,
            binding.runtime_binding.input_arg_count);
  EXPECT_EQ(metadata.operands.operand_kinds,
            binding.runtime_binding.operand_kinds);
  EXPECT_EQ(metadata.operands.operand_arg_indices,
            binding.runtime_binding.operand_arg_indices);
  EXPECT_EQ(metadata.operands.scalar_args, std::vector<int32_t>({16, 4}));

  ASSERT_EQ(ov::gfx_plugin::infer_kernel_arg_count_from_module(
                module, /*fallback=*/999,
                binding.stage_manifest.custom_kernel.entry_point),
            binding.runtime_binding.operand_kinds.size());
}

TEST(GfxMlir,
     TypedMslRuntimeMetadataRejectsFallbackForMismatchedStageManifest) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  const auto typed_binding =
      ov::gfx_plugin::make_backend_custom_kernel_binding_plan(
          "Tile", "tile_kernel", {16, 4});
  ASSERT_TRUE(typed_binding.valid);
  const auto stale_binding =
      ov::gfx_plugin::make_backend_custom_kernel_binding_plan("Softmax",
                                                              "softmax_kernel");
  ASSERT_TRUE(stale_binding.valid);

  auto input = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 4}, ov::element::f32, ov::gfx_plugin::GfxStageStorageKind::Buffer,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  auto output = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 4}, ov::element::f32, ov::gfx_plugin::GfxStageStorageKind::Buffer,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);

  ov::gfx_plugin::GfxMpsrtStageDesc stage{};
  stage.kind = ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch;
  stage.domain = ov::gfx_plugin::GfxStageBackendDomain::AppleMsl;
  stage.input_storage = ov::gfx_plugin::GfxMpsrtStorage::Buffer;
  stage.output_storage = ov::gfx_plugin::GfxMpsrtStorage::Buffer;
  stage.layout = ov::gfx_plugin::GfxMpsrtLayout::Linear;
  stage.kernel_name = typed_binding.stage_manifest.custom_kernel.entry_point;
  stage.stage_manifest = typed_binding.stage_manifest;

  ov::gfx_plugin::GfxMpsrtProgram program{};
  program.valid = true;
  program.record_key = "typed_msl_missing_manifest_runtime_metadata";
  program.inputs = {input};
  program.output_values = {1u};
  program.stages.push_back({stage, {0u}, {1u}, {output}});
  program.external_buffer_abi =
      ov::gfx_plugin::gfx_mpsrt_make_external_buffer_abi_from_roles(
          {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
           ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput,
           ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams});
  ASSERT_TRUE(ov::gfx_plugin::materialize_module_mpsrt_ops(module, program));

  ASSERT_TRUE(
      ov::gfx_plugin::annotate_backend_custom_kernel_module_with_binding_plan(
          module, stale_binding));
  const auto abi =
      ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
  ASSERT_TRUE(abi.valid);
  ASSERT_TRUE(abi.has_buffer_roles);

  const auto metadata = ov::gfx_plugin::extract_kernel_runtime_metadata(
      module,
      /*output_arg_count=*/1,
      /*fallback_input_arg_count=*/999,
      stale_binding.stage_manifest.custom_kernel.entry_point);
  EXPECT_FALSE(metadata.valid);
  EXPECT_EQ(ov::gfx_plugin::infer_kernel_arg_count_from_module(
                module, /*fallback=*/999,
                stale_binding.stage_manifest.custom_kernel.entry_point),
            999u);
}

TEST(GfxMlir, TypedMpsrtProgramMaterializationErasesLegacyExternalAbiAttrs) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
  mlir::Builder builder(module.getContext());
  module->setAttr("gfx.mpsrt.external_buffer_count",
                  builder.getI32IntegerAttr(99));
  module->setAttr("gfx.mpsrt.external_output_buffer_count",
                  builder.getI32IntegerAttr(7));
  module->setAttr("gfx.mpsrt.external_buffer_roles",
                  builder.getArrayAttr({builder.getI32IntegerAttr(0)}));
  module->setAttr("gfx.mpsrt.runtime_abi.external_buffer_count",
                  builder.getI32IntegerAttr(99));
  module->setAttr("gfx.msl.kernel_family",
                  builder.getStringAttr("legacy_family"));
  module->setAttr("gfx.apple.pipeline.program.kind",
                  builder.getStringAttr("legacy_pipeline"));

  auto lhs = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 4}, ov::element::f32, ov::gfx_plugin::GfxStageStorageKind::Buffer,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  auto rhs = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 4}, ov::element::f32, ov::gfx_plugin::GfxStageStorageKind::Buffer,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  auto output = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 4}, ov::element::f32, ov::gfx_plugin::GfxStageStorageKind::Buffer,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "MatMul", nullptr,
      ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  auto stage = ov::gfx_plugin::gfx_mpsrt_make_stage_desc(plan, "MatMul");
  ASSERT_EQ(stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);

  ov::gfx_plugin::GfxMpsrtProgram program{};
  program.valid = true;
  program.record_key = "typed_program_external_abi_wins";
  program.inputs = {lhs, rhs};
  program.output_values = {2u};
  program.stages.push_back({stage, {0u, 1u}, {2u}, {output}});
  program.external_buffer_abi =
      ov::gfx_plugin::gfx_mpsrt_make_external_buffer_abi_from_roles(
          {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
           ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
           ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams,
           ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput});

  ASSERT_TRUE(ov::gfx_plugin::materialize_module_mpsrt_ops(module, program));
  EXPECT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_count"));
  EXPECT_FALSE(module->hasAttr("gfx.mpsrt.external_output_buffer_count"));
  EXPECT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_roles"));
  EXPECT_FALSE(module->hasAttr("gfx.mpsrt.runtime_abi.external_buffer_count"));
  EXPECT_FALSE(module->hasAttr("gfx.msl.kernel_family"));
  EXPECT_FALSE(module->hasAttr("gfx.apple.pipeline.program.kind"));

  auto ops_func = module.lookupSymbol<mlir::func::FuncOp>(
      ov::gfx_plugin::kGfxMpsrtOpsSymbol);
  ASSERT_TRUE(ops_func);
  EXPECT_EQ(ops_func
                ->getAttrOfType<mlir::IntegerAttr>(
                    "gfx.mpsrt.ops.external_buffer_count")
                .getInt(),
            4);
  const auto abi =
      ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
  ASSERT_TRUE(abi.valid);
  EXPECT_EQ(abi.buffer_count, 4u);
  EXPECT_EQ(abi.output_buffer_count, 1u);
  EXPECT_EQ(abi.buffer_roles,
            std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput}));
}

TEST(GfxMlir, BackendCustomKernelBindingPlanDoesNotWriteLegacyOperandAttrs) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
  mlir::Builder builder(module.getContext());
  module->setAttr("gfx.kernel_operand_kinds",
                  builder.getArrayAttr({builder.getI32IntegerAttr(1)}));
  module->setAttr("gfx.kernel_operand_arg_indices",
                  builder.getArrayAttr({builder.getI32IntegerAttr(42)}));
  module->setAttr("gfx.kernel_scalar_values",
                  builder.getArrayAttr({builder.getI32IntegerAttr(7)}));

  const auto plan = ov::gfx_plugin::make_backend_custom_kernel_binding_plan(
      "Add", "eltwise_kernel");
  ASSERT_TRUE(plan.valid);
  ASSERT_TRUE(
      ov::gfx_plugin::annotate_backend_custom_kernel_module_with_binding_plan(
          module, plan));

  ASSERT_EQ(module
                ->getAttrOfType<mlir::StringAttr>(
                    "gfx.stage_manifest.kernel.entry_point")
                .str(),
            "eltwise_fused_buffer");
  ASSERT_TRUE(
      module->hasAttr("gfx.stage_manifest.kernel.external_buffer_abi.roles"));
  ASSERT_FALSE(module->hasAttr("gfx.kernel_operand_kinds"));
  ASSERT_FALSE(module->hasAttr("gfx.kernel_operand_arg_indices"));
  ASSERT_FALSE(module->hasAttr("gfx.kernel_scalar_values"));

  const auto metadata = ov::gfx_plugin::extract_kernel_runtime_metadata(
      module,
      /*output_arg_count=*/1,
      /*fallback_input_arg_count=*/999, "eltwise_fused_buffer");
  ASSERT_TRUE(metadata.valid);
  ASSERT_EQ(metadata.kernel_input_arg_count, 5u);
  ASSERT_EQ(metadata.operands.operand_kinds,
            std::vector<int32_t>({1, 1, 1, 0, 0, 1, 1, 1}));
  ASSERT_EQ(metadata.operands.operand_arg_indices,
            std::vector<int32_t>({0, 1, 5, -1, -1, 2, 3, 4}));
}

TEST(GfxMlir, StageKernelBindingHelpersOwnDirectAndCustomRuntimeBinding) {
  const auto direct = ov::gfx_plugin::make_stage_direct_kernel_runtime_binding(
      {0, 2}, 3, {1, 0, 1}, {0, -1, 3}, {17});
  EXPECT_EQ(direct.inputs, std::vector<size_t>({0, 2}));
  EXPECT_EQ(direct.input_arg_count, 3u);
  EXPECT_EQ(direct.operand_kinds, std::vector<int32_t>({1, 0, 1}));
  EXPECT_EQ(direct.operand_arg_indices, std::vector<int32_t>({0, -1, 3}));
  EXPECT_EQ(direct.scalar_args, std::vector<int32_t>({17}));

  const auto compact =
      ov::gfx_plugin::make_stage_compact_buffer_kernel_runtime_binding(3);
  EXPECT_EQ(compact.inputs, std::vector<size_t>({0, 1, 2}));
  EXPECT_EQ(compact.input_arg_count, 3u);
  EXPECT_TRUE(compact.operand_kinds.empty());
  EXPECT_TRUE(compact.operand_arg_indices.empty());

  const auto custom =
      ov::gfx_plugin::require_stage_backend_custom_kernel_runtime_binding(
          /*is_opencl_backend=*/false, "Tile", "tile_kernel", {16, 4},
          "tile_stage");
  EXPECT_EQ(custom.scalar_args, std::vector<int32_t>({16, 4}));
  EXPECT_EQ(custom.operand_kinds,
            std::vector<int32_t>({1, 1, 0, 0, 1, 1, 1, 1}));
  EXPECT_EQ(custom.operand_arg_indices,
            std::vector<int32_t>({0, 5, -1, -1, 1, 2, 3, 4}));
}

TEST(GfxMlir, MslScalarRuntimeArgsAreStoredOnStageManifest) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  const auto plan = ov::gfx_plugin::make_backend_custom_kernel_binding_plan(
      "Tile", "tile_kernel", {16, 4});
  ASSERT_TRUE(plan.valid);
  ASSERT_TRUE(
      ov::gfx_plugin::annotate_backend_custom_kernel_module_with_binding_plan(
          module, plan));

  ASSERT_FALSE(module->hasAttr("gfx.kernel_scalar_values"));
  auto scalar_attr = module->getAttrOfType<mlir::ArrayAttr>(
      "gfx.stage_manifest.kernel.scalar_args");
  ASSERT_TRUE(scalar_attr);
  ASSERT_EQ(scalar_attr.size(), 2u);

  const auto metadata = ov::gfx_plugin::extract_kernel_runtime_metadata(
      module,
      /*output_arg_count=*/1,
      /*fallback_input_arg_count=*/999, "gather_scatter_indexed");
  ASSERT_TRUE(metadata.valid);
  EXPECT_EQ(metadata.operands.scalar_args, std::vector<int32_t>({16, 4}));
  EXPECT_EQ(metadata.operands.operand_kinds,
            std::vector<int32_t>({1, 1, 0, 0, 1, 1, 1, 1}));
  EXPECT_EQ(metadata.operands.operand_arg_indices,
            std::vector<int32_t>({0, 5, -1, -1, 1, 2, 3, 4}));
}

TEST(GfxMlir, MslSliceManifestKeepsOnlyDataInputAsKernelInput) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  const auto plan = ov::gfx_plugin::make_backend_custom_kernel_binding_plan(
      "Slice", "slice_kernel");
  ASSERT_TRUE(plan.valid);
  ASSERT_TRUE(
      ov::gfx_plugin::annotate_backend_custom_kernel_module_with_binding_plan(
          module, plan));

  const auto metadata = ov::gfx_plugin::extract_kernel_runtime_metadata(
      module,
      /*output_arg_count=*/1,
      /*fallback_input_arg_count=*/999, "gather_scatter_indexed",
      ov::gfx_plugin::GfxKernelBackendDomain::AppleMsl);
  ASSERT_TRUE(metadata.valid);
  EXPECT_EQ(metadata.kernel_inputs, std::vector<size_t>({0}));
  EXPECT_EQ(metadata.kernel_input_arg_count, 7u);
  EXPECT_EQ(metadata.operands.operand_kinds,
            std::vector<int32_t>({1, 1, 1, 1, 1, 1, 1, 1}));
  EXPECT_EQ(metadata.operands.operand_arg_indices,
            std::vector<int32_t>({0, 7, 1, 2, 3, 4, 5, 6}));
}

TEST(GfxMlir, MslStaticSliceDirectIoManifestMatchesInlineConstants) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  const auto plan =
      ov::gfx_plugin::make_backend_custom_kernel_direct_io_binding_plan(
          "Slice", "slice_kernel",
          /*tensor_input_count=*/1,
          /*output_count=*/1, ov::gfx_plugin::GfxKernelBackendDomain::AppleMsl,
          ov::gfx_plugin::GfxKernelStorageKind::Buffer, "apple_msl:buffer:");
  ASSERT_TRUE(plan.valid);
  ASSERT_TRUE(
      ov::gfx_plugin::annotate_backend_custom_kernel_module_with_binding_plan(
          module, plan));

  const auto metadata = ov::gfx_plugin::extract_kernel_runtime_metadata(
      module,
      /*output_arg_count=*/1,
      /*fallback_input_arg_count=*/999, "slice_kernel",
      ov::gfx_plugin::GfxKernelBackendDomain::AppleMsl);
  ASSERT_TRUE(metadata.valid);
  EXPECT_EQ(metadata.kernel_inputs, std::vector<size_t>({0}));
  EXPECT_EQ(metadata.kernel_input_arg_count, 1u);
  EXPECT_EQ(metadata.operands.operand_kinds, std::vector<int32_t>({1, 1}));
  EXPECT_EQ(metadata.operands.operand_arg_indices,
            std::vector<int32_t>({0, 1}));

  ov::gfx_plugin::KernelSource source;
  ASSERT_TRUE(ov::gfx_plugin::configure_backend_custom_kernel_source_signature(
      source, plan));
  EXPECT_EQ(source.signature.arg_count, 2u);
  EXPECT_EQ(source.signature.output_arg_count, 1u);

  auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                      ov::Shape{2, 3, 4});
  auto slice = std::make_shared<ov::op::v8::Slice>(
      data, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 0}),
      ov::op::v0::Constant::create(ov::element::i64, {3}, {2, 3, 4}),
      ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 1, 2}),
      ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
  auto source_module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
  auto source_plan =
      ov::gfx_plugin::make_direct_static_slice_msl_kernel_source_plan(
          slice, ov::element::f32, source_module);
  ASSERT_TRUE(source_plan.valid());
  EXPECT_EQ(source_plan.source.signature.arg_count, 2u);
  EXPECT_EQ(source_plan.source.signature.output_arg_count, 1u);
  EXPECT_EQ(source_plan.binding.runtime_binding.input_arg_count, 1u);
  EXPECT_EQ(source_plan.binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 1}));
  EXPECT_NE(source_plan.source.msl_source.find("constant uint TOTAL_C"),
            std::string::npos);
  EXPECT_EQ(source_plan.source.msl_source.find("constant uint& TOTAL"),
            std::string::npos);
}

TEST(GfxMlir, StageManifestSuppliesElementwiseRoleAbiWithoutLegacyMpsrtAttrs) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "Add", nullptr,
      ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Add");
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_count"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_output_buffer_count"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_roles"));

  const auto abi =
      ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
  ASSERT_TRUE(abi.valid);
  ASSERT_TRUE(abi.has_buffer_roles);
  ASSERT_EQ(abi.buffer_count, 6u);
  ASSERT_EQ(abi.output_buffer_count, 1u);
}

TEST(GfxMlir,
     StageManifestSuppliesRoleBasedExternalBufferAbiWithoutModuleMpsrtAttrs) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "Softmax", nullptr,
      ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Softmax");
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_count"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_output_buffer_count"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_roles"));

  const auto abi =
      ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
  ASSERT_TRUE(abi.valid);
  ASSERT_EQ(abi.buffer_count, 2u);
  ASSERT_EQ(abi.output_buffer_count, 1u);
  ASSERT_TRUE(abi.has_buffer_roles);
  ASSERT_EQ(abi.buffer_roles,
            std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput}));
}

TEST(
    GfxMlir,
    StageManifestSuppliesRoleBasedGatherExternalBufferAbiWithoutModuleMpsrtAttrs) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "Gather", nullptr,
      ov::element::f32,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Gather");
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_count"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_output_buffer_count"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_roles"));

  const auto abi =
      ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
  ASSERT_TRUE(abi.valid);
  ASSERT_EQ(abi.buffer_count, 4u);
  ASSERT_EQ(abi.output_buffer_count, 1u);
  ASSERT_EQ(abi.buffer_roles,
            std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams}));
}

TEST(GfxMlir, SoftmaxMslMetadataUsesRoleBasedExternalAbi) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "Softmax", nullptr,
      ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Softmax");

  ASSERT_FALSE(module->hasAttr("gfx.msl.kernel_family"));
  ASSERT_EQ(
      module
          ->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.kernel.family")
          .str(),
      "softmax_buffer");
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_count"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_output_buffer_count"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_roles"));

  const auto abi =
      ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
  ASSERT_TRUE(abi.valid);
  EXPECT_TRUE(abi.has_buffer_count);
  EXPECT_TRUE(abi.has_output_buffer_count);
  EXPECT_EQ(abi.buffer_count, 2u);
  EXPECT_EQ(abi.output_buffer_count, 1u);
  ASSERT_TRUE(abi.has_buffer_roles);
  EXPECT_EQ(abi.buffer_roles,
            std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput}));
}

TEST(GfxMlir, MslSourcePlanKeepsExactManifestAbiOverLegacySignature) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4});
  auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);
  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(add, ctx);
  ASSERT_TRUE(module);

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "Add", add, ov::element::f32,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Add");

  auto source = ov::gfx_plugin::make_kernel_source(
      module, "legacy_add_entry",
      "kernel void legacy_add_entry(device const float* lhs [[buffer(0)]], "
      "device const float* rhs [[buffer(1)]], "
      "device float* output [[buffer(2)]]) {}",
      /*arg_count=*/99);
  source.signature.output_arg_count = 7;

  const auto source_plan = ov::gfx_plugin::configure_msl_kernel_source_plan(
      std::move(source), "Add");
  ASSERT_TRUE(source_plan.valid());
  EXPECT_EQ(source_plan.source.entry_point, "eltwise_fused_buffer");
  EXPECT_EQ(source_plan.source.signature.arg_count, 8u);
  EXPECT_EQ(source_plan.source.signature.output_arg_count, 1u);

  const auto abi =
      ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
  ASSERT_TRUE(abi.valid);
  EXPECT_EQ(abi.buffer_roles,
            std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams}));
}

TEST(GfxMlir,
     TypedMslSourcePlanUsesStageManifestRuntimeSignatureOverExternalAbi) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  const auto binding = ov::gfx_plugin::make_backend_custom_kernel_binding_plan(
      "Tile", "tile_kernel", {16, 4});
  ASSERT_TRUE(binding.valid);
  ASSERT_TRUE(
      ov::gfx_plugin::annotate_backend_custom_kernel_module_with_binding_plan(
          module, binding));

  ov::gfx_plugin::GfxMpsrtExternalBufferAbiPlan external_buffer_abi{};
  ASSERT_TRUE(
      ov::gfx_plugin::gfx_mpsrt_external_buffer_abi_from_kernel_manifest(
          module, external_buffer_abi));
  ASSERT_TRUE(external_buffer_abi.valid);
  EXPECT_EQ(external_buffer_abi.buffer_count, 6u);
  EXPECT_EQ(binding.runtime_binding.operand_kinds.size(), 8u);

  auto input = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 4}, ov::element::f32, ov::gfx_plugin::GfxStageStorageKind::Buffer,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  auto output = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 4}, ov::element::f32, ov::gfx_plugin::GfxStageStorageKind::Buffer,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);

  ov::gfx_plugin::GfxMpsrtStageDesc stage{};
  stage.kind = ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch;
  stage.domain = ov::gfx_plugin::GfxStageBackendDomain::AppleMsl;
  stage.input_storage = ov::gfx_plugin::GfxMpsrtStorage::Buffer;
  stage.output_storage = ov::gfx_plugin::GfxMpsrtStorage::Buffer;
  stage.layout = ov::gfx_plugin::GfxMpsrtLayout::Linear;
  stage.kernel_name = binding.stage_manifest.custom_kernel.entry_point;
  stage.stage_manifest = binding.stage_manifest;

  ov::gfx_plugin::GfxMpsrtProgram program{};
  program.valid = true;
  program.record_key = "typed_msl_tile_manifest_runtime_signature";
  program.inputs = {input};
  program.output_values = {1u};
  program.stages.push_back({stage, {0u}, {1u}, {output}});
  program.external_buffer_abi = external_buffer_abi;
  ASSERT_TRUE(ov::gfx_plugin::materialize_module_mpsrt_ops(module, program));

  ov::gfx_plugin::KernelSource source;
  source.module = module;
  source.entry_point = "tile_kernel";
  source.msl_source =
      "#include <metal_stdlib>\n"
      "using namespace metal;\n"
      "kernel void tile_kernel(device const float* input [[buffer(0)]], "
      "device float* output [[buffer(1)]], "
      "constant int& repeat0 [[buffer(2)]], "
      "constant int& repeat1 [[buffer(3)]]) {}\n";
  source.signature.arg_count = 3u;
  source.signature.output_arg_count = 1u;

  const auto source_plan =
      ov::gfx_plugin::make_mpsrt_kernel_source_plan_from_configured_source(
          std::move(source));
  ASSERT_TRUE(source_plan.valid());
  EXPECT_EQ(source_plan.source.entry_point,
            binding.stage_manifest.custom_kernel.entry_point);
  EXPECT_EQ(source_plan.source.signature.arg_count,
            binding.runtime_binding.operand_kinds.size());
  EXPECT_EQ(source_plan.source.signature.output_arg_count, 1u);

  const auto typed_abi =
      ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
  ASSERT_TRUE(typed_abi.valid);
  EXPECT_EQ(typed_abi.buffer_count, 6u);
}

TEST(GfxMlir,
     ConfiguredMslSourcePlanRejectsTypedCustomStageWithoutExactManifestAbi) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  auto input = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 4}, ov::element::f32, ov::gfx_plugin::GfxStageStorageKind::Buffer,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  auto output = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 4}, ov::element::f32, ov::gfx_plugin::GfxStageStorageKind::Buffer,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);

  ov::gfx_plugin::GfxMpsrtStageDesc stage{};
  stage.kind = ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch;
  stage.domain = ov::gfx_plugin::GfxStageBackendDomain::AppleMsl;
  stage.input_storage = ov::gfx_plugin::GfxMpsrtStorage::Buffer;
  stage.output_storage = ov::gfx_plugin::GfxMpsrtStorage::Buffer;
  stage.layout = ov::gfx_plugin::GfxMpsrtLayout::Linear;
  stage.kernel_name = "typed_incomplete_manifest";
  stage.stage_manifest.valid = true;
  stage.stage_manifest.stage_family =
      ov::gfx_plugin::GfxKernelStageFamily::Eltwise;
  stage.stage_manifest.backend_domain =
      ov::gfx_plugin::GfxKernelBackendDomain::AppleMsl;
  stage.stage_manifest.execution_kind =
      ov::gfx_plugin::GfxKernelExecutionKind::CustomKernel;
  stage.stage_manifest.storage = ov::gfx_plugin::GfxKernelStorageKind::Buffer;
  stage.stage_manifest.specialization_key =
      "apple_msl:buffer:typed_incomplete_manifest";
  stage.stage_manifest.custom_kernel.valid = true;
  stage.stage_manifest.custom_kernel.kernel_family = "eltwise_fused_buffer";
  stage.stage_manifest.custom_kernel.kernel_family_id = static_cast<uint32_t>(
      ov::gfx_plugin::GfxKernelFamily::EltwiseFusedBuffer);
  stage.stage_manifest.custom_kernel.entry_point = "typed_incomplete_manifest";
  stage.stage_manifest.custom_kernel.external_buffer_abi.valid = true;
  stage.stage_manifest.custom_kernel.dispatch_policy =
      ov::gfx_plugin::make_gfx_kernel_linear_dispatch_policy(
          /*threads_per_threadgroup=*/64,
          /*precompiled_binary_required=*/true);

  ov::gfx_plugin::GfxMpsrtProgram program{};
  program.valid = true;
  program.record_key = "typed_incomplete_manifest";
  program.inputs = {input};
  program.output_values = {1u};
  program.stages.push_back({stage, {0u}, {1u}, {output}});
  program.external_buffer_abi =
      ov::gfx_plugin::gfx_mpsrt_make_external_buffer_abi_from_roles(
          {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
           ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput});
  ASSERT_TRUE(ov::gfx_plugin::materialize_module_mpsrt_ops(module, program));

  ov::gfx_plugin::KernelSource source;
  source.module = module;
  source.entry_point = "typed_incomplete_manifest";
  source.msl_source = "#include <metal_stdlib>\n"
                      "using namespace metal;\n"
                      "kernel void typed_incomplete_manifest(device const "
                      "float* input [[buffer(0)]], "
                      "device float* output [[buffer(1)]]) {}\n";
  source.signature.arg_count = 99;
  source.signature.output_arg_count = 7;

  const auto source_plan =
      ov::gfx_plugin::make_mpsrt_kernel_source_plan_from_configured_source(
          std::move(source));
  EXPECT_FALSE(source_plan.valid());
}

TEST(GfxMlir,
     DirectMslSourceWithTypedExternalAbiIgnoresMismatchedLegacySignature) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4});
  auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);

  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  auto lhs_desc = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 4}, ov::element::f32, ov::gfx_plugin::GfxStageStorageKind::Buffer,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  auto rhs_desc = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 4}, ov::element::f32, ov::gfx_plugin::GfxStageStorageKind::Buffer,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  auto output_desc = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 4}, ov::element::f32, ov::gfx_plugin::GfxStageStorageKind::Buffer,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);

  ov::gfx_plugin::GfxMpsrtStageDesc stage{};
  stage.kind = ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch;
  stage.domain = ov::gfx_plugin::GfxStageBackendDomain::AppleMsl;
  stage.input_storage = ov::gfx_plugin::GfxMpsrtStorage::Buffer;
  stage.output_storage = ov::gfx_plugin::GfxMpsrtStorage::Buffer;
  stage.layout = ov::gfx_plugin::GfxMpsrtLayout::Linear;
  stage.kernel_name = "direct_exact_add";
  stage.stage_manifest.valid = true;
  stage.stage_manifest.stage_family =
      ov::gfx_plugin::GfxKernelStageFamily::Eltwise;
  stage.stage_manifest.backend_domain =
      ov::gfx_plugin::GfxKernelBackendDomain::AppleMsl;
  stage.stage_manifest.execution_kind =
      ov::gfx_plugin::GfxKernelExecutionKind::CustomKernel;
  stage.stage_manifest.storage = ov::gfx_plugin::GfxKernelStorageKind::Buffer;

  ov::gfx_plugin::GfxMpsrtProgram program{};
  program.valid = true;
  program.record_key = "direct_exact_add";
  program.inputs = {lhs_desc, rhs_desc};
  program.output_values = {2u};
  program.stages.push_back({stage, {0u, 1u}, {2u}, {output_desc}});
  program.external_buffer_abi =
      ov::gfx_plugin::gfx_mpsrt_make_external_buffer_abi_from_roles(
          {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
           ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
           ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput});
  ASSERT_TRUE(ov::gfx_plugin::materialize_module_mpsrt_ops(module, program));

  ov::gfx_plugin::KernelSource source;
  source.module = module;
  source.entry_point = "direct_exact_add";
  source.msl_source =
      "#include <metal_stdlib>\n"
      "using namespace metal;\n"
      "kernel void direct_exact_add(device const float* lhs [[buffer(0)]], "
      "device const float* rhs [[buffer(1)]], "
      "device float* output [[buffer(2)]]) {}\n";
  source.signature.arg_count = 99;
  source.signature.output_arg_count = 7;

  const auto source_plan =
      ov::gfx_plugin::configure_apple_metal_kernel_source_plan_for_stage(
          source, add, nullptr, "Add",
          /*has_bias=*/false,
          /*has_activation=*/false,
          /*has_batchnorm=*/false, ov::gfx_plugin::ActivationKind::Identity,
          ov::element::f32,
          /*has_runtime_slice_params=*/false);
  EXPECT_FALSE(source_plan.valid());
  EXPECT_EQ(source.entry_point, "direct_exact_add");
  EXPECT_NE(source.msl_source.find("kernel void direct_exact_add"),
            std::string::npos);
  EXPECT_EQ(source.msl_source.find("kernel void eltwise_fused_buffer"),
            std::string::npos);
  EXPECT_EQ(source.signature.arg_count, 99u);
  EXPECT_EQ(source.signature.output_arg_count, 7u);
}

TEST(GfxMlir,
     IncompleteMslManifestDoesNotMaterializeTypedAbiFromLegacySignature) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  ov::gfx_plugin::GfxKernelStageManifest manifest{};
  manifest.valid = true;
  manifest.stage_family = ov::gfx_plugin::GfxKernelStageFamily::Reduction;
  manifest.backend_domain = ov::gfx_plugin::GfxKernelBackendDomain::AppleMsl;
  manifest.execution_kind =
      ov::gfx_plugin::GfxKernelExecutionKind::CustomKernel;
  manifest.storage = ov::gfx_plugin::GfxKernelStorageKind::Buffer;
  manifest.specialization_key = "apple_msl:buffer:legacy_leading_io";
  manifest.custom_kernel.valid = true;
  manifest.custom_kernel.kernel_family = "reduction_buffer";
  manifest.custom_kernel.entry_point = "legacy_leading_io";
  manifest.custom_kernel.external_buffer_abi.valid = true;
  manifest.custom_kernel.dispatch_policy =
      ov::gfx_plugin::make_gfx_kernel_linear_dispatch_policy(
          /*threads_per_threadgroup=*/128,
          /*precompiled_binary_required=*/false);
  ov::gfx_plugin::detail::gfx_mpsrt_set_stage_manifest_attrs(module, manifest);
  mlir::OpBuilder builder(module.getContext());
  module->setAttr(
      "gfx.stage_manifest.kernel.external_buffer_abi.leading_input_count",
      builder.getI32IntegerAttr(1));
  module->setAttr(
      "gfx.stage_manifest.kernel.external_buffer_abi.leading_output_count",
      builder.getI32IntegerAttr(1));

  ov::gfx_plugin::KernelSource source;
  source.module = module;
  source.entry_point = "legacy_leading_io";
  source.msl_source =
      "#include <metal_stdlib>\n"
      "using namespace metal;\n"
      "kernel void legacy_leading_io(device const float* input [[buffer(0)]], "
      "device float* output [[buffer(1)]]) {}\n";
  source.signature.arg_count = 99;
  source.signature.output_arg_count = 7;

  const auto source_plan =
      ov::gfx_plugin::configure_msl_kernel_source_plan(source, "ReduceSum");
  EXPECT_FALSE(source_plan.valid());

  EXPECT_EQ(source.entry_point, "legacy_leading_io");
  EXPECT_EQ(source.signature.arg_count, 99u);
  EXPECT_EQ(source.signature.output_arg_count, 7u);
  EXPECT_FALSE(ov::gfx_plugin::module_has_mpsrt_ops_program(module));

  ov::gfx_plugin::GfxMpsrtExternalBufferAbiPlan abi{};
  EXPECT_FALSE(
      ov::gfx_plugin::gfx_mpsrt_external_buffer_abi_from_kernel_manifest(module,
                                                                         abi));
  const auto readback_abi =
      ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
  EXPECT_FALSE(readback_abi.valid);
}

TEST(GfxMlir, MslSourcePlanKeepsSourceBindingWhenLegacySignatureIsWider) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  auto lhs = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 4}, ov::element::f32, ov::gfx_plugin::GfxStageStorageKind::Buffer,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  auto rhs = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 4}, ov::element::f32, ov::gfx_plugin::GfxStageStorageKind::Buffer,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  auto output = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
      {1, 4}, ov::element::f32, ov::gfx_plugin::GfxStageStorageKind::Buffer,
      ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
  ov::gfx_plugin::GfxMpsrtStageDesc stage{};
  stage.kind = ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch;
  stage.domain = ov::gfx_plugin::GfxStageBackendDomain::AppleMsl;
  stage.input_storage = ov::gfx_plugin::GfxMpsrtStorage::Buffer;
  stage.output_storage = ov::gfx_plugin::GfxMpsrtStorage::Buffer;
  stage.layout = ov::gfx_plugin::GfxMpsrtLayout::Linear;
  stage.kernel_name = "legacy_add_entry";
  stage.stage_manifest.valid = true;
  stage.stage_manifest.stage_family =
      ov::gfx_plugin::GfxKernelStageFamily::Eltwise;
  stage.stage_manifest.backend_domain =
      ov::gfx_plugin::GfxKernelBackendDomain::AppleMsl;
  stage.stage_manifest.execution_kind =
      ov::gfx_plugin::GfxKernelExecutionKind::CustomKernel;
  stage.stage_manifest.storage = ov::gfx_plugin::GfxKernelStorageKind::Buffer;

  ov::gfx_plugin::GfxMpsrtProgram program{};
  program.valid = true;
  program.record_key = "legacy_msl_dispatch";
  program.inputs = {lhs, rhs};
  program.output_values = {2u};
  program.stages.push_back({stage, {0u, 1u}, {2u}, {output}});
  program.external_buffer_abi.valid = true;
  program.external_buffer_abi.has_buffer_count = true;
  program.external_buffer_abi.has_output_buffer_count = true;
  program.external_buffer_abi.has_buffer_roles = true;
  program.external_buffer_abi.buffer_count = 3u;
  program.external_buffer_abi.output_buffer_count = 1u;
  program.external_buffer_abi.buffer_roles = {
      ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
      ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
      ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput};
  ASSERT_TRUE(ov::gfx_plugin::materialize_module_mpsrt_ops(module, program));

  ov::gfx_plugin::GfxMpsrtModuleStagePlan initial_stage_plan{};
  ASSERT_TRUE(
      ov::gfx_plugin::read_module_mpsrt_stage_plan(module, initial_stage_plan));
  ASSERT_EQ(initial_stage_plan.stage.kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
  ASSERT_TRUE(initial_stage_plan.stage.stage_manifest.valid);
  ASSERT_FALSE(initial_stage_plan.stage.stage_manifest.custom_kernel.valid);

  ov::gfx_plugin::KernelSource source;
  source.module = module;
  source.entry_point = "legacy_add_entry";
  source.msl_source = "#include <metal_stdlib>\n"
                      "using namespace metal;\n"
                      "kernel void legacy_add_entry(device const float* lhs "
                      "[[buffer(0)]]) {}\n";
  source.signature.arg_count = 99;
  source.signature.output_arg_count = 7;

  const auto factory_binding =
      ov::gfx_plugin::make_backend_custom_kernel_binding_plan(
          "Add", source.entry_point);
  ASSERT_TRUE(factory_binding.valid);
  const auto source_binding =
      ov::gfx_plugin::make_backend_custom_kernel_source_binding_plan(
          source,
          /*is_opencl_backend=*/false, "Add", source.entry_point);
  ASSERT_TRUE(source_binding.valid);

  auto source_plan =
      ov::gfx_plugin::make_msl_generated_custom_kernel_source_plan(
          std::move(source), "Add");
  ASSERT_TRUE(source_plan.valid());
  source = std::move(source_plan.source);
  EXPECT_EQ(source.entry_point, "eltwise_fused_buffer");
  EXPECT_NE(source.msl_source.find("kernel void eltwise_fused_buffer"),
            std::string::npos);
  EXPECT_EQ(source.msl_source.find("kernel void legacy_add_entry"),
            std::string::npos);
  EXPECT_EQ(source.signature.arg_count, 8u);
  EXPECT_EQ(source.signature.output_arg_count, 1u);

  ov::gfx_plugin::GfxKernelStageManifest rewritten_manifest{};
  ASSERT_TRUE(ov::gfx_plugin::detail::gfx_mpsrt_read_stage_manifest_attrs(
      module, rewritten_manifest));
  ASSERT_TRUE(rewritten_manifest.custom_kernel.valid);
  EXPECT_EQ(rewritten_manifest.custom_kernel.entry_point,
            "eltwise_fused_buffer");
  EXPECT_EQ(rewritten_manifest.custom_kernel.external_buffer_abi.roles,
            std::vector<ov::gfx_plugin::GfxKernelBufferRole>(
                {ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
                 ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
                 ov::gfx_plugin::GfxKernelBufferRole::TensorOutput,
                 ov::gfx_plugin::GfxKernelBufferRole::ScalarParam,
                 ov::gfx_plugin::GfxKernelBufferRole::ScalarParam,
                 ov::gfx_plugin::GfxKernelBufferRole::RuntimeParams,
                 ov::gfx_plugin::GfxKernelBufferRole::RuntimeParams,
                 ov::gfx_plugin::GfxKernelBufferRole::RuntimeParams}));
}

TEST(GfxMlir, SdpaMslSourcePlansUseManifestRolesWithoutSignatureHints) {
  const auto masked =
      ov::gfx_plugin::make_sdpa_msl_kernel_source_plan(ov::element::f16,
                                                       /*has_mask=*/true);
  ASSERT_TRUE(masked.valid());
  EXPECT_NE(masked.source.msl_source.find("device const scalar_t* mask"),
            std::string::npos);
  EXPECT_EQ(masked.binding.runtime_binding.inputs,
            std::vector<size_t>({0u, 1u, 2u, 3u}));
  EXPECT_EQ(masked.binding.runtime_binding.input_arg_count, 5u);
  EXPECT_EQ(masked.binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(masked.source.signature.arg_count, 6u);
  EXPECT_EQ(masked.source.signature.output_arg_count, 1u);

  const auto nomask =
      ov::gfx_plugin::make_sdpa_msl_kernel_source_plan(ov::element::f16,
                                                       /*has_mask=*/false);
  ASSERT_TRUE(nomask.valid());
  EXPECT_EQ(nomask.source.msl_source.find("device const scalar_t* mask"),
            std::string::npos);
  EXPECT_EQ(nomask.binding.runtime_binding.inputs,
            std::vector<size_t>({0u, 1u, 2u}));
  EXPECT_EQ(nomask.binding.runtime_binding.input_arg_count, 4u);
  EXPECT_EQ(nomask.binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 1, 2, 3, 4}));
  EXPECT_EQ(nomask.source.signature.arg_count, 5u);
  EXPECT_EQ(nomask.source.signature.output_arg_count, 1u);
}

TEST(GfxMlir, TopKMslMetadataUsesRoleBasedExternalAbi) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "TopK", nullptr,
      ov::element::f32,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "TopK");

  ASSERT_FALSE(module->hasAttr("gfx.msl.kernel_family"));
  ASSERT_EQ(
      module
          ->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.kernel.family")
          .str(),
      "gather_scatter_indexed");
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_count"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_output_buffer_count"));
  ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_roles"));

  const auto abi =
      ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
  ASSERT_TRUE(abi.valid);
  EXPECT_TRUE(abi.has_buffer_count);
  EXPECT_TRUE(abi.has_output_buffer_count);
  EXPECT_EQ(abi.buffer_count, 3u);
  EXPECT_EQ(abi.output_buffer_count, 2u);
  ASSERT_TRUE(abi.has_buffer_roles);
  EXPECT_EQ(abi.buffer_roles,
            std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput}));
}

TEST(GfxMlir, TopKMslSupportsI64IndexPackingWithoutMlirModule) {
  ov::gfx_plugin::TopKCodegenDesc desc{};
  desc.axis_len = 8400;
  desc.k = 300;
  desc.outer = 1;
  desc.inner = 1;
  desc.mode_max = true;
  desc.sort_type = ov::gfx_plugin::TopKSortType::SortValues;
  desc.index_type = ov::element::i64;

  const auto source = ov::gfx_plugin::generate_msl_for_topk(desc, {});

  EXPECT_NE(source.find("device int* out_idx [[buffer(2)]]"),
            std::string::npos);
  EXPECT_NE(source.find("gfx_topk_value_better"), std::string::npos);
  EXPECT_NE(source.find("return candidate_index < current_index"),
            std::string::npos);
  EXPECT_NE(source.find("out_idx[out_idx_flat * 2]"), std::string::npos);
}

TEST(GfxMlir, TopKI64IndexMlirUsesTrailingLaneStorage) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 24000});
  auto k = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{},
                                        std::vector<int32_t>{300});
  auto topk = std::make_shared<ov::op::v3::TopK>(
      input, k, 1, ov::op::TopKMode::MAX, ov::op::TopKSortType::SORT_VALUES,
      ov::element::i64);
  auto values = std::make_shared<ov::op::v0::Result>(topk->output(0));
  auto indices = std::make_shared<ov::op::v0::Result>(topk->output(1));
  auto model = std::make_shared<ov::Model>(ov::ResultVector{values, indices},
                                           ov::ParameterVector{input},
                                           "topk_i64_lane_storage");

  mlir::MLIRContext ctx;
  auto module = ov::gfx_plugin::build_mlir_topk_from_model(model, ctx);
  ASSERT_TRUE(module);
  auto func = module.lookupSymbol<mlir::func::FuncOp>("topk_main");
  ASSERT_TRUE(func);
  EXPECT_EQ(func.getFunctionType().getNumInputs(), 3u);
  EXPECT_EQ(func.getFunctionType().getNumResults(), 0u);

  std::string text;
  llvm::raw_string_ostream os(text);
  module.print(os);
  EXPECT_NE(text.find("memref<1x300x2xi32>"), std::string::npos);
  EXPECT_EQ(text.find("memref<1x600xi32>"), std::string::npos);
  EXPECT_EQ(text.find("memref<1x300xi64>"), std::string::npos);
  EXPECT_NE(text.find("iter_args"), std::string::npos);
}

TEST(GfxMlir, GatherMslMetadataUsesRolePatternWithRuntimeParams) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "Gather", nullptr,
      ov::element::f32,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Gather");

  ASSERT_FALSE(module->hasAttr("gfx.msl.kernel_family"));
  ASSERT_EQ(
      module
          ->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.kernel.family")
          .str(),
      "gather_scatter_indexed");
  const auto abi =
      ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
  ASSERT_TRUE(abi.valid);
  EXPECT_EQ(abi.buffer_count, 4u);
  EXPECT_EQ(abi.output_buffer_count, 1u);
  ASSERT_TRUE(abi.has_buffer_roles);
  EXPECT_EQ(abi.buffer_roles,
            std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams}));
}

TEST(GfxMlir, SliceMslMetadataUsesRolePatternForTrailingRuntimeParams) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "Slice", nullptr,
      ov::element::f32,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Slice");

  ASSERT_FALSE(module->hasAttr("gfx.msl.kernel_family"));
  ASSERT_EQ(
      module
          ->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.kernel.family")
          .str(),
      "gather_scatter_indexed");
  const auto abi =
      ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
  ASSERT_TRUE(abi.valid);
  EXPECT_EQ(abi.buffer_count, 8u);
  EXPECT_EQ(abi.output_buffer_count, 1u);
  ASSERT_TRUE(abi.has_buffer_roles);
  ASSERT_EQ(abi.buffer_roles.size(), 8u);
  EXPECT_EQ(abi.buffer_roles[0],
            ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput);
  EXPECT_EQ(abi.buffer_roles[1],
            ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput);
  for (size_t i = 2; i < abi.buffer_roles.size(); ++i) {
    EXPECT_EQ(abi.buffer_roles[i],
              ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams);
  }
}

TEST(GfxMlir, AppleMslStructuralArgCountsComeFromStageManifest) {
  struct Case {
    const char *stage_type;
    const char *entry_point;
    std::vector<int32_t> scalar_args;
    size_t expected_arg_count;
  };
  const std::vector<Case> cases = {
      {"ShapeOf", "shapeof_kernel", {0}, 4u},
      {"Tile", "tile_kernel", {16, 4}, 8u},
      {"Gather", "gather_kernel", {}, 4u},
      {"GatherND", "gathernd_kernel", {}, 4u},
      {"GatherElements", "gather_elements_kernel", {}, 4u},
      {"MaxPool", "pool2d_kernel", {}, 3u},
      {"AvgPool", "pool2d_kernel", {}, 3u},
      {"Concat", "concat_kernel", {}, 3u},
  };

  for (const auto &c : cases) {
    mlir::MLIRContext ctx;
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mlir::OpBuilder builder(&ctx);
    auto plan = ov::gfx_plugin::make_backend_custom_kernel_binding_plan(
        c.stage_type, c.entry_point, c.scalar_args);
    ASSERT_TRUE(plan.valid) << c.stage_type;
    ASSERT_TRUE(
        ov::gfx_plugin::annotate_backend_custom_kernel_module_with_binding_plan(
            module, plan))
        << c.stage_type;
    module->setAttr("gfx.fixed_arg_count", builder.getI32IntegerAttr(999));

    EXPECT_EQ(ov::gfx_plugin::infer_backend_custom_kernel_arg_count(
                  module, ov::gfx_plugin::GfxKernelBackendDomain::AppleMsl,
                  /*fallback=*/2),
              c.expected_arg_count)
        << c.stage_type;
    EXPECT_EQ(ov::gfx_plugin::require_backend_manifest_arg_count(
                  module, ov::gfx_plugin::GfxKernelBackendDomain::AppleMsl,
                  std::string_view{}, c.stage_type),
              c.expected_arg_count)
        << c.stage_type;
  }
}

TEST(GfxMlir, BackendManifestArgCountRejectsMissingManifestAbi) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
  EXPECT_THROW(
      (void)ov::gfx_plugin::require_backend_manifest_arg_count(
          module, ov::gfx_plugin::GfxKernelBackendDomain::AppleMsl,
          "missing_kernel", "MissingManifest"),
      ov::Exception);
}

TEST(GfxMlir, MslKernelSourceSignatureCanBeConfiguredFromModuleManifest) {
  struct Case {
    const char *stage_type;
    const char *entry_point;
    std::vector<int32_t> scalar_args;
    uint32_t expected_arg_count;
    uint32_t expected_output_arg_count;
  };
  const std::vector<Case> cases = {
      {"Convert", "convert_kernel", {16}, 3u, 1u},
      {"Gather", "gather_kernel", {}, 4u, 1u},
      {"GatherND", "gathernd_kernel", {}, 4u, 1u},
      {"GatherElements", "gather_elements_kernel", {}, 4u, 1u},
      {"Tile", "tile_kernel", {16, 4}, 8u, 1u},
      {"MatMul", "matmul_kernel", {}, 3u, 1u},
  };

  for (const auto &c : cases) {
    mlir::MLIRContext ctx;
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    auto plan = ov::gfx_plugin::make_backend_custom_kernel_binding_plan(
        c.stage_type, c.entry_point, c.scalar_args);
    ASSERT_TRUE(plan.valid) << c.stage_type;
    ASSERT_TRUE(
        ov::gfx_plugin::annotate_backend_custom_kernel_module_with_binding_plan(
            module, plan))
        << c.stage_type;

    ov::gfx_plugin::KernelSource source;
    source.module = module;
    source.entry_point = c.entry_point;

    ASSERT_TRUE(
        ov::gfx_plugin::
            configure_backend_custom_kernel_source_signature_from_module(
                source))
        << c.stage_type;
    EXPECT_EQ(source.signature.arg_count, c.expected_arg_count) << c.stage_type;
    EXPECT_EQ(source.signature.output_arg_count, c.expected_output_arg_count)
        << c.stage_type;
  }
}

TEST(GfxMlir, MslKernelManifestAdapterResolvesExactRolesWithoutSignatureHints) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  const auto plan = select_test_stage_optimization_plan(
      nullptr, ov::gfx_plugin::GpuBackend::Metal, "GatherElements", nullptr,
      ov::element::f32,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan,
                                                      "GatherElements");
  EXPECT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_count"));

  ov::gfx_plugin::KernelSource source;
  source.module = module;
  source.entry_point = "gather_elements_kernel";
  source.msl_source =
      "#include <metal_stdlib>\n"
      "using namespace metal;\n"
      "kernel void gather_elements_kernel(device const float* data "
      "[[buffer(0)]]) {}\n";
  auto source_plan =
      ov::gfx_plugin::make_msl_generated_custom_kernel_source_plan(
          std::move(source), "GatherElements");
  ASSERT_TRUE(source_plan.valid());
  source = std::move(source_plan.source);

  EXPECT_EQ(source.entry_point, "gather_scatter_indexed");
  ov::gfx_plugin::GfxMpsrtExternalBufferAbiPlan abi;
  ASSERT_TRUE(
      ov::gfx_plugin::gfx_mpsrt_external_buffer_abi_from_kernel_manifest(module,
                                                                         abi));
  ASSERT_TRUE(abi.valid);
  EXPECT_EQ(abi.buffer_count, 4u);
  EXPECT_EQ(abi.output_buffer_count, 1u);
  EXPECT_EQ(abi.buffer_roles,
            std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams}));
}

TEST(GfxMlir, MslKernelSourceConfigurationPreservesExistingExactManifestRoles) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  auto binding = ov::gfx_plugin::make_backend_custom_kernel_roles_binding_plan(
      "GatherElements", "gather_elements_kernel",
      {ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
       ov::gfx_plugin::GfxKernelBufferRole::RuntimeParams,
       ov::gfx_plugin::GfxKernelBufferRole::TensorOutput,
       ov::gfx_plugin::GfxKernelBufferRole::RuntimeParams});
  ASSERT_TRUE(binding.valid);
  ASSERT_TRUE(
      ov::gfx_plugin::annotate_backend_custom_kernel_module_with_binding_plan(
          module, binding));

  ov::gfx_plugin::KernelSource source;
  source.module = module;
  source.entry_point = "gather_elements_kernel";
  source.msl_source =
      "#include <metal_stdlib>\n"
      "using namespace metal;\n"
      "kernel void gather_elements_kernel(device const float* data "
      "[[buffer(0)]]) {}\n";
  source.signature.arg_count = 2;
  source.signature.output_arg_count = 1;
  auto source_plan =
      ov::gfx_plugin::make_msl_generated_custom_kernel_source_plan(
          std::move(source), "GatherElements");
  ASSERT_TRUE(source_plan.valid());
  source = std::move(source_plan.source);

  EXPECT_EQ(source.entry_point, "gather_elements_kernel");
  ov::gfx_plugin::GfxMpsrtExternalBufferAbiPlan abi;
  ASSERT_TRUE(
      ov::gfx_plugin::gfx_mpsrt_external_buffer_abi_from_kernel_manifest(module,
                                                                         abi));
  ASSERT_TRUE(abi.valid);
  EXPECT_EQ(abi.buffer_count, 4u);
  EXPECT_EQ(abi.output_buffer_count, 1u);
  EXPECT_EQ(abi.buffer_roles,
            std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams}));
}

TEST(GfxMlir, MslKernelSourceConfigurationDoesNotWidenExactManifestRoles) {
  mlir::MLIRContext ctx;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  auto binding = ov::gfx_plugin::make_backend_custom_kernel_roles_binding_plan(
      "GatherElements", "gather_elements_kernel",
      {ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
       ov::gfx_plugin::GfxKernelBufferRole::TensorOutput,
       ov::gfx_plugin::GfxKernelBufferRole::RuntimeParams});
  ASSERT_TRUE(binding.valid);
  ASSERT_TRUE(
      ov::gfx_plugin::annotate_backend_custom_kernel_module_with_binding_plan(
          module, binding));

  ov::gfx_plugin::KernelSource source;
  source.module = module;
  source.entry_point = "gather_elements_kernel";
  source.msl_source =
      "#include <metal_stdlib>\n"
      "using namespace metal;\n"
      "kernel void gather_elements_kernel(device const float* data "
      "[[buffer(0)]]) {}\n";
  source.signature.arg_count = 3;
  source.signature.output_arg_count = 1;
  auto source_plan =
      ov::gfx_plugin::make_msl_generated_custom_kernel_source_plan(
          std::move(source), "GatherElements");
  ASSERT_TRUE(source_plan.valid());
  source = std::move(source_plan.source);

  EXPECT_EQ(source.entry_point, "gather_elements_kernel");
  ov::gfx_plugin::GfxMpsrtExternalBufferAbiPlan abi;
  ASSERT_TRUE(
      ov::gfx_plugin::gfx_mpsrt_external_buffer_abi_from_kernel_manifest(module,
                                                                         abi));
  ASSERT_TRUE(abi.valid);
  EXPECT_EQ(abi.buffer_count, 3u);
  EXPECT_EQ(abi.output_buffer_count, 1u);
  EXPECT_EQ(abi.buffer_roles,
            std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput,
                 ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams}));
}

TEST(GfxMlir, MatMulCodegenProducesMsl) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 2});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 2});
  auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, true);

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(matmul, ctx);
  ASSERT_TRUE(module);

  ov::gfx_plugin::MatMulCodegenDesc desc{};
  desc.element_type = ov::element::f32;
  desc.input_a_type = ov::element::f32;
  desc.input_b_type = ov::element::f32;
  desc.output_type = ov::element::f32;
  desc.batch = 1;
  desc.batch_a = 1;
  desc.batch_b = 1;
  desc.M = 4;
  desc.N = 4;
  desc.K = 2;
  desc.a_transpose = false;
  desc.b_transpose = true;
  desc.b_is_nk_layout = true;

  const auto msl = ov::gfx_plugin::generate_msl_from_mlir(module, desc);
  ASSERT_FALSE(msl.empty());
  ASSERT_NE(msl.find("kernel void matmul_kernel"), std::string::npos);
}

TEST(GfxMlir, MatMulPipelineSucceeds) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 2});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 2});
  auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, true);

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(matmul, ctx);
  ASSERT_TRUE(module);
  ASSERT_NO_THROW(ov::gfx_plugin::run_mlir_pipeline(
      module, /*use_alloca=*/true, /*use_parallel_loops=*/false));
}

TEST(GfxMlir, MatMulKernelPlanPipelineSucceeds) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 2});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 2});
  auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, true);

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(matmul, ctx);
  ASSERT_TRUE(module);

  auto plan_ctx = ov::gfx_plugin::build_mlir_kernel_plan(
      module, std::string{}, matmul,
      /*output_args_override=*/0,
      /*extra_inputs=*/0, "matmul_plan_test", "gfx_kernel",
      [&](const ov::gfx_plugin::KernelArgMappingInfo &info) -> size_t {
        size_t func_results = info.func_results;
        if (func_results == 0) {
          func_results = matmul->get_output_size();
        }
        const auto sig = info.signature;
        return sig.total() ? sig.total() : (info.func_inputs + func_results);
      });
  auto src = plan_ctx.build_info.plan.to_source();
  ASSERT_TRUE(src.module);
  ASSERT_NO_THROW(ov::gfx_plugin::run_mlir_pipeline(
      src.module, /*use_alloca=*/true, /*use_parallel_loops=*/false));
}

TEST(GfxMlir, MatMulPipelineSucceedsAfterFusionPlan) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 2});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 2});
  auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, true);
  auto res = std::make_shared<ov::op::v0::Result>(matmul);
  auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                           ov::ParameterVector{lhs, rhs},
                                           "matmul_fusion_ctx");

  ov::gfx_plugin::FusionConfig fusion_cfg;
  fusion_cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, fusion_cfg);
  EXPECT_TRUE(plan.groups.empty());

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(matmul, ctx);
  ASSERT_TRUE(module);
  ASSERT_NO_THROW(ov::gfx_plugin::run_mlir_pipeline(
      module, /*use_alloca=*/true, /*use_parallel_loops=*/false));
}

TEST(GfxMlir, BiasBroadcastAddMlirPipelineUsesGenericAddPath) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{2, 17, 5, 4});
  std::vector<float> bias_vals(17, 0.0f);
  for (size_t i = 0; i < bias_vals.size(); ++i) {
    bias_vals[i] = static_cast<float>((static_cast<int>(i % 13) - 6)) * 0.125f;
  }
  auto bias = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{1, 17, 1, 1}, bias_vals);
  auto add = std::make_shared<ov::op::v1::Add>(param, bias);

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(add, ctx);
  ASSERT_TRUE(module);
  ASSERT_NO_THROW(ov::gfx_plugin::run_mlir_pipeline(
      module, /*use_alloca=*/true, /*use_parallel_loops=*/false));
}

TEST(GfxMlir, CompactAbiPreserveRequiresSameLaunchOperandKinds) {
  const std::vector<int32_t> existing_buffer_only_kinds{1, 1, 1, 1, 1,
                                                        1, 1, 1, 1};
  const std::vector<int32_t> existing_indices{0, 1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<int32_t> launch_scalar_buffer_kinds{0, 0, 0, 0, 1,
                                                        1, 0, 0, 1};

  EXPECT_FALSE(ov::gfx_plugin::compact_kernel_operand_layout_matches_launch(
      existing_buffer_only_kinds, existing_indices,
      launch_scalar_buffer_kinds));
  EXPECT_TRUE(ov::gfx_plugin::compact_kernel_operand_layout_matches_launch(
      existing_buffer_only_kinds, existing_indices,
      existing_buffer_only_kinds));
}

TEST(GfxMlir, BiasBroadcastAddI32MlirLoweringUsesGenericAddPath) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i32,
                                                       ov::Shape{2, 17, 5, 4});
  std::vector<int32_t> bias_vals(17, 0);
  for (size_t i = 0; i < bias_vals.size(); ++i) {
    bias_vals[i] = static_cast<int32_t>(i) - 8;
  }
  auto bias = std::make_shared<ov::op::v0::Constant>(
      ov::element::i32, ov::Shape{1, 17, 1, 1}, bias_vals);
  auto add = std::make_shared<ov::op::v1::Add>(param, bias);

  auto &ctx = ov::gfx_plugin::gfx_mlir_context();
  auto module = ov::gfx_plugin::build_mlir_for_node(add, ctx);
  ASSERT_TRUE(module);
  ASSERT_NO_THROW(ov::gfx_plugin::run_mlir_pipeline(
      module, /*use_alloca=*/true, /*use_parallel_loops=*/false));
}

TEST(GfxTransforms, MlirFusionConvBatchNormReluPlan) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 3, 4, 4});
  auto weights = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{2, 3, 3, 3},
      std::vector<float>(2 * 3 * 3 * 3, 1.f));
  auto conv = std::make_shared<ov::op::v1::Convolution>(
      param, weights, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});
  auto gamma = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{2}, std::vector<float>{1.f, 1.f});
  auto beta = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{2}, std::vector<float>{0.f, 0.f});
  auto mean = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{2}, std::vector<float>{0.f, 0.f});
  auto var = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{2}, std::vector<float>{1.f, 1.f});
  auto bn = std::make_shared<ov::op::v5::BatchNormInference>(conv, gamma, beta,
                                                             mean, var, 1e-5);
  auto relu = std::make_shared<ov::op::v0::Relu>(bn);
  auto res = std::make_shared<ov::op::v0::Result>(relu);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{param}, "conv_bn_relu");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "ConvBatchNormAct" || group.node_indices.size() != 3) {
      continue;
    }
    const auto conv_idx = group.node_indices[0];
    const auto bn_idx = group.node_indices[1];
    const auto act_idx = group.node_indices[2];
    ASSERT_LT(conv_idx, ordered.size());
    ASSERT_LT(bn_idx, ordered.size());
    ASSERT_LT(act_idx, ordered.size());
    const auto &conv_node = ordered[conv_idx];
    const auto &bn_node = ordered[bn_idx];
    const auto &act_node = ordered[act_idx];
    if (ov::as_type_ptr<const ov::op::v1::Convolution>(conv_node) &&
        ov::as_type_ptr<const ov::op::v5::BatchNormInference>(bn_node) &&
        ov::as_type_ptr<const ov::op::v0::Relu>(act_node) &&
        group.activation.has_value() &&
        group.activation.value() == ov::gfx_plugin::ActivationKind::Relu) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionMatMulReluPlan) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{2, 4});
  auto weights = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{4, 4}, std::vector<float>(16, 0.25f));
  auto mm = std::make_shared<ov::op::v0::MatMul>(param, weights, false, false);
  auto relu = std::make_shared<ov::op::v0::Relu>(mm);
  auto res = std::make_shared<ov::op::v0::Result>(relu);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{param}, "matmul_relu");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "MatMulActivation" || group.node_indices.size() != 2) {
      continue;
    }
    const auto mm_idx = group.node_indices[0];
    const auto act_idx = group.node_indices[1];
    ASSERT_LT(mm_idx, ordered.size());
    ASSERT_LT(act_idx, ordered.size());
    const auto &mm_node = ordered[mm_idx];
    const auto &act_node = ordered[act_idx];
    if (ov::as_type_ptr<const ov::op::v0::MatMul>(mm_node) &&
        ov::as_type_ptr<const ov::op::v0::Relu>(act_node) &&
        group.activation.has_value() &&
        group.activation.value() == ov::gfx_plugin::ActivationKind::Relu) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionMatMulGeluPlan) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{2, 4});
  auto weights = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{4, 4}, std::vector<float>(16, 0.25f));
  auto mm = std::make_shared<ov::op::v0::MatMul>(param, weights, false, false);
  auto gelu = std::make_shared<ov::op::v7::Gelu>(
      mm, ov::op::GeluApproximationMode::TANH);
  auto res = std::make_shared<ov::op::v0::Result>(gelu);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{param}, "matmul_gelu");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "MatMulActivation" || group.node_indices.size() != 2) {
      continue;
    }
    const auto mm_idx = group.node_indices[0];
    const auto act_idx = group.node_indices[1];
    ASSERT_LT(mm_idx, ordered.size());
    ASSERT_LT(act_idx, ordered.size());
    const auto &mm_node = ordered[mm_idx];
    const auto &act_node = ordered[act_idx];
    if (ov::as_type_ptr<const ov::op::v0::MatMul>(mm_node) &&
        ov::as_type_ptr<const ov::op::v7::Gelu>(act_node) &&
        group.activation.has_value() &&
        group.activation.value() == ov::gfx_plugin::ActivationKind::Gelu) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionMatMulSwishPlan) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{2, 4});
  auto weights = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{4, 4}, std::vector<float>(16, 0.25f));
  auto mm = std::make_shared<ov::op::v0::MatMul>(param, weights, false, false);
  auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(mm);
  auto mul = std::make_shared<ov::op::v1::Multiply>(mm, sigmoid);
  auto res = std::make_shared<ov::op::v0::Result>(mul);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{param}, "matmul_swish");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "MatMulActivation" || group.node_indices.size() != 3) {
      continue;
    }
    const auto mm_idx = group.node_indices[0];
    const auto sig_idx = group.node_indices[1];
    const auto mul_idx = group.node_indices[2];
    ASSERT_LT(mm_idx, ordered.size());
    ASSERT_LT(sig_idx, ordered.size());
    ASSERT_LT(mul_idx, ordered.size());
    const auto &mm_node = ordered[mm_idx];
    const auto &sig_node = ordered[sig_idx];
    const auto &mul_node = ordered[mul_idx];
    if (ov::as_type_ptr<const ov::op::v0::MatMul>(mm_node) &&
        ov::as_type_ptr<const ov::op::v0::Sigmoid>(sig_node) &&
        ov::as_type_ptr<const ov::op::v1::Multiply>(mul_node) &&
        group.activation.has_value() &&
        group.activation.value() == ov::gfx_plugin::ActivationKind::Swish) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionMatMulBiasReluPlan) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{2, 4});
  auto weights = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{4, 4}, std::vector<float>(16, 0.5f));
  auto bias = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{1, 4}, std::vector<float>(4, 0.1f));
  auto mm = std::make_shared<ov::op::v0::MatMul>(param, weights, false, false);
  auto add = std::make_shared<ov::op::v1::Add>(mm, bias);
  auto relu = std::make_shared<ov::op::v0::Relu>(add);
  auto res = std::make_shared<ov::op::v0::Result>(relu);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{param}, "matmul_bias_relu");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "MatMulBiasActivation" ||
        group.node_indices.size() != 3) {
      continue;
    }
    const auto mm_idx = group.node_indices[0];
    const auto add_idx = group.node_indices[1];
    const auto act_idx = group.node_indices[2];
    ASSERT_LT(mm_idx, ordered.size());
    ASSERT_LT(add_idx, ordered.size());
    ASSERT_LT(act_idx, ordered.size());
    const auto &mm_node = ordered[mm_idx];
    const auto &add_node = ordered[add_idx];
    const auto &act_node = ordered[act_idx];
    if (ov::as_type_ptr<const ov::op::v0::MatMul>(mm_node) &&
        ov::as_type_ptr<const ov::op::v1::Add>(add_node) &&
        ov::as_type_ptr<const ov::op::v0::Relu>(act_node) &&
        group.activation.has_value() &&
        group.activation.value() == ov::gfx_plugin::ActivationKind::Relu) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionMatMulBiasSwishPlan) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{2, 4});
  auto weights = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{4, 4}, std::vector<float>(16, 0.5f));
  auto bias = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{1, 4}, std::vector<float>(4, 0.1f));
  auto mm = std::make_shared<ov::op::v0::MatMul>(param, weights, false, false);
  auto add = std::make_shared<ov::op::v1::Add>(mm, bias);
  auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(add);
  auto mul = std::make_shared<ov::op::v1::Multiply>(add, sigmoid);
  auto res = std::make_shared<ov::op::v0::Result>(mul);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{param}, "matmul_bias_swish");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "MatMulBiasActivation" ||
        group.node_indices.size() != 4) {
      continue;
    }
    const auto mm_idx = group.node_indices[0];
    const auto add_idx = group.node_indices[1];
    const auto sig_idx = group.node_indices[2];
    const auto mul_idx = group.node_indices[3];
    ASSERT_LT(mm_idx, ordered.size());
    ASSERT_LT(add_idx, ordered.size());
    ASSERT_LT(sig_idx, ordered.size());
    ASSERT_LT(mul_idx, ordered.size());
    const auto &mm_node = ordered[mm_idx];
    const auto &add_node = ordered[add_idx];
    const auto &sig_node = ordered[sig_idx];
    const auto &mul_node = ordered[mul_idx];
    if (ov::as_type_ptr<const ov::op::v0::MatMul>(mm_node) &&
        ov::as_type_ptr<const ov::op::v1::Add>(add_node) &&
        ov::as_type_ptr<const ov::op::v0::Sigmoid>(sig_node) &&
        ov::as_type_ptr<const ov::op::v1::Multiply>(mul_node) &&
        group.activation.has_value() &&
        group.activation.value() == ov::gfx_plugin::ActivationKind::Swish) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionConvBiasReluPlan) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 3, 4, 4});
  auto weights = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{2, 3, 3, 3},
      std::vector<float>(2 * 3 * 3 * 3, 1.f));
  auto bias = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{1, 2, 1, 1}, std::vector<float>(2, 0.25f));
  auto conv = std::make_shared<ov::op::v1::Convolution>(
      param, weights, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});
  auto add = std::make_shared<ov::op::v1::Add>(conv, bias);
  auto relu = std::make_shared<ov::op::v0::Relu>(add);
  auto res = std::make_shared<ov::op::v0::Result>(relu);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{param}, "conv_bias_relu");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "ConvBiasActivation" || group.node_indices.size() != 3) {
      continue;
    }
    const auto conv_idx = group.node_indices[0];
    const auto add_idx = group.node_indices[1];
    const auto act_idx = group.node_indices[2];
    ASSERT_LT(conv_idx, ordered.size());
    ASSERT_LT(add_idx, ordered.size());
    ASSERT_LT(act_idx, ordered.size());
    const auto &conv_node = ordered[conv_idx];
    const auto &add_node = ordered[add_idx];
    const auto &act_node = ordered[act_idx];
    if (ov::as_type_ptr<const ov::op::v1::Convolution>(conv_node) &&
        ov::as_type_ptr<const ov::op::v1::Add>(add_node) &&
        ov::as_type_ptr<const ov::op::v0::Relu>(act_node) &&
        group.activation.has_value() &&
        group.activation.value() == ov::gfx_plugin::ActivationKind::Relu) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms,
     MlirFusionConvActivationPolicyKeepsConvBiasWithoutActivation) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 3, 4, 4});
  auto weights = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{2, 3, 3, 3},
      std::vector<float>(2 * 3 * 3 * 3, 1.f));
  auto bias = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{1, 2, 1, 1}, std::vector<float>(2, 0.25f));
  auto conv = std::make_shared<ov::op::v1::Convolution>(
      param, weights, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});
  auto add = std::make_shared<ov::op::v1::Add>(conv, bias);
  auto relu = std::make_shared<ov::op::v0::Relu>(add);
  auto res = std::make_shared<ov::op::v0::Result>(relu);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{param},
      "conv_bias_relu_no_activation_fusion");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  cfg.enable_conv_activation_fusion = false;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found_conv_bias = false;
  bool found_conv_bias_activation = false;
  for (const auto &group : plan.groups) {
    found_conv_bias |=
        group.kind == "ConvBias" && group.node_indices.size() == 2;
    found_conv_bias_activation |= group.kind == "ConvBiasActivation";
  }
  EXPECT_TRUE(found_conv_bias);
  EXPECT_FALSE(found_conv_bias_activation);
}

TEST(GfxTransforms, MlirFusionConvDirectSwishBypassesGenericActivationPolicy) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 3, 4, 4});
  auto weights = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{2, 3, 3, 3},
      std::vector<float>(2 * 3 * 3 * 3, 1.f));
  auto conv = std::make_shared<ov::op::v1::Convolution>(
      param, weights, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});
  auto swish = std::make_shared<ov::op::v4::Swish>(conv);
  auto res = std::make_shared<ov::op::v0::Result>(swish);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{param}, "conv_direct_swish");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  cfg.enable_conv_activation_fusion = false;
  cfg.enable_conv_swish_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "ConvActivation" || group.node_indices.size() != 2) {
      continue;
    }
    const auto conv_idx = group.node_indices[0];
    const auto swish_idx = group.node_indices[1];
    ASSERT_LT(conv_idx, ordered.size());
    ASSERT_LT(swish_idx, ordered.size());
    if (ov::as_type_ptr<const ov::op::v1::Convolution>(ordered[conv_idx]) &&
        ov::as_type_ptr<const ov::op::v4::Swish>(ordered[swish_idx]) &&
        group.activation.has_value() &&
        group.activation.value() == ov::gfx_plugin::ActivationKind::Swish) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms,
     MlirFusionConvBiasDirectSwishBypassesGenericActivationPolicy) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 3, 4, 4});
  auto weights = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{2, 3, 3, 3},
      std::vector<float>(2 * 3 * 3 * 3, 1.f));
  auto bias = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{1, 2, 1, 1}, std::vector<float>(2, 0.25f));
  auto conv = std::make_shared<ov::op::v1::Convolution>(
      param, weights, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});
  auto add = std::make_shared<ov::op::v1::Add>(conv, bias);
  auto swish = std::make_shared<ov::op::v4::Swish>(add);
  auto res = std::make_shared<ov::op::v0::Result>(swish);
  auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                           ov::ParameterVector{param},
                                           "conv_bias_direct_swish");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  cfg.enable_conv_activation_fusion = false;
  cfg.enable_conv_swish_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "ConvBiasActivation" || group.node_indices.size() != 3) {
      continue;
    }
    const auto conv_idx = group.node_indices[0];
    const auto add_idx = group.node_indices[1];
    const auto swish_idx = group.node_indices[2];
    ASSERT_LT(conv_idx, ordered.size());
    ASSERT_LT(add_idx, ordered.size());
    ASSERT_LT(swish_idx, ordered.size());
    if (ov::as_type_ptr<const ov::op::v1::Convolution>(ordered[conv_idx]) &&
        ov::as_type_ptr<const ov::op::v1::Add>(ordered[add_idx]) &&
        ov::as_type_ptr<const ov::op::v4::Swish>(ordered[swish_idx]) &&
        group.activation.has_value() &&
        group.activation.value() == ov::gfx_plugin::ActivationKind::Swish &&
        group.bias.has_value()) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionConvBiasSwishPlan) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 3, 4, 4});
  auto weights = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{2, 3, 3, 3},
      std::vector<float>(2 * 3 * 3 * 3, 1.f));
  auto bias = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{1, 2, 1, 1}, std::vector<float>(2, 0.25f));
  auto conv = std::make_shared<ov::op::v1::Convolution>(
      param, weights, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});
  auto add = std::make_shared<ov::op::v1::Add>(conv, bias);
  auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(add);
  auto mul = std::make_shared<ov::op::v1::Multiply>(add, sigmoid);
  auto res = std::make_shared<ov::op::v0::Result>(mul);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{param}, "conv_bias_swish");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "ConvBiasActivation" || group.node_indices.size() != 4) {
      continue;
    }
    const auto conv_idx = group.node_indices[0];
    const auto add_idx = group.node_indices[1];
    const auto sig_idx = group.node_indices[2];
    const auto mul_idx = group.node_indices[3];
    ASSERT_LT(conv_idx, ordered.size());
    ASSERT_LT(add_idx, ordered.size());
    ASSERT_LT(sig_idx, ordered.size());
    ASSERT_LT(mul_idx, ordered.size());
    const auto &conv_node = ordered[conv_idx];
    const auto &add_node = ordered[add_idx];
    const auto &sig_node = ordered[sig_idx];
    const auto &mul_node = ordered[mul_idx];
    if (ov::as_type_ptr<const ov::op::v1::Convolution>(conv_node) &&
        ov::as_type_ptr<const ov::op::v1::Add>(add_node) &&
        ov::as_type_ptr<const ov::op::v0::Sigmoid>(sig_node) &&
        ov::as_type_ptr<const ov::op::v1::Multiply>(mul_node) &&
        group.activation.has_value() &&
        group.activation.value() == ov::gfx_plugin::ActivationKind::Swish) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxStagePolicyTest, MetalConvSwishSourcePlanUsesMpsTextureEpilogue) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                       ov::Shape{1, 3, 4, 4});
  auto weights = std::make_shared<ov::op::v0::Constant>(
      ov::element::f16, ov::Shape{2, 3, 3, 3},
      std::vector<ov::float16>(2 * 3 * 3 * 3, ov::float16(1.f)));
  auto conv = std::make_shared<ov::op::v1::Convolution>(
      param, weights, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});

  ov::gfx_plugin::KernelSource source;
  source.module = ov::gfx_plugin::build_mlir_for_node(
      conv, ov::gfx_plugin::gfx_mlir_context());
  ASSERT_TRUE(source.module);

  auto source_plan =
      ov::gfx_plugin::configure_apple_mps_vendor_kernel_source_plan_for_node(
          source, conv, nullptr, "Convolution",
          /*has_bias=*/false,
          /*has_activation=*/true,
          /*has_batchnorm=*/false, ov::gfx_plugin::ActivationKind::Swish,
          nullptr);
  ASSERT_TRUE(source_plan.valid());
  EXPECT_TRUE(source_plan.requires_mpsrt_model);
  EXPECT_EQ(source_plan.kind,
            ov::gfx_plugin::GfxMpsrtKernelSourcePlanKind::MultiStage);
  EXPECT_EQ(source_plan.first_stage_kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MPSConv2D);
  EXPECT_EQ(source_plan.last_stage_kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
  EXPECT_EQ(source_plan.source.entry_point,
            "gfx_mpsrt_conv_texture_swish_epilogue");
  EXPECT_EQ(source_plan.source.signature.arg_count, 2u);
  EXPECT_EQ(source_plan.source.signature.output_arg_count, 1u);
  EXPECT_TRUE(source_plan.has_runtime_binding);
  EXPECT_EQ(source_plan.runtime_binding.input_arg_count, 2u);
  EXPECT_EQ(source_plan.runtime_binding.operand_kinds,
            std::vector<int32_t>({1, 1, 1}));
  EXPECT_EQ(source_plan.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 1, 2}));
  EXPECT_NE(source_plan.source.msl_source.find("texture2d_array"),
            std::string::npos);
  EXPECT_NE(source_plan.source.msl_source.find("precise::exp(-x)"),
            std::string::npos);
}

TEST(GfxStagePolicyTest, MetalF32ConvSwishSourcePlanUsesMpsTextureEpilogue) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 3, 4, 4});
  auto weights = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{2, 3, 3, 3},
      std::vector<float>(2 * 3 * 3 * 3, 1.f));
  auto conv = std::make_shared<ov::op::v1::Convolution>(
      param, weights, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});

  ov::gfx_plugin::KernelSource source;
  source.module = ov::gfx_plugin::build_mlir_for_node(
      conv, ov::gfx_plugin::gfx_mlir_context());
  ASSERT_TRUE(source.module);

  const auto source_plan =
      ov::gfx_plugin::configure_apple_mps_vendor_kernel_source_plan_for_node(
          source, conv, nullptr, "Convolution",
          /*has_bias=*/false,
          /*has_activation=*/true,
          /*has_batchnorm=*/false, ov::gfx_plugin::ActivationKind::Swish,
          nullptr);
  ASSERT_TRUE(source_plan.valid());
  EXPECT_TRUE(source_plan.requires_mpsrt_model);
  EXPECT_EQ(source_plan.kind,
            ov::gfx_plugin::GfxMpsrtKernelSourcePlanKind::MultiStage);
  EXPECT_EQ(source_plan.first_stage_kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MPSConv2D);
  EXPECT_EQ(source_plan.last_stage_kind,
            ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
  EXPECT_EQ(source_plan.source.entry_point,
            "gfx_mpsrt_conv_texture_swish_epilogue");
  EXPECT_TRUE(source_plan.has_runtime_binding);
  EXPECT_NE(source_plan.source.msl_source.find("texture2d_array"),
            std::string::npos);
  EXPECT_NE(source_plan.source.msl_source.find("precise::exp(-x)"),
            std::string::npos);
}

TEST(GfxTransforms, MlirFusionConvBiasPlan) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 3, 4, 4});
  auto weights = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{2, 3, 3, 3},
      std::vector<float>(2 * 3 * 3 * 3, 1.f));
  auto bias = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{1, 2, 1, 1}, std::vector<float>(2, 0.125f));
  auto conv = std::make_shared<ov::op::v1::Convolution>(
      param, weights, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});
  auto add = std::make_shared<ov::op::v1::Add>(conv, bias);
  auto res = std::make_shared<ov::op::v0::Result>(add);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{param}, "conv_bias");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "ConvBias" || group.node_indices.size() != 2) {
      continue;
    }
    const auto conv_idx = group.node_indices[0];
    const auto add_idx = group.node_indices[1];
    ASSERT_LT(conv_idx, ordered.size());
    ASSERT_LT(add_idx, ordered.size());
    const auto &conv_node = ordered[conv_idx];
    const auto &add_node = ordered[add_idx];
    if (ov::as_type_ptr<const ov::op::v1::Convolution>(conv_node) &&
        ov::as_type_ptr<const ov::op::v1::Add>(add_node)) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionConvScalePlan) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 3, 4, 4});
  auto weights = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{2, 3, 3, 3},
      std::vector<float>(2 * 3 * 3 * 3, 1.f));
  auto scale = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{1, 2, 1, 1}, std::vector<float>{0.5f, 2.0f});
  auto conv = std::make_shared<ov::op::v1::Convolution>(
      param, weights, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});
  auto mul = std::make_shared<ov::op::v1::Multiply>(conv, scale);
  auto res = std::make_shared<ov::op::v0::Result>(mul);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{param}, "conv_scale");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "ConvScale" || group.node_indices.size() != 2 ||
        !group.batchnorm.has_value()) {
      continue;
    }
    const auto conv_idx = group.node_indices[0];
    const auto mul_idx = group.node_indices[1];
    ASSERT_LT(conv_idx, ordered.size());
    ASSERT_LT(mul_idx, ordered.size());
    const auto &conv_node = ordered[conv_idx];
    const auto &mul_node = ordered[mul_idx];
    if (ov::as_type_ptr<const ov::op::v1::Convolution>(conv_node) &&
        ov::as_type_ptr<const ov::op::v1::Multiply>(mul_node)) {
      ASSERT_EQ(group.batchnorm->gamma.size(), 2u);
      EXPECT_FLOAT_EQ(group.batchnorm->gamma[0], 0.5f);
      EXPECT_FLOAT_EQ(group.batchnorm->gamma[1], 2.0f);
      EXPECT_FLOAT_EQ(group.batchnorm->beta[0], 0.0f);
      EXPECT_FLOAT_EQ(group.batchnorm->beta[1], 0.0f);
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionAddBiasReluPlan) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 4, 4});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 4, 4});
  auto bias = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{1, 4, 1, 1}, std::vector<float>(4, 0.25f));
  auto add0 = std::make_shared<ov::op::v1::Add>(lhs, rhs);
  auto add1 = std::make_shared<ov::op::v1::Add>(add0, bias);
  auto relu = std::make_shared<ov::op::v0::Relu>(add1);
  auto res = std::make_shared<ov::op::v0::Result>(relu);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{lhs, rhs}, "add_bias_relu");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "EltwiseBiasActivation" ||
        group.node_indices.size() != 3) {
      continue;
    }
    const auto elt_idx = group.node_indices[0];
    const auto add_idx = group.node_indices[1];
    const auto act_idx = group.node_indices[2];
    ASSERT_LT(elt_idx, ordered.size());
    ASSERT_LT(add_idx, ordered.size());
    ASSERT_LT(act_idx, ordered.size());
    const auto &elt_node = ordered[elt_idx];
    const auto &add_node = ordered[add_idx];
    const auto &act_node = ordered[act_idx];
    if (ov::as_type_ptr<const ov::op::v1::Add>(elt_node) &&
        ov::as_type_ptr<const ov::op::v1::Add>(add_node) &&
        ov::as_type_ptr<const ov::op::v0::Relu>(act_node) &&
        group.activation.has_value() &&
        group.activation.value() == ov::gfx_plugin::ActivationKind::Relu) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionAddBiasPlan) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 4, 4});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{1, 4, 4, 4});
  auto bias = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{1, 4, 1, 1}, std::vector<float>(4, 0.25f));
  auto add0 = std::make_shared<ov::op::v1::Add>(lhs, rhs);
  auto add1 = std::make_shared<ov::op::v1::Add>(add0, bias);
  auto res = std::make_shared<ov::op::v0::Result>(add1);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{lhs, rhs}, "add_bias");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "EltwiseBias" || group.node_indices.size() != 2) {
      continue;
    }
    const auto elt_idx = group.node_indices[0];
    const auto add_idx = group.node_indices[1];
    ASSERT_LT(elt_idx, ordered.size());
    ASSERT_LT(add_idx, ordered.size());
    const auto &elt_node = ordered[elt_idx];
    const auto &add_node = ordered[add_idx];
    if (ov::as_type_ptr<const ov::op::v1::Add>(elt_node) &&
        ov::as_type_ptr<const ov::op::v1::Add>(add_node)) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionMatMulBiasPlan) {
  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{2, 4});
  auto weights = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{4, 4}, std::vector<float>(16, 0.5f));
  auto bias = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{1, 4}, std::vector<float>(4, 0.2f));
  auto mm = std::make_shared<ov::op::v0::MatMul>(param, weights, false, false);
  auto add = std::make_shared<ov::op::v1::Add>(mm, bias);
  auto res = std::make_shared<ov::op::v0::Result>(add);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{res}, ov::ParameterVector{param}, "matmul_bias");

  ov::gfx_plugin::FusionConfig cfg;
  cfg.enable_fusion = true;
  auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
  ASSERT_FALSE(plan.groups.empty());

  bool found = false;
  const auto ordered = model->get_ordered_ops();
  for (const auto &group : plan.groups) {
    if (group.kind != "MatMulBias" || group.node_indices.size() != 2) {
      continue;
    }
    const auto mm_idx = group.node_indices[0];
    const auto add_idx = group.node_indices[1];
    ASSERT_LT(mm_idx, ordered.size());
    ASSERT_LT(add_idx, ordered.size());
    const auto &mm_node = ordered[mm_idx];
    const auto &add_node = ordered[add_idx];
    if (ov::as_type_ptr<const ov::op::v0::MatMul>(mm_node) &&
        ov::as_type_ptr<const ov::op::v1::Add>(add_node)) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(GfxTransforms, CommonOptimizationsConstantFolding) {
  // Constant folding should remove Add/Relu when inputs are compile-time
  // constants.
  auto c0 = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{2}, std::vector<float>{1.f, -2.f});
  auto c1 = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{2}, std::vector<float>{3.f, 4.f});
  auto add = std::make_shared<ov::op::v1::Add>(c0, c1);
  auto relu = std::make_shared<ov::op::v0::Relu>(add);
  auto res = std::make_shared<ov::op::v0::Result>(relu);
  auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                           ov::ParameterVector{}, "const_fold");

  auto transformed = ov::gfx_plugin::transforms::run_pipeline(model);

  // Expect only Result + Constant to remain after folding (Add and Relu folded
  // away).
  size_t constants = 0;
  size_t adds = 0;
  size_t relus = 0;
  for (const auto &op : transformed->get_ops()) {
    if (ov::is_type<ov::op::v0::Constant>(op))
      ++constants;
    if (ov::is_type<ov::op::v1::Add>(op))
      ++adds;
    if (ov::is_type<ov::op::v0::Relu>(op))
      ++relus;
  }
  EXPECT_EQ(adds, 0u);
  EXPECT_EQ(relus, 0u);
  EXPECT_GE(constants, 1u);
}

TEST(GfxTransforms, PrecisionPolicyPreservesDeclaredF32ElementTypesByDefault) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{2, 3});
  auto bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 3},
                                           std::vector<float>{1.f, 2.f, 3.f});
  auto add = std::make_shared<ov::op::v1::Add>(input, bias);
  add->set_friendly_name("plain_f32_add");
  auto relu = std::make_shared<ov::op::v0::Relu>(add);
  relu->set_friendly_name("plain_f32_relu");
  auto res = std::make_shared<ov::op::v0::Result>(relu);
  auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                           ov::ParameterVector{input},
                                           "precision_policy_plain_f32");

  auto transformed = ov::gfx_plugin::transforms::run_pipeline(model);

  bool saw_add = false;
  bool saw_relu = false;
  for (const auto &op : transformed->get_ops()) {
    if (op->get_friendly_name() == "plain_f32_add") {
      saw_add = true;
      EXPECT_EQ(op->get_output_element_type(0), ov::element::f32);
    }
    if (op->get_friendly_name() == "plain_f32_relu") {
      saw_relu = true;
      EXPECT_EQ(op->get_output_element_type(0), ov::element::f32);
    }
  }
  EXPECT_TRUE(saw_add);
  EXPECT_TRUE(saw_relu);
}

TEST(GfxTransforms, PrecisionPolicyMarksExpReducePathForFp32) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 8});
  auto exp = std::make_shared<ov::op::v0::Exp>(input);
  auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1},
                                           std::vector<int64_t>{1});
  auto reduce = std::make_shared<ov::op::v1::ReduceSum>(exp, axes, false);
  auto res = std::make_shared<ov::op::v0::Result>(reduce);
  auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                           ov::ParameterVector{input},
                                           "precision_policy_exp_reduce");

  auto transformed = ov::gfx_plugin::transforms::run_pipeline(model);

  bool saw_marked_exp = false;
  bool saw_marked_reduce = false;
  for (const auto &op : transformed->get_ops()) {
    if (ov::is_type<ov::op::v0::Exp>(op)) {
      saw_marked_exp = ov::fp16_compression_is_disabled(op);
    }
    if (ov::is_type<ov::op::v1::ReduceSum>(op)) {
      saw_marked_reduce = ov::fp16_compression_is_disabled(op);
    }
  }
  EXPECT_TRUE(saw_marked_exp);
  EXPECT_TRUE(saw_marked_reduce);
}

TEST(GfxTransforms, PrecisionPolicyMarksLargeTopKIslandForFp32) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 8400});
  auto k = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{},
                                        std::vector<int32_t>{300});
  auto topk = std::make_shared<ov::op::v3::TopK>(
      input, k, 1, ov::op::TopKMode::MAX, ov::op::TopKSortType::SORT_VALUES,
      ov::element::i64);
  auto values = std::make_shared<ov::op::v0::Result>(topk->output(0));
  auto indices = std::make_shared<ov::op::v0::Result>(topk->output(1));
  auto model = std::make_shared<ov::Model>(ov::ResultVector{values, indices},
                                           ov::ParameterVector{input},
                                           "precision_policy_large_topk");

  auto transformed = ov::gfx_plugin::transforms::run_pipeline(model);

  bool saw_marked_topk = false;
  bool saw_marked_parameter = false;
  for (const auto &op : transformed->get_ops()) {
    if (ov::is_type<ov::op::v3::TopK>(op)) {
      saw_marked_topk = ov::fp16_compression_is_disabled(op);
    }
    if (ov::is_type<ov::op::v0::Parameter>(op)) {
      saw_marked_parameter = ov::fp16_compression_is_disabled(op);
    }
  }
  EXPECT_TRUE(saw_marked_topk);
  EXPECT_TRUE(saw_marked_parameter);
}

TEST(
    GfxTransforms,
    RankingCanonicalizationMovesSigmoidAfterReduceMaxAndTopKWhenIndicesUnused) {
  auto logits = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                        ov::Shape{1, 3, 8});
  logits->set_friendly_name("score_logits");
  auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(logits);
  sigmoid->set_friendly_name("score_sigmoid");
  auto reduce_axes = ov::op::v0::Constant::create(
      ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
  auto reduce =
      std::make_shared<ov::op::v1::ReduceMax>(sigmoid, reduce_axes, false);
  reduce->set_friendly_name("score_reduce");
  auto k = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{},
                                        std::vector<int32_t>{2});
  auto topk = std::make_shared<ov::op::v3::TopK>(
      reduce, k, 1, ov::op::TopKMode::MAX, ov::op::TopKSortType::SORT_VALUES,
      ov::element::i64);
  topk->set_friendly_name("score_topk");
  auto values = std::make_shared<ov::op::v0::Result>(topk->output(0));
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{values}, ov::ParameterVector{logits},
      "ranking_canonicalization_sigmoid_topk");

  auto transformed = ov::gfx_plugin::transforms::run_pipeline(
      model, ranking_pipeline_options());

  bool saw_topk = false;
  bool saw_values_sigmoid_after_topk = false;
  bool saw_topk_on_sigmoid = false;
  size_t sigmoid_count = 0;
  for (const auto &op : transformed->get_ops()) {
    if (auto rewritten_topk = ov::as_type_ptr<ov::op::util::TopKBase>(op)) {
      saw_topk = true;
      saw_topk_on_sigmoid |= ov::is_type<ov::op::v0::Sigmoid>(
          rewritten_topk->input_value(0).get_node_shared_ptr());
    }
    if (auto restored_sigmoid = ov::as_type_ptr<ov::op::v0::Sigmoid>(op)) {
      ++sigmoid_count;
      saw_values_sigmoid_after_topk |=
          ov::as_type_ptr<ov::op::util::TopKBase>(
              restored_sigmoid->input_value(0).get_node_shared_ptr()) !=
          nullptr;
    }
  }
  EXPECT_TRUE(saw_topk);
  EXPECT_TRUE(saw_values_sigmoid_after_topk);
  EXPECT_EQ(sigmoid_count, 1u);
  EXPECT_FALSE(saw_topk_on_sigmoid);
}

TEST(GfxTransforms,
     RankingCanonicalizationMovesSigmoidForObservableTopKValuesAndIndices) {
  auto logits = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                        ov::Shape{1, 3, 8});
  logits->set_friendly_name("score_logits");
  auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(logits);
  sigmoid->set_friendly_name("score_sigmoid");
  auto reduce_axes = ov::op::v0::Constant::create(
      ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
  auto reduce =
      std::make_shared<ov::op::v1::ReduceMax>(sigmoid, reduce_axes, false);
  reduce->set_friendly_name("score_reduce");
  auto k = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{},
                                        std::vector<int32_t>{2});
  auto topk = std::make_shared<ov::op::v3::TopK>(
      reduce, k, 1, ov::op::TopKMode::MAX, ov::op::TopKSortType::SORT_VALUES,
      ov::element::i64);
  topk->set_friendly_name("score_topk");
  auto values = std::make_shared<ov::op::v0::Result>(topk->output(0));
  auto indices = std::make_shared<ov::op::v0::Result>(topk->output(1));
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{values, indices}, ov::ParameterVector{logits},
      "ranking_canonicalization_observable_values_indices");

  auto transformed = ov::gfx_plugin::transforms::run_pipeline(
      model, ranking_pipeline_options());

  bool saw_restored_sigmoid_after_topk = false;
  bool saw_topk_on_sigmoid_domain = false;
  size_t sigmoid_count = 0;
  std::function<bool(const std::shared_ptr<ov::Node> &, size_t)>
      input_path_reaches_sigmoid =
          [&](const std::shared_ptr<ov::Node> &node, size_t depth) -> bool {
    if (!node || depth > 16) {
      return false;
    }
    if (node->get_type_name() == std::string("Sigmoid")) {
      return true;
    }
    if (node->get_input_size() == 0) {
      return false;
    }
    return input_path_reaches_sigmoid(
        node->input_value(0).get_node_shared_ptr(), depth + 1);
  };
  for (const auto &op : transformed->get_ops()) {
    if (auto rewritten_topk = ov::as_type_ptr<ov::op::util::TopKBase>(op)) {
      auto topk_input = rewritten_topk->input_value(0).get_node_shared_ptr();
      saw_topk_on_sigmoid_domain |= input_path_reaches_sigmoid(topk_input, 0);
    }
    if (auto restored_sigmoid = ov::as_type_ptr<ov::op::v0::Sigmoid>(op)) {
      ++sigmoid_count;
      saw_restored_sigmoid_after_topk |=
          ov::as_type_ptr<ov::op::util::TopKBase>(
              restored_sigmoid->input_value(0).get_node_shared_ptr()) !=
          nullptr;
    }
  }
  EXPECT_FALSE(saw_topk_on_sigmoid_domain);
  EXPECT_EQ(sigmoid_count, 1u);
  EXPECT_TRUE(saw_restored_sigmoid_after_topk);
}

TEST(GfxTransforms,
     RankingCanonicalizationMovesSigmoidWhenOnlyTopKIndicesAreObservable) {
  auto logits = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                        ov::Shape{1, 3, 8});
  logits->set_friendly_name("score_logits");
  auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(logits);
  sigmoid->set_friendly_name("score_sigmoid");
  auto reduce_axes = ov::op::v0::Constant::create(
      ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
  auto reduce =
      std::make_shared<ov::op::v1::ReduceMax>(sigmoid, reduce_axes, false);
  reduce->set_friendly_name("score_reduce");
  auto k = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{},
                                        std::vector<int32_t>{2});
  auto topk = std::make_shared<ov::op::v3::TopK>(
      reduce, k, 1, ov::op::TopKMode::MAX, ov::op::TopKSortType::SORT_VALUES,
      ov::element::i64);
  topk->set_friendly_name("score_topk");
  auto indices = std::make_shared<ov::op::v0::Result>(topk->output(1));
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{indices}, ov::ParameterVector{logits},
      "ranking_canonicalization_indices_only");

  auto transformed = ov::gfx_plugin::transforms::run_pipeline(
      model, ranking_pipeline_options());

  bool saw_topk = false;
  bool saw_topk_on_sigmoid_domain = false;
  size_t sigmoid_count = 0;
  std::function<bool(const std::shared_ptr<ov::Node> &, size_t)>
      input_path_reaches_sigmoid =
          [&](const std::shared_ptr<ov::Node> &node, size_t depth) -> bool {
    if (!node || depth > 16) {
      return false;
    }
    if (node->get_type_name() == std::string("Sigmoid")) {
      return true;
    }
    if (node->get_input_size() == 0) {
      return false;
    }
    return input_path_reaches_sigmoid(
        node->input_value(0).get_node_shared_ptr(), depth + 1);
  };
  for (const auto &op : transformed->get_ops()) {
    if (auto rewritten_topk = ov::as_type_ptr<ov::op::util::TopKBase>(op)) {
      saw_topk = true;
      auto topk_input = rewritten_topk->input_value(0).get_node_shared_ptr();
      saw_topk_on_sigmoid_domain |= input_path_reaches_sigmoid(topk_input, 0);
    }
    if (ov::as_type_ptr<ov::op::v0::Sigmoid>(op)) {
      ++sigmoid_count;
    }
  }
  EXPECT_TRUE(saw_topk);
  EXPECT_FALSE(saw_topk_on_sigmoid_domain);
  EXPECT_EQ(sigmoid_count, 0u);
}

TEST(GfxTransforms,
     RankingCanonicalizationTracesYoloConcatSplitScorePathWhenIndicesUnused) {
  auto boxes = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 2, 8});
  auto score_logits = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f32, ov::Shape{1, 3, 8});
  auto score_sigmoid = std::make_shared<ov::op::v0::Sigmoid>(score_logits);
  auto concat = std::make_shared<ov::op::v0::Concat>(
      ov::OutputVector{boxes, score_sigmoid}, 1);
  auto permutation = ov::op::v0::Constant::create(
      ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 2, 1});
  auto transpose = std::make_shared<ov::op::v1::Transpose>(concat, permutation);
  auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{},
                                                 std::vector<int64_t>{2});
  auto split_lengths = ov::op::v0::Constant::create(
      ov::element::i64, ov::Shape{2}, std::vector<int64_t>{2, 3});
  auto split = std::make_shared<ov::op::v1::VariadicSplit>(
      transpose, split_axis, split_lengths);
  auto reduce_axis = ov::op::v0::Constant::create(
      ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2});
  auto reduce = std::make_shared<ov::op::v1::ReduceMax>(split->output(1),
                                                        reduce_axis, false);
  auto k = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{},
                                        std::vector<int32_t>{2});
  auto topk = std::make_shared<ov::op::v3::TopK>(
      reduce, k, 1, ov::op::TopKMode::MAX, ov::op::TopKSortType::SORT_VALUES,
      ov::element::i64);
  auto values = std::make_shared<ov::op::v0::Result>(topk->output(0));
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{values}, ov::ParameterVector{boxes, score_logits},
      "ranking_canonicalization_yolo_score_path");

  auto transformed = ov::gfx_plugin::transforms::run_pipeline(
      model, ranking_pipeline_options());

  bool saw_topk = false;
  bool saw_values_sigmoid_after_topk = false;
  bool saw_topk_on_sigmoid = false;
  size_t sigmoid_count = 0;
  for (const auto &op : transformed->get_ops()) {
    if (auto rewritten_topk = ov::as_type_ptr<ov::op::util::TopKBase>(op)) {
      saw_topk = true;
      saw_topk_on_sigmoid |= ov::is_type<ov::op::v0::Sigmoid>(
          rewritten_topk->input_value(0).get_node_shared_ptr());
    }
    if (auto restored_sigmoid = ov::as_type_ptr<ov::op::v0::Sigmoid>(op)) {
      ++sigmoid_count;
      saw_values_sigmoid_after_topk |=
          ov::as_type_ptr<ov::op::util::TopKBase>(
              restored_sigmoid->input_value(0).get_node_shared_ptr()) !=
          nullptr;
    }
  }
  EXPECT_TRUE(saw_topk);
  EXPECT_TRUE(saw_values_sigmoid_after_topk);
  EXPECT_EQ(sigmoid_count, 1u);
  EXPECT_FALSE(saw_topk_on_sigmoid);
}

TEST(GfxTransforms,
     PrecisionPolicyKeepsFusibleConvBiasTopKDecodeFp32IncludingConv) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 4, 4, 4});
  auto weights = ov::op::v0::Constant::create(
      ov::element::f32, ov::Shape{8, 4, 1, 1}, std::vector<float>(8 * 4, 1.f));
  auto conv = std::make_shared<ov::op::v1::Convolution>(
      input, weights, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0},
      ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});
  auto bias = ov::op::v0::Constant::create(
      ov::element::f32, ov::Shape{1, 8, 1, 1}, std::vector<float>(8, 0.5f));
  auto add = std::make_shared<ov::op::v1::Add>(conv, bias);
  auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2},
                                            std::vector<int64_t>{1, 128});
  auto reshape = std::make_shared<ov::op::v1::Reshape>(add, shape, false);
  auto k = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{},
                                        std::vector<int32_t>{32});
  auto topk = std::make_shared<ov::op::v3::TopK>(
      reshape, k, 1, ov::op::TopKMode::MAX, ov::op::TopKSortType::SORT_VALUES,
      ov::element::i64);
  auto values = std::make_shared<ov::op::v0::Result>(topk->output(0));
  auto indices = std::make_shared<ov::op::v0::Result>(topk->output(1));
  auto model = std::make_shared<ov::Model>(ov::ResultVector{values, indices},
                                           ov::ParameterVector{input},
                                           "precision_policy_conv_bias_topk");

  auto transformed = ov::gfx_plugin::transforms::run_pipeline(model);

  bool saw_marked_add = false;
  bool saw_marked_topk = false;
  bool saw_marked_conv = false;
  for (const auto &op : transformed->get_ops()) {
    if (ov::is_type<ov::op::v1::Convolution>(op)) {
      saw_marked_conv = op->get_output_element_type(0) == ov::element::f32 &&
                        ov::fp16_compression_is_disabled(op);
    }
    if (ov::is_type<ov::op::v1::Add>(op)) {
      saw_marked_add = ov::fp16_compression_is_disabled(op);
    }
    if (ov::is_type<ov::op::v3::TopK>(op)) {
      saw_marked_topk = ov::fp16_compression_is_disabled(op);
    }
  }
  EXPECT_TRUE(saw_marked_conv);
  EXPECT_TRUE(saw_marked_add);
  EXPECT_TRUE(saw_marked_topk);
}

TEST(GfxTransforms,
     PrecisionPolicyExtendsSingleConsumerScoreHeadAroundConvForLargeTopK) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 4, 4, 4});
  auto weights0 =
      ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4, 4, 1, 1},
                                   std::vector<float>(4 * 4, 0.25f));
  auto conv0 = std::make_shared<ov::op::v1::Convolution>(
      input, weights0, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0},
      ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});
  conv0->set_friendly_name("score_conv0");
  auto bias0 = ov::op::v0::Constant::create(
      ov::element::f32, ov::Shape{1, 4, 1, 1}, std::vector<float>(4, 0.125f));
  auto add0 = std::make_shared<ov::op::v1::Add>(conv0, bias0);
  add0->set_friendly_name("score_add0");
  auto swish0 = std::make_shared<ov::op::v4::Swish>(add0);
  swish0->set_friendly_name("score_swish0");
  auto weights1 = ov::op::v0::Constant::create(
      ov::element::f32, ov::Shape{8, 4, 1, 1}, std::vector<float>(8 * 4, 0.5f));
  auto conv1 = std::make_shared<ov::op::v1::Convolution>(
      swish0, weights1, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0},
      ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});
  conv1->set_friendly_name("score_conv1");
  auto bias1 = ov::op::v0::Constant::create(
      ov::element::f32, ov::Shape{1, 8, 1, 1}, std::vector<float>(8, 0.25f));
  auto add1 = std::make_shared<ov::op::v1::Add>(conv1, bias1);
  add1->set_friendly_name("score_add1");
  auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2},
                                            std::vector<int64_t>{1, 128});
  auto reshape = std::make_shared<ov::op::v1::Reshape>(add1, shape, false);
  auto k = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{},
                                        std::vector<int32_t>{32});
  auto topk = std::make_shared<ov::op::v3::TopK>(
      reshape, k, 1, ov::op::TopKMode::MAX, ov::op::TopKSortType::SORT_VALUES,
      ov::element::i64);
  auto values = std::make_shared<ov::op::v0::Result>(topk->output(0));
  auto indices = std::make_shared<ov::op::v0::Result>(topk->output(1));
  auto model = std::make_shared<ov::Model>(ov::ResultVector{values, indices},
                                           ov::ParameterVector{input},
                                           "precision_policy_score_head_topk");

  auto transformed = ov::gfx_plugin::transforms::run_pipeline(model);

  bool saw_marked_add0 = false;
  bool saw_marked_swish0 = false;
  bool saw_marked_add1 = false;
  bool saw_marked_conv0 = false;
  bool saw_marked_conv1 = false;
  for (const auto &op : transformed->get_ops()) {
    if (op->get_friendly_name() == "score_conv0") {
      saw_marked_conv0 = op->get_output_element_type(0) == ov::element::f32 &&
                         ov::fp16_compression_is_disabled(op);
    }
    if (op->get_friendly_name() == "score_add0") {
      saw_marked_add0 = ov::fp16_compression_is_disabled(op);
    }
    if (op->get_friendly_name() == "score_swish0") {
      saw_marked_swish0 = ov::fp16_compression_is_disabled(op);
    }
    if (op->get_friendly_name() == "score_conv1") {
      saw_marked_conv1 = op->get_output_element_type(0) == ov::element::f32 &&
                         ov::fp16_compression_is_disabled(op);
    }
    if (op->get_friendly_name() == "score_add1") {
      saw_marked_add1 = ov::fp16_compression_is_disabled(op);
    }
  }
  EXPECT_TRUE(saw_marked_conv0);
  EXPECT_TRUE(saw_marked_add0);
  EXPECT_TRUE(saw_marked_swish0);
  EXPECT_TRUE(saw_marked_conv1);
  EXPECT_TRUE(saw_marked_add1);
}

TEST(GfxTransforms,
     PrecisionPolicyStopsAtSharedFeatureBoundaryWithoutCrossingInput) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 4, 4, 4});
  input->set_friendly_name("model_input");
  auto shared_weights = ov::op::v0::Constant::create(
      ov::element::f32, ov::Shape{4, 4, 1, 1}, std::vector<float>(4 * 4, 0.5f));
  auto shared_conv = std::make_shared<ov::op::v1::Convolution>(
      input, shared_weights, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0},
      ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});
  shared_conv->set_friendly_name("shared_feature_conv");
  auto tap = std::make_shared<ov::op::v0::Relu>(shared_conv);
  auto tap_result = std::make_shared<ov::op::v0::Result>(tap);

  auto score_weights =
      ov::op::v0::Constant::create(ov::element::f32, ov::Shape{8, 4, 1, 1},
                                   std::vector<float>(8 * 4, 0.25f));
  auto score_conv = std::make_shared<ov::op::v1::Convolution>(
      shared_conv, score_weights, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0},
      ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});
  score_conv->set_friendly_name("score_boundary_conv");
  auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2},
                                            std::vector<int64_t>{1, 128});
  auto reshape =
      std::make_shared<ov::op::v1::Reshape>(score_conv, shape, false);
  auto k = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{},
                                        std::vector<int32_t>{32});
  auto topk = std::make_shared<ov::op::v3::TopK>(
      reshape, k, 1, ov::op::TopKMode::MAX, ov::op::TopKSortType::SORT_VALUES,
      ov::element::i64);
  auto values = std::make_shared<ov::op::v0::Result>(topk->output(0));
  auto indices = std::make_shared<ov::op::v0::Result>(topk->output(1));
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{tap_result, values, indices}, ov::ParameterVector{input},
      "precision_policy_shared_boundary_topk");

  auto transformed = ov::gfx_plugin::transforms::run_pipeline(model);

  bool saw_shared_conv = false;
  bool saw_marked_score_conv = false;
  bool saw_marked_input = false;
  for (const auto &op : transformed->get_ops()) {
    if (op->get_friendly_name() == "model_input") {
      saw_marked_input = ov::fp16_compression_is_disabled(op);
    }
    if (op->get_friendly_name() == "shared_feature_conv") {
      saw_shared_conv = op->get_output_element_type(0) == ov::element::f32 &&
                        ov::fp16_compression_is_disabled(op);
    }
    if (op->get_friendly_name() == "score_boundary_conv") {
      saw_marked_score_conv =
          op->get_output_element_type(0) == ov::element::f32 &&
          ov::fp16_compression_is_disabled(op);
    }
  }
  EXPECT_FALSE(saw_marked_input);
  EXPECT_TRUE(saw_shared_conv);
  EXPECT_TRUE(saw_marked_score_conv);
}

TEST(GfxTransforms,
     PrecisionPolicyKeepsBoundedFp32IslandsForIndexAmplifiedTopK) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 4, 4, 4});
  input->set_friendly_name("model_input");
  auto shared_weights = ov::op::v0::Constant::create(
      ov::element::f32, ov::Shape{4, 4, 1, 1}, std::vector<float>(4 * 4, 0.5f));
  auto shared_conv = std::make_shared<ov::op::v1::Convolution>(
      input, shared_weights, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0},
      ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});
  shared_conv->set_friendly_name("shared_feature_conv");
  auto tap = std::make_shared<ov::op::v0::Relu>(shared_conv);
  auto tap_result = std::make_shared<ov::op::v0::Result>(tap);

  auto score_weights =
      ov::op::v0::Constant::create(ov::element::f32, ov::Shape{8, 4, 1, 1},
                                   std::vector<float>(8 * 4, 0.25f));
  auto score_conv = std::make_shared<ov::op::v1::Convolution>(
      shared_conv, score_weights, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0},
      ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});
  score_conv->set_friendly_name("score_boundary_conv");
  auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2},
                                            std::vector<int64_t>{1, 128});
  auto reshape =
      std::make_shared<ov::op::v1::Reshape>(score_conv, shape, false);
  auto k = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{},
                                        std::vector<int32_t>{32});
  auto topk = std::make_shared<ov::op::v3::TopK>(
      reshape, k, 1, ov::op::TopKMode::MAX, ov::op::TopKSortType::SORT_VALUES,
      ov::element::i64);
  auto gather_data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                             ov::Shape{1, 32});
  gather_data->set_friendly_name("gather_data");
  auto gather = std::make_shared<ov::op::v6::GatherElements>(
      gather_data, topk->output(1), 1);
  auto values = std::make_shared<ov::op::v0::Result>(topk->output(0));
  auto gathered = std::make_shared<ov::op::v0::Result>(gather);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{tap_result, values, gathered},
      ov::ParameterVector{input, gather_data},
      "precision_policy_index_amplified_topk");

  auto transformed = ov::gfx_plugin::transforms::run_pipeline(model);

  bool saw_marked_input = false;
  bool saw_marked_shared_conv = false;
  bool saw_marked_score_conv = false;
  bool saw_marked_gather_data = false;
  bool saw_marked_gather = false;
  for (const auto &op : transformed->get_ops()) {
    if (op->get_friendly_name() == "model_input") {
      saw_marked_input = ov::fp16_compression_is_disabled(op);
    }
    if (op->get_friendly_name() == "gather_data") {
      saw_marked_gather_data = ov::fp16_compression_is_disabled(op);
    }
    if (op->get_friendly_name() == "shared_feature_conv") {
      saw_marked_shared_conv =
          op->get_output_element_type(0) == ov::element::f32 &&
          ov::fp16_compression_is_disabled(op);
    }
    if (op->get_friendly_name() == "score_boundary_conv") {
      saw_marked_score_conv =
          op->get_output_element_type(0) == ov::element::f32 &&
          ov::fp16_compression_is_disabled(op);
    }
    if (ov::is_type<ov::op::v6::GatherElements>(op)) {
      saw_marked_gather = ov::fp16_compression_is_disabled(op);
    }
  }
  EXPECT_FALSE(saw_marked_input);
  EXPECT_TRUE(saw_marked_shared_conv);
  EXPECT_TRUE(saw_marked_score_conv);
  EXPECT_TRUE(saw_marked_gather_data);
  EXPECT_TRUE(saw_marked_gather);
}
