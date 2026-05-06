// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/hsigmoid.hpp"
#include "openvino/op/hswish.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#define HAS_OV_LAYER_NORM 0
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sigmoid.hpp"
#include "transforms/pipeline.hpp"
#include "transforms/fusion_pass.hpp"
#include "plugin/gfx_backend_config.hpp"
#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "mlir/gfx_mlir_kernel_metadata.hpp"
#include "mlir/gfx_apple_stage_pipeline.hpp"
#include "mlir/gfx_mpsrt_conv_metadata.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "mlir/gfx_mpsrt_matmul_metadata.hpp"
#include "mlir/gfx_mpsrt_ops.hpp"
#include "mlir/gfx_mpsrt_runtime_abi_pipeline.hpp"
#include "mlir/mlir_kernel_plan_utils.hpp"
#include "mlir/mlir_support.hpp"
#include "mlir/codegen_common.hpp"
#include "mlir/mlir_passes.hpp"
#include "mlir/msl_codegen.hpp"
#include "mlir/IR/Verifier.h"
#include "runtime/gfx_mpsrt_builder_plan.hpp"
#include "runtime/gfx_stage_policy.hpp"
#include "llvm/ADT/SmallVector.h"
#if GFX_BACKEND_VULKAN_AVAILABLE
#    include "mlir/spirv_codegen.hpp"
#endif
TEST(GfxTransforms, MlirFusionConvReluPlan) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 4, 4});
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                          ov::Shape{2, 3, 3, 3},
                                                          std::vector<float>(2 * 3 * 3 * 3, 1.f));
    auto conv = std::make_shared<ov::op::v1::Convolution>(param,
                                                          weights,
                                                          ov::Strides{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::Strides{1, 1});
    auto relu = std::make_shared<ov::op::v0::Relu>(conv);
    auto res = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "conv_relu");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "ConvActivation" || group.node_indices.size() != 2) {
            continue;
        }
        const auto conv_idx = group.node_indices[0];
        const auto act_idx = group.node_indices[1];
        ASSERT_LT(conv_idx, ordered.size());
        ASSERT_LT(act_idx, ordered.size());
        const auto& conv_node = ordered[conv_idx];
        const auto& act_node = ordered[act_idx];
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
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 4, 4});
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                          ov::Shape{2, 3, 3, 3},
                                                          std::vector<float>(2 * 3 * 3 * 3, 1.f));
    auto conv = std::make_shared<ov::op::v1::Convolution>(param,
                                                          weights,
                                                          ov::Strides{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::Strides{1, 1});
    auto gelu = std::make_shared<ov::op::v7::Gelu>(conv, ov::op::GeluApproximationMode::TANH);
    auto res = std::make_shared<ov::op::v0::Result>(gelu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "conv_gelu");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "ConvActivation" || group.node_indices.size() != 2) {
            continue;
        }
        const auto conv_idx = group.node_indices[0];
        const auto act_idx = group.node_indices[1];
        ASSERT_LT(conv_idx, ordered.size());
        ASSERT_LT(act_idx, ordered.size());
        const auto& conv_node = ordered[conv_idx];
        const auto& act_node = ordered[act_idx];
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
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 4, 4});
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                          ov::Shape{2, 3, 3, 3},
                                                          std::vector<float>(2 * 3 * 3 * 3, 1.f));
    auto conv = std::make_shared<ov::op::v1::Convolution>(param,
                                                          weights,
                                                          ov::Strides{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::Strides{1, 1});
    auto hswish = std::make_shared<ov::op::v4::HSwish>(conv);
    auto res = std::make_shared<ov::op::v0::Result>(hswish);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "conv_hswish");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "ConvActivation" || group.node_indices.size() != 2) {
            continue;
        }
        const auto conv_idx = group.node_indices[0];
        const auto act_idx = group.node_indices[1];
        ASSERT_LT(conv_idx, ordered.size());
        ASSERT_LT(act_idx, ordered.size());
        const auto& conv_node = ordered[conv_idx];
        const auto& act_node = ordered[act_idx];
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
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4, 4});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4, 4});
    auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);
    auto relu = std::make_shared<ov::op::v0::Relu>(add);
    auto res = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{lhs, rhs},
                                             "add_relu");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "EltwiseActivation" || group.node_indices.size() != 2) {
            continue;
        }
        const auto elt_idx = group.node_indices[0];
        const auto act_idx = group.node_indices[1];
        ASSERT_LT(elt_idx, ordered.size());
        ASSERT_LT(act_idx, ordered.size());
        const auto& elt_node = ordered[elt_idx];
        const auto& act_node = ordered[act_idx];
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
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4, 4});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4, 4});
    auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);
    auto gelu = std::make_shared<ov::op::v7::Gelu>(add, ov::op::GeluApproximationMode::TANH);
    auto res = std::make_shared<ov::op::v0::Result>(gelu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{lhs, rhs},
                                             "add_gelu");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "EltwiseActivation" || group.node_indices.size() != 2) {
            continue;
        }
        const auto elt_idx = group.node_indices[0];
        const auto act_idx = group.node_indices[1];
        ASSERT_LT(elt_idx, ordered.size());
        ASSERT_LT(act_idx, ordered.size());
        const auto& elt_node = ordered[elt_idx];
        const auto& act_node = ordered[act_idx];
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
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4, 4});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4, 4});
    auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);
    auto hsigmoid = std::make_shared<ov::op::v5::HSigmoid>(add);
    auto res = std::make_shared<ov::op::v0::Result>(hsigmoid);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{lhs, rhs}, "add_hsigmoid");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "EltwiseActivation" || group.node_indices.size() != 2) {
            continue;
        }
        const auto add_idx = group.node_indices[0];
        const auto act_idx = group.node_indices[1];
        ASSERT_LT(add_idx, ordered.size());
        ASSERT_LT(act_idx, ordered.size());
        const auto& add_node = ordered[add_idx];
        const auto& act_node = ordered[act_idx];
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
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4, 4});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4, 4});
    auto mul = std::make_shared<ov::op::v1::Multiply>(lhs, rhs);
    auto relu = std::make_shared<ov::op::v0::Relu>(mul);
    auto res = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{lhs, rhs},
                                             "mul_relu");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "EltwiseActivation" || group.node_indices.size() != 2) {
            continue;
        }
        const auto elt_idx = group.node_indices[0];
        const auto act_idx = group.node_indices[1];
        ASSERT_LT(elt_idx, ordered.size());
        ASSERT_LT(act_idx, ordered.size());
        const auto& elt_node = ordered[elt_idx];
        const auto& act_node = ordered[act_idx];
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
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4, 4});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4, 4});
    auto max = std::make_shared<ov::op::v1::Maximum>(lhs, rhs);
    auto relu = std::make_shared<ov::op::v0::Relu>(max);
    auto res = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{lhs, rhs},
                                             "max_relu");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "EltwiseActivation" || group.node_indices.size() != 2) {
            continue;
        }
        const auto elt_idx = group.node_indices[0];
        const auto act_idx = group.node_indices[1];
        ASSERT_LT(elt_idx, ordered.size());
        ASSERT_LT(act_idx, ordered.size());
        const auto& elt_node = ordered[elt_idx];
        const auto& act_node = ordered[act_idx];
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
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4, 4});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4, 4});
    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                       ov::Shape{1, 4, 1, 1},
                                                       std::vector<float>(4, 0.25f));
    auto mul = std::make_shared<ov::op::v1::Multiply>(lhs, rhs);
    auto add = std::make_shared<ov::op::v1::Add>(mul, bias);
    auto relu = std::make_shared<ov::op::v0::Relu>(add);
    auto res = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{lhs, rhs},
                                             "mul_bias_relu");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "EltwiseBiasActivation" || group.node_indices.size() != 3) {
            continue;
        }
        const auto elt_idx = group.node_indices[0];
        const auto add_idx = group.node_indices[1];
        const auto act_idx = group.node_indices[2];
        ASSERT_LT(elt_idx, ordered.size());
        ASSERT_LT(add_idx, ordered.size());
        ASSERT_LT(act_idx, ordered.size());
        const auto& elt_node = ordered[elt_idx];
        const auto& add_node = ordered[add_idx];
        const auto& act_node = ordered[act_idx];
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
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4, 4});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4, 4});
    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                       ov::Shape{1, 4, 1, 1},
                                                       std::vector<float>(4, -0.5f));
    auto max = std::make_shared<ov::op::v1::Maximum>(lhs, rhs);
    auto add = std::make_shared<ov::op::v1::Add>(max, bias);
    auto res = std::make_shared<ov::op::v0::Result>(add);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{lhs, rhs},
                                             "max_bias");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "EltwiseBias" || group.node_indices.size() != 2) {
            continue;
        }
        const auto elt_idx = group.node_indices[0];
        const auto add_idx = group.node_indices[1];
        ASSERT_LT(elt_idx, ordered.size());
        ASSERT_LT(add_idx, ordered.size());
        const auto& elt_node = ordered[elt_idx];
        const auto& add_node = ordered[add_idx];
        if (ov::as_type_ptr<const ov::op::v1::Maximum>(elt_node) &&
            ov::as_type_ptr<const ov::op::v1::Add>(add_node)) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionAttentionPlan) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto w1 = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                     ov::Shape{4, 4},
                                                     std::vector<float>(16, 0.5f));
    auto w2 = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                     ov::Shape{4, 4},
                                                     std::vector<float>(16, 0.25f));
    auto mm1 = std::make_shared<ov::op::v0::MatMul>(param, w1, false, false);
    auto sm = std::make_shared<ov::op::v1::Softmax>(mm1, 1);
    auto mm2 = std::make_shared<ov::op::v0::MatMul>(sm, w2, false, false);
    auto res = std::make_shared<ov::op::v0::Result>(mm2);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "attn_plan");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "Attention" || group.node_indices.size() != 3) {
            continue;
        }
        const auto mm1_idx = group.node_indices[0];
        const auto sm_idx = group.node_indices[1];
        const auto mm2_idx = group.node_indices[2];
        ASSERT_LT(mm1_idx, ordered.size());
        ASSERT_LT(sm_idx, ordered.size());
        ASSERT_LT(mm2_idx, ordered.size());
        const auto& mm1_node = ordered[mm1_idx];
        const auto& sm_node = ordered[sm_idx];
        const auto& mm2_node = ordered[mm2_idx];
        if (ov::as_type_ptr<const ov::op::v0::MatMul>(mm1_node) &&
            ov::as_type_ptr<const ov::op::v1::Softmax>(sm_node) &&
            ov::as_type_ptr<const ov::op::v0::MatMul>(mm2_node)) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionAttentionScaleMaskPlan) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto w1 = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                     ov::Shape{4, 4},
                                                     std::vector<float>(16, 0.5f));
    auto w2 = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                     ov::Shape{4, 4},
                                                     std::vector<float>(16, 0.25f));
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                        ov::Shape{1},
                                                        std::vector<float>{0.5f});
    auto mask = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                       ov::Shape{1, 4},
                                                       std::vector<float>(4, -1.0f));
    auto mm1 = std::make_shared<ov::op::v0::MatMul>(param, w1, false, false);
    auto scaled = std::make_shared<ov::op::v1::Multiply>(mm1, scale);
    auto add = std::make_shared<ov::op::v1::Add>(scaled, mask);
    auto sm = std::make_shared<ov::op::v1::Softmax>(add, 1);
    auto mm2 = std::make_shared<ov::op::v0::MatMul>(sm, w2, false, false);
    auto res = std::make_shared<ov::op::v0::Result>(mm2);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "attn_scale_mask");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
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
        const auto& mm1_node = ordered[mm1_idx];
        const auto& scale_node = ordered[scale_idx];
        const auto& add_node = ordered[add_idx];
        const auto& sm_node = ordered[sm_idx];
        const auto& mm2_node = ordered[mm2_idx];
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
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto w1 = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                     ov::Shape{4, 4},
                                                     std::vector<float>(16, 0.5f));
    auto w2 = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                     ov::Shape{4, 4},
                                                     std::vector<float>(16, 0.25f));
    auto scale_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16,
                                                              ov::Shape{1},
                                                              std::vector<ov::float16>{ov::float16(0.5f)});
    auto scale = std::make_shared<ov::op::v0::Convert>(scale_const, ov::element::f32);
    auto mm1 = std::make_shared<ov::op::v0::MatMul>(param, w1, false, false);
    auto scaled = std::make_shared<ov::op::v1::Multiply>(mm1, scale);
    auto sm = std::make_shared<ov::op::v1::Softmax>(scaled, 1);
    auto mm2 = std::make_shared<ov::op::v0::MatMul>(sm, w2, false, false);
    auto res = std::make_shared<ov::op::v0::Result>(mm2);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "attn_scale");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "AttentionScale" && group.kind != "AttentionScaleMask") {
            continue;
        }
        bool has_convert = false;
        for (const auto idx : group.node_indices) {
            ASSERT_LT(idx, ordered.size());
            has_convert = has_convert ||
                          static_cast<bool>(ov::as_type_ptr<const ov::op::v0::Convert>(ordered[idx]));
        }
        if (has_convert) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionAttentionScaleMaskPlanWithConvertedConstants) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto w1 = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                     ov::Shape{4, 4},
                                                     std::vector<float>(16, 0.5f));
    auto w2 = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                     ov::Shape{4, 4},
                                                     std::vector<float>(16, 0.25f));
    auto scale_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16,
                                                              ov::Shape{1},
                                                              std::vector<ov::float16>{ov::float16(0.5f)});
    auto mask_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16,
                                                             ov::Shape{1, 4},
                                                             std::vector<ov::float16>(4, ov::float16(-1.0f)));
    auto scale = std::make_shared<ov::op::v0::Convert>(scale_const, ov::element::f32);
    auto mask = std::make_shared<ov::op::v0::Convert>(mask_const, ov::element::f32);
    auto mm1 = std::make_shared<ov::op::v0::MatMul>(param, w1, false, false);
    auto scaled = std::make_shared<ov::op::v1::Multiply>(mm1, scale);
    auto add = std::make_shared<ov::op::v1::Add>(scaled, mask);
    auto sm = std::make_shared<ov::op::v1::Softmax>(add, 1);
    auto mm2 = std::make_shared<ov::op::v0::MatMul>(sm, w2, false, false);
    auto res = std::make_shared<ov::op::v0::Result>(mm2);
    auto model =
        std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "attn_scale_mask_convert");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
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
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 4, 4});
    auto perm = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                       ov::Shape{4},
                                                       std::vector<int64_t>{0, 1, 3, 2});
    auto transposed = std::make_shared<ov::op::v1::Transpose>(param, perm);
    auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto split = std::make_shared<ov::op::v1::Split>(transposed, axis, 3);
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                        ov::Shape{1},
                                                        std::vector<float>{0.5f});
    auto qk = std::make_shared<ov::op::v0::MatMul>(split->output(0), split->output(1), false, true);
    auto scaled = std::make_shared<ov::op::v1::Multiply>(qk, scale);
    auto softmax = std::make_shared<ov::op::v1::Softmax>(scaled, 3);
    auto attn = std::make_shared<ov::op::v0::MatMul>(softmax, split->output(2), false, false);
    auto post_transpose = std::make_shared<ov::op::v1::Transpose>(attn, perm);
    auto shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                        ov::Shape{3},
                                                        std::vector<int64_t>{1, 4, 4});
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(post_transpose, shape, false);
    auto res = std::make_shared<ov::op::v0::Result>(reshaped);
    auto model =
        std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "attn_layout_window");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if ((group.kind != "AttentionScale" && group.kind != "AttentionScaleMask") ||
            group.node_indices.size() < 7) {
            continue;
        }
        bool has_split = false;
        bool has_pre_transpose = false;
        bool has_post_reshape = false;
        for (const auto idx : group.node_indices) {
            ASSERT_LT(idx, ordered.size());
            const auto& node = ordered[idx];
            has_split = has_split || static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Split>(node));
            has_pre_transpose = has_pre_transpose || static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Transpose>(node));
            has_post_reshape = has_post_reshape || static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Reshape>(node));
        }
        if (has_split && has_pre_transpose && has_post_reshape) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionAttentionPreScalePlanExpandsLayoutWindow) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 4, 4});
    auto perm = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                       ov::Shape{4},
                                                       std::vector<int64_t>{0, 1, 3, 2});
    auto transposed = std::make_shared<ov::op::v1::Transpose>(param, perm);
    auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto split = std::make_shared<ov::op::v1::Split>(transposed, axis, 3);
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                        ov::Shape{1},
                                                        std::vector<float>{0.5f});
    auto pre_scaled_q = std::make_shared<ov::op::v1::Multiply>(split->output(1), scale);
    auto qk = std::make_shared<ov::op::v0::MatMul>(split->output(0), pre_scaled_q, false, true);
    auto softmax = std::make_shared<ov::op::v1::Softmax>(qk, 3);
    auto attn = std::make_shared<ov::op::v0::MatMul>(split->output(2), softmax, false, false);
    auto post_transpose = std::make_shared<ov::op::v1::Transpose>(attn, perm);
    auto shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                        ov::Shape{3},
                                                        std::vector<int64_t>{1, 4, 4});
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(post_transpose, shape, false);
    auto res = std::make_shared<ov::op::v0::Result>(reshaped);
    auto model =
        std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "attn_prescale_layout_window");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "AttentionScale" || group.node_indices.size() < 8) {
            continue;
        }
        bool has_split = false;
        bool has_pre_scale = false;
        bool has_matmul = false;
        bool has_post_reshape = false;
        for (const auto idx : group.node_indices) {
            ASSERT_LT(idx, ordered.size());
            const auto& node = ordered[idx];
            has_split = has_split || static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Split>(node));
            has_pre_scale = has_pre_scale || static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Multiply>(node));
            has_matmul = has_matmul || static_cast<bool>(ov::as_type_ptr<const ov::op::v0::MatMul>(node));
            has_post_reshape = has_post_reshape || static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Reshape>(node));
        }
        if (has_split && has_pre_scale && has_matmul && has_post_reshape) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST(GfxMlir, MatMulBuilderProducesModule) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, true);

    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
    auto module = ov::gfx_plugin::build_mlir_for_node(matmul, ctx);
    ASSERT_TRUE(module);
    auto func = module.lookupSymbol<mlir::func::FuncOp>("matmul_main");
    ASSERT_TRUE(static_cast<bool>(func));
    const auto func_type = func.getFunctionType();
    ASSERT_EQ(func_type.getNumInputs(), 2u);
    ASSERT_EQ(func_type.getNumResults(), 1u);
}

TEST(GfxMlir, MatMulMpsrtMetadataAnnotatesPlacementAndTensorDescriptors) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, true);

    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
    auto module = ov::gfx_plugin::build_mlir_for_node(matmul, ctx);
    ASSERT_TRUE(module);

    ASSERT_FALSE(static_cast<bool>(module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops")));
    const auto to_i64_dims = [](const ov::Shape& shape) {
        std::vector<int64_t> dims;
        dims.reserve(shape.size());
        for (const auto dim : shape) {
            dims.push_back(static_cast<int64_t>(dim));
        }
        return dims;
    };
    ov::gfx_plugin::GfxAppleMpsGemmProgramDesc program_desc{};
    program_desc.lhs = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
        to_i64_dims(lhs->get_shape()),
        ov::element::f32,
        ov::gfx_plugin::GfxStageStorageKind::Matrix,
        ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
    program_desc.rhs = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
        to_i64_dims(rhs->get_shape()),
        ov::element::f32,
        ov::gfx_plugin::GfxStageStorageKind::Matrix,
        ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
    program_desc.output = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc(
        to_i64_dims(matmul->get_output_shape(0)),
        ov::element::f32,
        ov::gfx_plugin::GfxStageStorageKind::Matrix,
        ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
    program_desc.gemm.transpose_rhs = 1;
    program_desc.gemm.alpha = 1.0f;
    const auto materialized = ov::gfx_plugin::materialize_apple_mps_gemm_program(module,
                                                                                program_desc);
    ASSERT_TRUE(materialized.valid);
    ASSERT_TRUE(materialized.typed_program_materialized);
    ASSERT_TRUE(static_cast<bool>(module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops")));
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.apple.pipeline.program.kind").str(),
              "single_stage");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.apple.pipeline.program.record_key").str(),
              "mps_gemm_model|MatMul");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.apple.pipeline.pass4.name").str(),
              "gfx-apple-vendor-descriptor");

    ASSERT_FALSE(module->hasAttr("gfx.backend"));
    ASSERT_FALSE(module->hasAttr("gfx.storage"));
    ASSERT_FALSE(module->hasAttr("gfx.stage_type"));
    ASSERT_FALSE(module->hasAttr("gfx.uses_vendor_primitive"));
    ASSERT_FALSE(module->hasAttr("gfx.uses_custom_kernel"));
    ASSERT_FALSE(module->hasAttr("gfx.specialization_key"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage_kind"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage_record_key"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.gemm.transpose_rhs"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.input_count"));

    ov::gfx_plugin::GfxMpsrtModuleStagePlan extracted;
    ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_stage_plan(module, extracted));
    ASSERT_TRUE(extracted.valid);
    ASSERT_EQ(extracted.stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
    ASSERT_EQ(extracted.stage.domain, ov::gfx_plugin::GfxStageBackendDomain::AppleMps);
    ASSERT_EQ(extracted.stage.kernel_name, "mps_gemm");
    ASSERT_EQ(extracted.stage.builder_symbol, "ovgfx_mpsrt_encode_gemm");
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
    ASSERT_EQ(extracted.stage_record_key,
              "mps_gemm|apple_mps|matrix|matrix|row_major|MatMul|apple_mps:matrix:MatMul|"
              "gemm:ta0:tb1:alpha1.000000:beta0.000000");
    ASSERT_EQ(extracted.stage.gemm_desc.transpose_lhs, 0u);
    ASSERT_EQ(extracted.stage.gemm_desc.transpose_rhs, 1u);
    ASSERT_EQ(extracted.stage.gemm_desc.alpha, 1.0f);

    ov::gfx_plugin::GfxMpsrtProgram program;
    ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_program(module, program));
    ASSERT_TRUE(program.valid);
    ASSERT_FALSE(program.multi_stage);
    ASSERT_EQ(program.record_key, "mps_gemm_model|MatMul");
    ASSERT_EQ(program.external_buffer_abi.buffer_roles,
              std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                  {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                   ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                   ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput}));
    ASSERT_EQ(extracted.stage.gemm_desc.beta, 0.0f);
    ASSERT_EQ(extracted.inputs.size(), 2u);
    ASSERT_EQ(extracted.outputs.size(), 1u);
    ASSERT_EQ(extracted.inputs[0].storage, ov::gfx_plugin::GfxMpsrtStorage::Matrix);
    ASSERT_EQ(extracted.inputs[0].layout, ov::gfx_plugin::GfxMpsrtLayout::RowMajor);
    ASSERT_EQ(extracted.inputs[0].dtype, ov::gfx_plugin::GfxMpsrtDType::F32);
    ASSERT_EQ(extracted.inputs[0].matrix_rows, 4u);
    ASSERT_EQ(extracted.inputs[0].matrix_columns, 2u);
    ASSERT_EQ(extracted.inputs[0].matrix_row_bytes, 8u);
    ASSERT_TRUE(extracted.stage.stage_manifest.valid);
    ASSERT_EQ(extracted.stage.stage_manifest.stage_family, ov::gfx_plugin::GfxKernelStageFamily::Gemm);
    ASSERT_EQ(extracted.stage.stage_manifest.backend_domain,
              ov::gfx_plugin::GfxKernelBackendDomain::AppleMps);
    ASSERT_EQ(extracted.stage.stage_manifest.execution_kind,
              ov::gfx_plugin::GfxKernelExecutionKind::VendorPrimitive);
    ASSERT_EQ(extracted.stage.stage_manifest.storage, ov::gfx_plugin::GfxKernelStorageKind::Matrix);
    ASSERT_FALSE(extracted.stage.stage_manifest.custom_kernel.valid);
    ASSERT_EQ(extracted.stage.input_storage, ov::gfx_plugin::GfxMpsrtStorage::Matrix);
    ASSERT_EQ(extracted.stage.output_storage, ov::gfx_plugin::GfxMpsrtStorage::Matrix);
    ASSERT_EQ(extracted.stage.layout, ov::gfx_plugin::GfxMpsrtLayout::RowMajor);
    ASSERT_EQ(extracted.stage_record_key,
              "mps_gemm|apple_mps|matrix|matrix|row_major|MatMul|apple_mps:matrix:MatMul|"
              "gemm:ta0:tb1:alpha1.000000:beta0.000000");
    ASSERT_EQ(extracted.inputs.size(), 2u);
    ASSERT_EQ(extracted.outputs.size(), 1u);
    ASSERT_EQ(extracted.inputs[0].dtype, ov::gfx_plugin::GfxMpsrtDType::F32);
    ASSERT_EQ(extracted.inputs[0].matrix_rows, 4u);
    ASSERT_EQ(extracted.inputs[0].matrix_columns, 2u);
    ASSERT_EQ(extracted.outputs[0].byte_length, 1u * 4u * 4u * 4u);

    const auto module_builder_plan = ov::gfx_plugin::build_module_mpsrt_builder_plan(module);
    ASSERT_TRUE(module_builder_plan.valid);
    const auto& builder_plan = module_builder_plan.builder_plan;
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
    ASSERT_EQ(builder_plan.records[0].kind, ov::gfx_plugin::GfxMpsrtBuilderRecordKind::ModelBegin);
    ASSERT_EQ(builder_plan.records[0].symbol, "ovgfx_mpsrt_model_begin");
    ASSERT_EQ(builder_plan.records[1].kind, ov::gfx_plugin::GfxMpsrtBuilderRecordKind::AddTensor);
    ASSERT_EQ(builder_plan.records[1].symbol, "ovgfx_mpsrt_add_tensor");
    ASSERT_EQ(builder_plan.records[1].value, 0u);
    ASSERT_EQ(builder_plan.records[1].tensor_descs[0].storage,
              static_cast<uint32_t>(ov::gfx_plugin::GfxMpsrtStorage::Matrix));
    ASSERT_EQ(builder_plan.records[3].kind, ov::gfx_plugin::GfxMpsrtBuilderRecordKind::EncodeStage);
    ASSERT_EQ(builder_plan.records[3].symbol, "ovgfx_mpsrt_encode_gemm");
    ASSERT_EQ(builder_plan.records[3].inputs.size(), 2u);
    ASSERT_EQ(builder_plan.records[3].outputs.size(), 1u);
    ASSERT_EQ(builder_plan.records[3].outputs[0], 2u);
    ASSERT_EQ(builder_plan.records[3].gemm_desc.transpose_lhs, 0u);
    ASSERT_EQ(builder_plan.records[3].gemm_desc.transpose_rhs, 1u);
    ASSERT_EQ(builder_plan.records[3].tensor_descs[0].byte_length, 1u * 4u * 4u * 4u);
    ASSERT_EQ(builder_plan.records[4].kind, ov::gfx_plugin::GfxMpsrtBuilderRecordKind::ModelEnd);
    ASSERT_EQ(builder_plan.records[4].symbol, "ovgfx_mpsrt_model_end");
}

TEST(GfxMlir, ConvMpsrtMetadataAnnotatesVendorDescriptorFromOpenVINONode) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 16, 32, 32});
    auto weights = ov::op::v0::Constant::create(ov::element::f16,
                                                ov::Shape{8, 16, 3, 3},
                                                std::vector<float>(8 * 16 * 3 * 3, 1.f));
    auto conv = std::make_shared<ov::op::v1::Convolution>(input,
                                                          weights,
                                                          ov::Strides{2, 1},
                                                          ov::CoordinateDiff{1, 2},
                                                          ov::CoordinateDiff{3, 4},
                                                          ov::Strides{1, 2});

    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
    auto module = ov::gfx_plugin::build_mlir_for_node(conv, ctx);
    ASSERT_TRUE(module);

    const auto plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                     ov::gfx_plugin::GpuBackend::Metal,
                                                                     "Convolution",
                                                                     conv,
                                                                     ov::element::f16,
                                                                     /*has_bias=*/false,
                                                                     /*has_activation=*/false,
                                                                     /*has_batchnorm=*/false,
                                                                     {});
    const auto lowering_kind = ov::gfx_plugin::annotate_module_with_conv_mpsrt_plan(module,
                                                                                   plan,
                                                                                   conv,
                                                                                   "Convolution");
    ASSERT_EQ(lowering_kind, ov::gfx_plugin::GfxConvMpsrtLoweringKind::MpsConv2D);

    ASSERT_FALSE(module->hasAttr("gfx.backend"));
    ASSERT_FALSE(module->hasAttr("gfx.storage"));
    ASSERT_FALSE(module->hasAttr("gfx.stage_type"));
    ASSERT_FALSE(module->hasAttr("gfx.uses_vendor_primitive"));
    ASSERT_FALSE(module->hasAttr("gfx.uses_custom_kernel"));
    ASSERT_FALSE(module->hasAttr("gfx.specialization_key"));
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.stage_family").str(),
              "convolution");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.backend_domain").str(),
              "apple_mps");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.execution_kind").str(),
              "vendor_primitive");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.storage").str(), "image");
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage_kind"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage_record_key"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.conv2d.groups"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.input1.storage"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.storage_bridge_count"));

    ov::gfx_plugin::GfxMpsrtModuleStagePlan extracted;
    ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_stage_plan(module, extracted));
    ASSERT_TRUE(extracted.valid);
    ASSERT_EQ(extracted.stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MPSConv2D);
    ASSERT_EQ(extracted.stage.builder_symbol, "ovgfx_mpsrt_encode_conv2d");
    ASSERT_EQ(extracted.stage_record_key,
              "mps_conv2d|apple_mps|image|image|nhwc4|Convolution|apple_mps:image:Convolution|"
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
    ASSERT_EQ(extracted.inputs[1].storage, ov::gfx_plugin::GfxMpsrtStorage::Buffer);
    ASSERT_EQ(extracted.inputs[1].flags, ov::gfx_plugin::GfxMpsrtTensorFlagConst);

    const auto module_builder_plan = ov::gfx_plugin::build_module_mpsrt_builder_plan(module);
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
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].stage_kind,
              ov::gfx_plugin::GfxMpsrtStageKind::MPSConv2D);
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].conv2d_desc.strides[0], 2u);
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].conv2d_desc.dilations[1], 2u);
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].conv2d_desc.pads[3], 4u);
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

    ASSERT_TRUE(ov::gfx_plugin::materialize_module_mpsrt_ops_from_stage_plan(module, extracted));
    auto ops_func = module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops");
    ASSERT_TRUE(static_cast<bool>(ops_func));

    size_t to_image_count = 0;
    size_t conv_count = 0;
    size_t to_buffer_count = 0;
    mlir::Operation* to_image_op = nullptr;
    mlir::Operation* to_buffer_op = nullptr;
    ops_func.walk([&](mlir::Operation* op) {
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
    ASSERT_TRUE(to_image_op->getAttrOfType<mlir::BoolAttr>(
                             "gfx.mpsrt.storage_bridge.generated")
                    .getValue());
    ASSERT_EQ(to_image_op->getAttrOfType<mlir::StringAttr>(
                             "gfx.mpsrt.storage_bridge.direction")
                  .str(),
              "buffer_to_image");
    ASSERT_EQ(to_image_op->getAttrOfType<mlir::StringAttr>(
                             "gfx.mpsrt.storage_bridge.target_storage")
                  .str(),
              "image");
    ASSERT_EQ(to_buffer_op->getAttrOfType<mlir::StringAttr>(
                              "gfx.mpsrt.storage_bridge.direction")
                  .str(),
              "image_to_buffer");
    ASSERT_EQ(to_buffer_op->getAttrOfType<mlir::StringAttr>(
                              "gfx.mpsrt.storage_bridge.target_storage")
                  .str(),
              "buffer");
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));

    ov::gfx_plugin::GfxMpsrtProgram typed_program;
    ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_ops_program(module, typed_program));
    ASSERT_TRUE(typed_program.has_storage_bridges);
    ASSERT_EQ(typed_program.storage_bridges.size(), 2u);
}

TEST(GfxMlir, GroupConvMpsrtMetadataDerivesStageTypeFromManifest) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4, 16, 16});
    auto weights = ov::op::v0::Constant::create(ov::element::f16,
                                                ov::Shape{4, 1, 1, 3, 3},
                                                std::vector<float>(4 * 1 * 1 * 3 * 3, 1.f));
    auto group_conv = std::make_shared<ov::op::v1::GroupConvolution>(input,
                                                                     weights,
                                                                     ov::Strides{1, 1},
                                                                     ov::CoordinateDiff{1, 1},
                                                                     ov::CoordinateDiff{1, 1},
                                                                     ov::Strides{1, 1});

    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
    auto module = ov::gfx_plugin::build_mlir_for_node(group_conv, ctx);
    ASSERT_TRUE(module);

    const auto plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                     ov::gfx_plugin::GpuBackend::Metal,
                                                                     "GroupConvolution",
                                                                     group_conv,
                                                                     ov::element::f16,
                                                                     /*has_bias=*/false,
                                                                     /*has_activation=*/false,
                                                                     /*has_batchnorm=*/false,
                                                                     {});
    const auto lowering_kind = ov::gfx_plugin::annotate_module_with_conv_mpsrt_plan(module,
                                                                                   plan,
                                                                                   group_conv,
                                                                                   "GroupConv2D");
    ASSERT_EQ(lowering_kind, ov::gfx_plugin::GfxConvMpsrtLoweringKind::MpsGroupConv2D);
    ASSERT_FALSE(module->hasAttr("gfx.stage_type"));
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.stage_family").str(),
              "group_convolution");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.backend_domain").str(),
              "apple_mps");
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage_kind"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.conv2d.groups"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.input1.storage"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage_record_key"));

    ov::gfx_plugin::GfxMpsrtModuleStagePlan extracted;
    ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_stage_plan(module, extracted));
    ASSERT_TRUE(extracted.valid);
    ASSERT_EQ(extracted.stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MPSGroupConv2D);
    ASSERT_EQ(extracted.stage.stage_type, "GroupConvolution");
    ASSERT_EQ(extracted.stage.conv2d_desc.groups, 4u);
    ASSERT_EQ(extracted.inputs[1].storage, ov::gfx_plugin::GfxMpsrtStorage::Buffer);
    ASSERT_EQ(extracted.inputs[1].flags, ov::gfx_plugin::GfxMpsrtTensorFlagConst);
    ASSERT_EQ(extracted.stage_record_key,
              "mps_group_conv2d|apple_mps|image|image|nhwc4|GroupConvolution|"
              "apple_mps:image:GroupConvolution|conv2d:g4:s1x1:d1x1:p1,1,1,1");
    ASSERT_EQ(extracted.stage.stage_manifest.stage_family,
              ov::gfx_plugin::GfxKernelStageFamily::GroupConvolution);
    ASSERT_EQ(extracted.stage.stage_manifest.semantic_input_roles,
              std::vector<ov::gfx_plugin::GfxKernelBufferRole>(
                  {ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
                   ov::gfx_plugin::GfxKernelBufferRole::ConstTensor}));

    const auto module_builder_plan = ov::gfx_plugin::build_module_mpsrt_builder_plan(module);
    ASSERT_TRUE(module_builder_plan.valid);
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].stage_kind,
              ov::gfx_plugin::GfxMpsrtStageKind::MPSGroupConv2D);
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].conv2d_desc.groups, 4u);
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

    auto lhs = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc({1, 8, 16},
                                                          ov::element::f16,
                                                          ov::gfx_plugin::GfxStageStorageKind::Matrix,
                                                          ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
    auto rhs = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc({1, 16, 4},
                                                          ov::element::f16,
                                                          ov::gfx_plugin::GfxStageStorageKind::Matrix,
                                                          ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
    auto output = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc({1, 8, 4},
                                                             ov::element::f16,
                                                             ov::gfx_plugin::GfxStageStorageKind::Matrix,
                                                             ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
    const auto plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                     ov::gfx_plugin::GpuBackend::Metal,
                                                                     "MatMul",
                                                                     nullptr,
                                                                     ov::element::f16,
                                                                     /*has_bias=*/false,
                                                                     /*has_activation=*/false,
                                                                     /*has_batchnorm=*/false,
                                                                     {});
    auto stage = ov::gfx_plugin::gfx_mpsrt_make_stage_desc(plan, "MatMul");
    ASSERT_EQ(stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);

    ov::gfx_plugin::GfxMpsrtProgram program{};
    program.valid = true;
    program.record_key = "matrix_storage_bridge_model";
    program.inputs = {lhs, rhs};
    program.output_values = {2u};
    program.stages.push_back({stage,
                              ov::gfx_plugin::gfx_mpsrt_stage_record_key(stage),
                              {0u, 1u},
                              {2u},
                              {output}});
    program.has_storage_bridges = true;

    ov::gfx_plugin::GfxMpsrtStorageBridgeDesc bridge{};
    ASSERT_TRUE(ov::gfx_plugin::gfx_mpsrt_make_storage_bridge_desc(
        0u,
        ov::gfx_plugin::gfx_mpsrt_to_abi_desc(lhs),
        ov::gfx_plugin::GfxMpsrtStorageBridgeDirection::BufferToMatrix,
        bridge));
    program.storage_bridges.push_back(bridge);
    ASSERT_TRUE(ov::gfx_plugin::gfx_mpsrt_make_storage_bridge_desc(
        1u,
        ov::gfx_plugin::gfx_mpsrt_to_abi_desc(rhs),
        ov::gfx_plugin::GfxMpsrtStorageBridgeDirection::BufferToMatrix,
        bridge));
    program.storage_bridges.push_back(bridge);
    ASSERT_TRUE(ov::gfx_plugin::gfx_mpsrt_make_storage_bridge_desc(
        2u,
        ov::gfx_plugin::gfx_mpsrt_to_abi_desc(output),
        ov::gfx_plugin::GfxMpsrtStorageBridgeDirection::MatrixToBuffer,
        bridge));
    program.storage_bridges.push_back(bridge);

    ASSERT_TRUE(ov::gfx_plugin::materialize_module_mpsrt_ops(module, program));
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));
    auto ops_func = module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops");
    ASSERT_TRUE(static_cast<bool>(ops_func));

    std::vector<std::string> mpsrt_ops;
    for (auto& op : ops_func.getBody().front().getOperations()) {
        const auto name = op.getName().getStringRef();
        if (name.starts_with("gfx.mpsrt.")) {
            mpsrt_ops.push_back(name.str());
        }
    }
    ASSERT_EQ(mpsrt_ops,
              std::vector<std::string>({"gfx.mpsrt.to_matrix",
                                        "gfx.mpsrt.to_matrix",
                                        "gfx.mpsrt.gemm",
                                        "gfx.mpsrt.to_buffer",
                                        "gfx.mpsrt.return"}));

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
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 128, 256});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 256, 64});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, false);

    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
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
        module,
        desc,
        lhs->get_shape(),
        rhs->get_shape());

    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.model_stage_count"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage0.backend"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage1.backend"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage1.stage_manifest.kernel.entry_point"));
    ASSERT_EQ(module->getAttrOfType<mlir::IntegerAttr>("gfx.apple.pipeline.pass_boundary_count").getInt(), 7);
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.apple.pipeline.program.kind").str(),
              "multi_stage");
    ASSERT_EQ(module->getAttrOfType<mlir::IntegerAttr>("gfx.apple.pipeline.program.stage_count").getInt(), 2);
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.apple.pipeline.program.stage0.backend_domain").str(),
              "apple_mps");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.apple.pipeline.program.stage1.execution_kind").str(),
              "custom_kernel");

    ov::gfx_plugin::GfxMpsrtProgram extracted;
    ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_program(module, extracted));
    ASSERT_TRUE(extracted.valid);
    ASSERT_TRUE(extracted.multi_stage);
    ASSERT_EQ(extracted.record_key, "mps_gemm_plus_msl_epilogue_model|MatMul");
    ASSERT_EQ(extracted.inputs.size(), 2u);
    ASSERT_EQ(extracted.stages.size(), 2u);
    ASSERT_EQ(extracted.stages[0].stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
    ASSERT_EQ(extracted.stages[0].stage.domain, ov::gfx_plugin::GfxStageBackendDomain::AppleMps);
    ASSERT_EQ(extracted.stages[0].stage.gemm_desc.alpha, 1.0f);
    ASSERT_EQ(extracted.stages[1].stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
    ASSERT_EQ(extracted.stages[1].stage.domain, ov::gfx_plugin::GfxStageBackendDomain::AppleMsl);
    ASSERT_EQ(extracted.stages[1].stage.dispatch_entry_point, "eltwise_fused_buffer");
    ASSERT_EQ(extracted.stages[1].stage.dispatch_threads_per_threadgroup, 256u);
    ASSERT_TRUE(extracted.stages[1].stage.dispatch_precompiled_kernel_required);
    ASSERT_EQ(extracted.stages[1].inputs, std::vector<ov::gfx_plugin::GfxMpsrtValue>({2u}));
    ASSERT_EQ(extracted.stages[1].outputs, std::vector<ov::gfx_plugin::GfxMpsrtValue>({3u}));

    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.program.symbol"));
    ASSERT_FALSE(static_cast<bool>(module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_program")));
    auto generated_ops = module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops");
    ASSERT_TRUE(static_cast<bool>(generated_ops));
    ASSERT_TRUE(generated_ops->getAttrOfType<mlir::BoolAttr>("gfx.mpsrt.ops.generated").getValue());
    ASSERT_EQ(generated_ops->getAttrOfType<mlir::StringAttr>("gfx.mpsrt.ops.kind").str(),
              "multi_stage");
    ASSERT_EQ(generated_ops->getAttrOfType<mlir::IntegerAttr>("gfx.mpsrt.ops.stage_count").getInt(), 2);

    mlir::Builder stale_builder(module.getContext());
    module->setAttr("gfx.mpsrt.model_record_key", stale_builder.getStringAttr("legacy_attrs_are_not_primary"));
    ov::gfx_plugin::GfxMpsrtProgram program;
    ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_program(module, program));
    ASSERT_TRUE(program.valid);
    ASSERT_TRUE(program.multi_stage);
    ASSERT_EQ(program.record_key, "mps_gemm_plus_msl_epilogue_model|MatMul");
    ASSERT_EQ(program.inputs.size(), 2u);
    ASSERT_EQ(program.stages.size(), 2u);
    ASSERT_EQ(program.stages[0].stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
    ASSERT_EQ(program.stages[1].stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
    ASSERT_TRUE(program.external_buffer_abi.valid);
    ASSERT_EQ(program.external_buffer_abi.buffer_roles,
              std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                  {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                   ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                   ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput}));

    ov::gfx_plugin::GfxMpsrtBuilderPlan program_builder_plan;
    ASSERT_TRUE(ov::gfx_plugin::gfx_mpsrt_build_builder_plan_from_program(program, program_builder_plan));
    ASSERT_TRUE(program_builder_plan.valid);
    ASSERT_EQ(program_builder_plan.records.size(), 6u);
    ASSERT_EQ(program_builder_plan.records[3].stage_kind,
              ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
    ASSERT_EQ(program_builder_plan.records[4].stage_kind,
              ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);

    ASSERT_TRUE(ov::gfx_plugin::materialize_module_mpsrt_ops(module, program));
    auto ops_func = module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops");
    ASSERT_TRUE(static_cast<bool>(ops_func));
    ASSERT_TRUE(ops_func->getAttrOfType<mlir::BoolAttr>("gfx.mpsrt.ops.generated").getValue());
    std::vector<std::string> mpsrt_ops;
    ops_func.walk([&](mlir::Operation* op) {
        if (op->getAttrOfType<mlir::BoolAttr>("gfx.mpsrt.op.generated")) {
            mpsrt_ops.push_back(op->getName().getStringRef().str());
        }
    });
    ASSERT_EQ(mpsrt_ops,
              std::vector<std::string>({"gfx.mpsrt.gemm", "gfx.mpsrt.dispatch", "gfx.mpsrt.return"}));
    mlir::Operation* gemm_op = nullptr;
    mlir::Operation* dispatch_op = nullptr;
    ops_func.walk([&](mlir::Operation* op) {
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
    ASSERT_FALSE(static_cast<bool>(module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_op_gemm")));
    ASSERT_FALSE(static_cast<bool>(module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_op_dispatch")));
    ASSERT_EQ(gemm_op->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.backend_domain").str(),
              "apple_mps");
    ASSERT_EQ(gemm_op->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.execution_kind").str(),
              "vendor_primitive");
    ASSERT_EQ(gemm_op->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.storage").str(),
              "matrix");
    ASSERT_EQ(dispatch_op->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.backend_domain").str(),
              "apple_msl");
    ASSERT_EQ(dispatch_op->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.execution_kind").str(),
              "custom_kernel");
    ASSERT_EQ(dispatch_op->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.storage").str(),
              "buffer");
    ASSERT_EQ(dispatch_op->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.kernel.entry_point").str(),
              "eltwise_fused_buffer");
    ASSERT_TRUE(mlir::succeeded(mlir::verify(ops_func)));

    mlir::Builder op_builder(module.getContext());
    const auto dispatch_entry_point =
        dispatch_op->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.kernel.entry_point");
    dispatch_op->removeAttr("gfx.stage_manifest.kernel.entry_point");
    ASSERT_TRUE(mlir::failed(mlir::verify(ops_func)));
    dispatch_op->setAttr("gfx.stage_manifest.kernel.entry_point", dispatch_entry_point);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(ops_func)));

    dispatch_op->setAttr("gfx.mpsrt.op.stage.stage_manifest.backend_domain",
                         op_builder.getStringAttr("apple_mps"));
    dispatch_op->setAttr("gfx.mpsrt.op.stage.stage_manifest.execution_kind",
                         op_builder.getStringAttr("vendor_primitive"));
    ov::gfx_plugin::GfxMpsrtProgram ops_program;
    ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_ops_program(module, ops_program));
    ASSERT_TRUE(ops_program.valid);
    ASSERT_TRUE(ops_program.multi_stage);
    ASSERT_EQ(ops_program.record_key, "mps_gemm_plus_msl_epilogue_model|MatMul");
    ASSERT_EQ(ops_program.stages.size(), 2u);
    ASSERT_EQ(ops_program.stages[0].stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
    ASSERT_EQ(ops_program.stages[1].stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
    ASSERT_EQ(ops_program.stages[1].stage.domain, ov::gfx_plugin::GfxStageBackendDomain::AppleMsl);
    ASSERT_TRUE(ops_program.stages[1].stage.uses_custom_kernel);
    dispatch_op->setAttr("gfx.mpsrt.op.stage_index", op_builder.getI32IntegerAttr(0));
    ov::gfx_plugin::GfxMpsrtProgram invalid_ops_program;
    ASSERT_FALSE(ov::gfx_plugin::read_module_mpsrt_ops_program(module, invalid_ops_program));
    dispatch_op->setAttr("gfx.mpsrt.op.stage_index", op_builder.getI32IntegerAttr(1));

    const auto module_builder_plan = ov::gfx_plugin::build_module_mpsrt_builder_plan(module);
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
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].stage_kind,
              ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
    ASSERT_EQ(module_builder_plan.builder_plan.records[4].stage_kind,
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

    auto lhs = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc({1, 8, 16},
                                                          ov::element::f16,
                                                          ov::gfx_plugin::GfxStageStorageKind::Matrix,
                                                          ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
    auto rhs = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc({1, 16, 4},
                                                          ov::element::f16,
                                                          ov::gfx_plugin::GfxStageStorageKind::Matrix,
                                                          ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);
    auto gemm_output = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc({1, 8, 4},
                                                                  ov::element::f16,
                                                                  ov::gfx_plugin::GfxStageStorageKind::Matrix);
    auto output = ov::gfx_plugin::gfx_mpsrt_make_tensor_desc({1, 8, 4},
                                                             ov::element::f16,
                                                             ov::gfx_plugin::GfxStageStorageKind::Buffer,
                                                             ov::gfx_plugin::GfxMpsrtTensorFlagExternalIo);

    const auto plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                     ov::gfx_plugin::GpuBackend::Metal,
                                                                     "MatMul",
                                                                     nullptr,
                                                                     ov::element::f16,
                                                                     /*has_bias=*/false,
                                                                     /*has_activation=*/false,
                                                                     /*has_batchnorm=*/false,
                                                                     {});
    auto gemm_stage = ov::gfx_plugin::gfx_mpsrt_make_stage_desc(plan, "MatMul");
    gemm_stage.gemm_desc.alpha = 1.0f;

    ov::gfx_plugin::GfxMpsrtStageDesc epilogue_stage{};
    epilogue_stage.kind = ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch;
    epilogue_stage.domain = ov::gfx_plugin::GfxStageBackendDomain::AppleMsl;
    epilogue_stage.input_storage = ov::gfx_plugin::GfxMpsrtStorage::Buffer;
    epilogue_stage.output_storage = ov::gfx_plugin::GfxMpsrtStorage::Buffer;
    epilogue_stage.layout = ov::gfx_plugin::GfxMpsrtLayout::Linear;
    epilogue_stage.uses_custom_kernel = true;
    epilogue_stage.stage_type = "MatMulEpilogue";
    epilogue_stage.builder_symbol = ov::gfx_plugin::gfx_mpsrt_builder_symbol(epilogue_stage.kind);
    epilogue_stage.specialization_key = "apple_msl:buffer:MatMulEpilogue";
    epilogue_stage.stage_manifest = ov::gfx_plugin::make_gfx_custom_kernel_stage_manifest(
        ov::gfx_plugin::GfxKernelStageFamily::Eltwise,
        ov::gfx_plugin::GfxKernelBackendDomain::AppleMsl,
        ov::gfx_plugin::GfxKernelStorageKind::Buffer,
        epilogue_stage.specialization_key,
        ov::gfx_plugin::make_gfx_custom_kernel_manifest(
            "eltwise_fused_buffer",
            static_cast<uint32_t>(ov::gfx_plugin::GfxKernelFamily::EltwiseFusedBuffer),
            "eltwise_fused_buffer",
            ov::gfx_plugin::make_gfx_kernel_roles_abi(
                {ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
                 ov::gfx_plugin::GfxKernelBufferRole::TensorOutput}),
            ov::gfx_plugin::make_gfx_kernel_linear_dispatch_policy(
                256,
                /*precompiled_binary_required=*/true)));

    ov::gfx_plugin::GfxAppleMpsrtProgramPlan program_plan{};
    program_plan.record_key = "explicit_apple_program_plan";
    program_plan.inputs = {lhs, rhs};
    program_plan.output_values = {3u};
    program_plan.stages.push_back({gemm_stage,
                                   {0u, 1u},
                                   {2u},
                                   {gemm_output}});
    program_plan.stages.push_back({epilogue_stage,
                                   {2u},
                                   {3u},
                                   {output}});

    const auto materialized =
        ov::gfx_plugin::materialize_apple_mpsrt_program_plan(module, program_plan);
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
    ASSERT_EQ(roundtrip.stages[1].stage.dispatch_entry_point, "eltwise_fused_buffer");
    ASSERT_EQ(roundtrip.stages[1].stage.dispatch_threads_per_threadgroup, 256u);

    auto invalid_plan = program_plan;
    invalid_plan.stages[0].stage.domain = ov::gfx_plugin::GfxStageBackendDomain::Spirv;
    auto invalid_module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    const auto rejected =
        ov::gfx_plugin::materialize_apple_mpsrt_program_plan(invalid_module, invalid_plan);
    ASSERT_FALSE(rejected.valid);
    ASSERT_FALSE(static_cast<bool>(invalid_module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops")));
}

TEST(GfxMlir, MatMulMpsrtLoweringEntryPointSelectsSingleAndMultiStagePlans) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 128, 256});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 256, 64});
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

    const auto plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                     ov::gfx_plugin::GpuBackend::Metal,
                                                                     "MatMul",
                                                                     matmul,
                                                                     ov::element::f16,
                                                                     /*has_bias=*/false,
                                                                     /*has_activation=*/false,
                                                                     /*has_batchnorm=*/false,
                                                                     {});
    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
    auto gemm_module = ov::gfx_plugin::build_mlir_for_node(matmul, ctx);
    ASSERT_TRUE(gemm_module);
    const auto gemm_lowering = ov::gfx_plugin::annotate_module_with_matmul_mpsrt_plan(
        gemm_module,
        plan,
        desc,
        lhs->get_shape(),
        rhs->get_shape());
    ASSERT_EQ(gemm_lowering, ov::gfx_plugin::GfxMatMulMpsrtLoweringKind::MpsGemm);

    ov::gfx_plugin::GfxMpsrtModuleStagePlan stage_plan;
    ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_stage_plan(gemm_module, stage_plan));
    ASSERT_EQ(stage_plan.stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
    ov::gfx_plugin::GfxMpsrtProgram gemm_program;
    ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_program(gemm_module, gemm_program));
    ASSERT_FALSE(gemm_program.multi_stage);
    ASSERT_EQ(gemm_program.record_key, "mps_gemm_model|MatMul");
    ASSERT_EQ(gemm_module->getAttrOfType<mlir::StringAttr>("gfx.apple.pipeline.program.kind").str(),
              "single_stage");
    ASSERT_EQ(gemm_module->getAttrOfType<mlir::StringAttr>("gfx.apple.pipeline.program.stage0.kind").str(),
              "mps_gemm");
    const auto generic_gemm_source_plan =
        ov::gfx_plugin::make_mpsrt_kernel_source_plan_from_module(gemm_module);
    ASSERT_TRUE(generic_gemm_source_plan.valid());
    ASSERT_EQ(generic_gemm_source_plan.kind, ov::gfx_plugin::GfxMpsrtKernelSourcePlanKind::SingleStage);
    ASSERT_EQ(generic_gemm_source_plan.first_stage_kind, ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
    ASSERT_EQ(generic_gemm_source_plan.last_stage_kind, ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
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
    const auto epilogue_lowering = ov::gfx_plugin::annotate_module_with_matmul_mpsrt_plan(
        epilogue_module,
        plan,
        epilogue_desc,
        lhs->get_shape(),
        rhs->get_shape());
    ASSERT_EQ(epilogue_lowering, ov::gfx_plugin::GfxMatMulMpsrtLoweringKind::MpsGemmWithMslEpilogue);

    ov::gfx_plugin::GfxMpsrtProgram multi_stage;
    ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_program(epilogue_module, multi_stage));
    ASSERT_TRUE(multi_stage.multi_stage);
    ASSERT_EQ(multi_stage.stages.size(), 2u);
    ASSERT_EQ(multi_stage.stages[0].stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
    ASSERT_EQ(multi_stage.stages[1].stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
    ASSERT_EQ(multi_stage.stages[1].stage.stage_manifest.execution_kind,
              ov::gfx_plugin::GfxKernelExecutionKind::CustomKernel);

    auto source_module = ov::gfx_plugin::build_mlir_for_node(matmul, ctx);
    ASSERT_TRUE(source_module);
    const auto source_plan = ov::gfx_plugin::lower_matmul_module_to_mpsrt_kernel_source(
        source_module,
        plan,
        epilogue_desc,
        lhs->get_shape(),
        rhs->get_shape());
    ASSERT_TRUE(source_plan.valid());
    ASSERT_TRUE(source_plan.requires_mpsrt_model);
    ASSERT_EQ(source_plan.lowering, ov::gfx_plugin::GfxMatMulMpsrtLoweringKind::MpsGemmWithMslEpilogue);
    ASSERT_TRUE(source_plan.mpsrt_plan.valid());
    ASSERT_EQ(source_plan.mpsrt_plan.kind, ov::gfx_plugin::GfxMpsrtKernelSourcePlanKind::MultiStage);
    ASSERT_EQ(source_plan.mpsrt_plan.first_stage_kind, ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
    ASSERT_EQ(source_plan.mpsrt_plan.last_stage_kind, ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
    ASSERT_EQ(source_plan.source.entry_point, "eltwise_fused_buffer");
    ASSERT_FALSE(source_plan.source.msl_source.empty());
    ASSERT_EQ(source_plan.source.signature.arg_count, 4u);
    ASSERT_EQ(source_plan.source.signature.output_arg_count, 1u);

    auto fallback_desc = desc;
    fallback_desc.batch = 3;
    fallback_desc.batch_a = 2;
    fallback_desc.batch_b = 3;
    const auto fallback_source_plan = ov::gfx_plugin::lower_matmul_node_to_metal_kernel_source(
        ctx,
        nullptr,
        matmul,
        fallback_desc,
        lhs->get_shape(),
        rhs->get_shape());
    ASSERT_TRUE(fallback_source_plan.valid());
    ASSERT_EQ(fallback_source_plan.kind,
              ov::gfx_plugin::GfxMatMulMetalKernelSourcePlanKind::MslFallback);
    ASSERT_FALSE(fallback_source_plan.uses_mpsrt_gemm());
    ASSERT_TRUE(fallback_source_plan.requires_mpsrt_model);
    ASSERT_EQ(fallback_source_plan.source.entry_point, "matmul_buffer");
    ASSERT_EQ(fallback_source_plan.source.signature.arg_count, 3u);
    ASSERT_EQ(fallback_source_plan.source.signature.output_arg_count, 1u);

    ov::gfx_plugin::GfxMpsrtModuleStagePlan fallback_stage_plan;
    ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_stage_plan(fallback_source_plan.source.module,
                                                             fallback_stage_plan));
    ASSERT_EQ(fallback_stage_plan.stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
    ASSERT_EQ(fallback_stage_plan.stage.stage_manifest.stage_family,
              ov::gfx_plugin::GfxKernelStageFamily::Gemm);
    ASSERT_EQ(fallback_stage_plan.stage.stage_manifest.backend_domain,
              ov::gfx_plugin::GfxKernelBackendDomain::AppleMsl);
    ASSERT_TRUE(fallback_stage_plan.stage.stage_manifest.custom_kernel.valid);
    ASSERT_EQ(fallback_stage_plan.stage.stage_manifest.custom_kernel.kernel_family,
              "matmul_buffer");
    ASSERT_TRUE(fallback_stage_plan.stage.stage_manifest.custom_kernel.external_buffer_abi.valid);
    EXPECT_FALSE(fallback_stage_plan.stage.stage_manifest.custom_kernel.external_buffer_abi.tail_outputs);
    EXPECT_EQ(fallback_stage_plan.stage.stage_manifest.custom_kernel.external_buffer_abi.roles,
              std::vector<ov::gfx_plugin::GfxKernelBufferRole>(
                  {ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
                   ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
                   ov::gfx_plugin::GfxKernelBufferRole::TensorOutput}));

}

TEST(GfxMlir, AppleMpsrtRuntimeAbiPipelineMaterializesMultiStageBuilderRecords) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 128, 256});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 256, 64});
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

    const auto plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                     ov::gfx_plugin::GpuBackend::Metal,
                                                                     "MatMul",
                                                                     matmul,
                                                                     ov::element::f16,
                                                                     /*has_bias=*/true,
                                                                     /*has_activation=*/true,
                                                                     /*has_batchnorm=*/false,
                                                                     {});
    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
    auto module = ov::gfx_plugin::build_mlir_for_node(matmul, ctx);
    ASSERT_TRUE(module);
    const auto lowering = ov::gfx_plugin::annotate_module_with_matmul_mpsrt_plan(module,
                                                                                plan,
                                                                                desc,
                                                                                lhs->get_shape(),
                                                                                rhs->get_shape());
    ASSERT_EQ(lowering, ov::gfx_plugin::GfxMatMulMpsrtLoweringKind::MpsGemmWithMslEpilogue);
    ASSERT_FALSE(static_cast<bool>(module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_program")));
    ASSERT_TRUE(static_cast<bool>(module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops")));
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.apple.pipeline.program.kind").str(),
              "multi_stage");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.apple.pipeline.program.stage0.kind").str(),
              "mps_gemm");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.apple.pipeline.program.stage1.kind").str(),
              "msl_dispatch");
    ov::gfx_plugin::GfxMpsrtProgram program;
    ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_program(module, program));
    ASSERT_EQ(program.stages.size(), 2u);
    const auto& epilogue_abi = program.stages[1].stage.stage_manifest.custom_kernel.external_buffer_abi;
    ASSERT_TRUE(epilogue_abi.valid);
    ASSERT_FALSE(epilogue_abi.tail_outputs);
    ASSERT_EQ(epilogue_abi.roles,
              std::vector<ov::gfx_plugin::GfxKernelBufferRole>(
                  {ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
                   ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
                   ov::gfx_plugin::GfxKernelBufferRole::TensorOutput}));
    module->removeAttr("gfx.mpsrt.stage_kind");
    module->removeAttr("gfx.mpsrt.model_record_key");
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage_kind"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.model_record_key"));

    mlir::PassManager pm(module.getContext());
    ov::gfx_plugin::populate_gfx_apple_mpsrt_runtime_abi_pipeline(pm);
    ASSERT_TRUE(mlir::succeeded(pm.run(module)));

    ASSERT_TRUE(module->getAttrOfType<mlir::BoolAttr>("gfx.mpsrt.runtime_abi.valid").getValue());
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.mpsrt.runtime_abi.kind").str(), "multi_stage");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.mpsrt.runtime_abi.record_key").str(),
              "mps_gemm_plus_msl_epilogue_model|MatMul");
    ASSERT_EQ(module->getAttrOfType<mlir::IntegerAttr>("gfx.mpsrt.runtime_abi.record_count").getInt(), 7);
    ASSERT_EQ(module->getAttrOfType<mlir::IntegerAttr>("gfx.mpsrt.runtime_abi.external_buffer_count").getInt(), 4);
    ASSERT_EQ(module->getAttrOfType<mlir::IntegerAttr>("gfx.mpsrt.runtime_abi.external_output_buffer_count").getInt(),
              1);
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.mpsrt.runtime_abi.record4.stage_kind").str(),
              "mps_gemm");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.mpsrt.runtime_abi.record5.stage_kind").str(),
              "msl_dispatch");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.mpsrt.runtime_abi.record5.kernel_name").str(),
              "eltwise_fused_buffer");
    const auto epilogue_order =
        module->getAttrOfType<mlir::ArrayAttr>("gfx.mpsrt.runtime_abi.record5.kernel_buffer_order");
    ASSERT_TRUE(epilogue_order);
    ASSERT_EQ(epilogue_order.size(), 3u);

    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.mpsrt.runtime_abi.call_plan_symbol").str(),
              "gfx_mpsrt_runtime_abi_plan");
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.model_stage_count"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.input_count"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.storage_bridge_count"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage0.backend"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage1.dispatch_entry_point"));
    ASSERT_FALSE(static_cast<bool>(module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_program")));
    ASSERT_TRUE(static_cast<bool>(module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops")));

    ov::gfx_plugin::GfxMpsrtProgram cleaned_multi_stage;
    ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_program(module, cleaned_multi_stage));
    ASSERT_TRUE(cleaned_multi_stage.valid);
    ASSERT_TRUE(cleaned_multi_stage.multi_stage);
    ASSERT_EQ(cleaned_multi_stage.record_key, "mps_gemm_plus_msl_epilogue_model|MatMul");
    ASSERT_EQ(cleaned_multi_stage.stages.size(), 2u);
    ASSERT_EQ(cleaned_multi_stage.stages[0].stage.kind,
              ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
    ASSERT_EQ(cleaned_multi_stage.stages[1].stage.kind,
              ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);

    auto call_plan = module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_runtime_abi_plan");
    ASSERT_TRUE(static_cast<bool>(call_plan));
    ASSERT_TRUE(call_plan->getAttrOfType<mlir::BoolAttr>("gfx.mpsrt.runtime_abi.generated").getValue());
    std::vector<mlir::func::CallOp> calls;
    call_plan.walk([&](mlir::func::CallOp call) {
        calls.push_back(call);
    });
    ASSERT_EQ(calls.size(), 7u);
    ASSERT_EQ(calls[0].getCallee(), "ovgfx_mpsrt_model_begin");
    ASSERT_EQ(calls[4].getCallee(), "ovgfx_mpsrt_encode_gemm");
    ASSERT_EQ(calls[5].getCallee(), "ovgfx_mpsrt_encode_dispatch");
    ASSERT_EQ(calls[5]->getAttrOfType<mlir::StringAttr>("gfx.mpsrt.runtime_abi.stage_kind").str(),
              "msl_dispatch");
    ASSERT_EQ(calls[5]->getAttrOfType<mlir::StringAttr>("gfx.mpsrt.runtime_abi.kernel_name").str(),
              "eltwise_fused_buffer");
    ASSERT_EQ(calls[5]->getAttrOfType<mlir::ArrayAttr>("gfx.mpsrt.runtime_abi.kernel_buffer_order").size(),
              3u);

    ov::gfx_plugin::GfxMpsrtBuilderPlan call_builder_plan;
    ASSERT_TRUE(ov::gfx_plugin::read_gfx_apple_mpsrt_runtime_abi_call_plan(module, call_builder_plan));
    ASSERT_TRUE(call_builder_plan.valid);
    ASSERT_EQ(call_builder_plan.stage_record_key, "mps_gemm_plus_msl_epilogue_model|MatMul");
    ASSERT_EQ(call_builder_plan.records.size(), 7u);
    ASSERT_EQ(call_builder_plan.input_values.size(), 3u);
    ASSERT_EQ(call_builder_plan.output_values.size(), 1u);
    ASSERT_TRUE(call_builder_plan.external_buffer_abi_valid);
    ASSERT_EQ(call_builder_plan.external_buffer_count, 4u);
    ASSERT_EQ(call_builder_plan.external_output_buffer_count, 1u);
    ASSERT_EQ(call_builder_plan.records[1].kind, ov::gfx_plugin::GfxMpsrtBuilderRecordKind::AddTensor);
    ASSERT_EQ(call_builder_plan.records[1].value, 0u);
    ASSERT_EQ(call_builder_plan.records[1].tensor_descs.size(), 1u);
    ASSERT_EQ(call_builder_plan.records[1].tensor_descs[0].storage,
              static_cast<uint32_t>(ov::gfx_plugin::GfxMpsrtStorage::Matrix));
    ASSERT_EQ(call_builder_plan.records[4].stage_kind, ov::gfx_plugin::GfxMpsrtStageKind::MPSGemm);
    ASSERT_EQ(call_builder_plan.records[4].tensor_descs.size(), 1u);
    ASSERT_EQ(call_builder_plan.records[4].gemm_desc.transpose_lhs, 0u);
    ASSERT_EQ(call_builder_plan.records[4].gemm_desc.transpose_rhs, 0u);
    ASSERT_EQ(call_builder_plan.records[5].stage_kind, ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
    ASSERT_EQ(call_builder_plan.records[5].kernel_name, "eltwise_fused_buffer");
    ASSERT_EQ(call_builder_plan.records[5].tensor_descs.size(), 1u);
    ASSERT_EQ(call_builder_plan.records[5].msl_dispatch_desc.kernel_family,
              call_builder_plan.records[5].dispatch_kernel_family_id);
    ASSERT_EQ(call_builder_plan.records[5].msl_dispatch_desc.input_count, 2u);
    ASSERT_EQ(call_builder_plan.records[5].msl_dispatch_desc.output_count, 1u);
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
    ASSERT_NE(msl.find("device const half* gemm [[buffer(0)]]"), std::string::npos);
    ASSERT_NE(msl.find("device const float* bias [[buffer(1)]]"), std::string::npos);
    ASSERT_NE(msl.find("device half* output [[buffer(2)]]"), std::string::npos);
    ASSERT_NE(msl.find("constant uint BATCH = 2;"), std::string::npos);
    ASSERT_NE(msl.find("constant uint M = 4;"), std::string::npos);
    ASSERT_NE(msl.find("constant uint N = 8;"), std::string::npos);
    ASSERT_NE(msl.find("x = (x >= 0.0f) ?"), std::string::npos);
}

TEST(GfxMlir, AddMslMetadataUsesRequiredMpsrtKernelFamily) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 64, 80, 80});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 64, 80, 80});
    auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);

    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
    auto module = ov::gfx_plugin::build_mlir_for_node(add, ctx);
    ASSERT_TRUE(module);
    mlir::Builder builder(module.getContext());
    module->setAttr("gfx.fixed_arg_count", builder.getI32IntegerAttr(3));
    module->setAttr("gfx.kernel_output_arg_count", builder.getI32IntegerAttr(1));

    const auto plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                     ov::gfx_plugin::GpuBackend::Metal,
                                                                     "Add",
                                                                     add,
                                                                     ov::element::f16,
                                                                     /*has_bias=*/false,
                                                                     /*has_activation=*/false,
                                                                     /*has_batchnorm=*/false,
                                                                     {});
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
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.stage_family").str(), "eltwise");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.backend_domain").str(), "apple_msl");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.execution_kind").str(),
              "custom_kernel");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.storage").str(), "buffer");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.kernel.family").str(),
              "eltwise_fused_buffer");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.kernel.entry_point").str(),
              "eltwise_fused_buffer");
    ASSERT_FALSE(module->getAttrOfType<mlir::BoolAttr>(
                          "gfx.stage_manifest.kernel.external_buffer_abi.tail_outputs")
                    .getValue());
    const auto add_role_values = ov::gfx_plugin::detail::gfx_mpsrt_read_u32_vector_attr(
        module,
        "gfx.stage_manifest.kernel.external_buffer_abi.roles");
    ASSERT_EQ(add_role_values.size(), 3u);
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage_kind"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.dispatch_kernel_family"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.dispatch_entry_point"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.dispatch_kernel_family_id"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.dispatch_flags"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.dispatch_precompiled_kernel_required"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.dispatch_threads_per_threadgroup"));
    ASSERT_FALSE(module->hasAttr("gfx.stage_manifest.kernel.precompiled_binary_required"));
    ASSERT_FALSE(module->hasAttr("gfx.stage_manifest.kernel.threads_per_threadgroup"));
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>(
                         "gfx.stage_manifest.kernel.dispatch_policy.grid")
                  .str(),
              "linear_1d");
    ASSERT_EQ(module->getAttrOfType<mlir::IntegerAttr>(
                         "gfx.stage_manifest.kernel.dispatch_policy.threads_per_threadgroup")
                  .getInt(),
              256);
    ASSERT_TRUE(module->getAttrOfType<mlir::BoolAttr>(
                          "gfx.stage_manifest.kernel.dispatch_policy.precompiled_binary_required")
                    .getValue());

    ov::gfx_plugin::GfxMpsrtModuleStagePlan extracted_stage;
    ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_stage_plan(module, extracted_stage));
    ASSERT_EQ(extracted_stage.stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
    ASSERT_EQ(extracted_stage.stage.dispatch_kernel_family, "eltwise_fused_buffer");
    ASSERT_EQ(extracted_stage.stage.dispatch_entry_point, "eltwise_fused_buffer");
    ASSERT_EQ(extracted_stage.stage.dispatch_kernel_family_id,
              static_cast<uint32_t>(ov::gfx_plugin::GfxKernelFamily::EltwiseFusedBuffer));
    ASSERT_EQ(extracted_stage.stage.dispatch_flags,
              ov::gfx_plugin::GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
    ASSERT_TRUE(extracted_stage.stage.dispatch_precompiled_kernel_required);
    ASSERT_EQ(extracted_stage.stage.dispatch_threads_per_threadgroup, 256u);

    const auto custom_kernel_plan = ov::gfx_plugin::make_gfx_custom_kernel_stage_plan("Add", "eltwise_kernel");
    ASSERT_TRUE(custom_kernel_plan.valid);
    ASSERT_EQ(custom_kernel_plan.family, ov::gfx_plugin::GfxKernelFamily::EltwiseFusedBuffer);
    ASSERT_EQ(custom_kernel_plan.stage_manifest.custom_kernel.entry_point, "eltwise_fused_buffer");

    ov::gfx_plugin::KernelSource source;
    source.module = module;
    source.entry_point = "eltwise_kernel";
    source.msl_source =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void eltwise_kernel(device const half* A [[buffer(0)]]) {}\n";
    ov::gfx_plugin::configure_msl_kernel_source_for_plan(source, "Add");
    ASSERT_EQ(source.entry_point, "eltwise_fused_buffer");
    ASSERT_NE(source.msl_source.find("kernel void eltwise_fused_buffer"), std::string::npos);
    ASSERT_EQ(source.msl_source.find("kernel void eltwise_kernel"), std::string::npos);

    const auto module_builder_plan = ov::gfx_plugin::build_module_mpsrt_builder_plan(module);
    ASSERT_TRUE(module_builder_plan.valid);
    ASSERT_TRUE(module_builder_plan.stage_plan.stage.stage_manifest.valid);
    ASSERT_EQ(module_builder_plan.stage_plan.stage.stage_manifest.execution_kind,
              ov::gfx_plugin::GfxKernelExecutionKind::CustomKernel);
    ASSERT_TRUE(module_builder_plan.stage_plan.stage.stage_manifest.custom_kernel.valid);
    ASSERT_EQ(module_builder_plan.stage_plan.stage.stage_manifest.custom_kernel.kernel_family,
              "eltwise_fused_buffer");
    ASSERT_FALSE(module_builder_plan.stage_plan.stage.stage_manifest.custom_kernel.external_buffer_abi.tail_outputs);
    ASSERT_TRUE(module_builder_plan.external_buffer_abi.valid);
    ASSERT_TRUE(module_builder_plan.external_buffer_abi.has_buffer_count);
    ASSERT_TRUE(module_builder_plan.external_buffer_abi.has_output_buffer_count);
    ASSERT_EQ(module_builder_plan.external_buffer_abi.buffer_count, 3u);
    ASSERT_EQ(module_builder_plan.external_buffer_abi.output_buffer_count, 1u);
    ASSERT_TRUE(module_builder_plan.external_buffer_abi.has_buffer_roles);
    ASSERT_EQ(module_builder_plan.external_buffer_abi.buffer_roles,
              std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                  {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                   ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                   ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput}));
    ASSERT_TRUE(module_builder_plan.builder_plan.external_buffer_abi_valid);
    ASSERT_EQ(module_builder_plan.builder_plan.external_buffer_count, 3u);
    ASSERT_EQ(module_builder_plan.builder_plan.external_output_buffer_count, 1u);
    ASSERT_EQ(module_builder_plan.builder_plan.external_buffer_roles,
              module_builder_plan.external_buffer_abi.buffer_roles);
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].kind,
              ov::gfx_plugin::GfxMpsrtBuilderRecordKind::EncodeStage);
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].symbol, "ovgfx_mpsrt_encode_dispatch");
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].kernel_name, "eltwise_fused_buffer");
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].dispatch_kernel_family, "eltwise_fused_buffer");
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].dispatch_entry_point, "eltwise_fused_buffer");
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].dispatch_kernel_family_id,
              static_cast<uint32_t>(ov::gfx_plugin::GfxKernelFamily::EltwiseFusedBuffer));
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].dispatch_flags,
              ov::gfx_plugin::GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].dispatch_threads_per_threadgroup, 256u);
    ASSERT_TRUE(module_builder_plan.builder_plan.records[3].dispatch_precompiled_kernel_required);
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].msl_dispatch_desc.kernel_family,
              static_cast<uint32_t>(ov::gfx_plugin::GfxKernelFamily::EltwiseFusedBuffer));
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].msl_dispatch_desc.storage,
              static_cast<uint32_t>(ov::gfx_plugin::GfxMpsrtStorage::Buffer));
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].msl_dispatch_desc.layout,
              static_cast<uint32_t>(ov::gfx_plugin::GfxMpsrtLayout::Linear));
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].msl_dispatch_desc.threads_per_threadgroup, 256u);
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].msl_dispatch_desc.input_count, 2u);
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].msl_dispatch_desc.output_count, 1u);
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].msl_dispatch_desc.flags,
              ov::gfx_plugin::GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
}

TEST(GfxMlir, AppleMslLoweringMaterializesStageManifestBeforeTypedProgram) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 8});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 8});
    auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);

    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
    auto module = ov::gfx_plugin::build_mlir_for_node(add, ctx);
    ASSERT_TRUE(module);
    mlir::Builder builder(module.getContext());
    module->setAttr("gfx.fixed_arg_count", builder.getI32IntegerAttr(3));
    module->setAttr("gfx.kernel_output_arg_count", builder.getI32IntegerAttr(1));

    const auto plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                     ov::gfx_plugin::GpuBackend::Metal,
                                                                     "Add",
                                                                     add,
                                                                     ov::element::f16,
                                                                     /*has_bias=*/false,
                                                                     /*has_activation=*/false,
                                                                     /*has_batchnorm=*/false,
                                                                     {});
    auto lowering_plan =
        ov::gfx_plugin::materialize_apple_msl_stage_manifest(module, plan, "Add", "eltwise_kernel");
    ASSERT_TRUE(lowering_plan.valid);
    ASSERT_EQ(lowering_plan.stage_plan.stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
    ASSERT_EQ(module->getAttrOfType<mlir::IntegerAttr>("gfx.apple.pipeline.pass_boundary_count")
                  .getInt(),
              6);
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.apple.pipeline.pass0.name").str(),
              "gfx-core-canonicalize");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.apple.pipeline.pass4.name").str(),
              "gfx-apple-vendor-descriptor");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.apple.pipeline.pass5.name").str(),
              "gfx-apple-stage-manifest");
    ASSERT_FALSE(module->hasAttr("gfx.apple.pipeline.pass6.name"));
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.apple.pipeline.vendor_descriptor.kind").str(),
              "none");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.kernel.family").str(),
              "eltwise_fused_buffer");
    ASSERT_FALSE(static_cast<bool>(module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops")));

    ov::gfx_plugin::GfxMpsrtModuleStagePlan extracted_stage;
    ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_stage_plan(module, extracted_stage));
    ASSERT_EQ(extracted_stage.stage.dispatch_entry_point, "eltwise_fused_buffer");

    ov::gfx_plugin::GfxMpsrtExternalBufferAbiPlan external_buffer_abi{};
    ASSERT_TRUE(ov::gfx_plugin::gfx_mpsrt_external_buffer_abi_from_kernel_manifest(module,
                                                                                  external_buffer_abi));
    ASSERT_TRUE(ov::gfx_plugin::materialize_apple_msl_typed_program(module,
                                                                    lowering_plan,
                                                                    external_buffer_abi));
    ASSERT_TRUE(static_cast<bool>(module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops")));
    ASSERT_FALSE(module->hasAttr("gfx.msl.kernel_family"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.dispatch_kernel_family"));
}

TEST(GfxMlir, AppleStagePipelineRunsNamedPassBoundariesBeforeTypedProgram) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 8});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 8});
    auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);

    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
    auto module = ov::gfx_plugin::build_mlir_for_node(add, ctx);
    ASSERT_TRUE(module);
    mlir::Builder builder(module.getContext());
    module->setAttr("gfx.fixed_arg_count", builder.getI32IntegerAttr(3));
    module->setAttr("gfx.kernel_output_arg_count", builder.getI32IntegerAttr(1));

    const auto plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                     ov::gfx_plugin::GpuBackend::Metal,
                                                                     "Add",
                                                                     add,
                                                                     ov::element::f16,
                                                                     /*has_bias=*/false,
                                                                     /*has_activation=*/false,
                                                                     /*has_batchnorm=*/false,
                                                                     {});
    const auto result = ov::gfx_plugin::run_gfx_apple_stage_pipeline(module, plan, "Add", "eltwise_kernel");
    ASSERT_TRUE(result.valid);
    ASSERT_TRUE(result.typed_program_materialized);
    ASSERT_TRUE(static_cast<bool>(module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops")));
    const auto passes = ov::gfx_plugin::gfx_apple_stage_pipeline_pass_boundaries(
        /*materialize_typed_program=*/true);
    ASSERT_EQ(passes.size(), 7u);
    ASSERT_EQ(ov::gfx_plugin::gfx_apple_stage_pipeline_pass_name(passes[0]),
              std::string("gfx-core-canonicalize"));
    ASSERT_EQ(ov::gfx_plugin::gfx_apple_stage_pipeline_pass_name(passes[5]),
              std::string("gfx-apple-stage-manifest"));
    ASSERT_EQ(ov::gfx_plugin::gfx_apple_stage_pipeline_pass_name(passes[6]),
              std::string("gfx-apple-runtime-abi"));
    ASSERT_EQ(module->getAttrOfType<mlir::IntegerAttr>("gfx.apple.pipeline.pass_boundary_count")
                  .getInt(),
              7);
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.apple.pipeline.pass5.name").str(),
              "gfx-apple-stage-manifest");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.apple.pipeline.pass6.name").str(),
              "gfx-apple-runtime-abi");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.apple.pipeline.vendor_descriptor.kind").str(),
              "none");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.apple.pipeline.placement.backend_domain").str(),
              "apple_msl");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.apple.pipeline.placement.execution_kind").str(),
              "custom_kernel");
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.apple.pipeline.storage.contract").str(),
              "buffer");
    ASSERT_EQ(module->getAttrOfType<mlir::IntegerAttr>(
                         "gfx.apple.pipeline.storage_assignment.storage_bridge_count")
                  .getInt(),
              0);
    ASSERT_TRUE(module->hasAttr("gfx.apple.pipeline.fusion.activation"));
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.backend_domain").str(),
              "apple_msl");
    ASSERT_EQ(result.stage_plan.stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
}

TEST(GfxMlir, AppleStagePipelineOwnsImageStorageBridgeAssignment) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 16, 32, 32});
    auto weights = ov::op::v0::Constant::create(ov::element::f16,
                                                ov::Shape{8, 16, 3, 3},
                                                std::vector<float>(8 * 16 * 3 * 3, 1.f));
    auto conv = std::make_shared<ov::op::v1::Convolution>(input,
                                                          weights,
                                                          ov::Strides{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::Strides{1, 1});

    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
    auto module = ov::gfx_plugin::build_mlir_for_node(conv, ctx);
    ASSERT_TRUE(module);

    const auto plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                     ov::gfx_plugin::GpuBackend::Metal,
                                                                     "Convolution",
                                                                     conv,
                                                                     ov::element::f16,
                                                                     /*has_bias=*/false,
                                                                     /*has_activation=*/false,
                                                                     /*has_batchnorm=*/false,
                                                                     {});
    ov::gfx_plugin::GfxAppleStagePipelineOptions options{};
    options.plan = plan;
    options.stage_type = "Convolution";
    options.semantic_input_roles = {ov::gfx_plugin::GfxKernelBufferRole::TensorInput,
                                    ov::gfx_plugin::GfxKernelBufferRole::ConstTensor};
    const auto result = ov::gfx_plugin::run_gfx_apple_stage_pipeline(module, options);
    ASSERT_TRUE(result.valid);
    ASSERT_TRUE(result.typed_program_materialized);
    ASSERT_EQ(module->getAttrOfType<mlir::IntegerAttr>(
                         "gfx.apple.pipeline.storage_assignment.storage_bridge_count")
                  .getInt(),
              2);

    std::vector<ov::gfx_plugin::GfxMpsrtStorageBridgeDesc> assignment;
    ASSERT_TRUE(ov::gfx_plugin::read_module_apple_storage_assignment(module, assignment));
    ASSERT_EQ(assignment.size(), 2u);
    ASSERT_EQ(assignment[0].direction, ov::gfx_plugin::GfxMpsrtStorageBridgeDirection::BufferToImage);
    ASSERT_EQ(assignment[1].direction, ov::gfx_plugin::GfxMpsrtStorageBridgeDirection::ImageToBuffer);

    ov::gfx_plugin::GfxMpsrtProgram program;
    ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_ops_program(module, program));
    ASSERT_TRUE(program.has_storage_bridges);
    ASSERT_EQ(program.storage_bridges.size(), assignment.size());
    ASSERT_EQ(program.storage_bridges[0].direction, assignment[0].direction);
    ASSERT_EQ(program.storage_bridges[1].direction, assignment[1].direction);
    ASSERT_EQ(program.inputs[1].flags, ov::gfx_plugin::GfxMpsrtTensorFlagConst);
    ASSERT_EQ(program.inputs[1].storage, ov::gfx_plugin::GfxMpsrtStorage::Buffer);
    ASSERT_EQ(program.stages.front().stage.stage_manifest.semantic_input_roles,
              options.semantic_input_roles);
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.storage_bridge_count"));
}

TEST(GfxMlir, AppleMpsPrimitiveMaterializersOwnDescriptorEnrichment) {
    auto pool_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                              ov::Shape{1, 4, 16, 16});
    auto max_pool = std::make_shared<ov::op::v1::MaxPool>(pool_input,
                                                          ov::Strides{2, 2},
                                                          ov::Shape{0, 0},
                                                          ov::Shape{0, 0},
                                                          ov::Shape{2, 2},
                                                          ov::op::RoundingType::FLOOR);
    auto softmax_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                                 ov::Shape{2, 8, 16});
    auto softmax = std::make_shared<ov::op::v1::Softmax>(softmax_input, 2);
    auto topk_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                              ov::Shape{2, 8, 16});
    auto k = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, std::vector<int32_t>{4});
    auto topk = std::make_shared<ov::op::v11::TopK>(topk_input,
                                                    k,
                                                    2,
                                                    ov::op::TopKMode::MAX,
                                                    ov::op::TopKSortType::SORT_VALUES,
                                                    ov::element::i32);

    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
    auto pool_module = ov::gfx_plugin::build_mlir_for_node(max_pool, ctx);
    auto softmax_module = ov::gfx_plugin::build_mlir_for_node(softmax, ctx);
    auto topk_module = ov::gfx_plugin::build_mlir_for_node(topk, ctx);
    ASSERT_TRUE(pool_module);
    ASSERT_TRUE(softmax_module);
    ASSERT_TRUE(topk_module);

    const auto pool_plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                          ov::gfx_plugin::GpuBackend::Metal,
                                                                          "MaxPool",
                                                                          max_pool,
                                                                          ov::element::f16,
                                                                          false,
                                                                          false,
                                                                          false,
                                                                          {});
    ov::gfx_plugin::GfxMpsrtPool2DAbiDesc pool_desc{};
    pool_desc.kernel[0] = 2;
    pool_desc.kernel[1] = 2;
    pool_desc.strides[0] = 2;
    pool_desc.strides[1] = 2;
    pool_desc.dilations[0] = 1;
    pool_desc.dilations[1] = 1;
    const auto pool_materialized = ov::gfx_plugin::materialize_apple_mps_pool2d_program(pool_module,
                                                                                       pool_plan,
                                                                                       "MaxPool",
                                                                                       pool_desc);
    ASSERT_TRUE(pool_materialized.valid);
    ASSERT_TRUE(pool_materialized.typed_program_materialized);

    const auto softmax_plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                             ov::gfx_plugin::GpuBackend::Metal,
                                                                             "Softmax",
                                                                             softmax,
                                                                             ov::element::f16,
                                                                             false,
                                                                             false,
                                                                             false,
                                                                             {});
    ov::gfx_plugin::GfxMpsrtSoftmaxAbiDesc softmax_desc{};
    softmax_desc.axis = 2;
    const auto softmax_materialized =
        ov::gfx_plugin::materialize_apple_mps_softmax_program(softmax_module,
                                                              softmax_plan,
                                                              "Softmax",
                                                              softmax_desc);
    ASSERT_TRUE(softmax_materialized.valid);
    ASSERT_TRUE(softmax_materialized.typed_program_materialized);

    const auto topk_plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                          ov::gfx_plugin::GpuBackend::Metal,
                                                                          "TopK",
                                                                          topk,
                                                                          ov::element::f16,
                                                                          false,
                                                                          false,
                                                                          false,
                                                                          {});
    ov::gfx_plugin::GfxMpsrtTopKAbiDesc topk_desc{};
    topk_desc.axis = 2;
    topk_desc.k = 4;
    topk_desc.mode_max = 1;
    topk_desc.sort_type = static_cast<uint32_t>(ov::gfx_plugin::TopKSortType::SortValues);
    const auto topk_materialized = ov::gfx_plugin::materialize_apple_mps_topk_program(topk_module,
                                                                                     topk_plan,
                                                                                     "TopK",
                                                                                     topk_desc);
    ASSERT_TRUE(topk_materialized.valid);
    ASSERT_TRUE(topk_materialized.typed_program_materialized);

    ov::gfx_plugin::GfxMpsrtProgram pool_program;
    ov::gfx_plugin::GfxMpsrtProgram softmax_program;
    ov::gfx_plugin::GfxMpsrtProgram topk_program;
    ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_ops_program(pool_module, pool_program));
    ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_ops_program(softmax_module, softmax_program));
    ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_ops_program(topk_module, topk_program));
    ASSERT_EQ(pool_module->getAttrOfType<mlir::StringAttr>(
                          "gfx.apple.pipeline.vendor_descriptor.kind")
                  .str(),
              "pool2d");
    ASSERT_EQ(softmax_module->getAttrOfType<mlir::StringAttr>(
                             "gfx.apple.pipeline.vendor_descriptor.kind")
                  .str(),
              "softmax");
    ASSERT_EQ(topk_module->getAttrOfType<mlir::StringAttr>(
                          "gfx.apple.pipeline.vendor_descriptor.kind")
                  .str(),
              "topk");

    ASSERT_EQ(pool_program.stages.front().stage.kind,
              ov::gfx_plugin::GfxMpsrtStageKind::MPSPool2D);
    ASSERT_EQ(pool_program.external_buffer_abi.buffer_roles,
              std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                  {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                   ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams,
                   ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput}));
    ASSERT_EQ(pool_program.stages.front().stage.pool2d_desc.kernel[0], 2u);
    ASSERT_EQ(pool_program.stages.front().stage.pool2d_desc.strides[1], 2u);
    ASSERT_EQ(pool_program.stages.front().stage.stage_manifest.semantic_input_roles,
              std::vector<ov::gfx_plugin::GfxKernelBufferRole>(
                  {ov::gfx_plugin::GfxKernelBufferRole::TensorInput}));

    ASSERT_EQ(softmax_program.stages.front().stage.kind,
              ov::gfx_plugin::GfxMpsrtStageKind::MPSSoftmax);
    ASSERT_EQ(softmax_program.external_buffer_abi.buffer_roles,
              std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                  {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                   ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput}));
    ASSERT_EQ(softmax_program.stages.front().stage.softmax_desc.axis, 2u);
    ASSERT_EQ(softmax_program.stages.front().stage.stage_manifest.semantic_output_roles,
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
    ASSERT_EQ(topk_program.stages.front().stage.stage_manifest.semantic_output_roles,
              std::vector<ov::gfx_plugin::GfxKernelBufferRole>(
                  {ov::gfx_plugin::GfxKernelBufferRole::TensorOutput,
                   ov::gfx_plugin::GfxKernelBufferRole::TensorOutput}));
    ASSERT_FALSE(pool_module->hasAttr("gfx.mpsrt.pool2d.kernel.0"));
    ASSERT_FALSE(softmax_module->hasAttr("gfx.mpsrt.softmax.axis"));
    ASSERT_FALSE(topk_module->hasAttr("gfx.mpsrt.topk.k"));
}

TEST(GfxMlir, StageManifestSuppliesMslDispatchMetadataWhenLegacyStageAttrsAreMissing) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 64, 80, 80});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 64, 80, 80});
    auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);

    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
    auto module = ov::gfx_plugin::build_mlir_for_node(add, ctx);
    ASSERT_TRUE(module);
    mlir::Builder builder(module.getContext());
    module->setAttr("gfx.fixed_arg_count", builder.getI32IntegerAttr(3));
    module->setAttr("gfx.kernel_output_arg_count", builder.getI32IntegerAttr(1));

    const auto plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                     ov::gfx_plugin::GpuBackend::Metal,
                                                                     "Add",
                                                                     add,
                                                                     ov::element::f16,
                                                                     /*has_bias=*/false,
                                                                     /*has_activation=*/false,
                                                                     /*has_batchnorm=*/false,
                                                                     {});
    ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Add");

    module->removeAttr("gfx.backend");
    module->removeAttr("gfx.specialization_key");
    module->removeAttr("gfx.uses_custom_kernel");
    module->removeAttr("gfx.uses_vendor_primitive");
    module->removeAttr("gfx.mpsrt.stage_kind");
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
    ASSERT_EQ(extracted.stage.domain, ov::gfx_plugin::GfxStageBackendDomain::AppleMsl);
    ASSERT_TRUE(extracted.stage.uses_custom_kernel);
    ASSERT_FALSE(extracted.stage.uses_vendor_primitive);
    ASSERT_EQ(extracted.stage.specialization_key, "apple_msl:buffer:Add");
    ASSERT_EQ(extracted.stage.dispatch_kernel_family, "eltwise_fused_buffer");
    ASSERT_EQ(extracted.stage.dispatch_entry_point, "eltwise_fused_buffer");
    ASSERT_EQ(extracted.stage.dispatch_kernel_family_id,
              static_cast<uint32_t>(ov::gfx_plugin::GfxKernelFamily::EltwiseFusedBuffer));
    ASSERT_EQ(extracted.stage.dispatch_flags,
              ov::gfx_plugin::GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
    ASSERT_EQ(extracted.stage.dispatch_threads_per_threadgroup, 256u);
    ASSERT_TRUE(extracted.stage.dispatch_precompiled_kernel_required);
    ASSERT_FALSE(extracted.stage_record_key.empty());

    const auto module_builder_plan = ov::gfx_plugin::build_module_mpsrt_builder_plan(module);
    ASSERT_TRUE(module_builder_plan.valid);
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].dispatch_kernel_family,
              "eltwise_fused_buffer");
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].dispatch_entry_point,
              "eltwise_fused_buffer");
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].dispatch_kernel_family_id,
              static_cast<uint32_t>(ov::gfx_plugin::GfxKernelFamily::EltwiseFusedBuffer));
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].dispatch_flags,
              ov::gfx_plugin::GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].dispatch_threads_per_threadgroup, 256u);
    ASSERT_TRUE(module_builder_plan.builder_plan.records[3].dispatch_precompiled_kernel_required);
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].msl_dispatch_desc.kernel_family,
              static_cast<uint32_t>(ov::gfx_plugin::GfxKernelFamily::EltwiseFusedBuffer));
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].msl_dispatch_desc.threads_per_threadgroup, 256u);
    ASSERT_EQ(module_builder_plan.builder_plan.records[3].msl_dispatch_desc.flags,
              ov::gfx_plugin::GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
}

TEST(GfxMlir, LegacyMpsrtStageAttrsWithoutManifestAreRejected) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 64, 80, 80});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 64, 80, 80});
    auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);

    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
    auto module = ov::gfx_plugin::build_mlir_for_node(add, ctx);
    ASSERT_TRUE(module);

    mlir::Builder builder(module.getContext());
    module->setAttr("gfx.backend", builder.getStringAttr("apple_msl"));
    module->setAttr("gfx.stage_type", builder.getStringAttr("Add"));
    module->setAttr("gfx.specialization_key", builder.getStringAttr("apple_msl:buffer:Add"));
    module->setAttr("gfx.uses_custom_kernel", builder.getBoolAttr(true));
    module->setAttr("gfx.mpsrt.stage_kind", builder.getStringAttr("msl_dispatch"));
    module->setAttr("gfx.mpsrt.stage_record_key", builder.getStringAttr("legacy_record"));
    module->setAttr("gfx.mpsrt.kernel_name", builder.getStringAttr("eltwise_fused_buffer"));
    module->setAttr("gfx.mpsrt.builder_symbol", builder.getStringAttr("ovgfx_mpsrt_encode_dispatch"));
    module->setAttr("gfx.mpsrt.dispatch_kernel_family", builder.getStringAttr("eltwise_fused_buffer"));
    module->setAttr("gfx.mpsrt.dispatch_entry_point", builder.getStringAttr("eltwise_fused_buffer"));
    ASSERT_FALSE(module->hasAttr("gfx.stage_manifest.stage_family"));

    ov::gfx_plugin::GfxMpsrtModuleStagePlan stage_plan;
    ASSERT_FALSE(ov::gfx_plugin::read_module_mpsrt_stage_plan(module, stage_plan));

    ov::gfx_plugin::GfxMpsrtProgram program;
    ASSERT_FALSE(ov::gfx_plugin::read_module_mpsrt_program(module, program));

    const auto module_builder_plan = ov::gfx_plugin::build_module_mpsrt_builder_plan(module);
    ASSERT_FALSE(module_builder_plan.valid);
}

TEST(GfxMlir, StageManifestOverridesConflictingGeneratedStageAttrs) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 64, 80, 80});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 64, 80, 80});
    auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);

    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
    auto module = ov::gfx_plugin::build_mlir_for_node(add, ctx);
    ASSERT_TRUE(module);
    mlir::Builder builder(module.getContext());
    module->setAttr("gfx.fixed_arg_count", builder.getI32IntegerAttr(3));
    module->setAttr("gfx.kernel_output_arg_count", builder.getI32IntegerAttr(1));

    const auto plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                     ov::gfx_plugin::GpuBackend::Metal,
                                                                     "Add",
                                                                     add,
                                                                     ov::element::f16,
                                                                     /*has_bias=*/false,
                                                                     /*has_activation=*/false,
                                                                     /*has_batchnorm=*/false,
                                                                     {});
    ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Add");

    auto generated_ops = module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops");
    ASSERT_TRUE(static_cast<bool>(generated_ops));
    mlir::Operation* dispatch_op = nullptr;
    generated_ops.walk([&](mlir::Operation* op) {
        if (op->getName().getStringRef() == "gfx.mpsrt.dispatch") {
            dispatch_op = op;
        }
    });
    ASSERT_NE(dispatch_op, nullptr);

    dispatch_op->setAttr("gfx.mpsrt.op.stage.backend", builder.getStringAttr("apple_mps"));
    dispatch_op->setAttr("gfx.mpsrt.op.stage.stage_kind", builder.getStringAttr("mps_gemm"));
    dispatch_op->setAttr("gfx.mpsrt.op.stage.stage_record_key", builder.getStringAttr("legacy_wrong_key"));
    dispatch_op->setAttr("gfx.mpsrt.op.stage.kernel_name", builder.getStringAttr("legacy_wrong_kernel"));
    dispatch_op->setAttr("gfx.mpsrt.op.stage.builder_symbol", builder.getStringAttr("ovgfx_mpsrt_encode_gemm"));

    ov::gfx_plugin::GfxMpsrtProgram program;
    ASSERT_TRUE(ov::gfx_plugin::read_module_mpsrt_program(module, program));
    ASSERT_TRUE(program.valid);
    ASSERT_FALSE(program.multi_stage);
    ASSERT_EQ(program.stages.size(), 1u);
    const auto& stage = program.stages.front();
    ASSERT_EQ(stage.stage.kind, ov::gfx_plugin::GfxMpsrtStageKind::MSLDispatch);
    ASSERT_EQ(stage.stage.domain, ov::gfx_plugin::GfxStageBackendDomain::AppleMsl);
    ASSERT_TRUE(stage.stage.uses_custom_kernel);
    ASSERT_FALSE(stage.stage.uses_vendor_primitive);
    ASSERT_EQ(stage.stage.kernel_name, "eltwise_fused_buffer");
    ASSERT_EQ(stage.stage.builder_symbol, "ovgfx_mpsrt_encode_dispatch");
    ASSERT_EQ(stage.stage_record_key,
              "msl_dispatch|apple_msl|buffer|buffer|linear|Add|apple_msl:buffer:Add|"
              "dispatch:eltwise_fused_buffer:eltwise_fused_buffer:linear_1d:tg256:metallib");
}

TEST(GfxMlir, StageManifestSuppliesTailOutputExternalBufferAbiWithoutModuleMpsrtAttrs) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 64, 80, 80});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 64, 80, 80});
    auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);

    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
    auto module = ov::gfx_plugin::build_mlir_for_node(add, ctx);
    ASSERT_TRUE(module);
    mlir::Builder builder(module.getContext());
    module->setAttr("gfx.fixed_arg_count", builder.getI32IntegerAttr(3));
    module->setAttr("gfx.kernel_output_arg_count", builder.getI32IntegerAttr(1));

    const auto plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                     ov::gfx_plugin::GpuBackend::Metal,
                                                                     "Add",
                                                                     add,
                                                                     ov::element::f16,
                                                                     /*has_bias=*/false,
                                                                     /*has_activation=*/false,
                                                                     /*has_batchnorm=*/false,
                                                                     {});
    ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Add");
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_count"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_output_buffer_count"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_roles"));

    const auto abi = ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
    ASSERT_TRUE(abi.valid);
    ASSERT_TRUE(abi.has_buffer_count);
    ASSERT_TRUE(abi.has_output_buffer_count);
    ASSERT_TRUE(abi.has_buffer_roles);
    ASSERT_EQ(abi.buffer_count, 3u);
    ASSERT_EQ(abi.output_buffer_count, 1u);
    ASSERT_EQ(abi.buffer_roles,
              std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                  {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                   ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                   ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput}));

    const auto module_builder_plan = ov::gfx_plugin::build_module_mpsrt_builder_plan(module);
    ASSERT_TRUE(module_builder_plan.valid);
    ASSERT_TRUE(module_builder_plan.builder_plan.external_buffer_abi_valid);
    ASSERT_EQ(module_builder_plan.builder_plan.external_buffer_count, 3u);
    ASSERT_EQ(module_builder_plan.builder_plan.external_output_buffer_count, 1u);
    ASSERT_EQ(module_builder_plan.builder_plan.external_buffer_roles, abi.buffer_roles);
}

TEST(GfxMlir, StageManifestSuppliesElementwiseRoleAbiWithoutLegacyMpsrtAttrs) {
    mlir::MLIRContext ctx;
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

    const auto plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                     ov::gfx_plugin::GpuBackend::Metal,
                                                                     "Add",
                                                                     nullptr,
                                                                     ov::element::f16,
                                                                     /*has_bias=*/false,
                                                                     /*has_activation=*/false,
                                                                     /*has_batchnorm=*/false,
                                                                     {});
    ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Add");
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_count"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_output_buffer_count"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_roles"));

    const auto abi = ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
    ASSERT_TRUE(abi.valid);
    ASSERT_TRUE(abi.has_buffer_roles);
    ASSERT_EQ(abi.buffer_count, 3u);
    ASSERT_EQ(abi.output_buffer_count, 1u);
}

TEST(GfxMlir, StageManifestSuppliesRoleBasedExternalBufferAbiWithoutModuleMpsrtAttrs) {
    mlir::MLIRContext ctx;
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

    const auto plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                     ov::gfx_plugin::GpuBackend::Metal,
                                                                     "Softmax",
                                                                     nullptr,
                                                                     ov::element::f16,
                                                                     /*has_bias=*/false,
                                                                     /*has_activation=*/false,
                                                                     /*has_batchnorm=*/false,
                                                                     {});
    ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Softmax");
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_count"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_output_buffer_count"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_roles"));

    const auto abi = ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
    ASSERT_TRUE(abi.valid);
    ASSERT_EQ(abi.buffer_count, 3u);
    ASSERT_EQ(abi.output_buffer_count, 1u);
    ASSERT_TRUE(abi.has_buffer_roles);
    ASSERT_EQ(abi.buffer_roles,
              std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                  {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                   ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput,
                   ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams}));
}

TEST(GfxMlir, StageManifestSuppliesLeadingIoExternalBufferAbiWithoutModuleMpsrtAttrs) {
    mlir::MLIRContext ctx;
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mlir::Builder builder(&ctx);
    module->setAttr("gfx.fixed_arg_count", builder.getI32IntegerAttr(4));

    const auto plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                     ov::gfx_plugin::GpuBackend::Metal,
                                                                     "Gather",
                                                                     nullptr,
                                                                     ov::element::f32,
                                                                     /*has_bias=*/false,
                                                                     /*has_activation=*/false,
                                                                     /*has_batchnorm=*/false,
                                                                     {});
    ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Gather");
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_count"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_output_buffer_count"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_roles"));

    const auto abi = ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
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

    const auto plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                     ov::gfx_plugin::GpuBackend::Metal,
                                                                     "Softmax",
                                                                     nullptr,
                                                                     ov::element::f16,
                                                                     /*has_bias=*/false,
                                                                     /*has_activation=*/false,
                                                                     /*has_batchnorm=*/false,
                                                                     {});
    ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Softmax");

    ASSERT_FALSE(module->hasAttr("gfx.msl.kernel_family"));
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.kernel.family").str(),
              "masked_softmax_attention");
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_count"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_output_buffer_count"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_roles"));

    const auto abi = ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
    ASSERT_TRUE(abi.valid);
    EXPECT_TRUE(abi.has_buffer_count);
    EXPECT_TRUE(abi.has_output_buffer_count);
    EXPECT_EQ(abi.buffer_count, 3u);
    EXPECT_EQ(abi.output_buffer_count, 1u);
    ASSERT_TRUE(abi.has_buffer_roles);
    EXPECT_EQ(abi.buffer_roles,
              std::vector<ov::gfx_plugin::GfxMpsrtExternalBufferRole>(
                  {ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput,
                   ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput,
                   ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams}));
}

TEST(GfxMlir, TopKMslMetadataUsesRoleBasedExternalAbi) {
    mlir::MLIRContext ctx;
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

    const auto plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                     ov::gfx_plugin::GpuBackend::Metal,
                                                                     "TopK",
                                                                     nullptr,
                                                                     ov::element::f32,
                                                                     /*has_bias=*/false,
                                                                     /*has_activation=*/false,
                                                                     /*has_batchnorm=*/false,
                                                                     {});
    ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "TopK");

    ASSERT_FALSE(module->hasAttr("gfx.msl.kernel_family"));
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.kernel.family").str(),
              "gather_scatter_indexed");
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_count"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_output_buffer_count"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_roles"));

    const auto abi = ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
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

TEST(GfxMlir, GatherMslMetadataUsesRolePatternWithRuntimeParams) {
    mlir::MLIRContext ctx;
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

    const auto plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                     ov::gfx_plugin::GpuBackend::Metal,
                                                                     "Gather",
                                                                     nullptr,
                                                                     ov::element::f32,
                                                                     /*has_bias=*/false,
                                                                     /*has_activation=*/false,
                                                                     /*has_batchnorm=*/false,
                                                                     {});
    ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Gather");

    ASSERT_FALSE(module->hasAttr("gfx.msl.kernel_family"));
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.kernel.family").str(),
              "gather_scatter_indexed");
    const auto abi = ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
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

    const auto plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                     ov::gfx_plugin::GpuBackend::Metal,
                                                                     "Slice",
                                                                     nullptr,
                                                                     ov::element::f32,
                                                                     /*has_bias=*/false,
                                                                     /*has_activation=*/false,
                                                                     /*has_batchnorm=*/false,
                                                                     {});
    ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "Slice");

    ASSERT_FALSE(module->hasAttr("gfx.msl.kernel_family"));
    ASSERT_EQ(module->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.kernel.family").str(),
              "gather_scatter_indexed");
    const auto abi = ov::gfx_plugin::read_module_mpsrt_external_buffer_abi(module);
    ASSERT_TRUE(abi.valid);
    EXPECT_EQ(abi.buffer_count, 8u);
    EXPECT_EQ(abi.output_buffer_count, 1u);
    ASSERT_TRUE(abi.has_buffer_roles);
    ASSERT_EQ(abi.buffer_roles.size(), 8u);
    EXPECT_EQ(abi.buffer_roles[0], ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorInput);
    EXPECT_EQ(abi.buffer_roles[1], ov::gfx_plugin::GfxMpsrtExternalBufferRole::TensorOutput);
    for (size_t i = 2; i < abi.buffer_roles.size(); ++i) {
        EXPECT_EQ(abi.buffer_roles[i], ov::gfx_plugin::GfxMpsrtExternalBufferRole::RuntimeParams);
    }
}

TEST(GfxMlir, MslKernelManifestAdapterResolvesExactRolesWithoutSignatureHints) {
    mlir::MLIRContext ctx;
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

    const auto plan = ov::gfx_plugin::select_stage_optimization_plan(nullptr,
                                                                     ov::gfx_plugin::GpuBackend::Metal,
                                                                     "GatherElements",
                                                                     nullptr,
                                                                     ov::element::f32,
                                                                     /*has_bias=*/false,
                                                                     /*has_activation=*/false,
                                                                     /*has_batchnorm=*/false,
                                                                     {});
    ov::gfx_plugin::annotate_msl_module_with_stage_plan(module, plan, "GatherElements");
    EXPECT_FALSE(module->hasAttr("gfx.mpsrt.external_buffer_count"));

    ov::gfx_plugin::KernelSource source;
    source.module = module;
    source.entry_point = "gather_elements_kernel";
    ov::gfx_plugin::configure_msl_kernel_source_for_plan(source, "GatherElements");

    EXPECT_EQ(source.entry_point, "gather_scatter_indexed");
    ov::gfx_plugin::GfxMpsrtExternalBufferAbiPlan abi;
    ASSERT_TRUE(ov::gfx_plugin::gfx_mpsrt_external_buffer_abi_from_kernel_manifest(module,
                                                                                  abi,
                                                                                  source.signature.arg_count,
                                                                                  source.signature.output_arg_count));
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

TEST(GfxMlir, ReduceSumBuilderProducesModule) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 8400, 4, 16});
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {3});
    auto reduce = std::make_shared<ov::op::v1::ReduceSum>(input, axes, false);

    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
    auto module = ov::gfx_plugin::build_mlir_for_node(reduce, ctx);
    ASSERT_TRUE(module);
    ASSERT_TRUE(static_cast<bool>(module.lookupSymbol<mlir::func::FuncOp>("reduce_main")));
    EXPECT_TRUE(ov::gfx_plugin::mlir_supports_node(reduce));
    ASSERT_NO_THROW(ov::gfx_plugin::run_mlir_pipeline(module, /*use_alloca=*/true, /*use_parallel_loops=*/false));
}

TEST(GfxMlir, MatMulCodegenProducesMsl) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, true);

    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
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
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, true);

    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
    auto module = ov::gfx_plugin::build_mlir_for_node(matmul, ctx);
    ASSERT_TRUE(module);
    ASSERT_NO_THROW(ov::gfx_plugin::run_mlir_pipeline(module, /*use_alloca=*/true, /*use_parallel_loops=*/false));
}

TEST(GfxMlir, MatMulKernelPlanPipelineSucceeds) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, true);

    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
    auto module = ov::gfx_plugin::build_mlir_for_node(matmul, ctx);
    ASSERT_TRUE(module);

    auto plan_ctx = ov::gfx_plugin::build_mlir_kernel_plan(
        module,
        std::string{},
        matmul,
        /*output_args_override=*/0,
        /*extra_inputs=*/0,
        "matmul_plan_test",
        "gfx_kernel",
        [&](const ov::gfx_plugin::KernelArgMappingInfo& info) -> size_t {
            size_t func_results = info.func_results;
            if (func_results == 0) {
                func_results = matmul->get_output_size();
            }
            const auto sig = info.signature;
            return sig.total() ? sig.total() : (info.func_inputs + func_results);
        });
    auto src = plan_ctx.build_info.plan.to_source();
    ASSERT_TRUE(src.module);
    ASSERT_NO_THROW(ov::gfx_plugin::run_mlir_pipeline(src.module, /*use_alloca=*/true, /*use_parallel_loops=*/false));
}

TEST(GfxMlir, MatMulPipelineSucceedsAfterFusionPlan) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, true);
    auto res = std::make_shared<ov::op::v0::Result>(matmul);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{lhs, rhs}, "matmul_fusion_ctx");

    ov::gfx_plugin::FusionConfig fusion_cfg;
    fusion_cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, fusion_cfg);
    EXPECT_TRUE(plan.groups.empty());

    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
    auto module = ov::gfx_plugin::build_mlir_for_node(matmul, ctx);
    ASSERT_TRUE(module);
    ASSERT_NO_THROW(ov::gfx_plugin::run_mlir_pipeline(module, /*use_alloca=*/true, /*use_parallel_loops=*/false));
}

#if GFX_BACKEND_VULKAN_AVAILABLE
TEST(GfxMlir, BinaryBiasAddSpirvLoweringSucceeds) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 17, 5, 4});
    std::vector<float> bias_vals(17, 0.0f);
    for (size_t i = 0; i < bias_vals.size(); ++i) {
        bias_vals[i] = static_cast<float>((static_cast<int>(i % 13) - 6)) * 0.125f;
    }
    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                       ov::Shape{1, 17, 1, 1},
                                                       bias_vals);
    auto add = std::make_shared<ov::op::v1::Add>(param, bias);

    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
    auto module = ov::gfx_plugin::build_mlir_for_node(add, ctx);
    ASSERT_TRUE(module);

    std::string log;
    const auto spirv = ov::gfx_plugin::lower_to_spirv(module, "binary_bias_add", &log);
    ASSERT_FALSE(spirv.empty()) << log;
}

TEST(GfxMlir, MatMulCompactAbiSpirvLoweringExpandsOperandMetadata) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 2, 32, 400});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 2, 32, 400});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, true, false);

    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
    auto module = ov::gfx_plugin::build_mlir_for_node(matmul, ctx);
    ASSERT_TRUE(module);

    auto func = ov::gfx_plugin::get_entry_func(module);
    ASSERT_TRUE(func);
    ASSERT_GE(func.getNumArguments(), 3u);
    for (unsigned arg_idx = 0; arg_idx < 2; ++arg_idx) {
        func.setArgAttr(arg_idx,
                        "gfx.kernel_runtime_arg_index",
                        mlir::IntegerAttr::get(mlir::IntegerType::get(module.getContext(), 32),
                                               static_cast<int32_t>(arg_idx)));
    }
    mlir::Builder b(module.getContext());
    module->setAttr("gfx.fixed_arg_count", b.getI32IntegerAttr(3));
    module->setAttr("gfx.kernel_output_arg_count", b.getI32IntegerAttr(1));

    std::string log;
    const auto spirv = ov::gfx_plugin::lower_to_spirv(module, "gfx_kernel", &log);
    ASSERT_FALSE(spirv.empty()) << log;

    const auto kinds = ov::gfx_plugin::extract_kernel_operand_kinds(module);
    const auto arg_indices = ov::gfx_plugin::extract_kernel_operand_arg_indices(module);
    const auto scalar_values = ov::gfx_plugin::extract_kernel_scalar_values(module);

    ASSERT_EQ(kinds.size(), 9u);
    ASSERT_EQ(arg_indices.size(), 9u);
    ASSERT_EQ(scalar_values.size(), 6u);

    EXPECT_EQ(kinds, std::vector<int32_t>({0, 0, 0, 0, 1, 1, 0, 0, 1}));
    EXPECT_EQ(arg_indices, std::vector<int32_t>({-1, -1, -1, -1, 0, 1, -1, -1, 2}));
    EXPECT_EQ(scalar_values, std::vector<int32_t>({1, 0, 8, 400, 32, 0}));
}
#endif

TEST(GfxMlir, BinaryBiasAddI32MlirLoweringSucceeds) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2, 17, 5, 4});
    std::vector<int32_t> bias_vals(17, 0);
    for (size_t i = 0; i < bias_vals.size(); ++i) {
        bias_vals[i] = static_cast<int32_t>(i) - 8;
    }
    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                       ov::Shape{1, 17, 1, 1},
                                                       bias_vals);
    auto add = std::make_shared<ov::op::v1::Add>(param, bias);

    auto& ctx = ov::gfx_plugin::gfx_mlir_context();
    auto module = ov::gfx_plugin::build_mlir_for_node(add, ctx);
    ASSERT_TRUE(module);
    ASSERT_NO_THROW(ov::gfx_plugin::run_mlir_pipeline(module, /*use_alloca=*/true, /*use_parallel_loops=*/false));
}

TEST(GfxTransforms, MlirFusionConvBatchNormReluPlan) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 4, 4});
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                          ov::Shape{2, 3, 3, 3},
                                                          std::vector<float>(2 * 3 * 3 * 3, 1.f));
    auto conv = std::make_shared<ov::op::v1::Convolution>(param,
                                                          weights,
                                                          ov::Strides{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::Strides{1, 1});
    auto gamma = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{2},
                                                        std::vector<float>{1.f, 1.f});
    auto beta = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{2},
                                                       std::vector<float>{0.f, 0.f});
    auto mean = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{2},
                                                       std::vector<float>{0.f, 0.f});
    auto var = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{2},
                                                      std::vector<float>{1.f, 1.f});
    auto bn = std::make_shared<ov::op::v5::BatchNormInference>(conv, gamma, beta, mean, var, 1e-5);
    auto relu = std::make_shared<ov::op::v0::Relu>(bn);
    auto res = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param},
                                             "conv_bn_relu");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "ConvBatchNormAct" || group.node_indices.size() != 3) {
            continue;
        }
        const auto conv_idx = group.node_indices[0];
        const auto bn_idx = group.node_indices[1];
        const auto act_idx = group.node_indices[2];
        ASSERT_LT(conv_idx, ordered.size());
        ASSERT_LT(bn_idx, ordered.size());
        ASSERT_LT(act_idx, ordered.size());
        const auto& conv_node = ordered[conv_idx];
        const auto& bn_node = ordered[bn_idx];
        const auto& act_node = ordered[act_idx];
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
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                          ov::Shape{4, 4},
                                                          std::vector<float>(16, 0.25f));
    auto mm = std::make_shared<ov::op::v0::MatMul>(param, weights, false, false);
    auto relu = std::make_shared<ov::op::v0::Relu>(mm);
    auto res = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param},
                                             "matmul_relu");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "MatMulActivation" || group.node_indices.size() != 2) {
            continue;
        }
        const auto mm_idx = group.node_indices[0];
        const auto act_idx = group.node_indices[1];
        ASSERT_LT(mm_idx, ordered.size());
        ASSERT_LT(act_idx, ordered.size());
        const auto& mm_node = ordered[mm_idx];
        const auto& act_node = ordered[act_idx];
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
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                          ov::Shape{4, 4},
                                                          std::vector<float>(16, 0.25f));
    auto mm = std::make_shared<ov::op::v0::MatMul>(param, weights, false, false);
    auto gelu = std::make_shared<ov::op::v7::Gelu>(mm, ov::op::GeluApproximationMode::TANH);
    auto res = std::make_shared<ov::op::v0::Result>(gelu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param},
                                             "matmul_gelu");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "MatMulActivation" || group.node_indices.size() != 2) {
            continue;
        }
        const auto mm_idx = group.node_indices[0];
        const auto act_idx = group.node_indices[1];
        ASSERT_LT(mm_idx, ordered.size());
        ASSERT_LT(act_idx, ordered.size());
        const auto& mm_node = ordered[mm_idx];
        const auto& act_node = ordered[act_idx];
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
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                          ov::Shape{4, 4},
                                                          std::vector<float>(16, 0.25f));
    auto mm = std::make_shared<ov::op::v0::MatMul>(param, weights, false, false);
    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(mm);
    auto mul = std::make_shared<ov::op::v1::Multiply>(mm, sigmoid);
    auto res = std::make_shared<ov::op::v0::Result>(mul);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param},
                                             "matmul_swish");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "MatMulActivation" || group.node_indices.size() != 3) {
            continue;
        }
        const auto mm_idx = group.node_indices[0];
        const auto sig_idx = group.node_indices[1];
        const auto mul_idx = group.node_indices[2];
        ASSERT_LT(mm_idx, ordered.size());
        ASSERT_LT(sig_idx, ordered.size());
        ASSERT_LT(mul_idx, ordered.size());
        const auto& mm_node = ordered[mm_idx];
        const auto& sig_node = ordered[sig_idx];
        const auto& mul_node = ordered[mul_idx];
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
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                          ov::Shape{4, 4},
                                                          std::vector<float>(16, 0.5f));
    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                       ov::Shape{1, 4},
                                                       std::vector<float>(4, 0.1f));
    auto mm = std::make_shared<ov::op::v0::MatMul>(param, weights, false, false);
    auto add = std::make_shared<ov::op::v1::Add>(mm, bias);
    auto relu = std::make_shared<ov::op::v0::Relu>(add);
    auto res = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param},
                                             "matmul_bias_relu");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "MatMulBiasActivation" || group.node_indices.size() != 3) {
            continue;
        }
        const auto mm_idx = group.node_indices[0];
        const auto add_idx = group.node_indices[1];
        const auto act_idx = group.node_indices[2];
        ASSERT_LT(mm_idx, ordered.size());
        ASSERT_LT(add_idx, ordered.size());
        ASSERT_LT(act_idx, ordered.size());
        const auto& mm_node = ordered[mm_idx];
        const auto& add_node = ordered[add_idx];
        const auto& act_node = ordered[act_idx];
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
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                          ov::Shape{4, 4},
                                                          std::vector<float>(16, 0.5f));
    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                       ov::Shape{1, 4},
                                                       std::vector<float>(4, 0.1f));
    auto mm = std::make_shared<ov::op::v0::MatMul>(param, weights, false, false);
    auto add = std::make_shared<ov::op::v1::Add>(mm, bias);
    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(add);
    auto mul = std::make_shared<ov::op::v1::Multiply>(add, sigmoid);
    auto res = std::make_shared<ov::op::v0::Result>(mul);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param},
                                             "matmul_bias_swish");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "MatMulBiasActivation" || group.node_indices.size() != 4) {
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
        const auto& mm_node = ordered[mm_idx];
        const auto& add_node = ordered[add_idx];
        const auto& sig_node = ordered[sig_idx];
        const auto& mul_node = ordered[mul_idx];
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
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 4, 4});
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                          ov::Shape{2, 3, 3, 3},
                                                          std::vector<float>(2 * 3 * 3 * 3, 1.f));
    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                       ov::Shape{1, 2, 1, 1},
                                                       std::vector<float>(2, 0.25f));
    auto conv = std::make_shared<ov::op::v1::Convolution>(param,
                                                          weights,
                                                          ov::Strides{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::Strides{1, 1});
    auto add = std::make_shared<ov::op::v1::Add>(conv, bias);
    auto relu = std::make_shared<ov::op::v0::Relu>(add);
    auto res = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param},
                                             "conv_bias_relu");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "ConvBiasActivation" || group.node_indices.size() != 3) {
            continue;
        }
        const auto conv_idx = group.node_indices[0];
        const auto add_idx = group.node_indices[1];
        const auto act_idx = group.node_indices[2];
        ASSERT_LT(conv_idx, ordered.size());
        ASSERT_LT(add_idx, ordered.size());
        ASSERT_LT(act_idx, ordered.size());
        const auto& conv_node = ordered[conv_idx];
        const auto& add_node = ordered[add_idx];
        const auto& act_node = ordered[act_idx];
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

TEST(GfxTransforms, MlirFusionConvBiasSwishPlan) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 4, 4});
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                          ov::Shape{2, 3, 3, 3},
                                                          std::vector<float>(2 * 3 * 3 * 3, 1.f));
    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                       ov::Shape{1, 2, 1, 1},
                                                       std::vector<float>(2, 0.25f));
    auto conv = std::make_shared<ov::op::v1::Convolution>(param,
                                                          weights,
                                                          ov::Strides{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::Strides{1, 1});
    auto add = std::make_shared<ov::op::v1::Add>(conv, bias);
    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(add);
    auto mul = std::make_shared<ov::op::v1::Multiply>(add, sigmoid);
    auto res = std::make_shared<ov::op::v0::Result>(mul);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param},
                                             "conv_bias_swish");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
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
        const auto& conv_node = ordered[conv_idx];
        const auto& add_node = ordered[add_idx];
        const auto& sig_node = ordered[sig_idx];
        const auto& mul_node = ordered[mul_idx];
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

TEST(GfxTransforms, MlirFusionConvBiasPlan) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 4, 4});
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                          ov::Shape{2, 3, 3, 3},
                                                          std::vector<float>(2 * 3 * 3 * 3, 1.f));
    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                       ov::Shape{1, 2, 1, 1},
                                                       std::vector<float>(2, 0.125f));
    auto conv = std::make_shared<ov::op::v1::Convolution>(param,
                                                          weights,
                                                          ov::Strides{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::Strides{1, 1});
    auto add = std::make_shared<ov::op::v1::Add>(conv, bias);
    auto res = std::make_shared<ov::op::v0::Result>(add);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param},
                                             "conv_bias");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "ConvBias" || group.node_indices.size() != 2) {
            continue;
        }
        const auto conv_idx = group.node_indices[0];
        const auto add_idx = group.node_indices[1];
        ASSERT_LT(conv_idx, ordered.size());
        ASSERT_LT(add_idx, ordered.size());
        const auto& conv_node = ordered[conv_idx];
        const auto& add_node = ordered[add_idx];
        if (ov::as_type_ptr<const ov::op::v1::Convolution>(conv_node) &&
            ov::as_type_ptr<const ov::op::v1::Add>(add_node)) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionConvScalePlan) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 4, 4});
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                          ov::Shape{2, 3, 3, 3},
                                                          std::vector<float>(2 * 3 * 3 * 3, 1.f));
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                        ov::Shape{1, 2, 1, 1},
                                                        std::vector<float>{0.5f, 2.0f});
    auto conv = std::make_shared<ov::op::v1::Convolution>(param,
                                                          weights,
                                                          ov::Strides{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::Strides{1, 1});
    auto mul = std::make_shared<ov::op::v1::Multiply>(conv, scale);
    auto res = std::make_shared<ov::op::v0::Result>(mul);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "conv_scale");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "ConvScale" || group.node_indices.size() != 2 || !group.batchnorm.has_value()) {
            continue;
        }
        const auto conv_idx = group.node_indices[0];
        const auto mul_idx = group.node_indices[1];
        ASSERT_LT(conv_idx, ordered.size());
        ASSERT_LT(mul_idx, ordered.size());
        const auto& conv_node = ordered[conv_idx];
        const auto& mul_node = ordered[mul_idx];
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
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4, 4});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4, 4});
    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                       ov::Shape{1, 4, 1, 1},
                                                       std::vector<float>(4, 0.25f));
    auto add0 = std::make_shared<ov::op::v1::Add>(lhs, rhs);
    auto add1 = std::make_shared<ov::op::v1::Add>(add0, bias);
    auto relu = std::make_shared<ov::op::v0::Relu>(add1);
    auto res = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{lhs, rhs},
                                             "add_bias_relu");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "EltwiseBiasActivation" || group.node_indices.size() != 3) {
            continue;
        }
        const auto elt_idx = group.node_indices[0];
        const auto add_idx = group.node_indices[1];
        const auto act_idx = group.node_indices[2];
        ASSERT_LT(elt_idx, ordered.size());
        ASSERT_LT(add_idx, ordered.size());
        ASSERT_LT(act_idx, ordered.size());
        const auto& elt_node = ordered[elt_idx];
        const auto& add_node = ordered[add_idx];
        const auto& act_node = ordered[act_idx];
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
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4, 4});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4, 4});
    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                       ov::Shape{1, 4, 1, 1},
                                                       std::vector<float>(4, 0.25f));
    auto add0 = std::make_shared<ov::op::v1::Add>(lhs, rhs);
    auto add1 = std::make_shared<ov::op::v1::Add>(add0, bias);
    auto res = std::make_shared<ov::op::v0::Result>(add1);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{lhs, rhs},
                                             "add_bias");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "EltwiseBias" || group.node_indices.size() != 2) {
            continue;
        }
        const auto elt_idx = group.node_indices[0];
        const auto add_idx = group.node_indices[1];
        ASSERT_LT(elt_idx, ordered.size());
        ASSERT_LT(add_idx, ordered.size());
        const auto& elt_node = ordered[elt_idx];
        const auto& add_node = ordered[add_idx];
        if (ov::as_type_ptr<const ov::op::v1::Add>(elt_node) &&
            ov::as_type_ptr<const ov::op::v1::Add>(add_node)) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST(GfxTransforms, MlirFusionMatMulBiasPlan) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                          ov::Shape{4, 4},
                                                          std::vector<float>(16, 0.5f));
    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                       ov::Shape{1, 4},
                                                       std::vector<float>(4, 0.2f));
    auto mm = std::make_shared<ov::op::v0::MatMul>(param, weights, false, false);
    auto add = std::make_shared<ov::op::v1::Add>(mm, bias);
    auto res = std::make_shared<ov::op::v0::Result>(add);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param},
                                             "matmul_bias");

    ov::gfx_plugin::FusionConfig cfg;
    cfg.enable_fusion = true;
    auto plan = ov::gfx_plugin::build_fusion_plan(model, cfg);
    ASSERT_FALSE(plan.groups.empty());

    bool found = false;
    const auto ordered = model->get_ordered_ops();
    for (const auto& group : plan.groups) {
        if (group.kind != "MatMulBias" || group.node_indices.size() != 2) {
            continue;
        }
        const auto mm_idx = group.node_indices[0];
        const auto add_idx = group.node_indices[1];
        ASSERT_LT(mm_idx, ordered.size());
        ASSERT_LT(add_idx, ordered.size());
        const auto& mm_node = ordered[mm_idx];
        const auto& add_node = ordered[add_idx];
        if (ov::as_type_ptr<const ov::op::v0::MatMul>(mm_node) &&
            ov::as_type_ptr<const ov::op::v1::Add>(add_node)) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST(GfxTransforms, CommonOptimizationsConstantFolding) {
    // Constant folding should remove Add/Relu when inputs are compile-time constants.
    auto c0 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{2}, std::vector<float>{1.f, -2.f});
    auto c1 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{2}, std::vector<float>{3.f, 4.f});
    auto add = std::make_shared<ov::op::v1::Add>(c0, c1);
    auto relu = std::make_shared<ov::op::v0::Relu>(add);
    auto res = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{}, "const_fold");

    auto transformed = ov::gfx_plugin::transforms::run_pipeline(model, ov::gfx_plugin::GpuBackend::Metal);

    // Expect only Result + Constant to remain after folding (Add and Relu folded away).
    size_t constants = 0;
    size_t adds = 0;
    size_t relus = 0;
    for (const auto& op : transformed->get_ops()) {
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
