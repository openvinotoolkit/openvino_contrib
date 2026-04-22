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
#include "mlir/mlir_kernel_plan_utils.hpp"
#include "mlir/mlir_support.hpp"
#include "mlir/codegen_common.hpp"
#include "mlir/mlir_passes.hpp"
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

    auto transformed = ov::gfx_plugin::transforms::run_pipeline(model);

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
