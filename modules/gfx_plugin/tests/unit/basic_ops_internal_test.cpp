// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/constant.hpp"
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
#include "openvino/op/gelu.hpp"
#include "openvino/op/hsigmoid.hpp"
#include "openvino/op/hswish.hpp"
#include "../gfx_test_utils.hpp"
#define HAS_OV_LAYER_NORM 0
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sigmoid.hpp"
#include "transforms/pipeline.hpp"
#include "transforms/fusion_pass.hpp"
#include "plugin/gfx_backend_config.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"

namespace {

inline void gfx_try_catch_fail(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const ov::Exception& e) {
        const std::string msg = e.what();
        if (msg.find("device-only") != std::string::npos ||
            msg.find("output tensors are device-only") != std::string::npos) {
            FAIL() << "GFX outputs are device-only; host readback required for test";
        }
        if (msg.find("GFX Vulkan") != std::string::npos ||
            msg.find("SPIR-V") != std::string::npos ||
            msg.find("spirv") != std::string::npos ||
            msg.find("vulkan") != std::string::npos) {
            FAIL() << "Vulkan backend did not support this case: " << msg;
            return;
        }
        throw;
    }
}

std::string gfx_skip_reason;

bool register_gfx_plugin(ov::Core& core) {
    gfx_skip_reason.clear();
    // Always require GFX plugin to be available; fail fast if not.
    try {
#ifdef GFX_PLUGIN_PATH
        const char* env_path = std::getenv("GFX_PLUGIN_PATH");
        const char* path = (env_path && *env_path) ? env_path : GFX_PLUGIN_PATH;
        core.register_plugin(path, "GFX");
#else
        // Assume default discovery if path macro is absent.
#endif
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        if (msg.find("already registered") == std::string::npos) {
            throw std::runtime_error(std::string("GFX plugin unavailable: ") + e.what());
        }
    }
    try {
        const auto backend = core.get_property("GFX", "GFX_BACKEND").as<std::string>();
        if (backend.empty()) {
            gfx_skip_reason = "GFX backend not available";
            return false;
        }
    } catch (const std::exception& e) {
        gfx_skip_reason = std::string("GFX backend property unavailable: ") + e.what();
        return false;
    }
    try {
        ov::test::utils::register_template_plugin(core);
    } catch (const std::exception& e) {
        gfx_skip_reason = std::string("TEMPLATE plugin unavailable: ") + e.what();
        return false;
    }
    return true;
}

std::string reference_device(const ov::Core& core) {
    const auto devices = core.get_available_devices();
    if (std::find(devices.begin(), devices.end(), "TEMPLATE") != devices.end()) {
        return "TEMPLATE";
    }
    throw std::runtime_error("TEMPLATE reference device not available");
}

void expect_allclose(const ov::Tensor& a, const ov::Tensor& b, float atol = 1e-5f, float rtol = 0.f) {
    ASSERT_EQ(a.get_element_type(), ov::element::f32);
    ASSERT_EQ(b.get_element_type(), ov::element::f32);
    ASSERT_EQ(a.get_byte_size(), b.get_byte_size());
    auto* pa = a.data<const float>();
    auto* pb = b.data<const float>();
    size_t count = a.get_size();
    for (size_t i = 0; i < count; ++i) {
        float diff = std::abs(pa[i] - pb[i]);
        float thresh = std::max(atol, rtol * std::abs(pa[i]));
        ASSERT_LE(diff, thresh) << "Mismatch at index " << i << ": " << pa[i] << " vs " << pb[i];
    }
    return true;
}

void expect_finite(const ov::Tensor& t) {
    const float* p = t.data<const float>();
    for (size_t i = 0; i < t.get_size(); ++i) {
        ASSERT_TRUE(std::isfinite(p[i])) << "Non-finite at " << i << ": " << p[i];
    }
    return true;
}

void expect_shape_type(const ov::Tensor& t, const ov::Shape& shape, ov::element::Type type = ov::element::f32) {
    ASSERT_EQ(t.get_element_type(), type);
    ASSERT_EQ(t.get_shape(), shape);
}

ov::Tensor get_output_or_skip(ov::InferRequest& req, size_t idx = 0) {
    return req.get_output_tensor(idx);
}



inline void expect_or_skip_allclose(const ov::Tensor& a, const ov::Tensor& b, float atol, float rtol, const char* /*msg*/) {
    expect_allclose(a, b, atol, rtol);
}

}  // namespace
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
