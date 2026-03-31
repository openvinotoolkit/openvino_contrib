// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "runtime/gfx_stage_policy.hpp"

using namespace ov::gfx_plugin;

namespace {

class FakeDeviceInfoBufferManager final : public GpuBufferManager {
public:
    explicit FakeDeviceInfoBufferManager(GpuExecutionDeviceInfo info) : m_info(std::move(info)) {}

    std::optional<GpuExecutionDeviceInfo> query_execution_device_info() const override {
        return m_info;
    }

private:
    GpuExecutionDeviceInfo m_info;
};

std::shared_ptr<const ov::Node> make_pointwise_conv_node() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 64, 64, 64});
    auto weights = ov::op::v0::Constant::create(ov::element::f16,
                                                ov::Shape{128, 64, 1, 1},
                                                std::vector<float>(128 * 64, 1.f));
    return std::make_shared<ov::op::v1::Convolution>(input,
                                                     weights,
                                                     ov::Strides{1, 1},
                                                     ov::CoordinateDiff{0, 0},
                                                     ov::CoordinateDiff{0, 0},
                                                     ov::Strides{1, 1});
}

std::shared_ptr<const ov::Node> make_depthwise_group_conv_node() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 32, 32, 32});
    auto weights = ov::op::v0::Constant::create(ov::element::f16,
                                                ov::Shape{32, 1, 1, 1, 1},
                                                std::vector<float>(32, 1.f));
    return std::make_shared<ov::op::v1::GroupConvolution>(input,
                                                          weights,
                                                          ov::Strides{1, 1},
                                                          ov::CoordinateDiff{0, 0},
                                                          ov::CoordinateDiff{0, 0},
                                                          ov::Strides{1, 1});
}

std::shared_ptr<const ov::Node> make_large_add_node() {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 64, 80, 80});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 64, 80, 80});
    return std::make_shared<ov::op::v1::Add>(lhs, rhs);
}

std::shared_ptr<const ov::Node> make_large_concat_node() {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 64, 80, 80});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 64, 80, 80});
    return std::make_shared<ov::op::v0::Concat>(ov::OutputVector{lhs, rhs}, 1);
}

std::shared_ptr<const ov::Node> make_large_transpose_node() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 32, 80, 80});
    auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 3, 1});
    return std::make_shared<ov::op::v1::Transpose>(input, perm);
}

std::shared_ptr<const ov::Node> make_large_chunked_conv_node() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 64, 64, 64});
    auto weights = ov::op::v0::Constant::create(ov::element::f16,
                                                ov::Shape{128, 64, 5, 5},
                                                std::vector<float>(128 * 64 * 5 * 5, 1.f));
    return std::make_shared<ov::op::v1::Convolution>(input,
                                                     weights,
                                                     ov::Strides{1, 1},
                                                     ov::CoordinateDiff{2, 2},
                                                     ov::CoordinateDiff{2, 2},
                                                     ov::Strides{1, 1});
}

std::shared_ptr<const ov::Node> make_medium_chunked_conv_node() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 64, 32, 32});
    auto weights = ov::op::v0::Constant::create(ov::element::f16,
                                                ov::Shape{64, 64, 5, 5},
                                                std::vector<float>(64 * 64 * 5 * 5, 1.f));
    return std::make_shared<ov::op::v1::Convolution>(input,
                                                     weights,
                                                     ov::Strides{1, 1},
                                                     ov::CoordinateDiff{2, 2},
                                                     ov::CoordinateDiff{2, 2},
                                                     ov::Strides{1, 1});
}

std::shared_ptr<const ov::Node> make_light_depthwise_group_conv_node() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 64, 32, 32});
    auto weights = ov::op::v0::Constant::create(ov::element::f16,
                                                ov::Shape{64, 1, 1, 3, 3},
                                                std::vector<float>(64 * 3 * 3, 1.f));
    return std::make_shared<ov::op::v1::GroupConvolution>(input,
                                                          weights,
                                                          ov::Strides{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::Strides{1, 1});
}

std::shared_ptr<const ov::Node> make_light_spatial3x3_conv_node() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 16, 160, 160});
    auto weights = ov::op::v0::Constant::create(ov::element::f16,
                                                ov::Shape{8, 16, 3, 3},
                                                std::vector<float>(8 * 16 * 3 * 3, 1.f));
    return std::make_shared<ov::op::v1::Convolution>(input,
                                                     weights,
                                                     ov::Strides{1, 1},
                                                     ov::CoordinateDiff{1, 1},
                                                     ov::CoordinateDiff{1, 1},
                                                     ov::Strides{1, 1});
}

std::pair<std::shared_ptr<const ov::Node>, std::shared_ptr<const ov::Node>> make_light_spatial3x3_conv_chain() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 16, 160, 160});
    auto weights0 = ov::op::v0::Constant::create(ov::element::f16,
                                                 ov::Shape{8, 16, 3, 3},
                                                 std::vector<float>(8 * 16 * 3 * 3, 1.f));
    auto conv0 = std::make_shared<ov::op::v1::Convolution>(input,
                                                           weights0,
                                                           ov::Strides{1, 1},
                                                           ov::CoordinateDiff{1, 1},
                                                           ov::CoordinateDiff{1, 1},
                                                           ov::Strides{1, 1});
    auto weights1 = ov::op::v0::Constant::create(ov::element::f16,
                                                 ov::Shape{8, 8, 3, 3},
                                                 std::vector<float>(8 * 8 * 3 * 3, 1.f));
    auto conv1 = std::make_shared<ov::op::v1::Convolution>(conv0,
                                                           weights1,
                                                           ov::Strides{1, 1},
                                                           ov::CoordinateDiff{1, 1},
                                                           ov::CoordinateDiff{1, 1},
                                                           ov::Strides{1, 1});
    return {conv0, conv1};
}

std::shared_ptr<const ov::Node> make_concat_fed_spatial3x3_conv_node() {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 8, 160, 160});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 8, 160, 160});
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{lhs, rhs}, 1);
    auto weights = ov::op::v0::Constant::create(ov::element::f16,
                                                ov::Shape{8, 16, 3, 3},
                                                std::vector<float>(8 * 16 * 3 * 3, 1.f));
    return std::make_shared<ov::op::v1::Convolution>(concat,
                                                     weights,
                                                     ov::Strides{1, 1},
                                                     ov::CoordinateDiff{1, 1},
                                                     ov::CoordinateDiff{1, 1},
                                                     ov::Strides{1, 1});
}

std::shared_ptr<const ov::Node> make_chunked_conv_with_reshape_consumer() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 64, 32, 32});
    auto weights = ov::op::v0::Constant::create(ov::element::f16,
                                                ov::Shape{64, 64, 5, 5},
                                                std::vector<float>(64 * 64 * 5 * 5, 1.f));
    auto conv = std::make_shared<ov::op::v1::Convolution>(input,
                                                          weights,
                                                          ov::Strides{1, 1},
                                                          ov::CoordinateDiff{2, 2},
                                                          ov::CoordinateDiff{2, 2},
                                                          ov::Strides{1, 1});
    auto pattern = ov::op::v0::Constant::create(ov::element::i64,
                                                ov::Shape{4},
                                                std::vector<int64_t>{1, 32, 32, 64});
    (void)std::make_shared<ov::op::v1::Reshape>(conv, pattern, false);
    return conv;
}

}  // namespace

TEST(GfxStagePolicyTest, VulkanPointwiseConvolutionStaysOnSharedMlirRouteAndAllowsBiasAndReluFusion) {
    const auto conv = make_pointwise_conv_node();
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Vulkan,
                                                     "Convolution",
                                                     conv,
                                                     ov::element::f16,
                                                     /*has_bias=*/true,
                                                     /*has_activation=*/true,
                                                     /*has_batchnorm=*/false,
                                                     {});

    EXPECT_TRUE(plan.post_ops.bias);
    EXPECT_TRUE(plan.post_ops.activation);
    EXPECT_TRUE(plan.post_ops.batchnorm);
    EXPECT_EQ(plan.conv.kind, GfxConvRouteKind::None);
    EXPECT_EQ(plan.conv.family, GfxConvFamily::Unknown);
    EXPECT_EQ(plan.conv.algorithm.kind, GfxConvAlgorithmKind::None);
    EXPECT_FALSE(plan.execution.submit.isolate);
    EXPECT_EQ(plan.execution.submit.weight, 4u);
}

TEST(GfxStagePolicyTest, VulkanConvolutionAllowsOnlyReluActivationFusion) {
    EXPECT_TRUE(allow_stage_activation_fusion(GpuBackend::Vulkan, "Convolution", ActivationKind::Relu));
    EXPECT_FALSE(allow_stage_activation_fusion(GpuBackend::Vulkan, "Convolution", ActivationKind::Sigmoid));
    EXPECT_FALSE(allow_stage_activation_fusion(GpuBackend::Vulkan, "GroupConvolution", ActivationKind::Relu));
}

TEST(GfxStagePolicyTest, VulkanGroupConvolutionStaysConservativeWithBiasOrActivation) {
    const auto gconv = make_depthwise_group_conv_node();
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Vulkan,
                                                     "GroupConvolution",
                                                     gconv,
                                                     ov::element::f16,
                                                     /*has_bias=*/true,
                                                     /*has_activation=*/true,
                                                     /*has_batchnorm=*/false,
                                                     {});

    EXPECT_FALSE(plan.post_ops.bias);
    EXPECT_FALSE(plan.post_ops.activation);
    EXPECT_EQ(plan.conv.kind, GfxConvRouteKind::None);
    EXPECT_EQ(plan.conv.algorithm.kind, GfxConvAlgorithmKind::None);
}

TEST(GfxStagePolicyTest, VulkanMatMulUsesAdaptiveSubmitWindow) {
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Vulkan,
                                                     "MatMul",
                                                     nullptr,
                                                     ov::element::f16,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     {});

    EXPECT_EQ(plan.archetype, GfxStageArchetype::MatMul);
    EXPECT_FALSE(plan.execution.submit.isolate);
    EXPECT_GE(plan.execution.submit.weight, 8u);
}

TEST(GfxStagePolicyTest, VulkanLargeBinaryChunkedStageUsesIsolatedSubmitWindow) {
    const auto add = make_large_add_node();
    GfxStageRuntimeTraits traits{};
    traits.binary_chunked = true;
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Vulkan,
                                                     "Add",
                                                     add,
                                                     ov::element::f16,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     traits);

    EXPECT_EQ(plan.archetype, GfxStageArchetype::BinaryElementwise);
    EXPECT_TRUE(plan.execution.submit.isolate);
    EXPECT_GE(plan.execution.submit.weight, 8u);
}

TEST(GfxStagePolicyTest, VulkanLargeConcatUsesIsolatedSubmitWindow) {
    const auto concat = make_large_concat_node();
    GfxStageRuntimeTraits traits{};
    traits.split_concat_chunked = true;
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Vulkan,
                                                     "Concat",
                                                     concat,
                                                     ov::element::f16,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     traits);

    EXPECT_EQ(plan.archetype, GfxStageArchetype::SplitConcat);
    EXPECT_TRUE(plan.execution.submit.isolate);
    EXPECT_GE(plan.execution.submit.weight, 8u);
}

TEST(GfxStagePolicyTest, VulkanConstrainedLargeBinaryChunkedStageCanShareSubmitWindow) {
    FakeDeviceInfoBufferManager buffer_manager({GpuBackend::Vulkan, "test-rpi", 16u, 16u, 256u, {256u, 256u, 64u}});
    const auto add = make_large_add_node();
    GfxStageRuntimeTraits traits{};
    traits.binary_chunked = true;
    const auto plan = select_stage_optimization_plan(&buffer_manager,
                                                     GpuBackend::Vulkan,
                                                     "Add",
                                                     add,
                                                     ov::element::f16,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     traits);

    EXPECT_EQ(plan.archetype, GfxStageArchetype::BinaryElementwise);
    EXPECT_FALSE(plan.execution.submit.isolate);
    EXPECT_GE(plan.execution.submit.weight, 8u);
}

TEST(GfxStagePolicyTest, VulkanConstrainedLargeConcatCanShareSubmitWindow) {
    FakeDeviceInfoBufferManager buffer_manager({GpuBackend::Vulkan, "test-rpi", 16u, 16u, 256u, {256u, 256u, 64u}});
    const auto concat = make_large_concat_node();
    GfxStageRuntimeTraits traits{};
    traits.split_concat_chunked = true;
    const auto plan = select_stage_optimization_plan(&buffer_manager,
                                                     GpuBackend::Vulkan,
                                                     "Concat",
                                                     concat,
                                                     ov::element::f16,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     traits);

    EXPECT_EQ(plan.archetype, GfxStageArchetype::SplitConcat);
    EXPECT_FALSE(plan.execution.submit.isolate);
    EXPECT_EQ(plan.execution.submit.weight, 4u);
}

TEST(GfxStagePolicyTest, VulkanLargeTransposeUsesChunkedSubmitTraits) {
    const auto transpose = make_large_transpose_node();
    GfxStageRuntimeTraits traits{};
    traits.transpose_chunked = true;
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Vulkan,
                                                     "Transpose",
                                                     transpose,
                                                     ov::element::f16,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     traits);

    EXPECT_EQ(plan.archetype, GfxStageArchetype::Layout);
    EXPECT_FALSE(plan.execution.submit.isolate);
    EXPECT_EQ(plan.execution.submit.weight, 4u);
}

TEST(GfxStagePolicyTest, VulkanChunkedConvolutionUsesIsolatedSubmitWindow) {
    const auto conv = make_large_chunked_conv_node();
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Vulkan,
                                                     "Convolution",
                                                     conv,
                                                     ov::element::f16,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     {});

    EXPECT_EQ(plan.archetype, GfxStageArchetype::Convolution);
    EXPECT_EQ(plan.conv.kind, GfxConvRouteKind::Chunked);
    EXPECT_TRUE(plan.execution.submit.isolate);
    EXPECT_GE(plan.execution.submit.weight, 8u);
}

TEST(GfxStagePolicyTest, VulkanMediumChunkedConvolutionUsesIsolatedSubmitWindow) {
    const auto conv = make_medium_chunked_conv_node();
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Vulkan,
                                                     "Convolution",
                                                     conv,
                                                     ov::element::f16,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     {});

    EXPECT_EQ(plan.archetype, GfxStageArchetype::Convolution);
    EXPECT_EQ(plan.conv.kind, GfxConvRouteKind::Chunked);
    EXPECT_TRUE(plan.execution.submit.isolate);
    EXPECT_GE(plan.execution.submit.weight, 8u);
}

TEST(GfxStagePolicyTest, VulkanLightDepthwiseChunkedConvolutionUsesIsolatedSubmitWindow) {
    const auto gconv = make_light_depthwise_group_conv_node();
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Vulkan,
                                                     "GroupConvolution",
                                                     gconv,
                                                     ov::element::f16,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     {});

    EXPECT_EQ(plan.archetype, GfxStageArchetype::GroupConvolution);
    EXPECT_EQ(plan.conv.kind, GfxConvRouteKind::GroupChunked);
    EXPECT_TRUE(plan.execution.submit.isolate);
    EXPECT_GE(plan.execution.submit.weight, 8u);
}

TEST(GfxStagePolicyTest, VulkanLightSpatial3x3ConvolutionFallsBackToSharedMlirRoute) {
    const auto conv = make_light_spatial3x3_conv_node();
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Vulkan,
                                                     "Convolution",
                                                     conv,
                                                     ov::element::f16,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     {});

    EXPECT_EQ(plan.archetype, GfxStageArchetype::Convolution);
    EXPECT_EQ(plan.conv.kind, GfxConvRouteKind::None);
    EXPECT_EQ(plan.conv.algorithm.kind, GfxConvAlgorithmKind::None);
    EXPECT_TRUE(plan.execution.submit.isolate);
}

TEST(GfxStagePolicyTest, VulkanSerialConvolutionChainCanShareAdaptiveSubmitWindow) {
    const auto [conv0, conv1] = make_light_spatial3x3_conv_chain();

    const auto first_plan = select_stage_optimization_plan(nullptr,
                                                           GpuBackend::Vulkan,
                                                           "Convolution",
                                                           conv0,
                                                           ov::element::f16,
                                                           /*has_bias=*/false,
                                                           /*has_activation=*/false,
                                                           /*has_batchnorm=*/false,
                                                           {});
    const auto second_plan = select_stage_optimization_plan(nullptr,
                                                            GpuBackend::Vulkan,
                                                            "Convolution",
                                                            conv1,
                                                            ov::element::f16,
                                                            /*has_bias=*/false,
                                                            /*has_activation=*/false,
                                                            /*has_batchnorm=*/false,
                                                            {});

    EXPECT_EQ(first_plan.conv.kind, GfxConvRouteKind::None);
    EXPECT_EQ(second_plan.conv.kind, GfxConvRouteKind::None);
    EXPECT_FALSE(first_plan.execution.submit.isolate);
    EXPECT_FALSE(second_plan.execution.submit.isolate);
    EXPECT_GE(first_plan.execution.submit.weight, 8u);
    EXPECT_GE(second_plan.execution.submit.weight, 8u);
}

TEST(GfxStagePolicyTest, VulkanConcatAdjacentConvolutionRemainsIsolated) {
    const auto conv = make_concat_fed_spatial3x3_conv_node();
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Vulkan,
                                                     "Convolution",
                                                     conv,
                                                     ov::element::f16,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     {});

    EXPECT_EQ(plan.conv.kind, GfxConvRouteKind::None);
    EXPECT_TRUE(plan.execution.submit.isolate);
}

TEST(GfxStagePolicyTest, VulkanLayoutAdjacentChunkedConvolutionRemainsIsolated) {
    const auto conv = make_chunked_conv_with_reshape_consumer();
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Vulkan,
                                                     "Convolution",
                                                     conv,
                                                     ov::element::f16,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     {});

    EXPECT_EQ(plan.conv.kind, GfxConvRouteKind::Chunked);
    EXPECT_TRUE(plan.execution.submit.isolate);
}
