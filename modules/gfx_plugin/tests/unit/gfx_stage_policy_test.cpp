// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "backends/metal/runtime/mpsrt/mpsrt_model.hpp"
#include "runtime/gfx_mpsrt_abi.hpp"
#include "runtime/gfx_mpsrt_builder_plan.hpp"
#include "runtime/gfx_mpsrt_plan.hpp"
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

std::shared_ptr<const ov::Node> make_matmul_node() {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 128, 256});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 256, 64});
    return std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, false);
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

TEST(GfxStagePolicyTest, MetalConvolutionPlansAppleMpsImageStorage) {
    const auto conv = make_pointwise_conv_node();
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Metal,
                                                     "Convolution",
                                                     conv,
                                                     ov::element::f16,
                                                     /*has_bias=*/true,
                                                     /*has_activation=*/true,
                                                     /*has_batchnorm=*/false,
                                                     {});

    EXPECT_EQ(plan.archetype, GfxStageArchetype::Convolution);
    EXPECT_EQ(plan.placement.domain, GfxStageBackendDomain::AppleMps);
    EXPECT_EQ(plan.placement.storage, GfxStageStorageKind::Image);
    EXPECT_TRUE(plan.placement.uses_vendor_primitive);
    EXPECT_FALSE(plan.placement.uses_custom_kernel);
    EXPECT_TRUE(plan.post_ops.bias);
    EXPECT_TRUE(plan.post_ops.activation);
}

TEST(GfxStagePolicyTest, MetalMatMulPlansAppleMpsMatrixStorage) {
    const auto matmul = make_matmul_node();
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Metal,
                                                     "MatMul",
                                                     matmul,
                                                     ov::element::f16,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     {});

    EXPECT_EQ(plan.archetype, GfxStageArchetype::MatMul);
    EXPECT_EQ(plan.placement.domain, GfxStageBackendDomain::AppleMps);
    EXPECT_EQ(plan.placement.storage, GfxStageStorageKind::Matrix);
    EXPECT_EQ(plan.placement.specialization_key, "apple_mps:matrix:MatMul");
}

TEST(GfxStagePolicyTest, MetalBinaryElementwiseStaysInMslBufferDomain) {
    const auto add = make_large_add_node();
    GfxStageRuntimeTraits traits{};
    traits.binary_chunked = true;
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Metal,
                                                     "Add",
                                                     add,
                                                     ov::element::f16,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     traits);

    EXPECT_EQ(plan.archetype, GfxStageArchetype::BinaryElementwise);
    EXPECT_EQ(plan.placement.domain, GfxStageBackendDomain::AppleMsl);
    EXPECT_EQ(plan.placement.storage, GfxStageStorageKind::Buffer);
    EXPECT_FALSE(plan.placement.uses_vendor_primitive);
    EXPECT_TRUE(plan.placement.uses_custom_kernel);
}

TEST(GfxStagePolicyTest, VulkanPlacementRemainsSharedSpirvBufferDomain) {
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

    EXPECT_EQ(plan.placement.domain, GfxStageBackendDomain::Spirv);
    EXPECT_EQ(plan.placement.storage, GfxStageStorageKind::Buffer);
    EXPECT_TRUE(plan.placement.uses_custom_kernel);
}

TEST(GfxStagePolicyTest, MpsrtImageTensorDescriptorUsesLogicalNchwShapeAndImageFields) {
    const auto desc = gfx_mpsrt_make_tensor_desc({1, 64, 32, 16},
                                                 ov::element::f16,
                                                 GfxStageStorageKind::Image);

    EXPECT_EQ(desc.dtype, GfxMpsrtDType::F16);
    EXPECT_EQ(desc.storage, GfxMpsrtStorage::Image);
    EXPECT_EQ(desc.layout, GfxMpsrtLayout::NCHW);
    EXPECT_EQ(desc.rank, 4u);
    EXPECT_EQ(desc.dims[0], 1u);
    EXPECT_EQ(desc.dims[1], 64u);
    EXPECT_EQ(desc.dims[2], 32u);
    EXPECT_EQ(desc.dims[3], 16u);
    EXPECT_EQ(desc.strides[0], 64 * 32 * 16);
    EXPECT_EQ(desc.strides[1], 32 * 16);
    EXPECT_EQ(desc.strides[2], 16);
    EXPECT_EQ(desc.strides[3], 1);
    EXPECT_EQ(desc.byte_length, 1u * 64u * 32u * 16u * 2u);
    EXPECT_EQ(desc.image_batch, 1u);
    EXPECT_EQ(desc.image_feature_channels, 64u);
    EXPECT_EQ(desc.image_height, 32u);
    EXPECT_EQ(desc.image_width, 16u);
}

TEST(GfxStagePolicyTest, MpsrtMatrixTensorDescriptorFlattensBatchToMatrixCount) {
    const auto desc = gfx_mpsrt_make_tensor_desc({2, 128, 64},
                                                 ov::element::f32,
                                                 GfxStageStorageKind::Matrix);

    EXPECT_EQ(desc.dtype, GfxMpsrtDType::F32);
    EXPECT_EQ(desc.storage, GfxMpsrtStorage::Matrix);
    EXPECT_EQ(desc.layout, GfxMpsrtLayout::RowMajor);
    EXPECT_EQ(desc.matrix_count, 2u);
    EXPECT_EQ(desc.matrix_rows, 128u);
    EXPECT_EQ(desc.matrix_columns, 64u);
    EXPECT_EQ(desc.matrix_row_bytes, 64u * 4u);
    EXPECT_EQ(desc.byte_length, 2u * 128u * 64u * 4u);
}

TEST(GfxStagePolicyTest, MpsrtUnsignedIntegerDescriptorKeepsDTypeAndByteLength) {
    const auto desc = gfx_mpsrt_make_tensor_desc({4, 8},
                                                 ov::element::u32,
                                                 GfxStageStorageKind::Buffer);

    EXPECT_EQ(desc.dtype, GfxMpsrtDType::U32);
    EXPECT_EQ(gfx_mpsrt_dtype_from_name(gfx_mpsrt_dtype_name(desc.dtype)), GfxMpsrtDType::U32);
    EXPECT_EQ(desc.storage, GfxMpsrtStorage::Buffer);
    EXPECT_EQ(desc.layout, GfxMpsrtLayout::Linear);
    EXPECT_EQ(desc.byte_length, 4u * 8u * 4u);
}

TEST(GfxStagePolicyTest, MpsrtTensorAbiDescriptorRoundTripsWithoutCppContainers) {
    auto desc = gfx_mpsrt_make_tensor_desc({2, 128, 64},
                                           ov::element::f16,
                                           GfxStageStorageKind::Matrix,
                                           GfxMpsrtTensorFlagExternalIo);
    desc.byte_offset = 256;
    const auto abi = gfx_mpsrt_to_abi_desc(desc);
    const auto restored = gfx_mpsrt_from_abi_desc(abi);

    EXPECT_EQ(abi.dtype, static_cast<uint32_t>(GfxMpsrtDType::F16));
    EXPECT_EQ(abi.storage, static_cast<uint32_t>(GfxMpsrtStorage::Matrix));
    EXPECT_EQ(abi.layout, static_cast<uint32_t>(GfxMpsrtLayout::RowMajor));
    EXPECT_EQ(abi.rank, 3u);
    EXPECT_EQ(abi.dims[0], 2u);
    EXPECT_EQ(abi.dims[1], 128u);
    EXPECT_EQ(abi.dims[2], 64u);
    EXPECT_EQ(abi.strides[0], 128 * 64);
    EXPECT_EQ(abi.byte_offset, 256u);
    EXPECT_EQ(abi.byte_length, 2u * 128u * 64u * 2u);
    EXPECT_EQ(restored.dtype, desc.dtype);
    EXPECT_EQ(restored.storage, desc.storage);
    EXPECT_EQ(restored.layout, desc.layout);
    EXPECT_EQ(restored.matrix_rows, 128u);
    EXPECT_EQ(restored.matrix_columns, 64u);
    EXPECT_EQ(restored.flags, GfxMpsrtTensorFlagExternalIo);
}

TEST(GfxStagePolicyTest, MpsrtStageRecordKeyIsStableForMetalMatMul) {
    const auto matmul = make_matmul_node();
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Metal,
                                                     "MatMul",
                                                     matmul,
                                                     ov::element::f16,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     {});
    const auto desc = gfx_mpsrt_make_stage_desc(plan, "MatMul");

    EXPECT_EQ(desc.kind, GfxMpsrtStageKind::MPSGemm);
    EXPECT_EQ(desc.kernel_name, "mps_gemm");
    EXPECT_EQ(desc.builder_symbol, "ovgfx_mpsrt_encode_gemm");
    EXPECT_EQ(gfx_mpsrt_stage_record_key(desc),
              "mps_gemm|apple_mps|matrix|matrix|row_major|MatMul|apple_mps:matrix:MatMul");
}

TEST(GfxStagePolicyTest, MpsrtStageRecordKeyUsesNhwc4ForMetalConvolutionStage) {
    const auto conv = make_pointwise_conv_node();
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Metal,
                                                     "Convolution",
                                                     conv,
                                                     ov::element::f16,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     {});
    const auto desc = gfx_mpsrt_make_stage_desc(plan, "Convolution");

    EXPECT_EQ(desc.kind, GfxMpsrtStageKind::MPSConv2D);
    EXPECT_EQ(desc.layout, GfxMpsrtLayout::NHWC4);
    EXPECT_EQ(desc.builder_symbol, "ovgfx_mpsrt_encode_conv2d");
    EXPECT_EQ(gfx_mpsrt_stage_kind_from_name(gfx_mpsrt_stage_kind_name(desc.kind)),
              GfxMpsrtStageKind::MPSConv2D);
    EXPECT_EQ(gfx_mpsrt_stage_record_key(desc),
              "mps_conv2d|apple_mps|image|image|nhwc4|Convolution|apple_mps:image:Convolution");
}

TEST(GfxStagePolicyTest, MpsrtStageRecordKeyKeepsElementwiseInMslDispatch) {
    const auto add = make_large_add_node();
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Metal,
                                                     "Add",
                                                     add,
                                                     ov::element::f16,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     {});
    const auto desc = gfx_mpsrt_make_stage_desc(plan, "Add");

    EXPECT_EQ(desc.kind, GfxMpsrtStageKind::MSLDispatch);
    EXPECT_EQ(desc.kernel_name, "Add");
    EXPECT_EQ(desc.dispatch_kernel_family, "eltwise_fused_buffer");
    EXPECT_EQ(desc.dispatch_entry_point, "eltwise_fused_buffer");
    EXPECT_EQ(desc.dispatch_kernel_family_id, static_cast<uint32_t>(GfxMslKernelFamily::EltwiseFusedBuffer));
    EXPECT_EQ(desc.dispatch_flags, GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
    EXPECT_EQ(desc.dispatch_threads_per_threadgroup, 256u);
    EXPECT_TRUE(desc.dispatch_precompiled_kernel_required);
    EXPECT_EQ(desc.builder_symbol, "ovgfx_mpsrt_encode_dispatch");
    EXPECT_EQ(gfx_mpsrt_stage_record_key(desc),
              "msl_dispatch|apple_msl|buffer|buffer|linear|Add|apple_msl:buffer:Add|"
              "dispatch:eltwise_fused_buffer:eltwise_fused_buffer:tg256:metallib");
}

TEST(GfxStagePolicyTest, MslKernelManifestCoversExistingUnaryAndBinaryMslOps) {
    const auto elu_plan = make_msl_kernel_plan("Elu", "unary_kernel");
    ASSERT_TRUE(elu_plan.valid);
    EXPECT_EQ(elu_plan.family, GfxMslKernelFamily::EltwiseFusedBuffer);
    EXPECT_EQ(elu_plan.required_entry_point, "eltwise_fused_buffer");
    EXPECT_EQ(elu_plan.abi_kernel_family, static_cast<uint32_t>(GfxMslKernelFamily::EltwiseFusedBuffer));
    ASSERT_TRUE(elu_plan.external_buffer_abi.valid);
    EXPECT_TRUE(elu_plan.external_buffer_abi.tail_outputs);
    EXPECT_TRUE(elu_plan.external_buffer_abi.roles.empty());

    const auto sqdiff_plan = make_msl_kernel_plan("SquaredDifference", "eltwise_kernel");
    ASSERT_TRUE(sqdiff_plan.valid);
    EXPECT_EQ(sqdiff_plan.family, GfxMslKernelFamily::EltwiseFusedBuffer);
    EXPECT_EQ(sqdiff_plan.required_entry_point, "eltwise_fused_buffer");

    const auto reduce_plan = make_msl_kernel_plan("ReduceMean", "reduce_kernel");
    ASSERT_TRUE(reduce_plan.valid);
    EXPECT_EQ(reduce_plan.family, GfxMslKernelFamily::ReductionBuffer);
    EXPECT_EQ(reduce_plan.required_entry_point, "reduction_buffer");
    EXPECT_EQ(reduce_plan.threads_per_threadgroup, 128u);
    ASSERT_TRUE(reduce_plan.external_buffer_abi.valid);
    EXPECT_EQ(reduce_plan.external_buffer_abi.leading_input_count, 1u);
    EXPECT_EQ(reduce_plan.external_buffer_abi.leading_output_count, 1u);

    const auto softmax_plan = make_msl_kernel_plan("Softmax", "softmax_kernel");
    ASSERT_TRUE(softmax_plan.valid);
    EXPECT_EQ(softmax_plan.family, GfxMslKernelFamily::MaskedSoftmaxAttention);
    ASSERT_TRUE(softmax_plan.external_buffer_abi.valid);
    EXPECT_FALSE(softmax_plan.external_buffer_abi.tail_outputs);
    EXPECT_EQ(softmax_plan.external_buffer_abi.roles,
              std::vector<GfxMpsrtExternalBufferRole>({GfxMpsrtExternalBufferRole::TensorInput,
                                                       GfxMpsrtExternalBufferRole::TensorOutput,
                                                       GfxMpsrtExternalBufferRole::RuntimeParams}));

    const auto topk_plan = make_msl_kernel_plan("TopK", "topk_kernel");
    ASSERT_TRUE(topk_plan.valid);
    EXPECT_EQ(topk_plan.family, GfxMslKernelFamily::GatherScatterIndexed);
    ASSERT_TRUE(topk_plan.external_buffer_abi.valid);
    EXPECT_FALSE(topk_plan.external_buffer_abi.tail_outputs);
    EXPECT_TRUE(topk_plan.external_buffer_abi.roles.empty());
    EXPECT_EQ(topk_plan.external_buffer_abi.leading_input_count, 1u);
    EXPECT_EQ(topk_plan.external_buffer_abi.leading_output_count, 2u);

    const auto gather_plan = make_msl_kernel_plan("Gather", "gather_kernel");
    ASSERT_TRUE(gather_plan.valid);
    EXPECT_EQ(gather_plan.family, GfxMslKernelFamily::GatherScatterIndexed);
    ASSERT_TRUE(gather_plan.external_buffer_abi.valid);
    EXPECT_EQ(gather_plan.external_buffer_abi.leading_input_count, 2u);
    EXPECT_EQ(gather_plan.external_buffer_abi.leading_output_count, 1u);

    const auto slice_plan = make_msl_kernel_plan("Slice", "slice_kernel");
    ASSERT_TRUE(slice_plan.valid);
    EXPECT_EQ(slice_plan.family, GfxMslKernelFamily::GatherScatterIndexed);
    ASSERT_TRUE(slice_plan.external_buffer_abi.valid);
    EXPECT_EQ(slice_plan.external_buffer_abi.leading_input_count, 1u);
    EXPECT_EQ(slice_plan.external_buffer_abi.leading_output_count, 1u);

    const auto conv3d_plan = make_msl_kernel_plan("Convolution", "conv3d_kernel");
    ASSERT_TRUE(conv3d_plan.valid);
    EXPECT_EQ(conv3d_plan.family, GfxMslKernelFamily::Conv3DDirectOrIm2col);
    EXPECT_EQ(conv3d_plan.required_entry_point, "conv3d_direct_or_im2col");
    EXPECT_EQ(conv3d_plan.threads_per_threadgroup, 128u);
}

TEST(GfxStagePolicyTest, MpsrtBuilderPlanSerializesMslDispatchStage) {
    const auto add = make_large_add_node();
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Metal,
                                                     "Add",
                                                     add,
                                                     ov::element::f16,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     {});
    const auto stage = gfx_mpsrt_make_stage_desc(plan, "Add");
    const auto lhs = gfx_mpsrt_make_tensor_desc({1, 64, 80, 80},
                                                ov::element::f16,
                                                GfxStageStorageKind::Buffer,
                                                GfxMpsrtTensorFlagExternalIo);
    const auto rhs = gfx_mpsrt_make_tensor_desc({1, 64, 80, 80},
                                                ov::element::f16,
                                                GfxStageStorageKind::Buffer,
                                                GfxMpsrtTensorFlagExternalIo);
    const auto out = gfx_mpsrt_make_tensor_desc({1, 64, 80, 80},
                                                ov::element::f16,
                                                GfxStageStorageKind::Buffer,
                                                GfxMpsrtTensorFlagTransient);
    const auto builder_plan = gfx_mpsrt_make_builder_plan(stage,
                                                          {lhs, rhs},
                                                          {out},
                                                          gfx_mpsrt_stage_record_key(stage));

    ASSERT_TRUE(builder_plan.valid);
    ASSERT_EQ(builder_plan.records.size(), 5u);
    EXPECT_EQ(builder_plan.records[0].symbol, "ovgfx_mpsrt_model_begin");
    EXPECT_EQ(builder_plan.records[1].symbol, "ovgfx_mpsrt_add_tensor");
    EXPECT_EQ(builder_plan.records[2].symbol, "ovgfx_mpsrt_add_tensor");
    EXPECT_EQ(builder_plan.records[3].kind, GfxMpsrtBuilderRecordKind::EncodeStage);
    EXPECT_EQ(builder_plan.records[3].symbol, "ovgfx_mpsrt_encode_dispatch");
    EXPECT_EQ(builder_plan.records[3].kernel_name, "eltwise_fused_buffer");
    EXPECT_EQ(builder_plan.records[3].dispatch_kernel_family, "eltwise_fused_buffer");
    EXPECT_EQ(builder_plan.records[3].dispatch_entry_point, "eltwise_fused_buffer");
    EXPECT_EQ(builder_plan.records[3].dispatch_kernel_family_id,
              static_cast<uint32_t>(GfxMslKernelFamily::EltwiseFusedBuffer));
    EXPECT_EQ(builder_plan.records[3].dispatch_flags, GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
    EXPECT_EQ(builder_plan.records[3].dispatch_threads_per_threadgroup, 256u);
    EXPECT_TRUE(builder_plan.records[3].dispatch_precompiled_kernel_required);
    EXPECT_EQ(builder_plan.records[3].msl_dispatch_desc.kernel_family,
              static_cast<uint32_t>(GfxMslKernelFamily::EltwiseFusedBuffer));
    EXPECT_EQ(builder_plan.records[3].msl_dispatch_desc.storage,
              static_cast<uint32_t>(GfxMpsrtStorage::Buffer));
    EXPECT_EQ(builder_plan.records[3].msl_dispatch_desc.layout,
              static_cast<uint32_t>(GfxMpsrtLayout::Linear));
    EXPECT_EQ(builder_plan.records[3].msl_dispatch_desc.threads_per_threadgroup, 256u);
    EXPECT_EQ(builder_plan.records[3].msl_dispatch_desc.input_count, 2u);
    EXPECT_EQ(builder_plan.records[3].msl_dispatch_desc.output_count, 1u);
    EXPECT_EQ(builder_plan.records[3].msl_dispatch_desc.flags,
              GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
    EXPECT_EQ(builder_plan.records[3].inputs, std::vector<GfxMpsrtValue>({0u, 1u}));
    EXPECT_EQ(builder_plan.records[3].outputs, std::vector<GfxMpsrtValue>({2u}));
    EXPECT_EQ(builder_plan.records[3].kernel_buffer_order, std::vector<GfxMpsrtValue>({0u, 1u, 2u}));
    EXPECT_EQ(builder_plan.records[3].tensor_descs[0].byte_length, 1u * 64u * 80u * 80u * 2u);
    EXPECT_EQ(builder_plan.records[4].symbol, "ovgfx_mpsrt_model_end");
}

TEST(GfxStagePolicyTest, MpsrtRuntimeModelBuildsMslDispatchStage) {
    const auto add = make_large_add_node();
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Metal,
                                                     "Add",
                                                     add,
                                                     ov::element::f16,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     {});
    const auto stage = gfx_mpsrt_make_stage_desc(plan, "Add");
    const auto lhs = gfx_mpsrt_make_tensor_desc({1, 64, 80, 80},
                                                ov::element::f16,
                                                GfxStageStorageKind::Buffer,
                                                GfxMpsrtTensorFlagExternalIo);
    const auto rhs = gfx_mpsrt_make_tensor_desc({1, 64, 80, 80},
                                                ov::element::f16,
                                                GfxStageStorageKind::Buffer,
                                                GfxMpsrtTensorFlagExternalIo);
    const auto out = gfx_mpsrt_make_tensor_desc({1, 64, 80, 80},
                                                ov::element::f16,
                                                GfxStageStorageKind::Buffer,
                                                GfxMpsrtTensorFlagTransient);
    const auto builder_plan = gfx_mpsrt_make_builder_plan(stage,
                                                          {lhs, rhs},
                                                          {out},
                                                          gfx_mpsrt_stage_record_key(stage));

    ov::gfx_plugin::metal::mpsrt::MpsrtModel model;
    std::string error;
    ASSERT_TRUE(ov::gfx_plugin::metal::mpsrt::build_mpsrt_model_from_builder_plan(builder_plan, model, &error))
        << error;

    ASSERT_EQ(model.tensors.size(), 3u);
    ASSERT_EQ(model.stages.size(), 1u);
    EXPECT_EQ(model.semantic_input_values, std::vector<GfxMpsrtValue>({0u, 1u}));
    EXPECT_EQ(model.semantic_output_values, std::vector<GfxMpsrtValue>({2u}));
    EXPECT_EQ(model.input_values, std::vector<GfxMpsrtValue>({0u, 1u}));
    EXPECT_EQ(model.output_values, std::vector<GfxMpsrtValue>({2u}));
    EXPECT_EQ(model.external_values, std::vector<GfxMpsrtValue>({0u, 1u, 2u}));
    EXPECT_EQ(model.external_input_values, std::vector<GfxMpsrtValue>({0u, 1u}));
    EXPECT_EQ(model.external_output_values, std::vector<GfxMpsrtValue>({2u}));
    const auto& runtime_stage = model.stages.front();
    EXPECT_EQ(runtime_stage.kind, GfxMpsrtStageKind::MSLDispatch);
    EXPECT_EQ(runtime_stage.kernel_name, "eltwise_fused_buffer");
    EXPECT_EQ(runtime_stage.dispatch_kernel_family, "eltwise_fused_buffer");
    EXPECT_EQ(runtime_stage.dispatch_entry_point, "eltwise_fused_buffer");
    EXPECT_EQ(runtime_stage.dispatch_kernel_family_id,
              static_cast<uint32_t>(GfxMslKernelFamily::EltwiseFusedBuffer));
    EXPECT_EQ(runtime_stage.dispatch_flags, GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
    EXPECT_EQ(runtime_stage.dispatch_threads_per_threadgroup, 256u);
    EXPECT_TRUE(runtime_stage.dispatch_precompiled_kernel_required);
    EXPECT_EQ(runtime_stage.msl_dispatch_desc.kernel_family,
              static_cast<uint32_t>(GfxMslKernelFamily::EltwiseFusedBuffer));
    EXPECT_EQ(runtime_stage.msl_dispatch_desc.input_count, 2u);
    EXPECT_EQ(runtime_stage.msl_dispatch_desc.output_count, 1u);
    EXPECT_EQ(runtime_stage.msl_dispatch_desc.flags, GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
    EXPECT_EQ(runtime_stage.inputs, std::vector<GfxMpsrtValue>({0u, 1u}));
    EXPECT_EQ(runtime_stage.outputs, std::vector<GfxMpsrtValue>({2u}));
    EXPECT_EQ(runtime_stage.kernel_buffer_order, std::vector<GfxMpsrtValue>({0u, 1u, 2u}));
    ASSERT_EQ(runtime_stage.output_descs.size(), 1u);
    EXPECT_EQ(runtime_stage.output_descs.front().byte_length, 1u * 64u * 80u * 80u * 2u);
}

TEST(GfxStagePolicyTest, MpsrtRuntimeModelRejectsMalformedMslDispatch) {
    const auto add = make_large_add_node();
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Metal,
                                                     "Add",
                                                     add,
                                                     ov::element::f16,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     {});
    const auto stage = gfx_mpsrt_make_stage_desc(plan, "Add");
    const auto lhs = gfx_mpsrt_make_tensor_desc({1, 64, 80, 80},
                                                ov::element::f16,
                                                GfxStageStorageKind::Buffer,
                                                GfxMpsrtTensorFlagExternalIo);
    const auto rhs = gfx_mpsrt_make_tensor_desc({1, 64, 80, 80},
                                                ov::element::f16,
                                                GfxStageStorageKind::Buffer,
                                                GfxMpsrtTensorFlagExternalIo);
    const auto out = gfx_mpsrt_make_tensor_desc({1, 64, 80, 80},
                                                ov::element::f16,
                                                GfxStageStorageKind::Buffer,
                                                GfxMpsrtTensorFlagTransient);
    auto builder_plan = gfx_mpsrt_make_builder_plan(stage,
                                                    {lhs, rhs},
                                                    {out},
                                                    gfx_mpsrt_stage_record_key(stage));
    ASSERT_EQ(builder_plan.records.size(), 5u);
    builder_plan.records[3].msl_dispatch_desc.kernel_family = 0;

    ov::gfx_plugin::metal::mpsrt::MpsrtModel model;
    std::string error;
    EXPECT_FALSE(ov::gfx_plugin::metal::mpsrt::build_mpsrt_model_from_builder_plan(builder_plan, model, &error));
    EXPECT_NE(error.find("MSL dispatch kernel family is not set"), std::string::npos);
}

TEST(GfxStagePolicyTest, MpsrtRuntimeModelAdaptsExpandedExternalBufferAbi) {
    const auto add = make_large_add_node();
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Metal,
                                                     "Add",
                                                     add,
                                                     ov::element::f16,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     {});
    const auto stage = gfx_mpsrt_make_stage_desc(plan, "Add");
    const auto lhs = gfx_mpsrt_make_tensor_desc({1, 64},
                                                ov::element::f16,
                                                GfxStageStorageKind::Buffer,
                                                GfxMpsrtTensorFlagExternalIo);
    const auto out = gfx_mpsrt_make_tensor_desc({1, 64},
                                                ov::element::f16,
                                                GfxStageStorageKind::Buffer,
                                                GfxMpsrtTensorFlagTransient);
    const auto builder_plan = gfx_mpsrt_make_builder_plan(stage,
                                                          {lhs},
                                                          {out},
                                                          gfx_mpsrt_stage_record_key(stage));

    ov::gfx_plugin::metal::mpsrt::MpsrtModel model;
    std::string error;
    ASSERT_TRUE(ov::gfx_plugin::metal::mpsrt::build_mpsrt_model_from_builder_plan(builder_plan, model, &error))
        << error;
    ASSERT_TRUE(ov::gfx_plugin::metal::mpsrt::adapt_mpsrt_model_to_external_buffer_abi(model,
                                                                                       3u,
                                                                                       1u,
                                                                                       &error))
        << error;

    EXPECT_EQ(model.semantic_input_values, std::vector<GfxMpsrtValue>({0u}));
    EXPECT_EQ(model.semantic_output_values, std::vector<GfxMpsrtValue>({1u}));
    EXPECT_EQ(model.input_values, std::vector<GfxMpsrtValue>({0u, 1u}));
    EXPECT_EQ(model.output_values, std::vector<GfxMpsrtValue>({2u}));
    EXPECT_EQ(model.external_values, std::vector<GfxMpsrtValue>({0u, 1u, 2u}));
    EXPECT_EQ(model.external_input_values, std::vector<GfxMpsrtValue>({0u, 1u}));
    EXPECT_EQ(model.external_output_values, std::vector<GfxMpsrtValue>({2u}));
    EXPECT_EQ(model.external_buffer_roles,
              std::vector<GfxMpsrtExternalBufferRole>({GfxMpsrtExternalBufferRole::TensorInput,
                                                       GfxMpsrtExternalBufferRole::TensorInput,
                                                       GfxMpsrtExternalBufferRole::TensorOutput}));
    ASSERT_EQ(model.stages.size(), 1u);
    EXPECT_EQ(model.stages.front().inputs, std::vector<GfxMpsrtValue>({0u, 1u}));
    EXPECT_EQ(model.stages.front().outputs, std::vector<GfxMpsrtValue>({2u}));
    EXPECT_EQ(model.stages.front().kernel_buffer_order, std::vector<GfxMpsrtValue>({0u, 1u, 2u}));
    EXPECT_EQ(model.stages.front().msl_dispatch_desc.input_count, 2u);
    EXPECT_EQ(model.stages.front().msl_dispatch_desc.output_count, 1u);
}

TEST(GfxStagePolicyTest, MpsrtRuntimeModelAdaptsExternalBufferRolesWithoutTailOutputAssumption) {
    const auto add = make_large_add_node();
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Metal,
                                                     "Softmax",
                                                     add,
                                                     ov::element::f16,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     {});
    const auto stage = gfx_mpsrt_make_stage_desc(plan, "Softmax");
    const auto input = gfx_mpsrt_make_tensor_desc({1, 64},
                                                  ov::element::f16,
                                                  GfxStageStorageKind::Buffer,
                                                  GfxMpsrtTensorFlagExternalIo);
    const auto output = gfx_mpsrt_make_tensor_desc({1, 64},
                                                   ov::element::f16,
                                                   GfxStageStorageKind::Buffer,
                                                   GfxMpsrtTensorFlagTransient);
    auto builder_plan = gfx_mpsrt_make_builder_plan(stage,
                                                    {input},
                                                    {output},
                                                    gfx_mpsrt_stage_record_key(stage));
    builder_plan.external_buffer_roles = {GfxMpsrtExternalBufferRole::TensorInput,
                                          GfxMpsrtExternalBufferRole::TensorOutput,
                                          GfxMpsrtExternalBufferRole::RuntimeParams};

    ov::gfx_plugin::metal::mpsrt::MpsrtModel model;
    std::string error;
    ASSERT_TRUE(ov::gfx_plugin::metal::mpsrt::build_mpsrt_model_from_builder_plan(builder_plan, model, &error))
        << error;
    ASSERT_TRUE(ov::gfx_plugin::metal::mpsrt::adapt_mpsrt_model_to_external_buffer_abi(model,
                                                                                       3u,
                                                                                       1u,
                                                                                       &error))
        << error;

    EXPECT_EQ(model.input_values, std::vector<GfxMpsrtValue>({0u, 2u}));
    EXPECT_EQ(model.output_values, std::vector<GfxMpsrtValue>({1u}));
    EXPECT_EQ(model.external_values, std::vector<GfxMpsrtValue>({0u, 1u, 2u}));
    EXPECT_EQ(model.external_input_values, std::vector<GfxMpsrtValue>({0u, 2u}));
    EXPECT_EQ(model.external_output_values, std::vector<GfxMpsrtValue>({1u}));
    EXPECT_EQ(model.external_buffer_roles,
              std::vector<GfxMpsrtExternalBufferRole>({GfxMpsrtExternalBufferRole::TensorInput,
                                                       GfxMpsrtExternalBufferRole::TensorOutput,
                                                       GfxMpsrtExternalBufferRole::RuntimeParams}));
    ASSERT_EQ(model.stages.size(), 1u);
    EXPECT_EQ(model.stages.front().inputs, std::vector<GfxMpsrtValue>({0u, 2u}));
    EXPECT_EQ(model.stages.front().outputs, std::vector<GfxMpsrtValue>({1u}));
    EXPECT_EQ(model.stages.front().kernel_buffer_order, std::vector<GfxMpsrtValue>({0u, 1u, 2u}));
    EXPECT_EQ(model.stages.front().msl_dispatch_desc.input_count, 2u);
    EXPECT_EQ(model.stages.front().msl_dispatch_desc.output_count, 1u);
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
    GpuExecutionDeviceInfo info;
    info.backend = GpuBackend::Vulkan;
    info.device_key = "test-rpi";
    info.preferred_simd_width = 16u;
    info.subgroup_size = 16u;
    info.max_total_threads_per_group = 256u;
    info.max_threads_per_group = {256u, 256u, 64u};
    FakeDeviceInfoBufferManager buffer_manager(info);
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
    GpuExecutionDeviceInfo info;
    info.backend = GpuBackend::Vulkan;
    info.device_key = "test-rpi";
    info.preferred_simd_width = 16u;
    info.subgroup_size = 16u;
    info.max_total_threads_per_group = 256u;
    info.max_threads_per_group = {256u, 256u, 64u};
    FakeDeviceInfoBufferManager buffer_manager(info);
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
