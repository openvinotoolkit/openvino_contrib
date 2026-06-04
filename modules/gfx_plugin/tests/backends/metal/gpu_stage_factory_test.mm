// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "backends/metal/runtime/mpsrt/mpsrt_msl_kernel_loader.hpp"
#include "backends/metal/runtime/metal_runtime_kernel_loader.hpp"
#include "backends/metal/runtime/stage_factory.hpp"
#include "kernel_ir/metal_kernels/mpsrt_image_bridge_kernels.hpp"
#include "runtime/gfx_backend_utils.hpp"
#include "runtime/execution_dispatcher.hpp"
#include "runtime/executable_descriptor.hpp"

using namespace ov::gfx_plugin;

namespace {

RuntimeStageExecutableDescriptor make_metal_test_descriptor(
    const std::shared_ptr<const ov::Node>& node) {
    RuntimeStageExecutableDescriptor descriptor;
    descriptor.stage_index = 0;
    descriptor.stage_record_key = 1;
    descriptor.artifact_descriptor_index = 0;
    descriptor.manifest_ref = "test_manifest";
    descriptor.abi_fingerprint = "test_abi";
    descriptor.artifact_key = "test_artifact";
    descriptor.backend_domain = "metal";
    descriptor.kernel_id =
        std::string("metal/generated/") + (node ? node->get_type_name() : "null");
    descriptor.op_family = node ? node->get_type_name() : "null";
    descriptor.origin = KernelArtifactOrigin::Generated;
    descriptor.payload_kind = KernelArtifactPayloadKind::None;
    descriptor.entry_point = descriptor.kernel_id;
    descriptor.dispatch_contract = "manifest";
    return descriptor;
}

}  // namespace

TEST(GpuStageFactory, CreatesStubForRelu) {
    ensure_metal_stage_factory_registered();
    auto p = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto relu = std::make_shared<ov::op::v0::Relu>(p);
    const auto descriptor = make_metal_test_descriptor(relu);

    auto stage = GpuStageFactory::create(relu, &descriptor, default_backend_kind());

    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->type(), std::string("Relu"));
    EXPECT_EQ(stage->name(), relu->get_friendly_name());
}

TEST(GpuStageFactory, ReturnsNullForUnsupportedParameter) {
    ensure_metal_stage_factory_registered();
    auto p = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    const auto descriptor = make_metal_test_descriptor(p);

    auto stage = GpuStageFactory::create(p, &descriptor, default_backend_kind());

    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->type(), std::string("Parameter"));
}

TEST(GpuStageFactory, MetalFactoryUsesSharedViewOnlyStageForMetadataDescriptor) {
    ensure_metal_stage_factory_registered();
    auto p = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{2, 3});
    auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1},
                                             {6});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(p, shape, false);

    RuntimeStageExecutableDescriptor descriptor;
    descriptor.stage_index = 0;
    descriptor.stage_record_key = 1;
    descriptor.artifact_descriptor_index = 0;
    descriptor.manifest_ref = "test/view_manifest";
    descriptor.abi_fingerprint = "test/view_abi";
    descriptor.artifact_key = "test/view_artifact";
    descriptor.backend_domain = "metal";
    descriptor.kernel_id = "metadata";
    descriptor.op_family = reshape->get_type_name();
    descriptor.origin = KernelArtifactOrigin::Metadata;
    descriptor.payload_kind = KernelArtifactPayloadKind::None;
    descriptor.layout_contract = "view_only";
    descriptor.tensor_view_only = true;

    auto stage = GpuStageFactory::create(reshape, &descriptor, default_backend_kind());

    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->type(), std::string("Reshape"));
    EXPECT_EQ(stage->name(), reshape->get_friendly_name());
}

TEST(GpuStageFactory, MpsrtMslKernelLoaderBuildsStageFromKernelSourceDescriptor) {
    const auto& source = mpsrt_image_bridge_kernel_source(
        MpsrtImageBridgeKernelKind::BufferToImageF32);

    const auto stage =
        metal::mpsrt::MpsrtMslKernelLoader::make_stage(source, 64);

    EXPECT_EQ(stage.kind, GfxMpsrtStageKind::MSLDispatch);
    EXPECT_EQ(stage.stage_record_key, source.kernel_id);
    EXPECT_EQ(stage.dispatch_entry_point, source.entry_point);
    EXPECT_EQ(stage.dispatch_threads_per_threadgroup, 64u);
    EXPECT_EQ(stage.dispatch_flags, GfxMpsrtMslDispatchFlagNone);
}

TEST(GpuStageFactory, MetalRuntimeKernelLoaderUsesDescriptorAbiForMslPayload) {
    const auto& source = mpsrt_image_bridge_kernel_source(
        MpsrtImageBridgeKernelKind::BufferToImageF32);

    RuntimeStageExecutableDescriptor descriptor;
    descriptor.manifest_ref = "test_manifest";
    descriptor.abi_fingerprint = "test_abi";
    descriptor.artifact_key = "test_artifact";
    descriptor.backend_domain = source.backend_domain;
    descriptor.kernel_id = source.kernel_id;
    descriptor.origin = KernelArtifactOrigin::HandwrittenException;
    descriptor.payload_kind = KernelArtifactPayloadKind::MslSource;
    descriptor.entry_point = source.entry_point;
    descriptor.abi_arg_count = 2;
    descriptor.abi_output_arg_count = 1;
    descriptor.tensor_roles = {"tensor_input", "tensor_output"};
    descriptor.payload = std::make_shared<GfxKernelSourcePayload>(source);

    const auto kernel_source =
        MetalRuntimeKernelLoader::load_msl_source(descriptor);

    EXPECT_EQ(kernel_source.entry_point, source.entry_point);
    EXPECT_EQ(kernel_source.msl_source, source.source);
    EXPECT_EQ(kernel_source.signature.arg_count, 2u);
    EXPECT_EQ(kernel_source.signature.output_arg_count, 1u);
    EXPECT_FALSE(static_cast<bool>(kernel_source.module));
}
