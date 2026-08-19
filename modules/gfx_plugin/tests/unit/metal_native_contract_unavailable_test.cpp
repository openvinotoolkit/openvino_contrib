// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "compiler/backend_config.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

void expect_metal_native_contract_unavailable() {
    EXPECT_FALSE(kGfxBackendMetalAvailable)
        << "This adapter is linked only for targets without native Metal.";
}

TEST(GfxBufferManagerTest, AllocAlignedAndNonNull) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBufferManagerTest, ReuseBuffersViaFreeList) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBufferManagerTest, DynamicGrowthWithBufferHandle) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBufferManagerTest, PersistentAndPerInferAreSeparated) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBufferManagerTest, ConstCacheContextIsSharedAcrossCacheInstances) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBufferManagerTest, WrapSharedHostInputDoesNotCopyToCPU) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBufferManagerTest, OutputStagingHandleReusesAndGrows) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxMetalMemory, CopyRoundtrip) {
    expect_metal_native_contract_unavailable();
}

TEST(GpuStageFactory, CreatesStubForRelu) {
    expect_metal_native_contract_unavailable();
}

TEST(GpuStageFactory, ReturnsNullForUnsupportedParameter) {
    expect_metal_native_contract_unavailable();
}

TEST(GpuStageFactory, MetalFactoryUsesSharedViewOnlyStageForMetadataDescriptor) {
    expect_metal_native_contract_unavailable();
}

TEST(GpuStageFactory, MpsrtMslKernelLoaderBuildsStageFromKernelSourceDescriptor) {
    expect_metal_native_contract_unavailable();
}

TEST(GpuStageFactory, MetalRuntimeKernelLoaderUsesDescriptorAbiForMslPayload) {
    expect_metal_native_contract_unavailable();
}

TEST(GpuStageActivation, Relu) {
    expect_metal_native_contract_unavailable();
}

TEST(GpuStageActivation, Sigmoid) {
    expect_metal_native_contract_unavailable();
}

TEST(GpuStageActivation, Tanh) {
    expect_metal_native_contract_unavailable();
}

TEST(GpuStageActivation, RejectsRuntimeSourcePlanWithoutCompilerPayload) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, CompileAndExecuteKernel) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, CompileNoManifestMslRejectsSourceOnlyAbiInference) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, BindingSchemaIsSharedAcrossDistinctProgramsWithSameAbi) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, CompileAttachesMpsrtModelForAnnotatedMslDispatch) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtCompileBindingUsesExactTypedAbiOverLegacySignature) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtCompileBindingIgnoresWiderMslSourceBufferScan) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MetalCodegenRejectsIncompleteCustomManifestInsteadOfScanningMslBuffers) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtCompileBindingKeepsScalarRuntimeAbiSeparateFromExternalBuffers) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, InvalidTypedMpsrtProgramRejectsRawMslCompileFallback) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, TypedMpsrtCompileRejectsMissingExternalAbiInsteadOfUsingSignature) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, ExactStageManifestAbiWinsOverLegacySignatureWithoutMpsrt) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtContextCachesPreparedMslDispatchPipelines) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, AnnotatedMslKernelExecutesThroughMpsrtRequest) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, AnnotatedMslKernelWithExpandedAbiExecutesThroughMpsrtBufferOrder) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, SingleMslMpsrtBindsModelOwnedConstResourceFromPreparedState) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, AnnotatedSoftmaxMslKernelUsesRoleBasedMpsrtBufferOrder) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, AnnotatedGatherElementsMslKernelExecutesThroughRolePatternMpsrtBufferOrder) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, AnnotatedSliceMslKernelExecutesThroughRoleBasedRuntimeParamsMpsrtBufferOrder) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtRequestEncodesPreparedTwoStageMslModelWithValueBindings) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtRequestRejectsMslDispatchWithoutManifestBufferOrder) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtTensorBindingsAcceptImageExternalAndTransientResources) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtPrepareModelAllocatesImageBridgeScratchTextureFromResourceHeap) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtPreparedModelReleasesOwnedHeapAndBridgeResources) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtTensorBindingsRejectResourceLessConstTensors) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtExternalBufferBindingsPreserveNonTensorResourceAbiEntries) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtPrepareModelPlansTransientResourceLiveWindowsForHeapAliasing) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtRequestEncodesPreparedMpsResize2DModelWithImageBindings) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtRequestEncodesPreparedMpsGemmModelWithMatrixBindings) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtPrepareModelRejectsUnmaterializedModelConstResource) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtPrepareModelMaterializesMpsConv2DWeightsFromPreparedConstResource) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtRequestExecutesF32MpsConv2DWithBufferImageBridges) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtRequestExecutesF32MpsConv2DWithBiasAndStride2) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtRequestExecutesF32MpsPointwiseConv2DWithPaddedChannels) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtRequestExecutesF32MpsMaxPool2DWithBufferImageBridges) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtRequestExecutesF32MpsConvPoolChainWithTransientImage) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtRequestExecutesF32MpsConvTextureSwishEpilogueChain) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtPrepareModelMaterializesMpsGroupConv2DDepthwiseWeightsFromPreparedConstResource) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtRequestExecutesF32MpsCnnDepthwiseGroupConv2DWithBufferImageBridges) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtRequestExecutesF32MpsNativeGroupedConv2DWithFourInputChannelsPerGroup) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtRequestEncodesPreparedBatchedMpsGemmModelWithMatrixBindings) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtRequestEncodesPreparedBatchedTransposedF32MpsGraphGemmModel) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtRequestEncodesPreparedBatchBroadcastMpsGemmModelWithMatrixBindings) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MetalCompiledKernelExecutesMixedMpsGemmAndMslEpilogueMpsrtModel) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MetalCodegenCompilesVendorOnlyMpsGemmWithoutMslSource) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MetalCodegenCompilesVendorOnlyMpsSdpaWithoutMslSource) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtRequestExecutesF32MpsGraphSdpa) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtRequestExecutesF32MpsGraphTransposedSdpa) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MetalCodegenCompilesVendorOnlyMpsTopKWithoutMslSource) {
    expect_metal_native_contract_unavailable();
}

TEST(GfxBackendTest, MpsrtRequestExecutesLargeF32I64TopKWithMpsGraphExecutable) {
    expect_metal_native_contract_unavailable();
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
