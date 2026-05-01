// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Metal/Metal.h>
#ifdef NO
#undef NO
#endif
#ifdef YES
#undef YES
#endif

#include <gtest/gtest.h>

#include <cstring>
#include <unordered_map>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "backends/metal/runtime/metal_command_encoder.hpp"
#include "backends/metal/runtime/mpsrt/mpsrt_context.hpp"
#include "backends/metal/runtime/mpsrt/mpsrt_request.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "mlir/msl_codegen.hpp"
#include "runtime/gfx_stage_policy.hpp"
#include "runtime/gfx_msl_kernel_manifest.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

TEST(GfxBackendTest, CompileAndExecuteKernel) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    // Simple kernel: add 1 to each element.
    const char* source = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void add1(device float* data [[buffer(0)]],
                 constant uint& n [[buffer(1)]],
                 uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  data[gid] += 1.0f;
}
)MSL";

    mlir::MLIRContext ctx;
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

    KernelSource ks;
    ks.module = module;
    ks.entry_point = "add1";
    ks.msl_source = source;
    ks.signature.arg_count = 2;

    MetalCodegenBackend backend((MetalDeviceHandle)device);
    std::string log;
    auto kernel = backend.compile(ks, &log);
    ASSERT_TRUE(kernel) << log;

    constexpr uint32_t kCount = 64;
    id<MTLBuffer> buf = [device newBufferWithLength:sizeof(float) * kCount
                                           options:MTLResourceStorageModeShared];
    ASSERT_NE(buf, nil);
    float* ptr = static_cast<float*>([buf contents]);
    ASSERT_NE(ptr, nullptr);
    for (uint32_t i = 0; i < kCount; ++i) {
        ptr[i] = static_cast<float>(i);
    }

    MetalBuffer gpu_buf{};
    gpu_buf.buffer = (__bridge void*)buf;
    gpu_buf.size = sizeof(float) * kCount;
    gpu_buf.type = ov::element::f32;

    KernelDispatch dispatch;
    dispatch.grid[0] = kCount;
    dispatch.threads_per_group[0] = kernel->clamp_threadgroup_size(64);

    uint32_t count = kCount;
    id<MTLBuffer> count_buf = [device newBufferWithLength:sizeof(uint32_t)
                                                 options:MTLResourceStorageModeShared];
    ASSERT_NE(count_buf, nil);
    void* count_ptr = [count_buf contents];
    ASSERT_NE(count_ptr, nullptr);
    std::memcpy(count_ptr, &count, sizeof(uint32_t));

    MetalBuffer gpu_count{};
    gpu_count.buffer = (__bridge void*)count_buf;
    gpu_count.size = sizeof(uint32_t);
    gpu_count.type = ov::element::u32;
    std::vector<KernelArg> args;
    args.reserve(2);
    args.push_back(make_buffer_arg(0, gpu_buf));
    args.push_back(make_buffer_arg(1, gpu_count));

    id<MTLCommandQueue> queue = [device newCommandQueue];
    ASSERT_NE(queue, nil);
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    ASSERT_NE(cmd, nil);
    kernel->execute((GpuCommandBufferHandle)cmd, dispatch, args, nullptr);
    metal_end_compute_encoder((GpuCommandBufferHandle)cmd);
    [cmd commit];
    [cmd waitUntilCompleted];

    EXPECT_EQ(kernel->args_count(), 2u);
    for (uint32_t i = 0; i < kCount; ++i) {
        EXPECT_FLOAT_EQ(ptr[i], static_cast<float>(i + 1));
    }
}

TEST(GfxBackendTest, BindingSchemaIsSharedAcrossDistinctProgramsWithSameAbi) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    const char* source = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void add1(device float* data [[buffer(0)]],
                 constant uint& n [[buffer(1)]],
                 uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  data[gid] += 1.0f;
}
kernel void add2(device float* data [[buffer(0)]],
                 constant uint& n [[buffer(1)]],
                 uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  data[gid] += 2.0f;
}
)MSL";

    mlir::MLIRContext ctx;
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

    KernelSource first_ks;
    first_ks.module = module;
    first_ks.entry_point = "add1";
    first_ks.msl_source = source;
    first_ks.signature.arg_count = 2;

    KernelSource second_ks = first_ks;
    second_ks.entry_point = "add2";

    MetalCodegenBackend backend((MetalDeviceHandle)device);
    std::string log;
    auto first = backend.compile(first_ks, &log);
    ASSERT_TRUE(first) << log;
    auto second = backend.compile(second_ks, &log);
    ASSERT_TRUE(second) << log;

    auto* first_metal = dynamic_cast<MetalCompiledKernel*>(first.get());
    auto* second_metal = dynamic_cast<MetalCompiledKernel*>(second.get());
    ASSERT_NE(first_metal, nullptr);
    ASSERT_NE(second_metal, nullptr);
    EXPECT_EQ(first_metal->shared_binding_schema_identity(),
              second_metal->shared_binding_schema_identity());
}

TEST(GfxBackendTest, CompileAttachesMpsrtModelForAnnotatedMslDispatch) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    const char* source = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void add1(device float* data [[buffer(0)]],
                 uint gid [[thread_position_in_grid]]) {
  data[gid] += 1.0f;
}
)MSL";

    mlir::MLIRContext ctx;
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mlir::Builder builder(&ctx);
    module->setAttr("gfx.backend", builder.getStringAttr("apple_msl"));
    module->setAttr("gfx.stage_type", builder.getStringAttr("Add"));
    module->setAttr("gfx.mpsrt.stage_kind", builder.getStringAttr("msl_dispatch"));
    module->setAttr("gfx.mpsrt.kernel_name", builder.getStringAttr("add1"));
    module->setAttr("gfx.mpsrt.builder_symbol", builder.getStringAttr("ovgfx_mpsrt_encode_dispatch"));
    module->setAttr("gfx.specialization_key", builder.getStringAttr("apple_msl:buffer:Add"));
    module->setAttr("gfx.mpsrt.dispatch_kernel_family", builder.getStringAttr("eltwise_fused_buffer"));
    module->setAttr("gfx.mpsrt.dispatch_entry_point", builder.getStringAttr("add1"));
    module->setAttr("gfx.mpsrt.dispatch_kernel_family_id",
                    builder.getI32IntegerAttr(static_cast<int32_t>(GfxMslKernelFamily::EltwiseFusedBuffer)));
    module->setAttr("gfx.mpsrt.dispatch_flags",
                    builder.getI32IntegerAttr(GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired));
    module->setAttr("gfx.mpsrt.dispatch_threads_per_threadgroup", builder.getI32IntegerAttr(64));
    module->setAttr("gfx.mpsrt.dispatch_precompiled_kernel_required", builder.getBoolAttr(true));
    module->setAttr("gfx.mpsrt.stage_record_key",
                    builder.getStringAttr("msl_dispatch|apple_msl|buffer|buffer|linear|Add|"
                                          "apple_msl:buffer:Add|dispatch:eltwise_fused_buffer:add1:tg64:metallib"));
    module->setAttr("gfx.mpsrt.input_count", builder.getI32IntegerAttr(1));
    module->setAttr("gfx.mpsrt.output_count", builder.getI32IntegerAttr(1));

    const auto input_desc = gfx_mpsrt_make_tensor_desc({64},
                                                       ov::element::f32,
                                                       GfxStageStorageKind::Buffer,
                                                       GfxMpsrtTensorFlagExternalIo);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({64},
                                                        ov::element::f32,
                                                        GfxStageStorageKind::Buffer,
                                                        GfxMpsrtTensorFlagTransient);
    detail::gfx_mpsrt_set_tensor_desc_attrs(module, "gfx.mpsrt.input0", input_desc);
    detail::gfx_mpsrt_set_tensor_desc_attrs(module, "gfx.mpsrt.output0", output_desc);

    KernelSource ks;
    ks.module = module;
    ks.entry_point = "add1";
    ks.msl_source = source;
    ks.signature.arg_count = 1;

    MetalCodegenBackend backend((MetalDeviceHandle)device);
    std::string log;
    auto kernel = backend.compile(ks, &log);
    ASSERT_TRUE(kernel) << log;

    auto* metal_kernel = dynamic_cast<MetalCompiledKernel*>(kernel.get());
    ASSERT_NE(metal_kernel, nullptr);
    const auto* mpsrt_model = metal_kernel->mpsrt_model();
    ASSERT_NE(mpsrt_model, nullptr);
    ASSERT_EQ(mpsrt_model->stages.size(), 1u);
    EXPECT_EQ(mpsrt_model->tensors.size(), 2u);
    EXPECT_EQ(mpsrt_model->semantic_input_values, std::vector<GfxMpsrtValue>({0u}));
    EXPECT_EQ(mpsrt_model->semantic_output_values, std::vector<GfxMpsrtValue>({1u}));
    EXPECT_EQ(mpsrt_model->input_values, std::vector<GfxMpsrtValue>({0u}));
    EXPECT_EQ(mpsrt_model->output_values, std::vector<GfxMpsrtValue>({1u}));
    const auto& stage = mpsrt_model->stages.front();
    EXPECT_EQ(stage.kind, GfxMpsrtStageKind::MSLDispatch);
    EXPECT_EQ(stage.kernel_name, "add1");
    EXPECT_EQ(stage.dispatch_kernel_family, "eltwise_fused_buffer");
    EXPECT_EQ(stage.dispatch_entry_point, "add1");
    EXPECT_EQ(stage.msl_dispatch_desc.kernel_family,
              static_cast<uint32_t>(GfxMslKernelFamily::EltwiseFusedBuffer));
    EXPECT_EQ(stage.msl_dispatch_desc.input_count, 1u);
    EXPECT_EQ(stage.msl_dispatch_desc.output_count, 1u);
    EXPECT_EQ(stage.msl_dispatch_desc.threads_per_threadgroup, 64u);
}

TEST(GfxBackendTest, MpsrtContextCachesPreparedMslDispatchPipelines) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    const char* source = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void add1(device float* data [[buffer(0)]],
                 uint gid [[thread_position_in_grid]]) {
  data[gid] += 1.0f;
}
)MSL";

    mlir::MLIRContext ctx;
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mlir::Builder builder(&ctx);
    module->setAttr("gfx.backend", builder.getStringAttr("apple_msl"));
    module->setAttr("gfx.stage_type", builder.getStringAttr("Add"));
    module->setAttr("gfx.mpsrt.stage_kind", builder.getStringAttr("msl_dispatch"));
    module->setAttr("gfx.mpsrt.kernel_name", builder.getStringAttr("add1"));
    module->setAttr("gfx.mpsrt.builder_symbol", builder.getStringAttr("ovgfx_mpsrt_encode_dispatch"));
    module->setAttr("gfx.specialization_key", builder.getStringAttr("apple_msl:buffer:Add"));
    module->setAttr("gfx.mpsrt.dispatch_kernel_family", builder.getStringAttr("eltwise_fused_buffer"));
    module->setAttr("gfx.mpsrt.dispatch_entry_point", builder.getStringAttr("add1"));
    module->setAttr("gfx.mpsrt.dispatch_kernel_family_id",
                    builder.getI32IntegerAttr(static_cast<int32_t>(GfxMslKernelFamily::EltwiseFusedBuffer)));
    module->setAttr("gfx.mpsrt.dispatch_flags",
                    builder.getI32IntegerAttr(GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired));
    module->setAttr("gfx.mpsrt.dispatch_threads_per_threadgroup", builder.getI32IntegerAttr(64));
    module->setAttr("gfx.mpsrt.dispatch_precompiled_kernel_required", builder.getBoolAttr(true));
    module->setAttr("gfx.mpsrt.stage_record_key",
                    builder.getStringAttr("msl_dispatch|apple_msl|buffer|buffer|linear|Add|"
                                          "apple_msl:buffer:Add|dispatch:eltwise_fused_buffer:add1:tg64:metallib"));
    module->setAttr("gfx.mpsrt.input_count", builder.getI32IntegerAttr(1));
    module->setAttr("gfx.mpsrt.output_count", builder.getI32IntegerAttr(1));

    const auto input_desc = gfx_mpsrt_make_tensor_desc({64},
                                                       ov::element::f32,
                                                       GfxStageStorageKind::Buffer,
                                                       GfxMpsrtTensorFlagExternalIo);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({64},
                                                        ov::element::f32,
                                                        GfxStageStorageKind::Buffer,
                                                        GfxMpsrtTensorFlagTransient);
    detail::gfx_mpsrt_set_tensor_desc_attrs(module, "gfx.mpsrt.input0", input_desc);
    detail::gfx_mpsrt_set_tensor_desc_attrs(module, "gfx.mpsrt.output0", output_desc);

    KernelSource ks;
    ks.module = module;
    ks.entry_point = "add1";
    ks.msl_source = source;
    ks.signature.arg_count = 1;

    MetalCodegenBackend backend((MetalDeviceHandle)device);
    std::string log;
    auto kernel = backend.compile(ks, &log);
    ASSERT_TRUE(kernel) << log;

    auto* metal_kernel = dynamic_cast<MetalCompiledKernel*>(kernel.get());
    ASSERT_NE(metal_kernel, nullptr);
    const auto* mpsrt_model = metal_kernel->mpsrt_model();
    ASSERT_NE(mpsrt_model, nullptr);

    metal::mpsrt::MpsrtContext runtime_context(device);
    metal::mpsrt::MpsrtPreparedModel first_prepare;
    ASSERT_TRUE(runtime_context.prepare_model(*mpsrt_model, source, first_prepare, &log)) << log;
    ASSERT_EQ(first_prepare.msl_dispatches.size(), 1u);
    EXPECT_FALSE(first_prepare.msl_dispatches.front().pipeline_cache_hit);
    EXPECT_EQ(first_prepare.msl_dispatches.front().dispatch_entry_point, "add1");
    EXPECT_EQ(first_prepare.msl_dispatches.front().dispatch_kernel_family_id,
              static_cast<uint32_t>(GfxMslKernelFamily::EltwiseFusedBuffer));
    EXPECT_EQ(first_prepare.msl_dispatches.front().dispatch_threads_per_threadgroup, 64u);
    EXPECT_GT(first_prepare.msl_dispatches.front().thread_execution_width, 0u);
    EXPECT_GT(first_prepare.msl_dispatches.front().max_total_threads_per_threadgroup, 0u);
    EXPECT_NE(first_prepare.msl_dispatches.front().pipeline, nil);
    EXPECT_EQ(runtime_context.pipeline_cache_size(), 1u);
    EXPECT_EQ(runtime_context.pipeline_cache_misses(), 1u);

    metal::mpsrt::MpsrtPreparedModel second_prepare;
    ASSERT_TRUE(runtime_context.prepare_model(*mpsrt_model, source, second_prepare, &log)) << log;
    ASSERT_EQ(second_prepare.msl_dispatches.size(), 1u);
    EXPECT_TRUE(second_prepare.msl_dispatches.front().pipeline_cache_hit);
    EXPECT_EQ(second_prepare.msl_dispatches.front().pipeline, first_prepare.msl_dispatches.front().pipeline);
    EXPECT_EQ(runtime_context.pipeline_cache_size(), 1u);
    EXPECT_EQ(runtime_context.pipeline_cache_hits(), 1u);
}

TEST(GfxBackendTest, AnnotatedMslKernelExecutesThroughMpsrtRequest) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    const char* source = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void add1(device float* data [[buffer(0)]],
                 uint gid [[thread_position_in_grid]]) {
  data[gid] += 1.0f;
}
)MSL";

    mlir::MLIRContext ctx;
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mlir::Builder builder(&ctx);
    module->setAttr("gfx.backend", builder.getStringAttr("apple_msl"));
    module->setAttr("gfx.stage_type", builder.getStringAttr("Add"));
    module->setAttr("gfx.mpsrt.stage_kind", builder.getStringAttr("msl_dispatch"));
    module->setAttr("gfx.mpsrt.kernel_name", builder.getStringAttr("add1"));
    module->setAttr("gfx.mpsrt.builder_symbol", builder.getStringAttr("ovgfx_mpsrt_encode_dispatch"));
    module->setAttr("gfx.specialization_key", builder.getStringAttr("apple_msl:buffer:Add"));
    module->setAttr("gfx.mpsrt.dispatch_kernel_family", builder.getStringAttr("eltwise_fused_buffer"));
    module->setAttr("gfx.mpsrt.dispatch_entry_point", builder.getStringAttr("add1"));
    module->setAttr("gfx.mpsrt.dispatch_kernel_family_id",
                    builder.getI32IntegerAttr(static_cast<int32_t>(GfxMslKernelFamily::EltwiseFusedBuffer)));
    module->setAttr("gfx.mpsrt.dispatch_flags",
                    builder.getI32IntegerAttr(GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired));
    module->setAttr("gfx.mpsrt.dispatch_threads_per_threadgroup", builder.getI32IntegerAttr(64));
    module->setAttr("gfx.mpsrt.dispatch_precompiled_kernel_required", builder.getBoolAttr(true));
    module->setAttr("gfx.mpsrt.stage_record_key",
                    builder.getStringAttr("msl_dispatch|apple_msl|buffer|buffer|linear|Add|"
                                          "apple_msl:buffer:Add|dispatch:eltwise_fused_buffer:add1:tg64:metallib"));
    module->setAttr("gfx.mpsrt.input_count", builder.getI32IntegerAttr(1));
    module->setAttr("gfx.mpsrt.output_count", builder.getI32IntegerAttr(1));

    const auto input_desc = gfx_mpsrt_make_tensor_desc({8},
                                                       ov::element::f32,
                                                       GfxStageStorageKind::Buffer,
                                                       GfxMpsrtTensorFlagExternalIo);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({8},
                                                        ov::element::f32,
                                                        GfxStageStorageKind::Buffer,
                                                        GfxMpsrtTensorFlagTransient);
    detail::gfx_mpsrt_set_tensor_desc_attrs(module, "gfx.mpsrt.input0", input_desc);
    detail::gfx_mpsrt_set_tensor_desc_attrs(module, "gfx.mpsrt.output0", output_desc);

    KernelSource ks;
    ks.module = module;
    ks.entry_point = "add1";
    ks.msl_source = source;
    ks.signature.arg_count = 1;

    MetalCodegenBackend backend((MetalDeviceHandle)device);
    std::string log;
    auto kernel = backend.compile(ks, &log);
    ASSERT_TRUE(kernel) << log;

    constexpr uint32_t kCount = 8;
    id<MTLBuffer> buf = [device newBufferWithLength:sizeof(float) * kCount
                                           options:MTLResourceStorageModeShared];
    ASSERT_NE(buf, nil);
    float* ptr = static_cast<float*>([buf contents]);
    ASSERT_NE(ptr, nullptr);
    for (uint32_t i = 0; i < kCount; ++i) {
        ptr[i] = static_cast<float>(i);
    }

    MetalBuffer gpu_buf{};
    gpu_buf.buffer = (__bridge void*)buf;
    gpu_buf.size = sizeof(float) * kCount;
    gpu_buf.type = ov::element::f32;

    KernelDispatch dispatch;
    dispatch.grid[0] = kCount;
    dispatch.threads_per_group[0] = kernel->clamp_threadgroup_size(64);

    std::vector<KernelArg> args;
    args.push_back(make_buffer_arg(0, gpu_buf));

    std::unordered_map<std::string, uint64_t> counters;
    KernelExecutionHooks hooks;
    hooks.on_counter = [&counters](std::string_view name, uint64_t delta) {
        counters[std::string(name)] += delta;
    };

    id<MTLCommandQueue> queue = [device newCommandQueue];
    ASSERT_NE(queue, nil);
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    ASSERT_NE(cmd, nil);
    kernel->execute((GpuCommandBufferHandle)cmd, dispatch, args, &hooks);
    metal_end_compute_encoder((GpuCommandBufferHandle)cmd);
    [cmd commit];
    [cmd waitUntilCompleted];

    EXPECT_EQ(counters["mpsrt_model_request_encode_count"], 1u);
    EXPECT_EQ(counters["mpsrt_model_request_msl_stage_encode_count"], 1u);
    EXPECT_EQ(counters["mpsrt_msl_request_encode_count"], 1u);
    EXPECT_EQ(counters["mpsrt_binding_external_input_count"], 1u);
    EXPECT_EQ(counters["mpsrt_binding_external_output_count"], 1u);
    EXPECT_EQ(counters["mpsrt_binding_transient_alloc_count"], 0u);
    for (uint32_t i = 0; i < kCount; ++i) {
        EXPECT_FLOAT_EQ(ptr[i], static_cast<float>(i + 1));
    }
}

TEST(GfxBackendTest, AnnotatedMslKernelWithExpandedAbiExecutesThroughMpsrtBufferOrder) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    const char* source = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void add_bias(device const float* input [[buffer(0)]],
                     constant float& bias [[buffer(1)]],
                     device float* output [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
  output[gid] = input[gid] + bias;
}
)MSL";

    mlir::MLIRContext ctx;
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mlir::Builder builder(&ctx);
    module->setAttr("gfx.backend", builder.getStringAttr("apple_msl"));
    module->setAttr("gfx.stage_type", builder.getStringAttr("Add"));
    module->setAttr("gfx.mpsrt.stage_kind", builder.getStringAttr("msl_dispatch"));
    module->setAttr("gfx.mpsrt.kernel_name", builder.getStringAttr("add_bias"));
    module->setAttr("gfx.mpsrt.builder_symbol", builder.getStringAttr("ovgfx_mpsrt_encode_dispatch"));
    module->setAttr("gfx.specialization_key", builder.getStringAttr("apple_msl:buffer:Add"));
    module->setAttr("gfx.mpsrt.dispatch_kernel_family", builder.getStringAttr("eltwise_fused_buffer"));
    module->setAttr("gfx.mpsrt.dispatch_entry_point", builder.getStringAttr("add_bias"));
    module->setAttr("gfx.mpsrt.dispatch_kernel_family_id",
                    builder.getI32IntegerAttr(static_cast<int32_t>(GfxMslKernelFamily::EltwiseFusedBuffer)));
    module->setAttr("gfx.mpsrt.dispatch_flags",
                    builder.getI32IntegerAttr(GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired));
    module->setAttr("gfx.mpsrt.dispatch_threads_per_threadgroup", builder.getI32IntegerAttr(64));
    module->setAttr("gfx.mpsrt.dispatch_precompiled_kernel_required", builder.getBoolAttr(true));
    module->setAttr("gfx.mpsrt.external_buffer_count", builder.getI32IntegerAttr(3));
    module->setAttr("gfx.mpsrt.external_output_buffer_count", builder.getI32IntegerAttr(1));
    module->setAttr("gfx.mpsrt.stage_record_key",
                    builder.getStringAttr("msl_dispatch|apple_msl|buffer|buffer|linear|Add|"
                                          "apple_msl:buffer:Add|dispatch:eltwise_fused_buffer:add_bias:tg64:metallib"));
    module->setAttr("gfx.mpsrt.input_count", builder.getI32IntegerAttr(1));
    module->setAttr("gfx.mpsrt.output_count", builder.getI32IntegerAttr(1));

    const auto input_desc = gfx_mpsrt_make_tensor_desc({8},
                                                       ov::element::f32,
                                                       GfxStageStorageKind::Buffer,
                                                       GfxMpsrtTensorFlagExternalIo);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({8},
                                                        ov::element::f32,
                                                        GfxStageStorageKind::Buffer,
                                                        GfxMpsrtTensorFlagTransient);
    detail::gfx_mpsrt_set_tensor_desc_attrs(module, "gfx.mpsrt.input0", input_desc);
    detail::gfx_mpsrt_set_tensor_desc_attrs(module, "gfx.mpsrt.output0", output_desc);

    KernelSource ks;
    ks.module = module;
    ks.entry_point = "add_bias";
    ks.msl_source = source;
    ks.signature.arg_count = 3;

    MetalCodegenBackend backend((MetalDeviceHandle)device);
    std::string log;
    auto kernel = backend.compile(ks, &log);
    ASSERT_TRUE(kernel) << log;

    auto* metal_kernel = dynamic_cast<MetalCompiledKernel*>(kernel.get());
    ASSERT_NE(metal_kernel, nullptr);
    const auto* mpsrt_model = metal_kernel->mpsrt_model();
    ASSERT_NE(mpsrt_model, nullptr);
    ASSERT_EQ(mpsrt_model->stages.size(), 1u);
    EXPECT_EQ(mpsrt_model->semantic_input_values, std::vector<GfxMpsrtValue>({0u}));
    EXPECT_EQ(mpsrt_model->semantic_output_values, std::vector<GfxMpsrtValue>({1u}));
    EXPECT_EQ(mpsrt_model->input_values, std::vector<GfxMpsrtValue>({0u, 1u}));
    EXPECT_EQ(mpsrt_model->output_values, std::vector<GfxMpsrtValue>({2u}));
    EXPECT_EQ(mpsrt_model->external_values, std::vector<GfxMpsrtValue>({0u, 1u, 2u}));
    EXPECT_EQ(mpsrt_model->external_input_values, std::vector<GfxMpsrtValue>({0u, 1u}));
    EXPECT_EQ(mpsrt_model->external_output_values, std::vector<GfxMpsrtValue>({2u}));
    EXPECT_EQ(mpsrt_model->external_buffer_roles,
              std::vector<GfxMpsrtExternalBufferRole>({GfxMpsrtExternalBufferRole::TensorInput,
                                                       GfxMpsrtExternalBufferRole::TensorInput,
                                                       GfxMpsrtExternalBufferRole::TensorOutput}));
    EXPECT_EQ(mpsrt_model->stages.front().kernel_buffer_order,
              std::vector<GfxMpsrtValue>({0u, 1u, 2u}));
    EXPECT_EQ(mpsrt_model->stages.front().msl_dispatch_desc.input_count, 2u);
    EXPECT_EQ(mpsrt_model->stages.front().msl_dispatch_desc.output_count, 1u);

    constexpr uint32_t kCount = 8;
    id<MTLBuffer> input = [device newBufferWithLength:sizeof(float) * kCount
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> bias = [device newBufferWithLength:sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:sizeof(float) * kCount
                                              options:MTLResourceStorageModeShared];
    ASSERT_NE(input, nil);
    ASSERT_NE(bias, nil);
    ASSERT_NE(output, nil);
    float* input_ptr = static_cast<float*>([input contents]);
    float* bias_ptr = static_cast<float*>([bias contents]);
    float* output_ptr = static_cast<float*>([output contents]);
    ASSERT_NE(input_ptr, nullptr);
    ASSERT_NE(bias_ptr, nullptr);
    ASSERT_NE(output_ptr, nullptr);
    *bias_ptr = 2.5f;
    for (uint32_t i = 0; i < kCount; ++i) {
        input_ptr[i] = static_cast<float>(i);
        output_ptr[i] = -1.0f;
    }

    MetalBuffer input_buf{};
    input_buf.buffer = (__bridge void*)input;
    input_buf.size = sizeof(float) * kCount;
    input_buf.type = ov::element::f32;
    MetalBuffer bias_buf{};
    bias_buf.buffer = (__bridge void*)bias;
    bias_buf.size = sizeof(float);
    bias_buf.type = ov::element::f32;
    MetalBuffer output_buf{};
    output_buf.buffer = (__bridge void*)output;
    output_buf.size = sizeof(float) * kCount;
    output_buf.type = ov::element::f32;

    KernelDispatch dispatch;
    dispatch.grid[0] = kCount;
    dispatch.threads_per_group[0] = kernel->clamp_threadgroup_size(64);
    std::vector<KernelArg> args = {
        make_buffer_arg(0, input_buf),
        make_buffer_arg(1, bias_buf),
        make_buffer_arg(2, output_buf),
    };

    std::unordered_map<std::string, uint64_t> counters;
    KernelExecutionHooks hooks;
    hooks.on_counter = [&counters](std::string_view name, uint64_t delta) {
        counters[std::string(name)] += delta;
    };

    id<MTLCommandQueue> queue = [device newCommandQueue];
    ASSERT_NE(queue, nil);
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    ASSERT_NE(cmd, nil);
    kernel->execute((GpuCommandBufferHandle)cmd, dispatch, args, &hooks);
    metal_end_compute_encoder((GpuCommandBufferHandle)cmd);
    [cmd commit];
    [cmd waitUntilCompleted];

    EXPECT_EQ(counters["mpsrt_model_request_encode_count"], 1u);
    EXPECT_EQ(counters["mpsrt_binding_external_input_count"], 2u);
    EXPECT_EQ(counters["mpsrt_binding_external_output_count"], 1u);
    for (uint32_t i = 0; i < kCount; ++i) {
        EXPECT_FLOAT_EQ(output_ptr[i], static_cast<float>(i) + 2.5f);
    }
}

TEST(GfxBackendTest, AnnotatedSoftmaxMslKernelUsesRoleBasedMpsrtBufferOrder) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    const char* source = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct SoftmaxParams {
  uint rows;
  uint cols;
  uint inner;
  uint log_softmax;
};
kernel void softmax_kernel(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant SoftmaxParams& p [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
  uint total = p.rows * p.cols * p.inner;
  if (gid >= total) return;
  output[gid] = input[gid];
}
)MSL";

    mlir::MLIRContext ctx;
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Metal,
                                                     "Softmax",
                                                     nullptr,
                                                     ov::element::f32,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     {});
    annotate_msl_module_with_stage_plan(module, plan, "Softmax");

    constexpr uint32_t kCount = 8;
    const auto input_desc = gfx_mpsrt_make_tensor_desc({kCount},
                                                       ov::element::f32,
                                                       GfxStageStorageKind::Buffer,
                                                       GfxMpsrtTensorFlagExternalIo);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({kCount},
                                                        ov::element::f32,
                                                        GfxStageStorageKind::Buffer,
                                                        GfxMpsrtTensorFlagTransient);
    mlir::Builder builder(&ctx);
    module->setAttr("gfx.mpsrt.input_count", builder.getI32IntegerAttr(1));
    module->setAttr("gfx.mpsrt.output_count", builder.getI32IntegerAttr(1));
    detail::gfx_mpsrt_set_tensor_desc_attrs(module, "gfx.mpsrt.input0", input_desc);
    detail::gfx_mpsrt_set_tensor_desc_attrs(module, "gfx.mpsrt.output0", output_desc);

    KernelSource ks;
    ks.module = module;
    ks.entry_point = "softmax_kernel";
    ks.msl_source = source;
    ks.signature.arg_count = 3;
    configure_msl_kernel_source_for_plan(ks, "Softmax");
    ASSERT_EQ(ks.entry_point, "masked_softmax_attention");
    ASSERT_NE(ks.msl_source.find("kernel void masked_softmax_attention"), std::string::npos);
    ASSERT_EQ(ks.msl_source.find("kernel void softmax_kernel"), std::string::npos);

    MetalCodegenBackend backend((MetalDeviceHandle)device);
    std::string log;
    auto kernel = backend.compile(ks, &log);
    ASSERT_TRUE(kernel) << log;

    auto* metal_kernel = dynamic_cast<MetalCompiledKernel*>(kernel.get());
    ASSERT_NE(metal_kernel, nullptr);
    const auto* mpsrt_model = metal_kernel->mpsrt_model();
    ASSERT_NE(mpsrt_model, nullptr);
    ASSERT_EQ(mpsrt_model->stages.size(), 1u);
    EXPECT_EQ(mpsrt_model->semantic_input_values, std::vector<GfxMpsrtValue>({0u}));
    EXPECT_EQ(mpsrt_model->semantic_output_values, std::vector<GfxMpsrtValue>({1u}));
    EXPECT_EQ(mpsrt_model->input_values, std::vector<GfxMpsrtValue>({0u, 2u}));
    EXPECT_EQ(mpsrt_model->output_values, std::vector<GfxMpsrtValue>({1u}));
    EXPECT_EQ(mpsrt_model->external_values, std::vector<GfxMpsrtValue>({0u, 1u, 2u}));
    EXPECT_EQ(mpsrt_model->external_input_values, std::vector<GfxMpsrtValue>({0u, 2u}));
    EXPECT_EQ(mpsrt_model->external_output_values, std::vector<GfxMpsrtValue>({1u}));
    EXPECT_EQ(mpsrt_model->external_buffer_roles,
              std::vector<GfxMpsrtExternalBufferRole>({GfxMpsrtExternalBufferRole::TensorInput,
                                                       GfxMpsrtExternalBufferRole::TensorOutput,
                                                       GfxMpsrtExternalBufferRole::RuntimeParams}));
    EXPECT_EQ(mpsrt_model->stages.front().kernel_buffer_order,
              std::vector<GfxMpsrtValue>({0u, 1u, 2u}));
    EXPECT_EQ(mpsrt_model->stages.front().msl_dispatch_desc.input_count, 2u);
    EXPECT_EQ(mpsrt_model->stages.front().msl_dispatch_desc.output_count, 1u);

    id<MTLBuffer> input = [device newBufferWithLength:sizeof(float) * kCount
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:sizeof(float) * kCount
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> params = [device newBufferWithLength:sizeof(uint32_t) * 4
                                             options:MTLResourceStorageModeShared];
    ASSERT_NE(input, nil);
    ASSERT_NE(output, nil);
    ASSERT_NE(params, nil);
    float* input_ptr = static_cast<float*>([input contents]);
    float* output_ptr = static_cast<float*>([output contents]);
    uint32_t* params_ptr = static_cast<uint32_t*>([params contents]);
    ASSERT_NE(input_ptr, nullptr);
    ASSERT_NE(output_ptr, nullptr);
    ASSERT_NE(params_ptr, nullptr);
    params_ptr[0] = 1;
    params_ptr[1] = kCount;
    params_ptr[2] = 1;
    params_ptr[3] = 0;
    for (uint32_t i = 0; i < kCount; ++i) {
        input_ptr[i] = static_cast<float>(i) * 0.5f;
        output_ptr[i] = -1.0f;
    }

    MetalBuffer input_buf{};
    input_buf.buffer = (__bridge void*)input;
    input_buf.size = sizeof(float) * kCount;
    input_buf.type = ov::element::f32;
    MetalBuffer output_buf{};
    output_buf.buffer = (__bridge void*)output;
    output_buf.size = sizeof(float) * kCount;
    output_buf.type = ov::element::f32;
    MetalBuffer params_buf{};
    params_buf.buffer = (__bridge void*)params;
    params_buf.size = sizeof(uint32_t) * 4;
    params_buf.type = ov::element::u32;

    KernelDispatch dispatch;
    dispatch.grid[0] = kCount;
    dispatch.threads_per_group[0] = kernel->clamp_threadgroup_size(64);
    std::vector<KernelArg> args = {
        make_buffer_arg(0, input_buf),
        make_buffer_arg(1, output_buf),
        make_buffer_arg(2, params_buf),
    };

    std::unordered_map<std::string, uint64_t> counters;
    KernelExecutionHooks hooks;
    hooks.on_counter = [&counters](std::string_view name, uint64_t delta) {
        counters[std::string(name)] += delta;
    };

    id<MTLCommandQueue> queue = [device newCommandQueue];
    ASSERT_NE(queue, nil);
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    ASSERT_NE(cmd, nil);
    kernel->execute((GpuCommandBufferHandle)cmd, dispatch, args, &hooks);
    metal_end_compute_encoder((GpuCommandBufferHandle)cmd);
    [cmd commit];
    [cmd waitUntilCompleted];

    EXPECT_EQ(counters["mpsrt_model_request_encode_count"], 1u);
    EXPECT_EQ(counters["mpsrt_binding_external_input_count"], 2u);
    EXPECT_EQ(counters["mpsrt_binding_external_output_count"], 1u);
    for (uint32_t i = 0; i < kCount; ++i) {
        EXPECT_FLOAT_EQ(output_ptr[i], input_ptr[i]);
    }
}

TEST(GfxBackendTest, MpsrtRequestEncodesPreparedTwoStageMslModelWithValueBindings) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    const char* source = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void add1(device const float* input [[buffer(0)]],
                 device float* temp [[buffer(1)]],
                 uint gid [[thread_position_in_grid]]) {
  temp[gid] = input[gid] + 1.0f;
}
kernel void mul2(device const float* temp [[buffer(0)]],
                 device float* output [[buffer(1)]],
                 uint gid [[thread_position_in_grid]]) {
  output[gid] = temp[gid] * 2.0f;
}
)MSL";

    auto make_stage = [](size_t index,
                         const char* key,
                         const char* entry,
                         GfxMpsrtValue input,
                         GfxMpsrtValue output) {
        metal::mpsrt::MpsrtRuntimeStage stage;
        stage.kind = GfxMpsrtStageKind::MSLDispatch;
        stage.stage_record_key = key;
        stage.kernel_name = entry;
        stage.dispatch_kernel_family = "eltwise_fused_buffer";
        stage.dispatch_entry_point = entry;
        stage.dispatch_kernel_family_id = static_cast<uint32_t>(GfxMslKernelFamily::EltwiseFusedBuffer);
        stage.dispatch_flags = GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired;
        stage.dispatch_threads_per_threadgroup = 64;
        stage.dispatch_precompiled_kernel_required = true;
        stage.msl_dispatch_desc.kernel_family = stage.dispatch_kernel_family_id;
        stage.msl_dispatch_desc.flags = stage.dispatch_flags;
        stage.msl_dispatch_desc.input_count = 1;
        stage.msl_dispatch_desc.output_count = 1;
        stage.msl_dispatch_desc.threads_per_threadgroup = 64;
        stage.inputs = {input};
        stage.outputs = {output};
        (void)index;
        return stage;
    };

    metal::mpsrt::MpsrtModel model;
    constexpr uint32_t kCount = 16;
    model.stage_record_key = "two_stage_msl_model";
    model.input_values = {0};
    model.output_values = {2};
    const auto input_desc = gfx_mpsrt_make_tensor_desc({kCount},
                                                       ov::element::f32,
                                                       GfxStageStorageKind::Buffer,
                                                       GfxMpsrtTensorFlagExternalIo);
    const auto temp_desc = gfx_mpsrt_make_tensor_desc({kCount},
                                                      ov::element::f32,
                                                      GfxStageStorageKind::Buffer,
                                                      GfxMpsrtTensorFlagTransient);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({kCount},
                                                        ov::element::f32,
                                                        GfxStageStorageKind::Buffer,
                                                        GfxMpsrtTensorFlagExternalIo);
    model.tensors.push_back({0, gfx_mpsrt_to_abi_desc(input_desc)});
    model.tensors.push_back({1, gfx_mpsrt_to_abi_desc(temp_desc)});
    model.tensors.push_back({2, gfx_mpsrt_to_abi_desc(output_desc)});
    model.stages.push_back(make_stage(0, "stage0_add1", "add1", 0, 1));
    model.stages.push_back(make_stage(1, "stage1_mul2", "mul2", 1, 2));

    metal::mpsrt::MpsrtContext context(device);
    metal::mpsrt::MpsrtPreparedModel prepared_model;
    std::string log;
    ASSERT_TRUE(context.prepare_model(model, source, prepared_model, &log)) << log;
    ASSERT_EQ(prepared_model.msl_dispatches.size(), 2u);

    id<MTLBuffer> input = [device newBufferWithLength:sizeof(float) * kCount
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:sizeof(float) * kCount
                                              options:MTLResourceStorageModeShared];
    ASSERT_NE(input, nil);
    ASSERT_NE(output, nil);

    float* input_ptr = static_cast<float*>([input contents]);
    float* output_ptr = static_cast<float*>([output contents]);
    ASSERT_NE(input_ptr, nullptr);
    ASSERT_NE(output_ptr, nullptr);
    for (uint32_t i = 0; i < kCount; ++i) {
        input_ptr[i] = static_cast<float>(i);
        output_ptr[i] = -1.0f;
    }

    metal::mpsrt::MpsrtTensorBindings bindings;
    std::vector<id<MTLBuffer>> transient_buffers;
    auto transient_allocator = [&](const metal::mpsrt::MpsrtRuntimeTensor& tensor) {
        id<MTLBuffer> buffer =
            [device newBufferWithLength:static_cast<NSUInteger>(tensor.desc.byte_length)
                                options:MTLResourceStorageModeShared];
        transient_buffers.push_back(buffer);
        return metal::mpsrt::MpsrtBoundBuffer{(__bridge void*)buffer,
                                              static_cast<size_t>(tensor.desc.byte_offset)};
    };
    metal::mpsrt::MpsrtBindingBuildResult binding_result;
    ASSERT_TRUE(metal::mpsrt::build_mpsrt_tensor_bindings(model,
                                                          {{(__bridge void*)input, 0}},
                                                          {{(__bridge void*)output, 0}},
                                                          transient_allocator,
                                                          bindings,
                                                          &binding_result,
                                                          &log))
        << log;
    EXPECT_EQ(binding_result.external_inputs_bound, 1u);
    EXPECT_EQ(binding_result.external_outputs_bound, 1u);
    EXPECT_EQ(binding_result.transient_buffers_allocated, 1u);
    EXPECT_EQ(binding_result.const_tensors_skipped, 0u);
    EXPECT_EQ(transient_buffers.size(), 1u);

    KernelDispatch dispatch;
    dispatch.grid[0] = kCount;
    dispatch.threads_per_group[0] = 64;
    std::vector<KernelDispatch> stage_dispatches = {dispatch, dispatch};

    std::unordered_map<std::string, uint64_t> counters;
    KernelExecutionHooks hooks;
    hooks.on_counter = [&counters](std::string_view name, uint64_t delta) {
        counters[std::string(name)] += delta;
    };

    id<MTLCommandQueue> queue = [device newCommandQueue];
    ASSERT_NE(queue, nil);
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    ASSERT_NE(cmd, nil);

    metal::mpsrt::MpsrtRequest request;
    metal::mpsrt::MpsrtModelEncodeResult result;
    ASSERT_TRUE(request.encode_prepared_model((GpuCommandBufferHandle)cmd,
                                              model,
                                              prepared_model,
                                              stage_dispatches,
                                              bindings,
                                              &hooks,
                                              &result,
                                              &log))
        << log;
    metal_end_compute_encoder((GpuCommandBufferHandle)cmd);
    [cmd commit];
    [cmd waitUntilCompleted];

    EXPECT_EQ(result.encoded_msl_dispatches, 2u);
    EXPECT_EQ(result.skipped_non_msl_stages, 0u);
    EXPECT_EQ(counters["mpsrt_model_request_encode_count"], 1u);
    EXPECT_EQ(counters["mpsrt_model_request_msl_stage_encode_count"], 2u);
    EXPECT_EQ(counters["mpsrt_msl_request_encode_count"], 2u);
    for (uint32_t i = 0; i < kCount; ++i) {
        EXPECT_FLOAT_EQ(output_ptr[i], static_cast<float>((i + 1) * 2));
    }
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
