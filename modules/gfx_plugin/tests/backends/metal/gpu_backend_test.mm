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
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "backends/metal/codegen/metal_compiler.hpp"
#include "backends/metal/runtime/metal_command_encoder.hpp"
#include "backends/metal/runtime/mpsrt/mpsrt_context.hpp"
#include "backends/metal/runtime/mpsrt/mpsrt_request.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "mlir/msl_codegen.hpp"
#include "runtime/gfx_stage_policy.hpp"
#include "kernel_ir/gfx_custom_kernel_families.hpp"

namespace ov {
namespace gfx_plugin {

namespace runtime_mpsrt = ::ov::gfx_plugin::mpsrt;

namespace {

void annotate_test_msl_dispatch_module(
    mlir::ModuleOp module,
    std::string_view stage_type,
    std::string_view entry_point,
    GfxKernelExternalBufferAbiSpec external_buffer_abi = {},
    uint32_t threads_per_threadgroup = 64) {
    if (!external_buffer_abi.valid) {
        external_buffer_abi = make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                                         GfxKernelBufferRole::TensorOutput});
    }
    std::string specialization_key = "apple_msl:buffer:";
    specialization_key += stage_type;

    detail::gfx_mpsrt_set_stage_manifest_attrs(
        module,
        make_gfx_custom_kernel_stage_manifest(
            GfxKernelStageFamily::Eltwise,
            GfxKernelBackendDomain::AppleMsl,
            GfxKernelStorageKind::Buffer,
            specialization_key,
            make_gfx_custom_kernel_manifest(
                "eltwise_fused_buffer",
                static_cast<uint32_t>(GfxKernelFamily::EltwiseFusedBuffer),
                std::string(entry_point),
                std::move(external_buffer_abi),
                make_gfx_kernel_linear_dispatch_policy(
                    threads_per_threadgroup,
                    /*precompiled_binary_required=*/true))));
}

void annotate_test_mps_vendor_module(mlir::ModuleOp module,
                                     std::string_view stage_type,
                                     GfxKernelStageFamily stage_family,
                                     GfxKernelStorageKind storage = GfxKernelStorageKind::Matrix) {
    std::string specialization_key = "apple_mps:";
    specialization_key += gfx_kernel_storage_kind_name(storage);
    specialization_key += ":";
    specialization_key += stage_type;

    detail::gfx_mpsrt_set_stage_manifest_attrs(
        module,
        make_gfx_vendor_stage_manifest(stage_family,
                                       GfxKernelBackendDomain::AppleMps,
                                       storage,
                                       specialization_key));
}

void materialize_test_mpsrt_stage_module(mlir::ModuleOp module,
                                         const std::vector<GfxMpsrtTensorDesc>& inputs,
                                         const std::vector<GfxMpsrtTensorDesc>& outputs) {
    GfxMpsrtModuleStagePlan stage_plan{};
    ASSERT_TRUE(read_module_mpsrt_stage_plan(module, stage_plan));
    stage_plan.inputs = inputs;
    stage_plan.outputs = outputs;
    if (!stage_plan.inputs.empty()) {
        stage_plan.stage.input_storage = stage_plan.inputs.front().storage;
    }
    if (!stage_plan.outputs.empty()) {
        stage_plan.stage.output_storage = stage_plan.outputs.front().storage;
    } else {
        stage_plan.stage.output_storage = stage_plan.stage.input_storage;
    }
    stage_plan.stage.layout = stage_plan.stage.output_storage != GfxMpsrtStorage::Unknown
                                  ? gfx_mpsrt_stage_layout_for_storage(stage_plan.stage.output_storage)
                                  : GfxMpsrtLayout::Unknown;
    stage_plan.stage_record_key = gfx_mpsrt_stage_record_key(stage_plan.stage);
    stage_plan.valid = stage_plan.stage.domain != GfxStageBackendDomain::Unknown &&
                       stage_plan.stage.kind != GfxMpsrtStageKind::Unknown &&
                       gfx_mpsrt_stage_has_builder_symbol(stage_plan.stage.kind) &&
                       !stage_plan.stage_record_key.empty();
    ASSERT_TRUE(stage_plan.valid);
    ASSERT_TRUE(materialize_module_mpsrt_ops_from_stage_plan(module, stage_plan));
}

GfxAppleMpsStageLoweringPlan make_test_mps_vendor_lowering(
    mlir::ModuleOp module,
    const std::vector<GfxMpsrtTensorDesc>& inputs,
    const std::vector<GfxMpsrtTensorDesc>& outputs) {
    GfxAppleMpsStageLoweringPlan lowering_plan{};
    if (!read_module_mpsrt_stage_plan(module, lowering_plan.stage_plan)) {
        return {};
    }
    lowering_plan.stage_plan.inputs = inputs;
    lowering_plan.stage_plan.outputs = outputs;
    if (!lowering_plan.stage_plan.inputs.empty()) {
        lowering_plan.stage_plan.stage.input_storage = lowering_plan.stage_plan.inputs.front().storage;
    }
    if (!lowering_plan.stage_plan.outputs.empty()) {
        lowering_plan.stage_plan.stage.output_storage = lowering_plan.stage_plan.outputs.front().storage;
    } else {
        lowering_plan.stage_plan.stage.output_storage = lowering_plan.stage_plan.stage.input_storage;
    }
    lowering_plan.stage_plan.stage.layout =
        lowering_plan.stage_plan.stage.output_storage != GfxMpsrtStorage::Unknown
            ? gfx_mpsrt_stage_layout_for_storage(lowering_plan.stage_plan.stage.output_storage)
            : GfxMpsrtLayout::Unknown;
    lowering_plan.valid = true;
    if (!refresh_apple_mps_stage_record_key(lowering_plan)) {
        return {};
    }
    return lowering_plan;
}

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
    annotate_test_msl_dispatch_module(module, "Add", "add1");

    const auto input_desc = gfx_mpsrt_make_tensor_desc({64},
                                                       ov::element::f32,
                                                       GfxStageStorageKind::Buffer,
                                                       GfxMpsrtTensorFlagExternalIo);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({64},
                                                        ov::element::f32,
                                                        GfxStageStorageKind::Buffer,
                                                        GfxMpsrtTensorFlagTransient);
    materialize_test_mpsrt_stage_module(module, {input_desc}, {output_desc});
    ASSERT_TRUE(static_cast<bool>(module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_ops")));
    ASSERT_FALSE(static_cast<bool>(module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_program")));
    module->removeAttr("gfx.mpsrt.stage_kind");
    module->removeAttr("gfx.mpsrt.model_record_key");
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.stage_kind"));
    ASSERT_FALSE(module->hasAttr("gfx.mpsrt.model_record_key"));

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
              static_cast<uint32_t>(GfxKernelFamily::EltwiseFusedBuffer));
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
    annotate_test_msl_dispatch_module(module, "Add", "add1");

    const auto input_desc = gfx_mpsrt_make_tensor_desc({64},
                                                       ov::element::f32,
                                                       GfxStageStorageKind::Buffer,
                                                       GfxMpsrtTensorFlagExternalIo);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({64},
                                                        ov::element::f32,
                                                        GfxStageStorageKind::Buffer,
                                                        GfxMpsrtTensorFlagTransient);
    materialize_test_mpsrt_stage_module(module, {input_desc}, {output_desc});

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
              static_cast<uint32_t>(GfxKernelFamily::EltwiseFusedBuffer));
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
    annotate_test_msl_dispatch_module(module, "Add", "add1");

    const auto input_desc = gfx_mpsrt_make_tensor_desc({8},
                                                       ov::element::f32,
                                                       GfxStageStorageKind::Buffer,
                                                       GfxMpsrtTensorFlagExternalIo);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({8},
                                                        ov::element::f32,
                                                        GfxStageStorageKind::Buffer,
                                                        GfxMpsrtTensorFlagTransient);
    materialize_test_mpsrt_stage_module(module, {input_desc}, {output_desc});

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
    EXPECT_EQ(counters["mpsrt_binding_prepared_transient_buffer_count"], 0u);
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
    annotate_test_msl_dispatch_module(
        module,
        "Add",
        "add_bias",
        make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                   GfxKernelBufferRole::TensorInput,
                                   GfxKernelBufferRole::TensorOutput}));

    const auto input_desc = gfx_mpsrt_make_tensor_desc({8},
                                                       ov::element::f32,
                                                       GfxStageStorageKind::Buffer,
                                                       GfxMpsrtTensorFlagExternalIo);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({8},
                                                        ov::element::f32,
                                                        GfxStageStorageKind::Buffer,
                                                        GfxMpsrtTensorFlagTransient);
    materialize_test_mpsrt_stage_module(module, {input_desc}, {output_desc});

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

TEST(GfxBackendTest, SingleMslMpsrtBindsModelOwnedConstResourceFromPreparedState) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    const char* source = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void add_const_buffer(device const float* input [[buffer(0)]],
                             device const float* bias [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
  if (gid >= 8) return;
  output[gid] = input[gid] + bias[0];
}
)MSL";

    std::string log;
    MetalKernelCompiler compiler(device);
    id<MTLComputePipelineState> pipeline =
        compiler.compile_msl_from_source(source, "add_const_buffer", log);
    ASSERT_NE(pipeline, nil) << log;

    constexpr uint32_t kCount = 8;
    runtime_mpsrt::MpsrtModel model;
    model.stage_record_key = "single_msl_const_resource_model";
    model.semantic_input_values = {0};
    model.semantic_output_values = {2};
    model.input_values = {0};
    model.output_values = {2};
    model.external_values = {0, 2};
    model.external_input_values = {0};
    model.external_output_values = {2};
    model.external_buffer_roles = {GfxMpsrtExternalBufferRole::TensorInput,
                                   GfxMpsrtExternalBufferRole::TensorOutput};

    const auto input_desc = gfx_mpsrt_make_tensor_desc({kCount},
                                                       ov::element::f32,
                                                       GfxStageStorageKind::Buffer,
                                                       GfxMpsrtTensorFlagExternalIo);
    const auto bias_desc = gfx_mpsrt_make_tensor_desc({1},
                                                      ov::element::f32,
                                                      GfxStageStorageKind::Buffer,
                                                      GfxMpsrtTensorFlagConst);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({kCount},
                                                        ov::element::f32,
                                                        GfxStageStorageKind::Buffer,
                                                        GfxMpsrtTensorFlagExternalIo);
    const auto input_abi = gfx_mpsrt_to_abi_desc(input_desc);
    const auto bias_abi = gfx_mpsrt_to_abi_desc(bias_desc);
    const auto output_abi = gfx_mpsrt_to_abi_desc(output_desc);
    model.tensors.push_back({0, input_abi});
    model.tensors.push_back({1, bias_abi});
    model.tensors.push_back({2, output_abi});
    model.resources = {
        {0u, GfxMpsrtExternalBufferRole::TensorInput, runtime_mpsrt::MpsrtRuntimeResourceLifetime::External, 0u, true, 0u, input_abi},
        {1u, GfxMpsrtExternalBufferRole::ConstBuffer, runtime_mpsrt::MpsrtRuntimeResourceLifetime::Model, 0u, true, 1u, bias_abi},
        {2u, GfxMpsrtExternalBufferRole::TensorOutput, runtime_mpsrt::MpsrtRuntimeResourceLifetime::External, 1u, true, 2u, output_abi},
    };
    model.external_buffer_bindings = {
        {0u, 0u},
        {1u, 2u},
    };

    runtime_mpsrt::MpsrtRuntimeStage stage;
    stage.kind = GfxMpsrtStageKind::MSLDispatch;
    stage.stage_record_key = "msl_dispatch|apple_msl|buffer|buffer|linear|AddConst|"
                              "apple_msl:buffer:AddConst|dispatch:add_const_buffer:"
                              "add_const_buffer:tg64:metallib";
    stage.kernel_name = "add_const_buffer";
    stage.dispatch_kernel_family = "eltwise_fused_buffer";
    stage.dispatch_entry_point = "add_const_buffer";
    stage.dispatch_kernel_family_id = static_cast<uint32_t>(GfxKernelFamily::EltwiseFusedBuffer);
    stage.dispatch_threads_per_threadgroup = 64;
    stage.dispatch_flags = GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired;
    stage.dispatch_precompiled_kernel_required = true;
    stage.msl_dispatch_desc.kernel_family = stage.dispatch_kernel_family_id;
    stage.msl_dispatch_desc.storage = static_cast<uint32_t>(GfxMpsrtStorage::Buffer);
    stage.msl_dispatch_desc.layout = static_cast<uint32_t>(GfxMpsrtLayout::Linear);
    stage.msl_dispatch_desc.threads_per_threadgroup = 64;
    stage.msl_dispatch_desc.input_count = 2;
    stage.msl_dispatch_desc.output_count = 1;
    stage.msl_dispatch_desc.flags = GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired;
    stage.inputs = {0, 1};
    stage.outputs = {2};
    stage.kernel_buffer_order = {0, 1, 2};
    stage.output_descs = {output_abi};
    model.stages.push_back(stage);

    float bias = 3.5f;
    auto kernel = std::make_shared<MetalCompiledKernel>((MetalDeviceHandle)device, (void*)pipeline, 2);
    kernel->set_mpsrt_model(std::make_shared<runtime_mpsrt::MpsrtModel>(model));
    ASSERT_TRUE(kernel->register_mpsrt_const_tensor_data(1,
                                                         bias_abi,
                                                         &bias,
                                                         sizeof(bias),
                                                         &log))
        << log;

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

    MetalBuffer input_buf{};
    input_buf.buffer = (__bridge void*)input;
    input_buf.size = sizeof(float) * kCount;
    input_buf.type = ov::element::f32;
    MetalBuffer output_buf{};
    output_buf.buffer = (__bridge void*)output;
    output_buf.size = sizeof(float) * kCount;
    output_buf.type = ov::element::f32;

    std::vector<KernelArg> args = {
        make_buffer_arg(0, input_buf),
        make_buffer_arg(1, output_buf),
    };
    KernelDispatch dispatch;
    dispatch.grid[0] = kCount;
    dispatch.threads_per_group[0] = kernel->clamp_threadgroup_size(64);

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
    ASSERT_EQ([cmd status], MTLCommandBufferStatusCompleted);

    EXPECT_EQ(counters["mpsrt_model_request_encode_count"], 1u);
    EXPECT_EQ(counters["mpsrt_model_request_msl_stage_encode_count"], 1u);
    EXPECT_EQ(counters["mpsrt_binding_external_input_count"], 1u);
    EXPECT_EQ(counters["mpsrt_binding_external_output_count"], 1u);
    EXPECT_EQ(counters["mpsrt_binding_model_resource_count"], 1u);
    for (uint32_t i = 0; i < kCount; ++i) {
        EXPECT_FLOAT_EQ(output_ptr[i], static_cast<float>(i) + bias);
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
    materialize_test_mpsrt_stage_module(module, {input_desc}, {output_desc});

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

TEST(GfxBackendTest, AnnotatedGatherElementsMslKernelExecutesThroughRolePatternMpsrtBufferOrder) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    const char* source = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct GatherParams {
  uint count;
  uint base;
  uint reserved0;
  uint reserved1;
};
kernel void gather_elements_kernel(device const float* data [[buffer(0)]],
                                   device const uint* indices [[buffer(1)]],
                                   device float* output [[buffer(2)]],
                                   constant GatherParams& p [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  output[gid] = data[indices[gid]] + float(p.base);
}
)MSL";

    mlir::MLIRContext ctx;
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Metal,
                                                     "GatherElements",
                                                     nullptr,
                                                     ov::element::f32,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     {});
    annotate_msl_module_with_stage_plan(module, plan, "GatherElements");

    constexpr uint32_t kCount = 8;
    const auto data_desc = gfx_mpsrt_make_tensor_desc({kCount},
                                                      ov::element::f32,
                                                      GfxStageStorageKind::Buffer,
                                                      GfxMpsrtTensorFlagExternalIo);
    const auto indices_desc = gfx_mpsrt_make_tensor_desc({kCount},
                                                         ov::element::u32,
                                                         GfxStageStorageKind::Buffer,
                                                         GfxMpsrtTensorFlagExternalIo);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({kCount},
                                                        ov::element::f32,
                                                        GfxStageStorageKind::Buffer,
                                                        GfxMpsrtTensorFlagTransient);
    materialize_test_mpsrt_stage_module(module, {data_desc, indices_desc}, {output_desc});

    KernelSource ks;
    ks.module = module;
    ks.entry_point = "gather_elements_kernel";
    ks.msl_source = source;
    ks.signature.arg_count = 4;
    configure_msl_kernel_source_for_plan(ks, "GatherElements");
    ASSERT_EQ(ks.entry_point, "gather_scatter_indexed");
    ASSERT_NE(ks.msl_source.find("kernel void gather_scatter_indexed"), std::string::npos);
    ASSERT_EQ(ks.msl_source.find("kernel void gather_elements_kernel"), std::string::npos);

    MetalCodegenBackend backend((MetalDeviceHandle)device);
    std::string log;
    auto kernel = backend.compile(ks, &log);
    ASSERT_TRUE(kernel) << log;

    auto* metal_kernel = dynamic_cast<MetalCompiledKernel*>(kernel.get());
    ASSERT_NE(metal_kernel, nullptr);
    const auto* mpsrt_model = metal_kernel->mpsrt_model();
    ASSERT_NE(mpsrt_model, nullptr);
    ASSERT_EQ(mpsrt_model->stages.size(), 1u);
    EXPECT_EQ(mpsrt_model->semantic_input_values, std::vector<GfxMpsrtValue>({0u, 1u}));
    EXPECT_EQ(mpsrt_model->semantic_output_values, std::vector<GfxMpsrtValue>({2u}));
    EXPECT_EQ(mpsrt_model->input_values, std::vector<GfxMpsrtValue>({0u, 1u, 3u}));
    EXPECT_EQ(mpsrt_model->output_values, std::vector<GfxMpsrtValue>({2u}));
    EXPECT_EQ(mpsrt_model->external_values, std::vector<GfxMpsrtValue>({0u, 1u, 2u, 3u}));
    EXPECT_EQ(mpsrt_model->external_input_values, std::vector<GfxMpsrtValue>({0u, 1u, 3u}));
    EXPECT_EQ(mpsrt_model->external_output_values, std::vector<GfxMpsrtValue>({2u}));
    EXPECT_EQ(mpsrt_model->external_buffer_roles,
              std::vector<GfxMpsrtExternalBufferRole>({GfxMpsrtExternalBufferRole::TensorInput,
                                                       GfxMpsrtExternalBufferRole::TensorInput,
                                                       GfxMpsrtExternalBufferRole::TensorOutput,
                                                       GfxMpsrtExternalBufferRole::RuntimeParams}));
    EXPECT_EQ(mpsrt_model->stages.front().kernel_buffer_order,
              std::vector<GfxMpsrtValue>({0u, 1u, 2u, 3u}));
    EXPECT_EQ(mpsrt_model->stages.front().msl_dispatch_desc.input_count, 3u);
    EXPECT_EQ(mpsrt_model->stages.front().msl_dispatch_desc.output_count, 1u);

    id<MTLBuffer> data = [device newBufferWithLength:sizeof(float) * kCount
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> indices = [device newBufferWithLength:sizeof(uint32_t) * kCount
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:sizeof(float) * kCount
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> params = [device newBufferWithLength:sizeof(uint32_t) * 4
                                             options:MTLResourceStorageModeShared];
    ASSERT_NE(data, nil);
    ASSERT_NE(indices, nil);
    ASSERT_NE(output, nil);
    ASSERT_NE(params, nil);

    float* data_ptr = static_cast<float*>([data contents]);
    uint32_t* indices_ptr = static_cast<uint32_t*>([indices contents]);
    float* output_ptr = static_cast<float*>([output contents]);
    uint32_t* params_ptr = static_cast<uint32_t*>([params contents]);
    ASSERT_NE(data_ptr, nullptr);
    ASSERT_NE(indices_ptr, nullptr);
    ASSERT_NE(output_ptr, nullptr);
    ASSERT_NE(params_ptr, nullptr);

    params_ptr[0] = kCount;
    params_ptr[1] = 10;
    params_ptr[2] = 0;
    params_ptr[3] = 0;
    for (uint32_t i = 0; i < kCount; ++i) {
        data_ptr[i] = static_cast<float>(i) * 3.0f;
        indices_ptr[i] = kCount - 1 - i;
        output_ptr[i] = -1.0f;
    }

    MetalBuffer data_buf{};
    data_buf.buffer = (__bridge void*)data;
    data_buf.size = sizeof(float) * kCount;
    data_buf.type = ov::element::f32;
    MetalBuffer indices_buf{};
    indices_buf.buffer = (__bridge void*)indices;
    indices_buf.size = sizeof(uint32_t) * kCount;
    indices_buf.type = ov::element::u32;
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
        make_buffer_arg(0, data_buf),
        make_buffer_arg(1, indices_buf),
        make_buffer_arg(2, output_buf),
        make_buffer_arg(3, params_buf),
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
    EXPECT_EQ(counters["mpsrt_model_request_msl_stage_encode_count"], 1u);
    EXPECT_EQ(counters["mpsrt_msl_request_encode_count"], 1u);
    EXPECT_EQ(counters["mpsrt_binding_external_input_count"], 3u);
    EXPECT_EQ(counters["mpsrt_binding_external_output_count"], 1u);
    EXPECT_EQ(counters["mpsrt_binding_prepared_transient_buffer_count"], 0u);
    for (uint32_t i = 0; i < kCount; ++i) {
        EXPECT_FLOAT_EQ(output_ptr[i], data_ptr[indices_ptr[i]] + 10.0f);
    }
}

TEST(GfxBackendTest, AnnotatedSliceMslKernelExecutesThroughLeadingIoRuntimeParamsMpsrtBufferOrder) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    const char* source = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void slice_kernel(device const float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         constant uint& start [[buffer(2)]],
                         constant uint& step [[buffer(3)]],
                         constant uint& count [[buffer(4)]],
                         constant uint& bias [[buffer(5)]],
                         constant uint& unused0 [[buffer(6)]],
                         constant uint& unused1 [[buffer(7)]],
                         uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  (void)unused0;
  (void)unused1;
  output[gid] = input[start + gid * step] + float(bias);
}
)MSL";

    mlir::MLIRContext ctx;
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    const auto plan = select_stage_optimization_plan(nullptr,
                                                     GpuBackend::Metal,
                                                     "Slice",
                                                     nullptr,
                                                     ov::element::f32,
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false,
                                                     {});
    annotate_msl_module_with_stage_plan(module, plan, "Slice");

    constexpr uint32_t kInputCount = 16;
    constexpr uint32_t kOutputCount = 6;
    const auto input_desc = gfx_mpsrt_make_tensor_desc({kInputCount},
                                                       ov::element::f32,
                                                       GfxStageStorageKind::Buffer,
                                                       GfxMpsrtTensorFlagExternalIo);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({kOutputCount},
                                                        ov::element::f32,
                                                        GfxStageStorageKind::Buffer,
                                                        GfxMpsrtTensorFlagTransient);
    materialize_test_mpsrt_stage_module(module, {input_desc}, {output_desc});

    KernelSource ks;
    ks.module = module;
    ks.entry_point = "slice_kernel";
    ks.msl_source = source;
    ks.signature.arg_count = 8;
    configure_msl_kernel_source_for_plan(ks, "Slice");
    ASSERT_EQ(ks.entry_point, "gather_scatter_indexed");
    ASSERT_NE(ks.msl_source.find("kernel void gather_scatter_indexed"), std::string::npos);
    ASSERT_EQ(ks.msl_source.find("kernel void slice_kernel"), std::string::npos);

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
    EXPECT_EQ(mpsrt_model->input_values, std::vector<GfxMpsrtValue>({0u, 2u, 3u, 4u, 5u, 6u, 7u}));
    EXPECT_EQ(mpsrt_model->output_values, std::vector<GfxMpsrtValue>({1u}));
    EXPECT_EQ(mpsrt_model->external_values, std::vector<GfxMpsrtValue>({0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u}));
    EXPECT_EQ(mpsrt_model->external_input_values, std::vector<GfxMpsrtValue>({0u, 2u, 3u, 4u, 5u, 6u, 7u}));
    EXPECT_EQ(mpsrt_model->external_output_values, std::vector<GfxMpsrtValue>({1u}));
    ASSERT_EQ(mpsrt_model->external_buffer_roles.size(), 8u);
    EXPECT_EQ(mpsrt_model->external_buffer_roles[0], GfxMpsrtExternalBufferRole::TensorInput);
    EXPECT_EQ(mpsrt_model->external_buffer_roles[1], GfxMpsrtExternalBufferRole::TensorOutput);
    for (size_t i = 2; i < mpsrt_model->external_buffer_roles.size(); ++i) {
        EXPECT_EQ(mpsrt_model->external_buffer_roles[i], GfxMpsrtExternalBufferRole::RuntimeParams);
    }
    EXPECT_EQ(mpsrt_model->stages.front().kernel_buffer_order,
              std::vector<GfxMpsrtValue>({0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u}));
    EXPECT_EQ(mpsrt_model->stages.front().msl_dispatch_desc.input_count, 7u);
    EXPECT_EQ(mpsrt_model->stages.front().msl_dispatch_desc.output_count, 1u);

    id<MTLBuffer> input = [device newBufferWithLength:sizeof(float) * kInputCount
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:sizeof(float) * kOutputCount
                                              options:MTLResourceStorageModeShared];
    ASSERT_NE(input, nil);
    ASSERT_NE(output, nil);
    float* input_ptr = static_cast<float*>([input contents]);
    float* output_ptr = static_cast<float*>([output contents]);
    ASSERT_NE(input_ptr, nullptr);
    ASSERT_NE(output_ptr, nullptr);
    for (uint32_t i = 0; i < kInputCount; ++i) {
        input_ptr[i] = static_cast<float>(i) * 2.0f;
    }
    for (uint32_t i = 0; i < kOutputCount; ++i) {
        output_ptr[i] = -1.0f;
    }

    const uint32_t param_values[] = {
        2u,
        2u,
        kOutputCount,
        4u,
        0u,
        0u,
    };
    std::vector<id<MTLBuffer>> param_buffers;
    param_buffers.reserve(6);
    for (const uint32_t value : param_values) {
        id<MTLBuffer> buffer = [device newBufferWithLength:sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];
        ASSERT_NE(buffer, nil);
        uint32_t* ptr = static_cast<uint32_t*>([buffer contents]);
        ASSERT_NE(ptr, nullptr);
        *ptr = value;
        param_buffers.push_back(buffer);
    }

    MetalBuffer input_buf{};
    input_buf.buffer = (__bridge void*)input;
    input_buf.size = sizeof(float) * kInputCount;
    input_buf.type = ov::element::f32;
    MetalBuffer output_buf{};
    output_buf.buffer = (__bridge void*)output;
    output_buf.size = sizeof(float) * kOutputCount;
    output_buf.type = ov::element::f32;

    std::vector<KernelArg> args;
    args.reserve(8);
    args.push_back(make_buffer_arg(0, input_buf));
    args.push_back(make_buffer_arg(1, output_buf));
    for (uint32_t i = 0; i < param_buffers.size(); ++i) {
        MetalBuffer param_buf{};
        param_buf.buffer = (__bridge void*)param_buffers[i];
        param_buf.size = sizeof(uint32_t);
        param_buf.type = ov::element::u32;
        args.push_back(make_buffer_arg(i + 2, param_buf));
    }

    KernelDispatch dispatch;
    dispatch.grid[0] = kOutputCount;
    dispatch.threads_per_group[0] = kernel->clamp_threadgroup_size(64);

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
    EXPECT_EQ(counters["mpsrt_binding_external_input_count"], 7u);
    EXPECT_EQ(counters["mpsrt_binding_external_output_count"], 1u);
    EXPECT_EQ(counters["mpsrt_binding_prepared_transient_buffer_count"], 0u);
    for (uint32_t i = 0; i < kOutputCount; ++i) {
        EXPECT_FLOAT_EQ(output_ptr[i], input_ptr[2u + i * 2u] + 4.0f);
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
        runtime_mpsrt::MpsrtRuntimeStage stage;
        stage.kind = GfxMpsrtStageKind::MSLDispatch;
        stage.stage_record_key = key;
        stage.kernel_name = entry;
        stage.dispatch_kernel_family = "eltwise_fused_buffer";
        stage.dispatch_entry_point = entry;
        stage.dispatch_kernel_family_id = static_cast<uint32_t>(GfxKernelFamily::EltwiseFusedBuffer);
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
        stage.kernel_buffer_order = {input, output};
        (void)index;
        return stage;
    };

    runtime_mpsrt::MpsrtModel model;
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
    ASSERT_TRUE(runtime_mpsrt::finalize_mpsrt_model_resources(model, &log)) << log;
    ASSERT_TRUE(context.prepare_model(model, source, prepared_model, &log)) << log;
    ASSERT_EQ(prepared_model.msl_dispatches.size(), 2u);
    EXPECT_NE(prepared_model.resource_heap, nil);
    EXPECT_GT(prepared_model.resource_heap_size, 0u);
    EXPECT_EQ(prepared_model.resource_heap_size, prepared_model.resource_heap_unaliased_size);
    EXPECT_EQ(prepared_model.resource_heap_aliasable_size, prepared_model.resource_heap_unaliased_size);
    EXPECT_EQ(prepared_model.transient_buffer_resource_count, 1u);
    EXPECT_EQ(prepared_model.transient_image_resource_count, 0u);

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
    metal::mpsrt::MpsrtBindingBuildResult binding_result;
    ASSERT_TRUE(metal::mpsrt::build_mpsrt_tensor_bindings(model,
                                                          {{(__bridge void*)input, 0}},
                                                          {{(__bridge void*)output, 0}},
                                                          bindings,
                                                          &binding_result,
                                                          &log,
                                                          &prepared_model))
        << log;
    EXPECT_EQ(binding_result.external_inputs_bound, 1u);
    EXPECT_EQ(binding_result.external_outputs_bound, 1u);
    EXPECT_EQ(binding_result.transient_buffers_allocated, 1u);
    ASSERT_NE(bindings.lookup(1), nullptr);
    EXPECT_NE(bindings.lookup(1)->buffer, nullptr);

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

TEST(GfxBackendTest, MpsrtRequestRejectsMslDispatchWithoutManifestBufferOrder) {
    runtime_mpsrt::MpsrtRuntimeStage stage;
    stage.kind = GfxMpsrtStageKind::MSLDispatch;
    stage.stage_record_key = "missing_buffer_order";
    stage.kernel_name = "eltwise_fused_buffer";
    stage.dispatch_entry_point = "eltwise_fused_buffer";
    stage.msl_dispatch_desc.input_count = 1;
    stage.msl_dispatch_desc.output_count = 1;
    stage.inputs = {0};
    stage.outputs = {1};

    int input = 0;
    int output = 0;
    metal::mpsrt::MpsrtTensorBindings bindings;
    bindings.bind(0, metal::mpsrt::MpsrtBoundBuffer{&input, 0});
    bindings.bind(1, metal::mpsrt::MpsrtBoundBuffer{&output, 0});

    metal::mpsrt::MpsrtRequest request;
    std::vector<metal::mpsrt::MpsrtBoundBuffer> buffers;
    std::string error;
    EXPECT_FALSE(request.build_msl_stage_buffers(stage, bindings, buffers, &error));
    EXPECT_NE(error.find("kernel buffer order is not materialized"), std::string::npos);
    EXPECT_TRUE(buffers.empty());
}

TEST(GfxBackendTest, MpsrtTensorBindingsAcceptImageExternalAndTransientResources) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    runtime_mpsrt::MpsrtModel model;
    model.stage_record_key = "image_binding_smoke";
    model.input_values = {0};
    model.output_values = {2};
    const auto input_desc = gfx_mpsrt_make_tensor_desc({1, 4, 8, 8},
                                                       ov::element::f16,
                                                       GfxStageStorageKind::Image,
                                                       GfxMpsrtTensorFlagExternalIo);
    const auto temp_desc = gfx_mpsrt_make_tensor_desc({1, 4, 8, 8},
                                                      ov::element::f16,
                                                      GfxStageStorageKind::Image,
                                                      GfxMpsrtTensorFlagTransient);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({1, 4, 8, 8},
                                                        ov::element::f16,
                                                        GfxStageStorageKind::Image,
                                                        GfxMpsrtTensorFlagExternalIo);
    model.tensors.push_back({0, gfx_mpsrt_to_abi_desc(input_desc)});
    model.tensors.push_back({1, gfx_mpsrt_to_abi_desc(temp_desc)});
    model.tensors.push_back({2, gfx_mpsrt_to_abi_desc(output_desc)});

    auto make_texture = [&]() {
        MTLTextureDescriptor* texture_desc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                              width:8
                                                             height:8
                                                          mipmapped:false];
        texture_desc.textureType = MTLTextureType2D;
        texture_desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        texture_desc.storageMode = MTLStorageModePrivate;
        return [device newTextureWithDescriptor:texture_desc];
    };

    id<MTLTexture> input_texture = make_texture();
    id<MTLTexture> output_texture = make_texture();
    ASSERT_NE(input_texture, nil);
    ASSERT_NE(output_texture, nil);

    metal::mpsrt::MpsrtTensorBindings bindings;
    metal::mpsrt::MpsrtBindingBuildResult binding_result;
    std::string log;
    ASSERT_TRUE(runtime_mpsrt::finalize_mpsrt_model_resources(model, &log)) << log;
    metal::mpsrt::MpsrtContext context(device);
    metal::mpsrt::MpsrtPreparedModel prepared_model;
    ASSERT_TRUE(context.prepare_model_resources(model, prepared_model, &log)) << log;
    EXPECT_NE(prepared_model.resource_heap, nil);
    EXPECT_GT(prepared_model.resource_heap_size, 0u);
    EXPECT_EQ(prepared_model.resource_heap_size, prepared_model.resource_heap_unaliased_size);
    EXPECT_EQ(prepared_model.resource_heap_aliasable_size, prepared_model.resource_heap_unaliased_size);
    EXPECT_EQ(prepared_model.transient_buffer_resource_count, 0u);
    EXPECT_EQ(prepared_model.transient_image_resource_count, 1u);
    ASSERT_TRUE(metal::mpsrt::build_mpsrt_tensor_bindings(
                    model,
                    {metal::mpsrt::make_mpsrt_bound_image((__bridge void*)input_texture)},
                    {metal::mpsrt::make_mpsrt_bound_image((__bridge void*)output_texture)},
                    bindings,
                    &binding_result,
                    &log,
                    &prepared_model))
        << log;

    EXPECT_EQ(binding_result.external_inputs_bound, 1u);
    EXPECT_EQ(binding_result.external_outputs_bound, 1u);
    EXPECT_EQ(binding_result.transient_buffers_allocated, 0u);
    EXPECT_EQ(binding_result.transient_images_allocated, 1u);
    ASSERT_NE(bindings.lookup(0), nullptr);
    ASSERT_NE(bindings.lookup(1), nullptr);
    ASSERT_NE(bindings.lookup(2), nullptr);
    EXPECT_NE(bindings.lookup(0)->texture, nullptr);
    EXPECT_NE(bindings.lookup(1)->texture, nullptr);
    EXPECT_NE(bindings.lookup(2)->texture, nullptr);
}

TEST(GfxBackendTest, MpsrtPrepareModelAllocatesImageBridgeScratchTextureFromResourceHeap) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    runtime_mpsrt::MpsrtModel model;
    model.stage_record_key = "image_bridge_prepared_scratch";
    model.input_values = {0};
    model.output_values = {1};
    model.external_values = {0, 1};
    model.external_input_values = {0};
    model.external_output_values = {1};
    model.external_buffer_roles = {GfxMpsrtExternalBufferRole::TensorInput,
                                   GfxMpsrtExternalBufferRole::TensorOutput};

    const auto image_input_desc = gfx_mpsrt_make_tensor_desc({1, 4, 8, 8},
                                                             ov::element::f16,
                                                             GfxStageStorageKind::Image,
                                                             GfxMpsrtTensorFlagExternalIo);
    const auto buffer_output_desc = gfx_mpsrt_make_tensor_desc({1, 4, 8, 8},
                                                               ov::element::f16,
                                                               GfxStageStorageKind::Buffer,
                                                               GfxMpsrtTensorFlagExternalIo);
    const auto image_input_abi = gfx_mpsrt_to_abi_desc(image_input_desc);
    const auto buffer_output_abi = gfx_mpsrt_to_abi_desc(buffer_output_desc);
    model.tensors.push_back({0, image_input_abi});
    model.tensors.push_back({1, buffer_output_abi});
    model.resources = {
        {0u, GfxMpsrtExternalBufferRole::TensorInput, runtime_mpsrt::MpsrtRuntimeResourceLifetime::External, 0u, true, 0u, image_input_abi},
        {1u, GfxMpsrtExternalBufferRole::TensorOutput, runtime_mpsrt::MpsrtRuntimeResourceLifetime::External, 1u, true, 1u, buffer_output_abi},
    };
    model.external_buffer_bindings = {
        {0u, 0u},
        {1u, 1u},
    };
    GfxMpsrtStorageBridgeDesc bridge{};
    ASSERT_TRUE(gfx_mpsrt_make_image_bridge_desc(0,
                                                 image_input_abi,
                                                 GfxMpsrtStorageBridgeDirection::BufferToImage,
                                                 bridge));
    model.storage_bridges.push_back(bridge);

    metal::mpsrt::MpsrtContext context(device);
    metal::mpsrt::MpsrtPreparedModel prepared_model;
    std::string log;
    ASSERT_TRUE(context.prepare_model_resources(model, prepared_model, &log)) << log;

    EXPECT_NE(prepared_model.resource_heap, nil);
    EXPECT_GT(prepared_model.resource_heap_size, 0u);
    EXPECT_EQ(prepared_model.resource_heap_size, prepared_model.resource_heap_unaliased_size);
    EXPECT_EQ(prepared_model.resource_heap_aliasable_size, prepared_model.resource_heap_unaliased_size);
    EXPECT_EQ(prepared_model.transient_buffer_resource_count, 0u);
    EXPECT_EQ(prepared_model.transient_image_resource_count, 0u);
    ASSERT_EQ(prepared_model.image_bridge_resource_count, 1u);
    ASSERT_EQ(prepared_model.image_bridge_resources.size(), 1u);
    EXPECT_EQ(prepared_model.image_bridge_resources[0].value, 0u);
    EXPECT_EQ(prepared_model.image_bridge_resources[0].direction,
              GfxMpsrtStorageBridgeDirection::BufferToImage);
    EXPECT_GT(prepared_model.image_bridge_resources[0].heap_allocation_size, 0u);
    EXPECT_GT(prepared_model.image_bridge_resources[0].heap_alignment, 0u);
    EXPECT_NE(prepared_model.image_bridge_resources[0].texture, nil);
}

TEST(GfxBackendTest, MpsrtTensorBindingsRejectResourceLessConstTensors) {
    runtime_mpsrt::MpsrtModel model;
    model.stage_record_key = "resource_less_const_rejected";
    model.input_values = {0};
    model.output_values = {2};

    const auto input_desc = gfx_mpsrt_make_tensor_desc({4},
                                                       ov::element::f32,
                                                       GfxStageStorageKind::Buffer,
                                                       GfxMpsrtTensorFlagExternalIo);
    const auto const_desc = gfx_mpsrt_make_tensor_desc({1},
                                                       ov::element::f32,
                                                       GfxStageStorageKind::Buffer,
                                                       GfxMpsrtTensorFlagConst);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({4},
                                                        ov::element::f32,
                                                        GfxStageStorageKind::Buffer,
                                                        GfxMpsrtTensorFlagExternalIo);
    model.tensors.push_back({0, gfx_mpsrt_to_abi_desc(input_desc)});
    model.tensors.push_back({1, gfx_mpsrt_to_abi_desc(const_desc)});
    model.tensors.push_back({2, gfx_mpsrt_to_abi_desc(output_desc)});

    int input = 0;
    int output = 0;
    metal::mpsrt::MpsrtTensorBindings bindings;
    metal::mpsrt::MpsrtBindingBuildResult binding_result;
    std::string error;
    EXPECT_FALSE(metal::mpsrt::build_mpsrt_tensor_bindings(
        model,
        {metal::mpsrt::MpsrtBoundBuffer{&input, 0}},
        {metal::mpsrt::MpsrtBoundBuffer{&output, 0}},
        bindings,
        &binding_result,
        &error));
    EXPECT_NE(error.find("runtime resource table is required"), std::string::npos)
        << error;

    error.clear();
    EXPECT_FALSE(metal::mpsrt::build_mpsrt_external_tensor_bindings(
        model,
        {metal::mpsrt::MpsrtBoundBuffer{&input, 0},
         metal::mpsrt::MpsrtBoundBuffer{&output, 0}},
        bindings,
        &binding_result,
        &error));
    EXPECT_NE(error.find("runtime resource table is required"), std::string::npos)
        << error;
}

TEST(GfxBackendTest, MpsrtExternalTensorBindingsPreserveNonTensorResourceAbiEntries) {
    runtime_mpsrt::MpsrtModel model;
    model.stage_record_key = "external_resource_abi_smoke";
    model.input_values = {0};
    model.output_values = {1};
    model.external_values = {0, 1};
    model.external_input_values = {0};
    model.external_output_values = {1};
    model.external_buffer_roles = {GfxMpsrtExternalBufferRole::TensorInput,
                                   GfxMpsrtExternalBufferRole::RuntimeParams,
                                   GfxMpsrtExternalBufferRole::TensorOutput};
    model.resources = {
        {0u, GfxMpsrtExternalBufferRole::TensorInput, runtime_mpsrt::MpsrtRuntimeResourceLifetime::External, 0u, true, 0u},
        {1u, GfxMpsrtExternalBufferRole::RuntimeParams, runtime_mpsrt::MpsrtRuntimeResourceLifetime::External, 1u, false, 0u},
        {2u, GfxMpsrtExternalBufferRole::TensorOutput, runtime_mpsrt::MpsrtRuntimeResourceLifetime::External, 2u, true, 1u},
        {3u, GfxMpsrtExternalBufferRole::Unknown, runtime_mpsrt::MpsrtRuntimeResourceLifetime::Transient, 0u, true, 2u},
    };
    model.external_buffer_bindings = {
        {0u, 0u},
        {1u, 1u},
        {2u, 2u},
    };

    const auto input_desc = gfx_mpsrt_make_tensor_desc({4},
                                                       ov::element::f32,
                                                       GfxStageStorageKind::Buffer,
                                                       GfxMpsrtTensorFlagExternalIo);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({4},
                                                        ov::element::f32,
                                                        GfxStageStorageKind::Buffer,
                                                        GfxMpsrtTensorFlagExternalIo);
    const auto transient_desc = gfx_mpsrt_make_tensor_desc({4},
                                                           ov::element::f32,
                                                           GfxStageStorageKind::Buffer,
                                                           0);
    model.tensors.push_back({0, gfx_mpsrt_to_abi_desc(input_desc)});
    model.tensors.push_back({1, gfx_mpsrt_to_abi_desc(output_desc)});
    model.tensors.push_back({2, gfx_mpsrt_to_abi_desc(transient_desc)});
    model.resources[0].tensor_desc = model.tensors[0].desc;
    model.resources[2].tensor_desc = model.tensors[1].desc;
    model.resources[3].tensor_desc = model.tensors[2].desc;

    int input = 0;
    int runtime_params = 0;
    int output = 0;
    metal::mpsrt::MpsrtTensorBindings bindings;
    metal::mpsrt::MpsrtBindingBuildResult binding_result;
    std::string error;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);
    metal::mpsrt::MpsrtContext context(device);
    metal::mpsrt::MpsrtPreparedModel prepared_model;
    ASSERT_TRUE(context.prepare_model_resources(model, prepared_model, &error)) << error;
    EXPECT_NE(prepared_model.resource_heap, nil);
    EXPECT_GT(prepared_model.resource_heap_size, 0u);
    EXPECT_EQ(prepared_model.resource_heap_size, prepared_model.resource_heap_unaliased_size);
    EXPECT_EQ(prepared_model.resource_heap_aliasable_size, prepared_model.resource_heap_unaliased_size);
    EXPECT_EQ(prepared_model.transient_buffer_resource_count, 1u);
    EXPECT_EQ(prepared_model.transient_image_resource_count, 0u);
    ASSERT_TRUE(metal::mpsrt::build_mpsrt_external_tensor_bindings(
                    model,
                    {metal::mpsrt::MpsrtBoundBuffer{&input, 0},
                     metal::mpsrt::MpsrtBoundBuffer{&runtime_params, 0},
                     metal::mpsrt::MpsrtBoundBuffer{&output, 0}},
                    bindings,
                    &binding_result,
                    &error,
                    &prepared_model))
        << error;

    EXPECT_EQ(binding_result.external_inputs_bound, 1u);
    EXPECT_EQ(binding_result.external_outputs_bound, 1u);
    EXPECT_EQ(binding_result.external_resources_bound, 1u);
    EXPECT_EQ(binding_result.transient_buffers_allocated, 1u);
    ASSERT_NE(bindings.lookup(0), nullptr);
    ASSERT_NE(bindings.lookup(1), nullptr);
    ASSERT_NE(bindings.lookup(2), nullptr);
    EXPECT_EQ(bindings.lookup(0)->buffer, &input);
    EXPECT_EQ(bindings.lookup(1)->buffer, &output);
    EXPECT_NE(bindings.lookup(2)->buffer, nullptr);
    EXPECT_EQ(bindings.size(), 3u);
}

TEST(GfxBackendTest, MpsrtPrepareModelPlansTransientResourceLiveWindowsForHeapAliasing) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    runtime_mpsrt::MpsrtModel model;
    model.stage_record_key = "transient_live_window_plan";
    model.input_values = {0};
    model.output_values = {2, 4};

    const auto external_desc = gfx_mpsrt_make_tensor_desc({16},
                                                          ov::element::f32,
                                                          GfxStageStorageKind::Buffer,
                                                          GfxMpsrtTensorFlagExternalIo);
    const auto transient_desc = gfx_mpsrt_make_tensor_desc({16},
                                                           ov::element::f32,
                                                           GfxStageStorageKind::Buffer,
                                                           GfxMpsrtTensorFlagTransient);
    model.tensors.push_back({0, gfx_mpsrt_to_abi_desc(external_desc)});
    model.tensors.push_back({1, gfx_mpsrt_to_abi_desc(transient_desc)});
    model.tensors.push_back({2, gfx_mpsrt_to_abi_desc(external_desc)});
    model.tensors.push_back({3, gfx_mpsrt_to_abi_desc(transient_desc)});
    model.tensors.push_back({4, gfx_mpsrt_to_abi_desc(external_desc)});

    auto make_stage = [](GfxMpsrtValue input, GfxMpsrtValue output) {
        runtime_mpsrt::MpsrtRuntimeStage stage;
        stage.kind = GfxMpsrtStageKind::MSLDispatch;
        stage.inputs = {input};
        stage.outputs = {output};
        stage.kernel_buffer_order = {input, output};
        return stage;
    };
    model.stages.push_back(make_stage(0, 1));
    model.stages.push_back(make_stage(1, 2));
    model.stages.push_back(make_stage(0, 3));
    model.stages.push_back(make_stage(3, 4));

    std::string log;
    ASSERT_TRUE(runtime_mpsrt::finalize_mpsrt_model_resources(model, &log)) << log;

    metal::mpsrt::MpsrtContext context(device);
    metal::mpsrt::MpsrtPreparedModel prepared_model;
    ASSERT_TRUE(context.prepare_model_resources(model, prepared_model, &log)) << log;

    EXPECT_NE(prepared_model.resource_heap, nil);
    EXPECT_GT(prepared_model.resource_heap_size, 0u);
    EXPECT_EQ(prepared_model.resource_heap_size, prepared_model.resource_heap_aliasable_size);
    EXPECT_LT(prepared_model.resource_heap_size, prepared_model.resource_heap_unaliased_size);
    EXPECT_LT(prepared_model.resource_heap_aliasable_size, prepared_model.resource_heap_unaliased_size);
    EXPECT_GT(prepared_model.resource_heap_alias_reuse_count, 0u);
    EXPECT_EQ(prepared_model.transient_buffer_resource_count, 2u);
    EXPECT_EQ(prepared_model.transient_image_resource_count, 0u);

    const auto find_prepared = [&](GfxMpsrtValue value) -> const metal::mpsrt::MpsrtPreparedResource* {
        for (const auto& resource : prepared_model.resources) {
            if (resource.has_tensor_value && resource.value == value) {
                return &resource;
            }
        }
        return nullptr;
    };
    const auto* first_transient = find_prepared(1);
    const auto* second_transient = find_prepared(3);
    ASSERT_NE(first_transient, nullptr);
    ASSERT_NE(second_transient, nullptr);
    EXPECT_EQ(first_transient->first_stage_index, 0u);
    EXPECT_EQ(first_transient->last_stage_index, 1u);
    EXPECT_EQ(second_transient->first_stage_index, 2u);
    EXPECT_EQ(second_transient->last_stage_index, 3u);
    EXPECT_GT(first_transient->heap_allocation_size, 0u);
    EXPECT_GT(second_transient->heap_allocation_size, 0u);
    EXPECT_GT(first_transient->heap_alignment, 0u);
    EXPECT_GT(second_transient->heap_alignment, 0u);
}

TEST(GfxBackendTest, MpsrtRequestEncodesPreparedMpsResize2DModelWithImageBindings) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    runtime_mpsrt::MpsrtModel model;
    model.stage_record_key =
        "mps_resize2d|apple_mps|image|image|nhwc4|Interpolate|apple_mps:image:Interpolate";
    model.semantic_input_values = {0};
    model.semantic_output_values = {1};
    model.input_values = {0};
    model.output_values = {1};
    model.external_input_values = {0};
    model.external_output_values = {1};
    model.external_values = {0, 1};
    model.external_buffer_roles = {GfxMpsrtExternalBufferRole::TensorInput,
                                   GfxMpsrtExternalBufferRole::TensorOutput};

    const auto input_desc = gfx_mpsrt_make_tensor_desc({1, 4, 8, 8},
                                                       ov::element::f16,
                                                       GfxStageStorageKind::Image,
                                                       GfxMpsrtTensorFlagExternalIo);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({1, 4, 16, 16},
                                                        ov::element::f16,
                                                        GfxStageStorageKind::Image,
                                                        GfxMpsrtTensorFlagExternalIo);
    const auto input_abi = gfx_mpsrt_to_abi_desc(input_desc);
    const auto output_abi = gfx_mpsrt_to_abi_desc(output_desc);
    model.tensors.push_back({0, input_abi});
    model.tensors.push_back({1, output_abi});

    runtime_mpsrt::MpsrtRuntimeStage stage;
    stage.kind = GfxMpsrtStageKind::MPSResize2D;
    stage.stage_record_key = model.stage_record_key;
    stage.kernel_name = "mps_resize2d";
    stage.resize2d_desc.nearest = 0;
    stage.resize2d_desc.align_corners = 0;
    stage.resize2d_desc.half_pixel_centers = 1;
    stage.inputs = {0};
    stage.outputs = {1};
    stage.output_descs = {output_abi};
    model.stages.push_back(stage);

    metal::mpsrt::MpsrtContext context(device);
    metal::mpsrt::MpsrtPreparedModel prepared_model;
    std::string log;
    ASSERT_TRUE(runtime_mpsrt::finalize_mpsrt_model_resources(model, &log)) << log;
    ASSERT_TRUE(context.prepare_model(model, "", prepared_model, &log)) << log;
    ASSERT_TRUE(prepared_model.msl_dispatches.empty());
    ASSERT_EQ(prepared_model.mps_resize2d_stages.size(), 1u);
    EXPECT_FALSE(prepared_model.mps_resize2d_stages.front().kernel_cache_hit);
    EXPECT_EQ(prepared_model.mps_resize2d_stages.front().input_width, 8u);
    EXPECT_EQ(prepared_model.mps_resize2d_stages.front().output_width, 16u);
    EXPECT_EQ(prepared_model.skipped_non_msl_stages, 0u);

    metal::mpsrt::MpsrtPreparedModel cached_prepared_model;
    ASSERT_TRUE(context.prepare_model(model, "", cached_prepared_model, &log)) << log;
    ASSERT_EQ(cached_prepared_model.mps_resize2d_stages.size(), 1u);
    EXPECT_TRUE(cached_prepared_model.mps_resize2d_stages.front().kernel_cache_hit);
    EXPECT_EQ(cached_prepared_model.mps_resize2d_stages.front().kernel,
              prepared_model.mps_resize2d_stages.front().kernel);

    auto make_texture = [&](uint32_t width, uint32_t height) {
        MTLTextureDescriptor* texture_desc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                              width:width
                                                             height:height
                                                          mipmapped:false];
        texture_desc.textureType = MTLTextureType2DArray;
        texture_desc.arrayLength = 1;
        texture_desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        texture_desc.storageMode = MTLStorageModePrivate;
        return [device newTextureWithDescriptor:texture_desc];
    };
    id<MTLTexture> input_texture = make_texture(8, 8);
    id<MTLTexture> output_texture = make_texture(16, 16);
    ASSERT_NE(input_texture, nil);
    ASSERT_NE(output_texture, nil);

    metal::mpsrt::MpsrtTensorBindings bindings;
    metal::mpsrt::MpsrtBindingBuildResult binding_result;
    ASSERT_TRUE(metal::mpsrt::build_mpsrt_tensor_bindings(
                    model,
                    {metal::mpsrt::make_mpsrt_bound_image((__bridge void*)input_texture)},
                    {metal::mpsrt::make_mpsrt_bound_image((__bridge void*)output_texture)},
                    bindings,
                    &binding_result,
                    &log,
                    &cached_prepared_model))
        << log;
    EXPECT_EQ(binding_result.external_inputs_bound, 1u);
    EXPECT_EQ(binding_result.external_outputs_bound, 1u);
    EXPECT_EQ(binding_result.transient_buffers_allocated, 0u);
    EXPECT_EQ(binding_result.transient_images_allocated, 0u);

    std::vector<KernelDispatch> stage_dispatches(1);
    std::unordered_map<std::string, uint64_t> counters;
    KernelExecutionHooks hooks;
    hooks.on_counter = [&counters](std::string_view name, uint64_t delta) {
        counters[std::string(name)] += delta;
    };

    id<MTLCommandBuffer> cmd = [context.command_queue() commandBuffer];
    ASSERT_NE(cmd, nil);
    metal::mpsrt::MpsrtRequest request;
    metal::mpsrt::MpsrtModelEncodeResult result;
    ASSERT_TRUE(request.encode_prepared_model((GpuCommandBufferHandle)cmd,
                                              model,
                                              cached_prepared_model,
                                              stage_dispatches,
                                              bindings,
                                              &hooks,
                                              &result,
                                              &log))
        << log;
    [cmd commit];
    [cmd waitUntilCompleted];
    ASSERT_EQ([cmd status], MTLCommandBufferStatusCompleted);

    EXPECT_EQ(result.encoded_msl_dispatches, 0u);
    EXPECT_EQ(result.encoded_mps_resize2d_stages, 1u);
    EXPECT_EQ(result.skipped_non_msl_stages, 0u);
    EXPECT_EQ(result.bound_buffers, 2u);
    EXPECT_EQ(counters["mpsrt_model_request_encode_count"], 1u);
    EXPECT_EQ(counters["mpsrt_model_request_mps_resize2d_stage_encode_count"], 1u);
    EXPECT_EQ(counters["mpsrt_mps_resize2d_request_encode_count"], 1u);
    EXPECT_EQ(counters["mpsrt_mps_resize2d_kernel_encode_count"], 1u);
}

TEST(GfxBackendTest, MpsrtRequestEncodesPreparedMpsGemmModelWithMatrixBindings) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    constexpr uint32_t kRows = 2;
    constexpr uint32_t kInner = 3;
    constexpr uint32_t kColumns = 2;

    runtime_mpsrt::MpsrtModel model;
    model.stage_record_key = "mps_gemm_model";
    model.semantic_input_values = {0, 1};
    model.semantic_output_values = {2};
    model.input_values = {0, 1};
    model.output_values = {2};
    model.external_input_values = {0, 1};
    model.external_output_values = {2};
    model.external_values = {0, 1, 2};
    model.external_buffer_roles = {GfxMpsrtExternalBufferRole::TensorInput,
                                   GfxMpsrtExternalBufferRole::TensorInput,
                                   GfxMpsrtExternalBufferRole::TensorOutput};

    const auto lhs_desc = gfx_mpsrt_make_tensor_desc({kRows, kInner},
                                                     ov::element::f32,
                                                     GfxStageStorageKind::Matrix,
                                                     GfxMpsrtTensorFlagExternalIo);
    const auto rhs_desc = gfx_mpsrt_make_tensor_desc({kInner, kColumns},
                                                     ov::element::f32,
                                                     GfxStageStorageKind::Matrix,
                                                     GfxMpsrtTensorFlagExternalIo);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({kRows, kColumns},
                                                        ov::element::f32,
                                                        GfxStageStorageKind::Matrix,
                                                        GfxMpsrtTensorFlagExternalIo);
    model.tensors.push_back({0, gfx_mpsrt_to_abi_desc(lhs_desc)});
    model.tensors.push_back({1, gfx_mpsrt_to_abi_desc(rhs_desc)});
    model.tensors.push_back({2, gfx_mpsrt_to_abi_desc(output_desc)});

    runtime_mpsrt::MpsrtRuntimeStage stage;
    stage.kind = GfxMpsrtStageKind::MPSGemm;
    stage.stage_record_key = "mps_gemm|apple_mps|matrix|matrix|row_major|MatMul|apple_mps:matrix:MatMul";
    stage.kernel_name = "mps_gemm";
    stage.gemm_desc.alpha = 1.0f;
    stage.gemm_desc.beta = 0.0f;
    stage.inputs = {0, 1};
    stage.outputs = {2};
    stage.output_descs = {gfx_mpsrt_to_abi_desc(output_desc)};
    model.stages.push_back(stage);

    metal::mpsrt::MpsrtContext context(device);
    metal::mpsrt::MpsrtPreparedModel prepared_model;
    std::string log;
    ASSERT_TRUE(runtime_mpsrt::finalize_mpsrt_model_resources(model, &log)) << log;
    ASSERT_TRUE(context.prepare_model(model, "", prepared_model, &log)) << log;
    ASSERT_EQ(prepared_model.msl_dispatches.size(), 0u);
    ASSERT_EQ(prepared_model.mps_gemm_stages.size(), 1u);
    EXPECT_FALSE(prepared_model.mps_gemm_stages.front().kernel_cache_hit);
    EXPECT_EQ(prepared_model.skipped_non_msl_stages, 0u);

    metal::mpsrt::MpsrtPreparedModel second_prepared_model;
    ASSERT_TRUE(context.prepare_model(model, "", second_prepared_model, &log)) << log;
    ASSERT_EQ(second_prepared_model.mps_gemm_stages.size(), 1u);
    EXPECT_TRUE(second_prepared_model.mps_gemm_stages.front().kernel_cache_hit);
    EXPECT_EQ(second_prepared_model.mps_gemm_stages.front().kernel,
              prepared_model.mps_gemm_stages.front().kernel);

    id<MTLBuffer> lhs = [device newBufferWithLength:sizeof(float) * kRows * kInner
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> rhs = [device newBufferWithLength:sizeof(float) * kInner * kColumns
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:sizeof(float) * kRows * kColumns
                                               options:MTLResourceStorageModeShared];
    ASSERT_NE(lhs, nil);
    ASSERT_NE(rhs, nil);
    ASSERT_NE(output, nil);

    float* lhs_ptr = static_cast<float*>([lhs contents]);
    float* rhs_ptr = static_cast<float*>([rhs contents]);
    float* output_ptr = static_cast<float*>([output contents]);
    ASSERT_NE(lhs_ptr, nullptr);
    ASSERT_NE(rhs_ptr, nullptr);
    ASSERT_NE(output_ptr, nullptr);
    const float lhs_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float rhs_values[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::memcpy(lhs_ptr, lhs_values, sizeof(lhs_values));
    std::memcpy(rhs_ptr, rhs_values, sizeof(rhs_values));
    for (uint32_t i = 0; i < kRows * kColumns; ++i) {
        output_ptr[i] = -1.0f;
    }

    metal::mpsrt::MpsrtTensorBindings bindings;
    metal::mpsrt::MpsrtBindingBuildResult binding_result;
    ASSERT_TRUE(metal::mpsrt::build_mpsrt_tensor_bindings(model,
                                                          {{(__bridge void*)lhs, 0}, {(__bridge void*)rhs, 0}},
                                                          {{(__bridge void*)output, 0}},
                                                          bindings,
                                                          &binding_result,
                                                          &log))
        << log;
    EXPECT_EQ(binding_result.external_inputs_bound, 2u);
    EXPECT_EQ(binding_result.external_outputs_bound, 1u);
    EXPECT_EQ(binding_result.transient_buffers_allocated, 0u);

    std::vector<KernelDispatch> stage_dispatches(1);
    std::unordered_map<std::string, uint64_t> counters;
    KernelExecutionHooks hooks;
    hooks.on_counter = [&counters](std::string_view name, uint64_t delta) {
        counters[std::string(name)] += delta;
    };

    id<MTLCommandBuffer> cmd = [context.command_queue() commandBuffer];
    ASSERT_NE(cmd, nil);
    metal::mpsrt::MpsrtRequest request;
    metal::mpsrt::MpsrtModelEncodeResult result;
    ASSERT_TRUE(request.encode_prepared_model((GpuCommandBufferHandle)cmd,
                                              model,
                                              second_prepared_model,
                                              stage_dispatches,
                                              bindings,
                                              &hooks,
                                              &result,
                                              &log))
        << log;
    metal_end_compute_encoder((GpuCommandBufferHandle)cmd);
    [cmd commit];
    [cmd waitUntilCompleted];
    ASSERT_EQ([cmd status], MTLCommandBufferStatusCompleted);

    EXPECT_EQ(result.encoded_msl_dispatches, 0u);
    EXPECT_EQ(result.encoded_mps_gemm_stages, 1u);
    EXPECT_EQ(result.skipped_non_msl_stages, 0u);
    EXPECT_EQ(result.bound_buffers, 3u);
    EXPECT_EQ(counters["mpsrt_model_request_encode_count"], 1u);
    EXPECT_EQ(counters["mpsrt_model_request_mps_gemm_stage_encode_count"], 1u);
    EXPECT_EQ(counters["mpsrt_mps_gemm_request_encode_count"], 1u);

    EXPECT_FLOAT_EQ(output_ptr[0], 58.0f);
    EXPECT_FLOAT_EQ(output_ptr[1], 64.0f);
    EXPECT_FLOAT_EQ(output_ptr[2], 139.0f);
    EXPECT_FLOAT_EQ(output_ptr[3], 154.0f);
}

TEST(GfxBackendTest, MpsrtPrepareModelRejectsMpsConv2DUntilConstPackOwnsWeights) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    runtime_mpsrt::MpsrtModel model;
    model.stage_record_key =
        "mps_conv2d|apple_mps|image|image|nhwc4|Convolution|apple_mps:image:Convolution|"
        "conv2d:g1:s1x1:d1x1:p1,1,1,1";
    model.semantic_input_values = {0};
    model.semantic_output_values = {2};
    model.input_values = {0};
    model.output_values = {2};
    model.external_input_values = {0};
    model.external_output_values = {2};
    model.external_values = {0, 2};
    model.external_buffer_roles = {GfxMpsrtExternalBufferRole::TensorInput,
                                   GfxMpsrtExternalBufferRole::TensorOutput};

    const auto input_desc = gfx_mpsrt_make_tensor_desc({1, 3, 32, 32},
                                                       ov::element::f16,
                                                       GfxStageStorageKind::Image,
                                                       GfxMpsrtTensorFlagExternalIo);
    const auto weights_desc = gfx_mpsrt_make_tensor_desc({2, 3, 3, 3},
                                                         ov::element::f16,
                                                         GfxStageStorageKind::Buffer,
                                                         GfxMpsrtTensorFlagConst);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({1, 2, 32, 32},
                                                        ov::element::f16,
                                                        GfxStageStorageKind::Image,
                                                        GfxMpsrtTensorFlagExternalIo);
    model.tensors.push_back({0, gfx_mpsrt_to_abi_desc(input_desc)});
    model.tensors.push_back({1, gfx_mpsrt_to_abi_desc(weights_desc)});
    model.tensors.push_back({2, gfx_mpsrt_to_abi_desc(output_desc)});
    model.resources = {
        {0u, GfxMpsrtExternalBufferRole::TensorInput, runtime_mpsrt::MpsrtRuntimeResourceLifetime::External, 0u, true, 0u, model.tensors[0].desc},
        {1u, GfxMpsrtExternalBufferRole::ConstBuffer, runtime_mpsrt::MpsrtRuntimeResourceLifetime::Model, 0u, true, 1u, model.tensors[1].desc},
        {2u, GfxMpsrtExternalBufferRole::TensorOutput, runtime_mpsrt::MpsrtRuntimeResourceLifetime::External, 1u, true, 2u, model.tensors[2].desc},
    };

    runtime_mpsrt::MpsrtRuntimeStage stage;
    stage.kind = GfxMpsrtStageKind::MPSConv2D;
    stage.stage_record_key = model.stage_record_key;
    stage.kernel_name = "mps_conv2d";
    stage.conv2d_desc.groups = 1;
    stage.conv2d_desc.strides[0] = 1;
    stage.conv2d_desc.strides[1] = 1;
    stage.conv2d_desc.dilations[0] = 1;
    stage.conv2d_desc.dilations[1] = 1;
    stage.conv2d_desc.pads[0] = 1;
    stage.conv2d_desc.pads[1] = 1;
    stage.conv2d_desc.pads[2] = 1;
    stage.conv2d_desc.pads[3] = 1;
    stage.inputs = {0, 1};
    stage.outputs = {2};
    stage.output_descs = {gfx_mpsrt_to_abi_desc(output_desc)};
    model.stages.push_back(stage);

    metal::mpsrt::MpsrtContext context(device);
    metal::mpsrt::MpsrtPreparedModel prepared_model;
    std::string log;
    EXPECT_FALSE(context.prepare_model(model, "", prepared_model, &log));
    EXPECT_NE(log.find("const-pack cache"), std::string::npos) << log;
    EXPECT_TRUE(prepared_model.msl_dispatches.empty());
    EXPECT_TRUE(prepared_model.mps_gemm_stages.empty());
    EXPECT_TRUE(prepared_model.mps_conv2d_stages.empty());
    EXPECT_EQ(prepared_model.skipped_non_msl_stages, 0u);
}

TEST(GfxBackendTest, MpsrtPrepareModelMaterializesMpsConv2DWeightsFromConstPack) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    runtime_mpsrt::MpsrtModel model;
    model.stage_record_key =
        "mps_conv2d|apple_mps|image|image|nhwc4|Convolution|apple_mps:image:Convolution|"
        "conv2d:g1:s1x1:d1x1:p1,1,1,1";
    model.semantic_input_values = {0};
    model.semantic_output_values = {2};
    model.input_values = {0};
    model.output_values = {2};
    model.external_input_values = {0};
    model.external_output_values = {2};
    model.external_values = {0, 2};
    model.external_buffer_roles = {GfxMpsrtExternalBufferRole::TensorInput,
                                   GfxMpsrtExternalBufferRole::TensorOutput};

    const auto input_desc = gfx_mpsrt_make_tensor_desc({1, 3, 32, 32},
                                                       ov::element::f16,
                                                       GfxStageStorageKind::Image,
                                                       GfxMpsrtTensorFlagExternalIo);
    const auto weights_desc = gfx_mpsrt_make_tensor_desc({2, 3, 3, 3},
                                                         ov::element::f16,
                                                         GfxStageStorageKind::Buffer,
                                                         GfxMpsrtTensorFlagConst);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({1, 2, 32, 32},
                                                        ov::element::f16,
                                                        GfxStageStorageKind::Image,
                                                        GfxMpsrtTensorFlagExternalIo);
    const auto input_abi = gfx_mpsrt_to_abi_desc(input_desc);
    const auto weights_abi = gfx_mpsrt_to_abi_desc(weights_desc);
    const auto output_abi = gfx_mpsrt_to_abi_desc(output_desc);
    model.tensors.push_back({0, input_abi});
    model.tensors.push_back({1, weights_abi});
    model.tensors.push_back({2, output_abi});
    model.resources = {
        {0u, GfxMpsrtExternalBufferRole::TensorInput, runtime_mpsrt::MpsrtRuntimeResourceLifetime::External, 0u, true, 0u, input_abi},
        {1u, GfxMpsrtExternalBufferRole::ConstBuffer, runtime_mpsrt::MpsrtRuntimeResourceLifetime::Model, 0u, true, 1u, weights_abi},
        {2u, GfxMpsrtExternalBufferRole::TensorOutput, runtime_mpsrt::MpsrtRuntimeResourceLifetime::External, 1u, true, 2u, output_abi},
    };

    runtime_mpsrt::MpsrtRuntimeStage stage;
    stage.kind = GfxMpsrtStageKind::MPSConv2D;
    stage.stage_record_key = model.stage_record_key;
    stage.kernel_name = "mps_conv2d";
    stage.conv2d_desc.groups = 1;
    stage.conv2d_desc.strides[0] = 1;
    stage.conv2d_desc.strides[1] = 1;
    stage.conv2d_desc.dilations[0] = 1;
    stage.conv2d_desc.dilations[1] = 1;
    stage.conv2d_desc.pads[0] = 1;
    stage.conv2d_desc.pads[1] = 1;
    stage.conv2d_desc.pads[2] = 1;
    stage.conv2d_desc.pads[3] = 1;
    stage.inputs = {0, 1};
    stage.outputs = {2};
    stage.output_descs = {output_abi};
    model.stages.push_back(stage);

    std::vector<ov::float16> weights(2 * 3 * 3 * 3, ov::float16(0.125f));
    metal::mpsrt::MpsrtContext context(device);
    std::string log;
    ASSERT_TRUE(context.register_const_tensor_data(1,
                                                   weights_abi,
                                                   weights.data(),
                                                   weights.size() * sizeof(ov::float16),
                                                   &log))
        << log;
    ASSERT_TRUE(context.has_const_tensor(1));

    metal::mpsrt::MpsrtPreparedModel prepared_model;
    ASSERT_TRUE(context.prepare_model(model, "", prepared_model, &log)) << log;
    ASSERT_EQ(prepared_model.resources.size(), 3u);
    EXPECT_EQ(prepared_model.resources[1].resource_index, 1u);
    EXPECT_EQ(prepared_model.resources[1].lifetime, runtime_mpsrt::MpsrtRuntimeResourceLifetime::Model);
    EXPECT_EQ(prepared_model.resources[1].role, GfxMpsrtExternalBufferRole::ConstBuffer);
    EXPECT_EQ(prepared_model.resources[1].value, 1u);
    EXPECT_EQ(prepared_model.resources[1].byte_length, weights.size() * sizeof(ov::float16));
    EXPECT_NE(prepared_model.resources[1].buffer, nil);
    ASSERT_TRUE(prepared_model.msl_dispatches.empty());
    ASSERT_TRUE(prepared_model.mps_gemm_stages.empty());
    ASSERT_EQ(prepared_model.mps_conv2d_stages.size(), 1u);
    const auto& prepared = prepared_model.mps_conv2d_stages.front();
    EXPECT_EQ(prepared.stage_index, 0u);
    EXPECT_EQ(prepared.weights_value, 1u);
    EXPECT_EQ(prepared.weights_byte_length, weights.size() * sizeof(ov::float16));
    EXPECT_EQ(prepared.input_feature_channels, 3u);
    EXPECT_EQ(prepared.output_feature_channels, 2u);
    EXPECT_EQ(prepared.output_width, 32u);
    EXPECT_EQ(prepared.output_height, 32u);
    EXPECT_EQ(prepared.output_batch, 1u);
    EXPECT_TRUE(prepared.weights_cache_hit);
    EXPECT_FALSE(prepared.kernel_cache_hit);
    EXPECT_NE(prepared.weights_buffer, nil);
    EXPECT_NE(prepared.kernel, nil);
    EXPECT_EQ(prepared_model.skipped_non_msl_stages, 0u);

    metal::mpsrt::MpsrtPreparedModel cached_prepared_model;
    ASSERT_TRUE(context.prepare_model(model, "", cached_prepared_model, &log)) << log;
    ASSERT_EQ(cached_prepared_model.mps_conv2d_stages.size(), 1u);
    EXPECT_TRUE(cached_prepared_model.mps_conv2d_stages.front().kernel_cache_hit);
    EXPECT_EQ(cached_prepared_model.mps_conv2d_stages.front().kernel, prepared.kernel);

    id<MTLCommandBuffer> cmd = [context.command_queue() commandBuffer];
    ASSERT_NE(cmd, nil);
    metal::mpsrt::MpsrtRequest request;
    metal::mpsrt::MpsrtTensorBindings bindings;
    metal::mpsrt::MpsrtModelEncodeResult encode_result;
    std::vector<KernelDispatch> dispatches(1);
    std::string encode_log;
    EXPECT_FALSE(request.encode_prepared_model((GpuCommandBufferHandle)cmd,
                                               model,
                                               prepared_model,
                                               dispatches,
                                               bindings,
                                               nullptr,
                                               &encode_result,
                                               &encode_log));
    EXPECT_NE(encode_log.find("input image"), std::string::npos) << encode_log;
    EXPECT_EQ(encode_result.skipped_non_msl_stages, 0u);

    MTLTextureDescriptor* input_texture_desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                          width:32
                                                         height:32
                                                      mipmapped:false];
    input_texture_desc.textureType = MTLTextureType2DArray;
    input_texture_desc.arrayLength = 1;
    input_texture_desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    input_texture_desc.storageMode = MTLStorageModePrivate;
    id<MTLTexture> input_texture = [device newTextureWithDescriptor:input_texture_desc];
    ASSERT_NE(input_texture, nil);

    MTLTextureDescriptor* output_texture_desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                          width:32
                                                         height:32
                                                      mipmapped:false];
    output_texture_desc.textureType = MTLTextureType2DArray;
    output_texture_desc.arrayLength = 1;
    output_texture_desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    output_texture_desc.storageMode = MTLStorageModePrivate;
    id<MTLTexture> output_texture = [device newTextureWithDescriptor:output_texture_desc];
    ASSERT_NE(output_texture, nil);

    metal::mpsrt::MpsrtTensorBindings missing_prepared_bindings;
    metal::mpsrt::MpsrtBindingBuildResult missing_prepared_result;
    std::string missing_prepared_log;
    EXPECT_FALSE(metal::mpsrt::build_mpsrt_tensor_bindings(
        model,
        {metal::mpsrt::make_mpsrt_bound_image((__bridge void*)input_texture)},
        {metal::mpsrt::make_mpsrt_bound_image((__bridge void*)output_texture)},
        missing_prepared_bindings,
        &missing_prepared_result,
        &missing_prepared_log));
    EXPECT_NE(missing_prepared_log.find("missing from prepared resources"), std::string::npos)
        << missing_prepared_log;
    EXPECT_EQ(missing_prepared_result.model_resources_bound, 0u);

    metal::mpsrt::MpsrtBindingBuildResult binding_result;
    ASSERT_TRUE(metal::mpsrt::build_mpsrt_tensor_bindings(
                    model,
                    {metal::mpsrt::make_mpsrt_bound_image((__bridge void*)input_texture)},
                    {metal::mpsrt::make_mpsrt_bound_image((__bridge void*)output_texture)},
                    bindings,
                    &binding_result,
                    &encode_log,
                    &prepared_model))
        << encode_log;
    EXPECT_EQ(binding_result.external_inputs_bound, 1u);
    EXPECT_EQ(binding_result.external_outputs_bound, 1u);
    EXPECT_EQ(binding_result.model_resources_bound, 1u);
    EXPECT_EQ(binding_result.transient_buffers_allocated, 0u);
    EXPECT_EQ(binding_result.transient_images_allocated, 0u);
    ASSERT_NE(bindings.lookup(1), nullptr);
    EXPECT_EQ(bindings.lookup(1)->buffer, (__bridge void*)prepared.weights_buffer);

    id<MTLCommandBuffer> conv_cmd = [context.command_queue() commandBuffer];
    ASSERT_NE(conv_cmd, nil);
    metal::mpsrt::MpsrtModelEncodeResult conv_encode_result;
    std::unordered_map<std::string, uint64_t> counters;
    KernelExecutionHooks hooks;
    hooks.on_counter = [&counters](std::string_view name, uint64_t delta) {
        counters[std::string(name)] += delta;
    };
    ASSERT_TRUE(request.encode_prepared_model((GpuCommandBufferHandle)conv_cmd,
                                              model,
                                              prepared_model,
                                              dispatches,
                                              bindings,
                                              &hooks,
                                              &conv_encode_result,
                                              &encode_log))
        << encode_log;
    [conv_cmd commit];
    [conv_cmd waitUntilCompleted];
    ASSERT_EQ([conv_cmd status], MTLCommandBufferStatusCompleted);
    EXPECT_EQ(conv_encode_result.encoded_mps_conv2d_stages, 1u);
    EXPECT_EQ(conv_encode_result.bound_buffers, 2u);
    EXPECT_EQ(counters["mpsrt_model_request_mps_conv2d_stage_encode_count"], 1u);
    EXPECT_EQ(counters["mpsrt_mps_conv2d_request_encode_count"], 1u);
}

TEST(GfxBackendTest, MpsrtPrepareModelMaterializesMpsGroupConv2DDepthwiseWeightsFromConstPack) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    runtime_mpsrt::MpsrtModel model;
    model.stage_record_key =
        "mps_group_conv2d|apple_mps|image|image|nhwc4|GroupConvolution|apple_mps:image:GroupConvolution|"
        "conv2d:g4:s1x1:d1x1:p1,1,1,1";
    model.semantic_input_values = {0};
    model.semantic_output_values = {2};
    model.input_values = {0};
    model.output_values = {2};
    model.external_input_values = {0};
    model.external_output_values = {2};
    model.external_values = {0, 2};
    model.external_buffer_roles = {GfxMpsrtExternalBufferRole::TensorInput,
                                   GfxMpsrtExternalBufferRole::TensorOutput};

    const auto input_desc = gfx_mpsrt_make_tensor_desc({1, 4, 16, 16},
                                                       ov::element::f16,
                                                       GfxStageStorageKind::Image,
                                                       GfxMpsrtTensorFlagExternalIo);
    const auto weights_desc = gfx_mpsrt_make_tensor_desc({4, 1, 1, 3, 3},
                                                         ov::element::f16,
                                                         GfxStageStorageKind::Buffer,
                                                         GfxMpsrtTensorFlagConst);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({1, 4, 16, 16},
                                                        ov::element::f16,
                                                        GfxStageStorageKind::Image,
                                                        GfxMpsrtTensorFlagExternalIo);
    const auto input_abi = gfx_mpsrt_to_abi_desc(input_desc);
    const auto weights_abi = gfx_mpsrt_to_abi_desc(weights_desc);
    const auto output_abi = gfx_mpsrt_to_abi_desc(output_desc);
    model.tensors.push_back({0, input_abi});
    model.tensors.push_back({1, weights_abi});
    model.tensors.push_back({2, output_abi});

    runtime_mpsrt::MpsrtRuntimeStage stage;
    stage.kind = GfxMpsrtStageKind::MPSGroupConv2D;
    stage.stage_record_key = model.stage_record_key;
    stage.kernel_name = "mps_group_conv2d";
    stage.conv2d_desc.groups = 4;
    stage.conv2d_desc.strides[0] = 1;
    stage.conv2d_desc.strides[1] = 1;
    stage.conv2d_desc.dilations[0] = 1;
    stage.conv2d_desc.dilations[1] = 1;
    stage.conv2d_desc.pads[0] = 1;
    stage.conv2d_desc.pads[1] = 1;
    stage.conv2d_desc.pads[2] = 1;
    stage.conv2d_desc.pads[3] = 1;
    stage.inputs = {0, 1};
    stage.outputs = {2};
    stage.output_descs = {output_abi};
    model.stages.push_back(stage);

    std::vector<ov::float16> weights(4 * 1 * 1 * 3 * 3, ov::float16(0.125f));
    metal::mpsrt::MpsrtContext context(device);
    std::string log;
    ASSERT_TRUE(context.register_const_tensor_data(1,
                                                   weights_abi,
                                                   weights.data(),
                                                   weights.size() * sizeof(ov::float16),
                                                   &log))
        << log;

    metal::mpsrt::MpsrtPreparedModel prepared_model;
    ASSERT_TRUE(runtime_mpsrt::finalize_mpsrt_model_resources(model, &log)) << log;
    ASSERT_TRUE(context.prepare_model(model, "", prepared_model, &log)) << log;
    ASSERT_TRUE(prepared_model.msl_dispatches.empty());
    ASSERT_TRUE(prepared_model.mps_gemm_stages.empty());
    ASSERT_EQ(prepared_model.mps_conv2d_stages.size(), 1u);
    const auto& prepared = prepared_model.mps_conv2d_stages.front();
    EXPECT_EQ(prepared.stage_index, 0u);
    EXPECT_EQ(prepared.weights_value, 1u);
    EXPECT_EQ(prepared.weights_byte_length, weights.size() * sizeof(ov::float16));
    EXPECT_EQ(prepared.input_feature_channels, 4u);
    EXPECT_EQ(prepared.output_feature_channels, 4u);
    EXPECT_EQ(prepared.output_width, 16u);
    EXPECT_EQ(prepared.output_height, 16u);
    EXPECT_EQ(prepared.output_batch, 1u);
    EXPECT_NE(prepared.weights_buffer, nil);
    EXPECT_NE(prepared.kernel, nil);
    EXPECT_EQ(prepared_model.skipped_non_msl_stages, 0u);
}

TEST(GfxBackendTest, MpsrtRequestEncodesPreparedBatchedMpsGemmModelWithMatrixBindings) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    constexpr uint32_t kBatch = 2;
    constexpr uint32_t kRows = 2;
    constexpr uint32_t kInner = 3;
    constexpr uint32_t kColumns = 2;

    runtime_mpsrt::MpsrtModel model;
    model.stage_record_key = "mps_batched_gemm_model";
    model.semantic_input_values = {0, 1};
    model.semantic_output_values = {2};
    model.input_values = {0, 1};
    model.output_values = {2};
    model.external_input_values = {0, 1};
    model.external_output_values = {2};
    model.external_values = {0, 1, 2};
    model.external_buffer_roles = {GfxMpsrtExternalBufferRole::TensorInput,
                                   GfxMpsrtExternalBufferRole::TensorInput,
                                   GfxMpsrtExternalBufferRole::TensorOutput};

    const auto lhs_desc = gfx_mpsrt_make_tensor_desc({kBatch, kRows, kInner},
                                                     ov::element::f32,
                                                     GfxStageStorageKind::Matrix,
                                                     GfxMpsrtTensorFlagExternalIo);
    const auto rhs_desc = gfx_mpsrt_make_tensor_desc({kBatch, kInner, kColumns},
                                                     ov::element::f32,
                                                     GfxStageStorageKind::Matrix,
                                                     GfxMpsrtTensorFlagExternalIo);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({kBatch, kRows, kColumns},
                                                        ov::element::f32,
                                                        GfxStageStorageKind::Matrix,
                                                        GfxMpsrtTensorFlagExternalIo);
    ASSERT_EQ(lhs_desc.matrix_count, kBatch);
    ASSERT_EQ(rhs_desc.matrix_count, kBatch);
    ASSERT_EQ(output_desc.matrix_count, kBatch);
    model.tensors.push_back({0, gfx_mpsrt_to_abi_desc(lhs_desc)});
    model.tensors.push_back({1, gfx_mpsrt_to_abi_desc(rhs_desc)});
    model.tensors.push_back({2, gfx_mpsrt_to_abi_desc(output_desc)});

    runtime_mpsrt::MpsrtRuntimeStage stage;
    stage.kind = GfxMpsrtStageKind::MPSGemm;
    stage.stage_record_key = "mps_gemm|apple_mps|matrix|matrix|row_major|MatMul|apple_mps:matrix:MatMul";
    stage.kernel_name = "mps_gemm";
    stage.inputs = {0, 1};
    stage.outputs = {2};
    stage.output_descs = {gfx_mpsrt_to_abi_desc(output_desc)};
    model.stages.push_back(stage);

    metal::mpsrt::MpsrtContext context(device);
    metal::mpsrt::MpsrtPreparedModel prepared_model;
    std::string log;
    ASSERT_TRUE(runtime_mpsrt::finalize_mpsrt_model_resources(model, &log)) << log;
    ASSERT_TRUE(context.prepare_model(model, "", prepared_model, &log)) << log;
    ASSERT_EQ(prepared_model.mps_gemm_stages.size(), 1u);

    id<MTLBuffer> lhs = [device newBufferWithLength:sizeof(float) * kBatch * kRows * kInner
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> rhs = [device newBufferWithLength:sizeof(float) * kBatch * kInner * kColumns
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:sizeof(float) * kBatch * kRows * kColumns
                                               options:MTLResourceStorageModeShared];
    ASSERT_NE(lhs, nil);
    ASSERT_NE(rhs, nil);
    ASSERT_NE(output, nil);

    float* lhs_ptr = static_cast<float*>([lhs contents]);
    float* rhs_ptr = static_cast<float*>([rhs contents]);
    float* output_ptr = static_cast<float*>([output contents]);
    const float lhs_values[] = {
        1.0f, 2.0f, 3.0f,
        -1.0f, 0.5f, 4.0f,
        0.25f, -2.0f, 1.0f,
        3.0f, 0.0f, -0.5f};
    const float rhs_values[] = {
        1.0f, 0.0f,
        2.0f, -1.0f,
        0.5f, 3.0f,
        -2.0f, 1.0f,
        4.0f, 0.25f,
        1.5f, -0.5f};
    const float expected[] = {
        6.5f, 7.0f,
        2.0f, 11.5f,
        -7.0f, -0.75f,
        -6.75f, 3.25f};
    std::memcpy(lhs_ptr, lhs_values, sizeof(lhs_values));
    std::memcpy(rhs_ptr, rhs_values, sizeof(rhs_values));
    for (uint32_t i = 0; i < kBatch * kRows * kColumns; ++i) {
        output_ptr[i] = -1.0f;
    }

    metal::mpsrt::MpsrtTensorBindings bindings;
    metal::mpsrt::MpsrtBindingBuildResult binding_result;
    ASSERT_TRUE(metal::mpsrt::build_mpsrt_tensor_bindings(model,
                                                          {{(__bridge void*)lhs, 0}, {(__bridge void*)rhs, 0}},
                                                          {{(__bridge void*)output, 0}},
                                                          bindings,
                                                          &binding_result,
                                                          &log))
        << log;

    std::unordered_map<std::string, uint64_t> counters;
    KernelExecutionHooks hooks;
    hooks.on_counter = [&counters](std::string_view name, uint64_t delta) {
        counters[std::string(name)] += delta;
    };

    id<MTLCommandBuffer> cmd = [context.command_queue() commandBuffer];
    ASSERT_NE(cmd, nil);
    std::vector<KernelDispatch> stage_dispatches(1);
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
    ASSERT_EQ([cmd status], MTLCommandBufferStatusCompleted);

    EXPECT_EQ(result.encoded_mps_gemm_stages, 1u);
    EXPECT_EQ(counters["mpsrt_mps_gemm_request_encode_count"], 1u);
    for (uint32_t i = 0; i < kBatch * kRows * kColumns; ++i) {
        EXPECT_FLOAT_EQ(output_ptr[i], expected[i]);
    }
}

TEST(GfxBackendTest, MpsrtRequestEncodesPreparedBatchBroadcastMpsGemmModelWithMatrixBindings) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    constexpr uint32_t kBatch = 2;
    constexpr uint32_t kRows = 2;
    constexpr uint32_t kInner = 3;
    constexpr uint32_t kColumns = 2;

    runtime_mpsrt::MpsrtModel model;
    model.stage_record_key = "mps_broadcast_gemm_model";
    model.semantic_input_values = {0, 1};
    model.semantic_output_values = {2};
    model.input_values = {0, 1};
    model.output_values = {2};
    model.external_input_values = {0, 1};
    model.external_output_values = {2};
    model.external_values = {0, 1, 2};
    model.external_buffer_roles = {GfxMpsrtExternalBufferRole::TensorInput,
                                   GfxMpsrtExternalBufferRole::TensorInput,
                                   GfxMpsrtExternalBufferRole::TensorOutput};

    const auto lhs_desc = gfx_mpsrt_make_tensor_desc({1, kRows, kInner},
                                                     ov::element::f32,
                                                     GfxStageStorageKind::Matrix,
                                                     GfxMpsrtTensorFlagExternalIo);
    const auto rhs_desc = gfx_mpsrt_make_tensor_desc({kBatch, kInner, kColumns},
                                                     ov::element::f32,
                                                     GfxStageStorageKind::Matrix,
                                                     GfxMpsrtTensorFlagExternalIo);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({kBatch, kRows, kColumns},
                                                        ov::element::f32,
                                                        GfxStageStorageKind::Matrix,
                                                        GfxMpsrtTensorFlagExternalIo);
    ASSERT_EQ(lhs_desc.matrix_count, 1u);
    ASSERT_EQ(rhs_desc.matrix_count, kBatch);
    ASSERT_EQ(output_desc.matrix_count, kBatch);
    model.tensors.push_back({0, gfx_mpsrt_to_abi_desc(lhs_desc)});
    model.tensors.push_back({1, gfx_mpsrt_to_abi_desc(rhs_desc)});
    model.tensors.push_back({2, gfx_mpsrt_to_abi_desc(output_desc)});

    runtime_mpsrt::MpsrtRuntimeStage stage;
    stage.kind = GfxMpsrtStageKind::MPSGemm;
    stage.stage_record_key = "mps_gemm|apple_mps|matrix|matrix|row_major|MatMul|apple_mps:matrix:MatMul";
    stage.kernel_name = "mps_gemm";
    stage.inputs = {0, 1};
    stage.outputs = {2};
    stage.output_descs = {gfx_mpsrt_to_abi_desc(output_desc)};
    model.stages.push_back(stage);

    metal::mpsrt::MpsrtContext context(device);
    metal::mpsrt::MpsrtPreparedModel prepared_model;
    std::string log;
    ASSERT_TRUE(runtime_mpsrt::finalize_mpsrt_model_resources(model, &log)) << log;
    ASSERT_TRUE(context.prepare_model(model, "", prepared_model, &log)) << log;
    ASSERT_EQ(prepared_model.mps_gemm_stages.size(), 1u);
    EXPECT_EQ(prepared_model.mps_gemm_stages.front().batch_count, kBatch);
    EXPECT_TRUE(prepared_model.mps_gemm_stages.front().lhs_batch_broadcast);
    EXPECT_FALSE(prepared_model.mps_gemm_stages.front().rhs_batch_broadcast);

    id<MTLBuffer> lhs = [device newBufferWithLength:sizeof(float) * kRows * kInner
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> rhs = [device newBufferWithLength:sizeof(float) * kBatch * kInner * kColumns
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:sizeof(float) * kBatch * kRows * kColumns
                                               options:MTLResourceStorageModeShared];
    ASSERT_NE(lhs, nil);
    ASSERT_NE(rhs, nil);
    ASSERT_NE(output, nil);

    const float lhs_values[] = {1.0f, -1.0f, 0.5f, 2.0f, 0.0f, -0.5f};
    const float rhs_values[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        6.0f, 7.0f,
        -1.0f, 0.5f,
        2.0f, -2.0f,
        1.0f, 3.0f};
    const float expected[] = {
        1.0f, 1.5f,
        -1.0f, 0.5f,
        -2.5f, 4.0f,
        -2.5f, -0.5f};
    std::memcpy([lhs contents], lhs_values, sizeof(lhs_values));
    std::memcpy([rhs contents], rhs_values, sizeof(rhs_values));
    float* output_ptr = static_cast<float*>([output contents]);
    for (uint32_t i = 0; i < kBatch * kRows * kColumns; ++i) {
        output_ptr[i] = -1.0f;
    }

    metal::mpsrt::MpsrtTensorBindings bindings;
    metal::mpsrt::MpsrtBindingBuildResult binding_result;
    ASSERT_TRUE(metal::mpsrt::build_mpsrt_tensor_bindings(model,
                                                          {{(__bridge void*)lhs, 0}, {(__bridge void*)rhs, 0}},
                                                          {{(__bridge void*)output, 0}},
                                                          bindings,
                                                          &binding_result,
                                                          &log))
        << log;

    std::unordered_map<std::string, uint64_t> counters;
    KernelExecutionHooks hooks;
    hooks.on_counter = [&counters](std::string_view name, uint64_t delta) {
        counters[std::string(name)] += delta;
    };

    id<MTLCommandBuffer> cmd = [context.command_queue() commandBuffer];
    ASSERT_NE(cmd, nil);
    std::vector<KernelDispatch> stage_dispatches(1);
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
    ASSERT_EQ([cmd status], MTLCommandBufferStatusCompleted);

    EXPECT_EQ(result.encoded_mps_gemm_stages, 1u);
    EXPECT_EQ(counters["mpsrt_mps_gemm_request_encode_count"], 1u);
    EXPECT_EQ(counters["mpsrt_mps_gemm_kernel_encode_count"], kBatch);
    for (uint32_t i = 0; i < kBatch * kRows * kColumns; ++i) {
        EXPECT_FLOAT_EQ(output_ptr[i], expected[i]);
    }
}

TEST(GfxBackendTest, MetalCompiledKernelExecutesMixedMpsGemmAndMslEpilogueMpsrtModel) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    ASSERT_NE(queue, nil);

    constexpr uint32_t kRows = 2;
    constexpr uint32_t kInner = 3;
    constexpr uint32_t kColumns = 2;
    constexpr uint32_t kElementCount = kRows * kColumns;

    const char* source = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void eltwise_fused_buffer(device const float* gemm [[buffer(0)]],
                                 device const float* bias [[buffer(1)]],
                                 device float* output [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= 4) return;
  output[gid] = gemm[gid] + bias[gid];
}
)MSL";

    runtime_mpsrt::MpsrtModel model;
    model.stage_record_key = "mps_gemm_plus_msl_epilogue_model";
    model.semantic_input_values = {0, 1, 3};
    model.semantic_output_values = {4};
    model.input_values = {0, 1, 3};
    model.output_values = {4};
    model.external_values = {0, 1, 3, 4};
    model.external_input_values = {0, 1, 3};
    model.external_output_values = {4};
    model.external_buffer_roles = {GfxMpsrtExternalBufferRole::TensorInput,
                                   GfxMpsrtExternalBufferRole::TensorInput,
                                   GfxMpsrtExternalBufferRole::TensorInput,
                                   GfxMpsrtExternalBufferRole::TensorOutput};

    const auto lhs_desc = gfx_mpsrt_make_tensor_desc({kRows, kInner},
                                                     ov::element::f32,
                                                     GfxStageStorageKind::Matrix,
                                                     GfxMpsrtTensorFlagExternalIo);
    const auto rhs_desc = gfx_mpsrt_make_tensor_desc({kInner, kColumns},
                                                     ov::element::f32,
                                                     GfxStageStorageKind::Matrix,
                                                     GfxMpsrtTensorFlagExternalIo);
    const auto gemm_desc = gfx_mpsrt_make_tensor_desc({kRows, kColumns},
                                                      ov::element::f32,
                                                      GfxStageStorageKind::Matrix,
                                                      GfxMpsrtTensorFlagTransient);
    const auto bias_desc = gfx_mpsrt_make_tensor_desc({kElementCount},
                                                      ov::element::f32,
                                                      GfxStageStorageKind::Buffer,
                                                      GfxMpsrtTensorFlagExternalIo);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({kElementCount},
                                                        ov::element::f32,
                                                        GfxStageStorageKind::Buffer,
                                                        GfxMpsrtTensorFlagExternalIo);
    model.tensors.push_back({0, gfx_mpsrt_to_abi_desc(lhs_desc)});
    model.tensors.push_back({1, gfx_mpsrt_to_abi_desc(rhs_desc)});
    model.tensors.push_back({2, gfx_mpsrt_to_abi_desc(gemm_desc)});
    model.tensors.push_back({3, gfx_mpsrt_to_abi_desc(bias_desc)});
    model.tensors.push_back({4, gfx_mpsrt_to_abi_desc(output_desc)});

    runtime_mpsrt::MpsrtRuntimeStage gemm_stage;
    gemm_stage.kind = GfxMpsrtStageKind::MPSGemm;
    gemm_stage.stage_record_key = "mps_gemm|apple_mps|matrix|matrix|row_major|MatMul|apple_mps:matrix:MatMul";
    gemm_stage.kernel_name = "mps_gemm";
    gemm_stage.inputs = {0, 1};
    gemm_stage.outputs = {2};
    gemm_stage.output_descs = {gfx_mpsrt_to_abi_desc(gemm_desc)};
    model.stages.push_back(gemm_stage);

    runtime_mpsrt::MpsrtRuntimeStage epilogue_stage;
    epilogue_stage.kind = GfxMpsrtStageKind::MSLDispatch;
    epilogue_stage.stage_record_key = "msl_dispatch|apple_msl|buffer|buffer|linear|MatMulEpilogue|"
                                      "apple_msl:buffer:MatMulEpilogue|dispatch:eltwise_fused_buffer:"
                                      "eltwise_fused_buffer:tg256:metallib";
    epilogue_stage.kernel_name = "eltwise_fused_buffer";
    epilogue_stage.dispatch_kernel_family = "eltwise_fused_buffer";
    epilogue_stage.dispatch_entry_point = "eltwise_fused_buffer";
    epilogue_stage.dispatch_kernel_family_id = static_cast<uint32_t>(GfxKernelFamily::EltwiseFusedBuffer);
    epilogue_stage.dispatch_threads_per_threadgroup = 256;
    epilogue_stage.dispatch_flags = GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired;
    epilogue_stage.dispatch_precompiled_kernel_required = true;
    epilogue_stage.msl_dispatch_desc.kernel_family = epilogue_stage.dispatch_kernel_family_id;
    epilogue_stage.msl_dispatch_desc.storage = static_cast<uint32_t>(GfxMpsrtStorage::Buffer);
    epilogue_stage.msl_dispatch_desc.layout = static_cast<uint32_t>(GfxMpsrtLayout::Linear);
    epilogue_stage.msl_dispatch_desc.threads_per_threadgroup = 256;
    epilogue_stage.msl_dispatch_desc.input_count = 2;
    epilogue_stage.msl_dispatch_desc.output_count = 1;
    epilogue_stage.msl_dispatch_desc.flags = GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired;
    epilogue_stage.inputs = {2, 3};
    epilogue_stage.outputs = {4};
    epilogue_stage.kernel_buffer_order = {2, 3, 4};
    epilogue_stage.output_descs = {gfx_mpsrt_to_abi_desc(output_desc)};
    model.stages.push_back(epilogue_stage);

    id<MTLBuffer> lhs = [device newBufferWithLength:sizeof(float) * kRows * kInner
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> rhs = [device newBufferWithLength:sizeof(float) * kInner * kColumns
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> bias = [device newBufferWithLength:sizeof(float) * kElementCount
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:sizeof(float) * kElementCount
                                               options:MTLResourceStorageModeShared];
    ASSERT_NE(lhs, nil);
    ASSERT_NE(rhs, nil);
    ASSERT_NE(bias, nil);
    ASSERT_NE(output, nil);

    const float lhs_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float rhs_values[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    const float bias_values[] = {0.5f, -1.0f, 2.0f, -3.0f};
    const float expected[] = {58.5f, 63.0f, 141.0f, 151.0f};
    std::memcpy([lhs contents], lhs_values, sizeof(lhs_values));
    std::memcpy([rhs contents], rhs_values, sizeof(rhs_values));
    std::memcpy([bias contents], bias_values, sizeof(bias_values));
    float* output_ptr = static_cast<float*>([output contents]);
    for (uint32_t i = 0; i < kElementCount; ++i) {
        output_ptr[i] = -1.0f;
    }

    std::string log;
    ASSERT_TRUE(runtime_mpsrt::finalize_mpsrt_model_resources(model, &log)) << log;
    auto kernel = std::make_shared<MetalCompiledKernel>((MetalDeviceHandle)device, nullptr, 4);
    kernel->set_mpsrt_model(std::make_shared<runtime_mpsrt::MpsrtModel>(model));
    kernel->set_mpsrt_msl_source(source);

    MetalBuffer lhs_gpu{};
    lhs_gpu.buffer = (__bridge void*)lhs;
    lhs_gpu.size = sizeof(lhs_values);
    lhs_gpu.type = ov::element::f32;
    MetalBuffer rhs_gpu{};
    rhs_gpu.buffer = (__bridge void*)rhs;
    rhs_gpu.size = sizeof(rhs_values);
    rhs_gpu.type = ov::element::f32;
    MetalBuffer bias_gpu{};
    bias_gpu.buffer = (__bridge void*)bias;
    bias_gpu.size = sizeof(bias_values);
    bias_gpu.type = ov::element::f32;
    MetalBuffer output_gpu{};
    output_gpu.buffer = (__bridge void*)output;
    output_gpu.size = sizeof(float) * kElementCount;
    output_gpu.type = ov::element::f32;

    std::vector<KernelArg> args = {
        make_buffer_arg(0, lhs_gpu),
        make_buffer_arg(1, rhs_gpu),
        make_buffer_arg(2, bias_gpu),
        make_buffer_arg(3, output_gpu),
    };
    KernelDispatch dispatch;
    dispatch.grid[0] = kElementCount;
    dispatch.threads_per_group[0] = 4;

    std::unordered_map<std::string, uint64_t> counters;
    KernelExecutionHooks hooks;
    hooks.on_counter = [&counters](std::string_view name, uint64_t delta) {
        counters[std::string(name)] += delta;
    };

    for (uint32_t iteration = 0; iteration < 4; ++iteration) {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        ASSERT_NE(cmd, nil);
        kernel->execute((GpuCommandBufferHandle)cmd, dispatch, args, &hooks);
        metal_end_compute_encoder((GpuCommandBufferHandle)cmd);
        [cmd commit];
        [cmd waitUntilCompleted];
        ASSERT_EQ([cmd status], MTLCommandBufferStatusCompleted);

        for (uint32_t i = 0; i < kElementCount; ++i) {
            EXPECT_FLOAT_EQ(output_ptr[i], expected[i]);
            if (iteration + 1 < 4) {
                output_ptr[i] = -1.0f;
            }
        }
    }

    EXPECT_EQ(counters["mpsrt_prepared_model_cache_miss_count"], 3u);
    EXPECT_EQ(counters["mpsrt_prepared_model_cache_hit_count"], 1u);
    EXPECT_EQ(counters["mpsrt_model_request_encode_count"], 4u);
    EXPECT_EQ(counters["mpsrt_model_request_mps_gemm_stage_encode_count"], 4u);
    EXPECT_EQ(counters["mpsrt_model_request_msl_stage_encode_count"], 4u);
    EXPECT_EQ(counters["mpsrt_binding_prepared_transient_buffer_count"], 4u);
}

TEST(GfxBackendTest, MetalCodegenCompilesVendorOnlyMpsGemmWithoutMslSource) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    constexpr uint32_t kRows = 2;
    constexpr uint32_t kInner = 3;
    constexpr uint32_t kColumns = 2;

    mlir::MLIRContext ctx;
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    annotate_test_mps_vendor_module(module, "MatMul", GfxKernelStageFamily::Gemm);
    const auto lhs_desc = gfx_mpsrt_make_tensor_desc({kRows, kInner},
                                                     ov::element::f32,
                                                     GfxStageStorageKind::Matrix,
                                                     GfxMpsrtTensorFlagExternalIo);
    const auto rhs_desc = gfx_mpsrt_make_tensor_desc({kColumns, kInner},
                                                     ov::element::f32,
                                                     GfxStageStorageKind::Matrix,
                                                     GfxMpsrtTensorFlagExternalIo);
    const auto output_desc = gfx_mpsrt_make_tensor_desc({kRows, kColumns},
                                                        ov::element::f32,
                                                        GfxStageStorageKind::Matrix,
                                                        GfxMpsrtTensorFlagExternalIo);
    auto lowering_plan = make_test_mps_vendor_lowering(module, {lhs_desc, rhs_desc}, {output_desc});
    GfxMpsrtGemmAbiDesc gemm_desc{};
    gemm_desc.transpose_rhs = 1;
    lowering_plan.stage_plan.stage.gemm_desc = gemm_desc;
    ASSERT_TRUE(refresh_apple_mps_stage_record_key(lowering_plan));
    ASSERT_TRUE(materialize_apple_mps_typed_program(module, lowering_plan));

    KernelSource source;
    source.module = module;
    source.entry_point = "mps_gemm";
    source.signature.arg_count = 3;
    source.signature.output_arg_count = 1;

    MetalCodegenBackend backend((MetalDeviceHandle)device);
    std::string log;
    auto kernel = backend.compile(source, &log);
    ASSERT_TRUE(kernel) << log;
    auto* metal_kernel = dynamic_cast<MetalCompiledKernel*>(kernel.get());
    ASSERT_NE(metal_kernel, nullptr);
    ASSERT_NE(metal_kernel->mpsrt_model(), nullptr);
    ASSERT_EQ(metal_kernel->mpsrt_model()->stages.size(), 1u);
    EXPECT_EQ(metal_kernel->mpsrt_model()->stages.front().kind, GfxMpsrtStageKind::MPSGemm);
    EXPECT_EQ(metal_kernel->mpsrt_model()->stages.front().gemm_desc.transpose_lhs, 0u);
    EXPECT_EQ(metal_kernel->mpsrt_model()->stages.front().gemm_desc.transpose_rhs, 1u);

    id<MTLBuffer> lhs = [device newBufferWithLength:sizeof(float) * kRows * kInner
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> rhs = [device newBufferWithLength:sizeof(float) * kColumns * kInner
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:sizeof(float) * kRows * kColumns
                                               options:MTLResourceStorageModeShared];
    ASSERT_NE(lhs, nil);
    ASSERT_NE(rhs, nil);
    ASSERT_NE(output, nil);

    float* lhs_ptr = static_cast<float*>([lhs contents]);
    float* rhs_ptr = static_cast<float*>([rhs contents]);
    float* output_ptr = static_cast<float*>([output contents]);
    const float lhs_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float rhs_values[] = {7.0f, 9.0f, 11.0f, 8.0f, 10.0f, 12.0f};
    std::memcpy(lhs_ptr, lhs_values, sizeof(lhs_values));
    std::memcpy(rhs_ptr, rhs_values, sizeof(rhs_values));
    for (uint32_t i = 0; i < kRows * kColumns; ++i) {
        output_ptr[i] = -1.0f;
    }

    MetalBuffer lhs_gpu{};
    lhs_gpu.buffer = (__bridge void*)lhs;
    lhs_gpu.size = sizeof(float) * kRows * kInner;
    lhs_gpu.type = ov::element::f32;
    MetalBuffer rhs_gpu{};
    rhs_gpu.buffer = (__bridge void*)rhs;
    rhs_gpu.size = sizeof(float) * kColumns * kInner;
    rhs_gpu.type = ov::element::f32;
    MetalBuffer output_gpu{};
    output_gpu.buffer = (__bridge void*)output;
    output_gpu.size = sizeof(float) * kRows * kColumns;
    output_gpu.type = ov::element::f32;

    std::vector<KernelArg> args;
    args.push_back(make_buffer_arg(0, lhs_gpu));
    args.push_back(make_buffer_arg(1, rhs_gpu));
    args.push_back(make_buffer_arg(2, output_gpu));

    std::unordered_map<std::string, uint64_t> counters;
    KernelExecutionHooks hooks;
    hooks.on_counter = [&counters](std::string_view name, uint64_t delta) {
        counters[std::string(name)] += delta;
    };

    id<MTLCommandQueue> queue = [device newCommandQueue];
    ASSERT_NE(queue, nil);
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    ASSERT_NE(cmd, nil);
    KernelDispatch dispatch;
    kernel->execute((GpuCommandBufferHandle)cmd, dispatch, args, &hooks);
    metal_end_compute_encoder((GpuCommandBufferHandle)cmd);
    [cmd commit];
    [cmd waitUntilCompleted];
    ASSERT_EQ([cmd status], MTLCommandBufferStatusCompleted);

    EXPECT_EQ(counters["mpsrt_model_request_encode_count"], 1u);
    EXPECT_EQ(counters["mpsrt_model_request_mps_gemm_stage_encode_count"], 1u);
    EXPECT_EQ(counters["mpsrt_mps_gemm_request_encode_count"], 1u);
    EXPECT_FLOAT_EQ(output_ptr[0], 58.0f);
    EXPECT_FLOAT_EQ(output_ptr[1], 64.0f);
    EXPECT_FLOAT_EQ(output_ptr[2], 139.0f);
    EXPECT_FLOAT_EQ(output_ptr[3], 154.0f);
}

TEST(GfxBackendTest, MetalCodegenCompilesVendorOnlyMpsTopKWithoutMslSource) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);

    constexpr uint32_t kRows = 2;
    constexpr uint32_t kColumns = 5;
    constexpr uint32_t kTopK = 3;

    mlir::MLIRContext ctx;
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    annotate_test_mps_vendor_module(module, "TopK", GfxKernelStageFamily::TopK);
    const auto input_desc = gfx_mpsrt_make_tensor_desc({kRows, kColumns},
                                                       ov::element::f32,
                                                       GfxStageStorageKind::Matrix,
                                                       GfxMpsrtTensorFlagExternalIo);
    const auto values_desc = gfx_mpsrt_make_tensor_desc({kRows, kTopK},
                                                        ov::element::f32,
                                                        GfxStageStorageKind::Matrix,
                                                        GfxMpsrtTensorFlagExternalIo);
    const auto indices_desc = gfx_mpsrt_make_tensor_desc({kRows, kTopK},
                                                         ov::element::i32,
                                                         GfxStageStorageKind::Matrix,
                                                         GfxMpsrtTensorFlagExternalIo);
    auto lowering_plan = make_test_mps_vendor_lowering(module, {input_desc}, {values_desc, indices_desc});
    GfxMpsrtTopKAbiDesc topk_desc{};
    topk_desc.axis = 1;
    topk_desc.k = kTopK;
    topk_desc.mode_max = 1;
    topk_desc.sort_type = 1;
    lowering_plan.stage_plan.stage.topk_desc = topk_desc;
    ASSERT_TRUE(refresh_apple_mps_stage_record_key(lowering_plan));
    ASSERT_TRUE(materialize_apple_mps_typed_program(module, lowering_plan));

    KernelSource source;
    source.module = module;
    source.entry_point = "mps_topk";
    source.signature.arg_count = 3;
    source.signature.output_arg_count = 2;

    MetalCodegenBackend backend((MetalDeviceHandle)device);
    std::string log;
    auto kernel = backend.compile(source, &log);
    ASSERT_TRUE(kernel) << log;
    auto* metal_kernel = dynamic_cast<MetalCompiledKernel*>(kernel.get());
    ASSERT_NE(metal_kernel, nullptr);
    ASSERT_NE(metal_kernel->mpsrt_model(), nullptr);
    ASSERT_EQ(metal_kernel->mpsrt_model()->stages.size(), 1u);
    EXPECT_EQ(metal_kernel->mpsrt_model()->stages.front().kind, GfxMpsrtStageKind::MPSTopK);
    EXPECT_EQ(metal_kernel->mpsrt_model()->stages.front().topk_desc.k, kTopK);

    id<MTLBuffer> input = [device newBufferWithLength:sizeof(float) * kRows * kColumns
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> values = [device newBufferWithLength:sizeof(float) * kRows * kTopK
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> indices = [device newBufferWithLength:sizeof(int32_t) * kRows * kTopK
                                              options:MTLResourceStorageModeShared];
    ASSERT_NE(input, nil);
    ASSERT_NE(values, nil);
    ASSERT_NE(indices, nil);

    const float input_values[] = {1.0f, 5.0f, 3.0f, 2.0f, 4.0f,
                                  -1.0f, 7.0f, 0.0f, 6.0f, 2.0f};
    std::memcpy([input contents], input_values, sizeof(input_values));
    std::fill_n(static_cast<float*>([values contents]), kRows * kTopK, -1.0f);
    std::fill_n(static_cast<int32_t*>([indices contents]), kRows * kTopK, -1);

    MetalBuffer input_gpu{};
    input_gpu.buffer = (__bridge void*)input;
    input_gpu.size = sizeof(input_values);
    input_gpu.type = ov::element::f32;
    MetalBuffer values_gpu{};
    values_gpu.buffer = (__bridge void*)values;
    values_gpu.size = sizeof(float) * kRows * kTopK;
    values_gpu.type = ov::element::f32;
    MetalBuffer indices_gpu{};
    indices_gpu.buffer = (__bridge void*)indices;
    indices_gpu.size = sizeof(int32_t) * kRows * kTopK;
    indices_gpu.type = ov::element::i32;

    std::vector<KernelArg> args;
    args.push_back(make_buffer_arg(0, input_gpu));
    args.push_back(make_buffer_arg(1, values_gpu));
    args.push_back(make_buffer_arg(2, indices_gpu));

    std::unordered_map<std::string, uint64_t> counters;
    KernelExecutionHooks hooks;
    hooks.on_counter = [&counters](std::string_view name, uint64_t delta) {
        counters[std::string(name)] += delta;
    };

    id<MTLCommandQueue> queue = [device newCommandQueue];
    ASSERT_NE(queue, nil);
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    ASSERT_NE(cmd, nil);
    KernelDispatch dispatch;
    kernel->execute((GpuCommandBufferHandle)cmd, dispatch, args, &hooks);
    metal_end_compute_encoder((GpuCommandBufferHandle)cmd);
    [cmd commit];
    [cmd waitUntilCompleted];
    ASSERT_EQ([cmd status], MTLCommandBufferStatusCompleted);

    EXPECT_EQ(counters["mpsrt_model_request_encode_count"], 1u);
    EXPECT_EQ(counters["mpsrt_model_request_mps_topk_stage_encode_count"], 1u);
    EXPECT_EQ(counters["mpsrt_mps_topk_request_encode_count"], 1u);

    const float* values_ptr = static_cast<const float*>([values contents]);
    const int32_t* indices_ptr = static_cast<const int32_t*>([indices contents]);
    const float expected_values[] = {5.0f, 4.0f, 3.0f, 7.0f, 6.0f, 2.0f};
    const int32_t expected_indices[] = {1, 4, 2, 1, 3, 4};
    for (uint32_t i = 0; i < kRows * kTopK; ++i) {
        EXPECT_FLOAT_EQ(values_ptr[i], expected_values[i]);
        EXPECT_EQ(indices_ptr[i], expected_indices[i]);
    }
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
