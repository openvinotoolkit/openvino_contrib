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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "backends/metal/runtime/backend.hpp"

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
    [cmd commit];
    [cmd waitUntilCompleted];

    EXPECT_EQ(kernel->args_count(), 2u);
    for (uint32_t i = 0; i < kCount; ++i) {
        EXPECT_FLOAT_EQ(ptr[i], static_cast<float>(i + 1));
    }
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
