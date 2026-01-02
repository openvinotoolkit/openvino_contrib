// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/codegen/metal_codegen_backend.hpp"

#include <vector>

#include "backends/metal/codegen/metal_compiler.hpp"
#include "openvino/core/except.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "kernel_ir/gfx_kernel_cache.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
inline id<MTLBuffer> to_mtl(const GpuBuffer& buf) {
    return (__bridge id<MTLBuffer>)buf.buffer;
}
}  // namespace

MetalCodegenBackend::MetalCodegenBackend(MetalDeviceHandle device) : m_device(device) {}

std::shared_ptr<ICompiledKernel> MetalCodegenBackend::compile(const KernelSource& source,
                                                              std::string* log) {
    std::string local_log;
    std::string* log_ptr = log ? log : &local_log;
    std::string msl = resolve_msl_source(source, log_ptr);
    OPENVINO_ASSERT(!msl.empty(), "MetalCodegenBackend: missing MSL source");
    OPENVINO_ASSERT(!source.entry_point.empty(), "MetalCodegenBackend: missing entry point");

    const uint32_t arg_count = source.signature.arg_count;
    const uintptr_t device_key = reinterpret_cast<uintptr_t>(m_device);
    return lookup_or_compile_kernel(GpuBackend::Metal,
                                    device_key,
                                    msl.data(),
                                    msl.size(),
                                    source.entry_point,
                                    arg_count,
                                    [&]() -> std::shared_ptr<ICompiledKernel> {
                                        MetalKernelCompiler compiler((id<MTLDevice>)m_device);
                                        std::string local_log;
                                        std::string& compile_log = log ? *log : local_log;
                                        id<MTLComputePipelineState> pipeline =
                                            compiler.compile_msl_from_source(msl,
                                                                             source.entry_point.c_str(),
                                                                             compile_log);
                                        if (!pipeline) {
                                            return nullptr;
                                        }
                                        return std::make_shared<MetalCompiledKernel>(m_device,
                                                                                     (void*)pipeline,
                                                                                     arg_count);
                                    });
}

MetalCompiledKernel::MetalCompiledKernel(MetalDeviceHandle device, void* pipeline, uint32_t arg_count)
    : m_device(device), m_pipeline(pipeline), m_args_count(arg_count) {}

void MetalCompiledKernel::set_args_count(uint32_t count) {
    if (count == 0) {
        return;
    }
    if (m_args_count == 0) {
        m_args_count = count;
        return;
    }
    OPENVINO_ASSERT(m_args_count == count,
                    "MetalCompiledKernel: arg count mismatch (expected ",
                    m_args_count,
                    ", got ",
                    count,
                    ")");
}

size_t MetalCompiledKernel::clamp_threadgroup_size(size_t desired) const {
    return metal_clamp_tg_size(m_pipeline, desired);
}

void MetalCompiledKernel::execute(GpuCommandBufferHandle command_buffer,
                                  const KernelDispatch& dispatch,
                                  const std::vector<KernelArg>& args,
                                  const KernelExecutionHooks* hooks) {
    id<MTLCommandBuffer> cb = static_cast<id<MTLCommandBuffer>>(command_buffer);
    OPENVINO_ASSERT(cb, "MetalCompiledKernel: command buffer is null");
    OPENVINO_ASSERT(m_pipeline, "MetalCompiledKernel: pipeline is null");
    const uint32_t runtime_count = ensure_kernel_args_dense(args, "MetalCompiledKernel");
    set_args_count(runtime_count);

    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:(id<MTLComputePipelineState>)m_pipeline];

    for (const auto& arg : args) {
        OPENVINO_ASSERT(arg.kind == KernelArg::Kind::Buffer,
                        "MetalCompiledKernel: bytes arguments must be materialized into buffers");
        [enc setBuffer:to_mtl(arg.buffer) offset:arg.offset atIndex:arg.index];
    }

    if (hooks && hooks->on_begin) {
        hooks->on_begin(enc);
    }

    const size_t grid_x = dispatch.grid[0];
    const size_t grid_y = dispatch.grid[1];
    const size_t grid_z = dispatch.grid[2];
    if (grid_x == 0 || grid_y == 0 || grid_z == 0) {
        if (hooks && hooks->on_end) {
            hooks->on_end(enc);
        }
        [enc endEncoding];
        return;
    }

    MTLSize grid = MTLSizeMake(grid_x, grid_y, grid_z);
    MTLSize tg = MTLSizeMake(dispatch.threads_per_group[0],
                             dispatch.threads_per_group[1],
                             dispatch.threads_per_group[2]);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];

    if (hooks && hooks->on_end) {
        hooks->on_end(enc);
    }
    [enc endEncoding];
}

}  // namespace gfx_plugin
}  // namespace ov
