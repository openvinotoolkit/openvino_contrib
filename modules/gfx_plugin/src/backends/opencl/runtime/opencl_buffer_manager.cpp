// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/runtime/opencl_buffer_manager.hpp"

#include "backends/opencl/runtime/memory_api.hpp"
#include "openvino/core/except.hpp"
#include "runtime/gfx_profiler.hpp"
#include "runtime/memory_manager.hpp"

namespace ov {
namespace gfx_plugin {

OpenClBufferManager::OpenClBufferManager(std::shared_ptr<OpenClRuntimeContext> context)
    : m_context(std::move(context)) {
    OPENVINO_ASSERT(m_context, "GFX OpenCL: buffer manager requires runtime context");
    m_const_cache = std::make_shared<ImmutableGpuBufferCache>([](GpuBuffer&& buf) {
        opencl_free_buffer(buf);
    });
}

OpenClBufferManager::~OpenClBufferManager() = default;

std::optional<GpuExecutionDeviceInfo> OpenClBufferManager::query_execution_device_info() const {
    return m_context ? std::optional<GpuExecutionDeviceInfo>{m_context->execution_device_info()} : std::nullopt;
}

GpuBuffer OpenClBufferManager::allocate_buffer(size_t bytes,
                                               ov::element::Type type,
                                               BufferUsage usage,
                                               bool host_visible,
                                               const char*) {
    if (bytes == 0) {
        return {};
    }
    cl_int status = CL_SUCCESS;
    cl_mem mem = m_context->api().fn().clCreateBuffer(m_context->context(),
                                                      CL_MEM_READ_WRITE,
                                                      bytes,
                                                      nullptr,
                                                      &status);
    opencl_check(status, "clCreateBuffer");
    OPENVINO_ASSERT(mem, "GFX OpenCL: clCreateBuffer returned null");
    GpuBuffer buf;
    buf.buffer = reinterpret_cast<GpuBufferHandle>(mem);
    buf.size = bytes;
    buf.type = type;
    buf.backend = GpuBackend::OpenCL;
    buf.host_visible = host_visible;
    buf.persistent = usage == BufferUsage::Const || usage == BufferUsage::IO;
    buf.owned = true;
    buf.allocation_uid = allocate_gpu_buffer_uid();
    return buf;
}

GpuBuffer OpenClBufferManager::wrap_const(const std::string& key,
                                          const void* data,
                                          size_t bytes,
                                          ov::element::Type type) {
    if (bytes == 0) {
        return {};
    }
    OPENVINO_ASSERT(data, "GFX OpenCL: const buffer data is null");
    return m_const_cache->get_or_create(key, data, bytes, type, [&]() {
        auto buf = allocate_buffer(bytes, type, BufferUsage::Const, false, key.c_str());
        opencl_check(m_context->api().fn().clEnqueueWriteBuffer(m_context->queue(),
                                                                reinterpret_cast<cl_mem>(buf.buffer),
                                                                CL_TRUE,
                                                                0,
                                                                bytes,
                                                                data,
                                                                0,
                                                                nullptr,
                                                                nullptr),
                     "clEnqueueWriteBuffer(const)");
        return buf;
    });
}

GpuBuffer OpenClBufferManager::allocate_temp(const GpuBufferDesc& desc) {
    validate_gpu_buffer_desc(desc, "GFX OpenCL");
    const bool host_visible = desc.cpu_read || desc.cpu_write || desc.usage == BufferUsage::IO ||
                              desc.usage == BufferUsage::Staging;
    return allocate_buffer(desc.bytes, desc.type, desc.usage, host_visible, desc.label);
}

void OpenClBufferManager::release_temp(GpuBuffer&& buf) {
    opencl_free_buffer(buf);
}

void OpenClBufferManager::flush_const_upload_batch(GpuCommandBufferHandle, GfxProfiler*) {
    if (m_context) {
        m_context->finish();
    }
}

}  // namespace gfx_plugin
}  // namespace ov
