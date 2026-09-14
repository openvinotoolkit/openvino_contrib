// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/plugin/remote_tensor.hpp"

#include "backends/opencl/runtime/memory_api.hpp"
#include "openvino/core/except.hpp"
#include "openvino/gfx_plugin/properties.hpp"
#include "plugin/gfx_remote_utils.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

template <typename T>
T query_opencl_mem_info(const OpenClRuntimeContext& context,
                        cl_mem mem,
                        cl_uint param,
                        const char* action) {
    T value{};
    opencl_check(context.api().fn().clGetMemObjectInfo(
                     mem, param, sizeof(T), &value, nullptr),
                 action);
    return value;
}

void validate_external_opencl_mem(const OpenClRuntimeContext& context,
                                  cl_mem mem,
                                  size_t required_bytes,
                                  size_t declared_bytes,
                                  size_t& actual_bytes) {
    OPENVINO_ASSERT(mem, "GFX OpenCL: remote cl_mem handle is null");

    const auto mem_context = query_opencl_mem_info<cl_context>(
        context, mem, CL_MEM_CONTEXT, "clGetMemObjectInfo(CL_MEM_CONTEXT)");
    OPENVINO_ASSERT(mem_context == context.context(),
                    "GFX OpenCL: remote cl_mem belongs to a different context");

    actual_bytes = query_opencl_mem_info<size_t>(
        context, mem, CL_MEM_SIZE, "clGetMemObjectInfo(CL_MEM_SIZE)");
    OPENVINO_ASSERT(actual_bytes >= required_bytes,
                    "GFX OpenCL: remote cl_mem is smaller than required (",
                    actual_bytes,
                    " < ",
                    required_bytes,
                    ")");
    if (declared_bytes) {
        OPENVINO_ASSERT(declared_bytes == actual_bytes,
                        "GFX OpenCL: remote cl_mem size mismatch (declared ",
                        declared_bytes,
                        ", actual ",
                        actual_bytes,
                        ")");
    }
}

GpuBuffer allocate_opencl_remote_buffer(const std::shared_ptr<OpenClRuntimeContext>& context,
                                        const ov::element::Type& type,
                                        size_t requested_bytes,
                                        bool host_visible) {
    const size_t allocation_bytes = opencl_allocation_bytes(requested_bytes, type);
    cl_int status = CL_SUCCESS;
    cl_mem mem = context->api().fn().clCreateBuffer(
        context->context(), CL_MEM_READ_WRITE, allocation_bytes, nullptr, &status);
    opencl_check(status, "clCreateBuffer(remote_tensor)");
    OPENVINO_ASSERT(mem, "GFX OpenCL: clCreateBuffer returned null for remote tensor");

    GpuBuffer buf;
    buf.buffer = reinterpret_cast<GpuBufferHandle>(mem);
    buf.size = allocation_bytes;
    buf.type = type;
    buf.backend = GpuBackend::OpenCL;
    buf.persistent = true;
    buf.owned = true;
    buf.host_visible = host_visible;
    buf.allocation_uid = allocate_gpu_buffer_uid();
    return buf;
}

void release_opencl_remote_tensor(GpuTensor& tensor) {
    if (!tensor.buf.owned || tensor.buf.backend != GpuBackend::OpenCL ||
        !tensor.buf.buffer) {
        return;
    }
    opencl_free_buffer(tensor.buf);
}

}  // namespace

RemoteTensorCreateResult create_opencl_remote_tensor(
    const ov::element::Type& type,
    const ov::Shape& shape,
    const ov::AnyMap& params,
    const std::shared_ptr<OpenClRuntimeContext>& context,
    size_t bytes) {
    OPENVINO_ASSERT(context, "GFX OpenCL: remote tensor requires runtime context");

    RemoteTensorCreateResult result;
    GpuTensor tensor;
    tensor.shape = shape;
    tensor.expected_type = type;
    tensor.prefer_private = false;
    tensor.buf.type = type;
    tensor.buf.backend = GpuBackend::OpenCL;

    void* external = find_any_ptr(params, {kGfxMemoryProperty, kGfxBufferProperty});
    const size_t declared_bytes = find_any_size(params, {kGfxBufferBytesProperty}, 0);

    if (external) {
        auto mem = reinterpret_cast<cl_mem>(external);
        size_t actual_bytes = 0;
        validate_external_opencl_mem(*context, mem, bytes, declared_bytes, actual_bytes);
        tensor.buf.buffer = reinterpret_cast<GpuBufferHandle>(mem);
        tensor.buf.size = actual_bytes;
        tensor.buf.from_handle = true;
        tensor.buf.external = true;
        tensor.buf.owned = false;
        tensor.buf.host_visible =
            find_any_bool(params, {kGfxHostVisibleProperty, "HOST_VISIBLE"}, true);
    } else {
        const size_t allocation_bytes = opencl_allocation_bytes(bytes, type);
        if (declared_bytes) {
            OPENVINO_ASSERT(declared_bytes == allocation_bytes,
                            "GFX OpenCL: remote tensor bytes mismatch (declared ",
                            declared_bytes,
                            ", required allocation ",
                            allocation_bytes,
                            ")");
        }
        const bool host_visible =
            find_any_bool(params, {kGfxHostVisibleProperty, "HOST_VISIBLE"}, true);
        tensor.buf = allocate_opencl_remote_buffer(context, type, bytes, host_visible);
    }

    result.tensor = tensor;
    result.properties[kGfxBufferProperty] = tensor.buf.buffer;
    result.properties[kGfxMemoryProperty] = tensor.buf.buffer;
    result.properties[kGfxBufferBytesProperty] = tensor.buf.size;
    result.properties[kGfxHostVisibleProperty] = tensor.buf.host_visible;
    result.release_fn = &release_opencl_remote_tensor;
    return result;
}

}  // namespace gfx_plugin
}  // namespace ov
