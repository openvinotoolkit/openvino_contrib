// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/except.hpp"
#include "openvino/runtime/tensor.hpp"
#include "runtime/gfx_backend_utils.hpp"
#include "runtime/gpu_buffer_pool.hpp"
#include "runtime/gpu_memory.hpp"
#include "runtime/gpu_tensor.hpp"
#include "backends/metal/runtime/memory.hpp"
#include "infer_pipeline.hpp"

namespace ov {
namespace gfx_plugin {

inline GpuTensor bind_host_input(const ov::Tensor& host,
                                 GpuBackend backend,
                                 MetalAllocatorCore* metal_core,
                                 GpuBufferPool* pool,
                                 BufferHandle* device_handle,
                                 BufferHandle* staging_handle,
                                 const char* error_prefix) {
    OPENVINO_ASSERT(host && host.data(), error_prefix, ": input host tensor is empty");
    const size_t bytes = host.get_byte_size();

    GpuTensor tensor{};
    tensor.shape = host.get_shape();
    tensor.expected_type = host.get_element_type();
    tensor.buf.type = host.get_element_type();
    tensor.buf.backend = backend;

    if (bytes == 0) {
        return tensor;
    }

    if (backend == GpuBackend::Metal) {
        OPENVINO_ASSERT(metal_core, error_prefix, ": Metal allocator core is null");
        tensor.buf = metal_core->wrap_shared(host.data(), bytes, host.get_element_type());
        tensor.buf.backend = GpuBackend::Metal;
        tensor.buf.host_visible = true;
        tensor.prefer_private = false;
        return tensor;
    }

    OPENVINO_ASSERT(pool && device_handle && staging_handle,
                    error_prefix, ": Vulkan input staging handles are missing");

    GpuBufferDesc staging_desc;
    staging_desc.bytes = bytes;
    staging_desc.type = host.get_element_type();
    staging_desc.usage = BufferUsage::Staging;
    staging_desc.cpu_read = false;
    staging_desc.cpu_write = true;
    staging_desc.prefer_device_local = false;
    GpuBuffer staging = pool->ensure(*staging_handle, staging_desc);
    gpu_copy_from_host(staging, host.data(), bytes);

    GpuBufferDesc device_desc;
    device_desc.bytes = bytes;
    device_desc.type = host.get_element_type();
    device_desc.usage = BufferUsage::IO;
    device_desc.cpu_read = false;
    device_desc.cpu_write = false;
    device_desc.prefer_device_local = true;
    GpuBuffer buf = pool->ensure(*device_handle, device_desc);
    if (bytes && staging.valid() && buf.valid()) {
        gpu_copy_buffer(nullptr, staging, buf, bytes);
    }

    tensor.buf = buf;
    tensor.buf.backend = backend;
    tensor.prefer_private = true;
    return tensor;
}

struct OutputBindingResult {
    GpuTensor device_tensor;
    ov::Tensor host_tensor;
};

inline OutputBindingResult bind_host_output(const GpuTensor& dev,
                                            const OutputViewInfo& info,
                                            const ov::Tensor* host_override,
                                            GpuBackend backend,
                                            MetalAllocatorCore* metal_core,
                                            MetalAllocator* metal_allocator,
                                            GpuCommandQueueHandle metal_queue,
                                            GpuBufferPool* pool,
                                            BufferHandle* staging_handle,
                                            const char* error_prefix) {
    OutputBindingResult result{};
    result.device_tensor = dev;

    size_t bytes = 0;
    if (info.type != ov::element::dynamic) {
        bytes = info.type.size();
        for (auto d : info.shape) {
            bytes *= d;
        }
    }

    if (backend == GpuBackend::Metal) {
        OPENVINO_ASSERT(metal_core && metal_allocator,
                        error_prefix, ": Metal allocator is not available");

        if (host_override && *host_override) {
            MetalBuffer shared = metal_core->wrap_shared(host_override->data(), bytes, info.type);
            if (bytes && dev.buf.buffer != shared.buffer) {
                gpu_copy_buffer(metal_queue, dev.buf, shared, bytes);
            }
            result.device_tensor = dev;
            result.device_tensor.buf = shared;
            result.device_tensor.expected_type = info.type;
            result.device_tensor.shape = info.shape;
            result.device_tensor.prefer_private = false;
            result.host_tensor = *host_override;
            return result;
        }

        if (dev.buf.host_visible && dev.buf.buffer) {
            void* ptr = metal_map_buffer(dev.buf);
            OPENVINO_ASSERT(ptr, error_prefix, ": shared output buffer has no CPU pointer");
            result.host_tensor = ov::Tensor(info.type, info.shape, ptr);
            result.device_tensor.expected_type = info.type;
            if (result.device_tensor.shape.empty()) {
                result.device_tensor.shape = info.shape;
            }
            result.device_tensor.prefer_private = false;
            return result;
        }

        if (bytes) {
            BufferDesc desc;
            desc.bytes = bytes;
            desc.type = info.type;
            desc.usage = BufferUsage::IO;
            desc.storage = MetalStorage::Shared;
            desc.cpu_read = true;
            desc.cpu_write = true;
            MetalBuffer shared = metal_allocator->allocate(desc, /*persistent=*/false);
            gpu_copy_buffer(metal_queue, dev.buf, shared, bytes);
            result.device_tensor = dev;
            result.device_tensor.buf = shared;
            result.device_tensor.expected_type = info.type;
            result.device_tensor.shape = info.shape;
            result.device_tensor.prefer_private = false;
            void* ptr = metal_map_buffer(shared);
            OPENVINO_ASSERT(ptr, error_prefix, ": shared output buffer has no CPU pointer");
            result.host_tensor = ov::Tensor(info.type, info.shape, ptr);
            return result;
        }

        result.device_tensor.expected_type = info.type;
        result.device_tensor.shape = info.shape;
        result.device_tensor.prefer_private = false;
        if (host_override && *host_override) {
            result.host_tensor = *host_override;
        } else {
            result.host_tensor = ov::Tensor(info.type, info.shape);
        }
        return result;
    }

    OPENVINO_ASSERT(pool && staging_handle,
                    error_prefix, ": Vulkan output staging handle is missing");

    ov::Tensor host = host_override && *host_override ? *host_override : ov::Tensor(info.type, info.shape);
    if (bytes) {
        GpuBufferDesc staging_desc;
        staging_desc.bytes = bytes;
        staging_desc.type = info.type;
        staging_desc.usage = BufferUsage::Staging;
        staging_desc.cpu_read = true;
        staging_desc.cpu_write = false;
        staging_desc.prefer_device_local = false;
        GpuBuffer staging = pool->ensure(*staging_handle, staging_desc);
        if (dev.buf.buffer != staging.buffer) {
            gpu_copy_buffer(nullptr, dev.buf, staging, bytes);
        }
        gpu_copy_to_host(staging, host.data(), bytes);
    }
    result.host_tensor = host;
    result.device_tensor.expected_type = info.type;
    if (result.device_tensor.shape.empty()) {
        result.device_tensor.shape = info.shape;
    }
    return result;
}

inline bool init_stage_output_desc(GpuBackend backend,
                                   InferStage& stage,
                                   size_t out_idx,
                                   GpuTensor& out_ref,
                                   GpuBufferDesc& desc,
                                   bool is_model_output,
                                   bool skip_view_ops,
                                   const char* error_prefix) {
    if (skip_view_ops && is_view_op(stage)) {
        return false;
    }
    const auto out_shape = ensure_stage_output_shape(stage, out_idx);
    if (out_shape.empty()) {
        return false;
    }
    const auto et = resolve_stage_output_type(stage, out_ref, out_idx, error_prefix);
    size_t bytes = et.size();
    for (auto d : out_shape) {
        bytes *= d;
    }

    if (backend == GpuBackend::Metal) {
        out_ref.prefer_private = !is_model_output;
        desc.cpu_read = !out_ref.prefer_private;
        desc.cpu_write = !out_ref.prefer_private;
        desc.prefer_device_local = out_ref.prefer_private;
    } else {
        out_ref.prefer_private = true;
        desc.cpu_read = false;
        desc.cpu_write = false;
        desc.prefer_device_local = true;
    }

    desc.bytes = bytes;
    desc.type = et;
    desc.usage = BufferUsage::Intermediate;
    return true;
}

inline void release_stage_output_handles(std::vector<BufferHandle>& handles, GpuBufferPool& pool) {
    for (auto& handle : handles) {
        pool.release(handle);
    }
}

inline void prepare_stage_output_handles(std::vector<std::vector<BufferHandle>>& stage_handles,
                                         const std::vector<InferStage>& pipeline,
                                         GpuBufferPool& pool,
                                         bool release_view_only) {
    if (stage_handles.size() != pipeline.size()) {
        stage_handles.assign(pipeline.size(), {});
    }
    for (size_t stage_idx = 0; stage_idx < pipeline.size(); ++stage_idx) {
        const auto& stage = pipeline[stage_idx];
        if (release_view_only && !is_view_op(stage)) {
            continue;
        }
        auto& handles = stage_handles[stage_idx];
        if (handles.size() < stage.outputs.size()) {
            handles.resize(stage.outputs.size());
        }
        release_stage_output_handles(handles, pool);
    }
}

template <typename OutputLookupFn,
          typename HostOverrideFn,
          typename RemoteSetterFn,
          typename DeviceSetterFn>
inline void bind_outputs_common(const std::vector<ov::Output<const ov::Node>>& public_outputs,
                                const std::shared_ptr<const ov::Model>& runtime_model,
                                const std::unordered_map<const ov::Node*, size_t>& node_map,
                                const std::unordered_map<const ov::Node*, size_t>& param_map,
                                std::vector<InferStage>& pipeline,
                                OutputLookupFn output_input_lookup,
                                std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_outputs,
                                HostOverrideFn host_override,
                                RemoteSetterFn remote_setter,
                                DeviceSetterFn device_setter,
                                bool allow_missing,
                                bool allow_fallback_one,
                                const char* error_prefix) {
    for_each_output_tensor(public_outputs,
                           runtime_model,
                           node_map,
                           param_map,
                           pipeline,
                           output_input_lookup,
                           remote_outputs,
                           host_override,
                           remote_setter,
                           device_setter,
                           allow_missing,
                           allow_fallback_one,
                           error_prefix);
}

}  // namespace gfx_plugin
}  // namespace ov
