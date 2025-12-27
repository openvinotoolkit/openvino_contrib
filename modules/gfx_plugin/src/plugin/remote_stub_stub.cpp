// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_stub.hpp"

#include "openvino/core/except.hpp"

#include "plugin/gfx_remote_utils.hpp"
#include "plugin/gfx_backend_config.hpp"
#include "plugin/gfx_remote_properties.hpp"
#include "backends/metal/plugin/properties.hpp"
#include "runtime/gpu_memory.hpp"
#include "backends/vulkan/runtime/gpu_memory.hpp"
#include "backends/vulkan/runtime/memory.hpp"

namespace ov {
namespace gfx_plugin {

GfxRemoteTensor::GfxRemoteTensor(const ov::element::Type& type,
                                 const ov::Shape& shape,
                                 const ov::AnyMap& params,
                                 const std::string& dev,
                                 const GpuTensor& tensor,
                                 bool owns_buffer)
    : m_type(type),
      m_shape(shape),
      m_params(params),
      m_device(dev),
      m_tensor(tensor),
      m_owns_buffer(owns_buffer) {
    recalc_strides();
}

GfxRemoteTensor::~GfxRemoteTensor() {
    if (m_owns_buffer && m_tensor.buf.backend == GpuBackend::Vulkan) {
        vulkan_free_buffer(m_tensor.buf);
    }
}

ov::SoPtr<ov::IRemoteTensor> GfxRemoteContext::create_tensor(const ov::element::Type& type,
                                                             const ov::Shape& shape,
                                                             const ov::AnyMap& params) {
    ov::AnyMap merged = m_params;
    merged.insert(params.begin(), params.end());
    merged[kGfxBackendProperty] = m_backend_name;

    if (m_backend != GpuBackend::Vulkan) {
        OPENVINO_THROW("GFX remote tensors are supported only with Vulkan backend");
    }
    if (!kGfxBackendVulkanAvailable) {
        OPENVINO_THROW("GFX Vulkan backend is not available");
    }

    const size_t bytes = std::max<size_t>(1, ov::shape_size(shape)) * type.size();
    GpuTensor tensor;
    tensor.shape = shape;
    tensor.expected_type = type;
    tensor.buf.type = type;
    tensor.buf.backend = GpuBackend::Vulkan;

    bool owns_buffer = false;
    void* external_buf = find_any_ptr(merged,
                                      {kVkBufferProperty, kVulkanBufferProperty, kGfxBufferProperty});
    void* external_mem = find_any_ptr(merged,
                                      {kVkMemoryProperty, kVulkanMemoryProperty, kGfxMemoryProperty});

    bool host_visible = find_any_bool(merged, {kHostVisibleProperty, kGfxHostVisibleProperty}, false);

    if (external_buf) {
        tensor.buf.buffer = external_buf;
        tensor.buf.heap = external_mem;
        tensor.buf.size = bytes;
        tensor.buf.host_visible = host_visible;
        tensor.buf.external = true;
        tensor.buf.from_handle = true;
    } else {
        VulkanGpuAllocator allocator;
        GpuBufferDesc desc;
        desc.bytes = bytes;
        desc.type = type;
        desc.usage = BufferUsage::IO;
        desc.cpu_read = host_visible;
        desc.cpu_write = host_visible;
        desc.prefer_device_local = !host_visible;
        tensor.buf = allocator.allocate(desc);
        owns_buffer = true;
    }

    auto t = std::make_shared<GfxRemoteTensor>(type, shape, merged, m_device, tensor, owns_buffer);
    return ov::SoPtr<ov::IRemoteTensor>{t, nullptr};
}

}  // namespace gfx_plugin
}  // namespace ov
