// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/metal_memory.hpp"

#include <cstdlib>
#include <string>
#include <vector>

#include "backends/metal/runtime/dtype.hpp"

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

namespace ov {
namespace gfx_plugin {

namespace {
#ifdef __OBJC__
class MetalDeviceCache {
public:
    MetalDeviceCache() {
        @autoreleasepool {
            NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
            if (devices) {
                for (id<MTLDevice> dev in devices) {
                    if (!dev) {
                        continue;
                    }
                    [dev retain];
                    m_devices.push_back(dev);
                    if (dev.name) {
                        m_names.emplace_back(std::string([[dev name] UTF8String]));
                    }
                }
                [devices release];
            }
        }
    }

    ~MetalDeviceCache() {
        for (id<MTLDevice> dev : m_devices) {
            [dev release];
        }
    }

    const std::vector<std::string>& names() const { return m_names; }

    MetalDeviceHandle device_by_id(int index) const {
        if (index >= 0 && index < static_cast<int>(m_devices.size())) {
            return m_devices[static_cast<size_t>(index)];
        }
        return nullptr;
    }

private:
    std::vector<id<MTLDevice>> m_devices;
    std::vector<std::string> m_names;
};

MetalDeviceCache& metal_device_cache() {
    static MetalDeviceCache cache;
    return cache;
}
#endif
}  // namespace

bool metal_safe_debug_enabled() {
    static bool cached = false;
    static bool inited = false;
    if (!inited) {
        const char* env = std::getenv("OV_GFX_SAFE_DEBUG");
        cached = (env && std::string(env) == "1");
        inited = true;
    }
    return cached;
}

std::vector<std::string> metal_get_device_names() {
    std::vector<std::string> names;
#ifdef __OBJC__
    names = metal_device_cache().names();
#else
    names.emplace_back("GFX");
#endif
    if (names.empty()) {
        names.emplace_back("GFX");
    }
    return names;
}

MetalDeviceHandle metal_get_device_by_id(int index) {
#ifdef __OBJC__
    return metal_device_cache().device_by_id(index);
#else
    (void)index;
    return nullptr;
#endif
}

MetalCommandQueueHandle metal_create_command_queue(MetalDeviceHandle device) {
#ifdef __OBJC__
    if (!device) {
        return nullptr;
    }
    @autoreleasepool {
        id<MTLDevice> dev = static_cast<id<MTLDevice>>(device);
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        return queue;
    }
#else
    (void)device;
    return nullptr;
#endif
}

void metal_release_command_queue(MetalCommandQueueHandle queue) {
#ifdef __OBJC__
    if (!queue) {
        return;
    }
    id<MTLCommandQueue> q = static_cast<id<MTLCommandQueue>>(queue);
    [q release];
#else
    (void)queue;
#endif
}

void metal_release_external_buffer(MetalBuffer& buf) {
#ifdef __OBJC__
    if (!buf.external || !buf.buffer || !buf.owned)
        return;
    auto mb = static_cast<id<MTLBuffer>>(buf.buffer);
    if (mb) {
        [mb release];
    }
    buf.buffer = nullptr;
#endif
}

void* metal_map_buffer(const MetalBuffer& buf) {
#ifdef __OBJC__
    if (!buf.buffer) {
        return nullptr;
    }
    if (!buf.host_visible && buf.storage_mode == static_cast<uint32_t>(MTLStorageModePrivate)) {
        return nullptr;
    }
    id<MTLBuffer> mtl_buf = static_cast<id<MTLBuffer>>(buf.buffer);
    return mtl_buf ? [mtl_buf contents] : nullptr;
#else
    (void)buf;
    return nullptr;
#endif
}

void metal_unmap_buffer(const MetalBuffer& /*buf*/) {
    // No-op for Metal; buffer contents are persistently mapped.
}

void metal_copy_buffer(MetalCommandQueueHandle queue,
                       const MetalBuffer& src,
                       const MetalBuffer& dst,
                       size_t bytes) {
#ifdef __OBJC__
    if (!queue || !src.buffer || !dst.buffer || bytes == 0) {
        return;
    }
    id<MTLCommandQueue> cq = static_cast<id<MTLCommandQueue>>(queue);
    id<MTLCommandBuffer> cb = [cq commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:static_cast<id<MTLBuffer>>(src.buffer)
           sourceOffset:0
               toBuffer:static_cast<id<MTLBuffer>>(dst.buffer)
      destinationOffset:0
                    size:bytes];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
#else
    (void)queue;
    (void)src;
    (void)dst;
    (void)bytes;
#endif
}

MetalTensor& MetalTensorMap::bind_input(size_t index, const ov::Tensor& host, MetalAllocatorCore& core) {
    auto& binding = m_inputs[index];
    binding.host = host;
    const size_t bytes = host.get_byte_size();
    OPENVINO_ASSERT(bytes > 0 && host.data(), "MetalTensorMap::bind_input: host tensor is empty");
    binding.dev = MetalTensor{core.wrap_shared(host.data(), bytes, host.get_element_type()),
                              host.get_shape(),
                              host.get_element_type()};
    return binding.dev;
}

MetalTensor& MetalTensorMap::bind_input_device(size_t index, const MetalTensor& dev) {
    auto& binding = m_inputs[index];
    binding.host = {};
    binding.dev = dev;
    if (binding.dev.expected_type == ov::element::dynamic)
        binding.dev.expected_type = dev.buf.type;
    return binding.dev;
}

MetalTensor& MetalTensorMap::ensure_output_device(size_t index,
                                                  const ov::Shape& shape,
                                                  ov::element::Type type,
                                                  MetalAllocator& alloc,
                                                  const MetalDeviceCaps& caps,
                                                  bool prefer_private) {
    auto& binding = m_outputs[index];
    MetalDType dtype = resolve_metal_dtype(type);
    const size_t elem_size = element_size(dtype);
    const size_t bytes = ov::shape_size(shape) * elem_size;

    BufferDesc desc;
    desc.bytes = bytes;
    desc.type = dtype.ov_type;
    desc.usage = BufferUsage::IO;
    desc.storage = (prefer_private && caps.prefer_private_intermediates) ? MetalStorage::Private : MetalStorage::Shared;
    desc.cpu_read = !prefer_private;
    desc.cpu_write = !prefer_private;

    binding.dev.buf = alloc.ensure_handle(binding.handle, desc, /*persistent=*/false);
    if (!binding.dev.buf.buffer) {
        OPENVINO_THROW("GFX: failed to allocate output buffer");
    }
    binding.dev.shape = shape;
    binding.dev.expected_type = type;
    binding.dev.prefer_private = prefer_private;
    return binding.dev;
}

bool MetalTensorMap::has_output_device(size_t index) const {
    return m_outputs.find(index) != m_outputs.end() && m_outputs.at(index).dev.buf.valid();
}

const MetalTensor& MetalTensorMap::get_output_device(size_t index) const {
    auto it = m_outputs.find(index);
    OPENVINO_ASSERT(it != m_outputs.end(), "MetalTensorMap: output device not found");
    return it->second.dev;
}

bool MetalTensorMap::has_host_for_output(size_t index) const {
    auto it = m_outputs.find(index);
    return it != m_outputs.end() && it->second.host;
}

ov::Tensor& MetalTensorMap::get_or_create_host_for_output(size_t index) {
    auto it = m_outputs.find(index);
    OPENVINO_ASSERT(it != m_outputs.end(), "MetalTensorMap: output binding missing");
    if (!it->second.host) {
        OPENVINO_THROW("GFX: host output access disabled (no CPU copies)");
    }
    return it->second.host;
}

void MetalTensorMap::bind_host_for_output(size_t index, ov::Tensor host) {
    auto& binding = m_outputs[index];
    binding.host = std::move(host);
}

MetalTensor& MetalTensorMap::bind_output_device(size_t index, const MetalTensor& dev) {
    auto& binding = m_outputs[index];
    binding.dev = dev;
    if (binding.dev.expected_type == ov::element::dynamic)
        binding.dev.expected_type = dev.buf.type;
    return binding.dev;
}

bool MetalTensorMap::has_input_device(size_t index) const {
    return m_inputs.find(index) != m_inputs.end() && m_inputs.at(index).dev.buf.valid();
}

MetalTensor& MetalTensorMap::get_input_device(size_t index) {
    auto it = m_inputs.find(index);
    OPENVINO_ASSERT(it != m_inputs.end(), "MetalTensorMap: input device not found");
    return it->second.dev;
}

const MetalTensor& MetalTensorMap::get_input_device(size_t index) const {
    auto it = m_inputs.find(index);
    OPENVINO_ASSERT(it != m_inputs.end(), "MetalTensorMap: input device not found");
    return it->second.dev;
}

bool MetalTensorMap::has_input_host(size_t index) const {
    auto it = m_inputs.find(index);
    return it != m_inputs.end() && it->second.host;
}

ov::Tensor& MetalTensorMap::get_input_host(size_t index) {
    auto it = m_inputs.find(index);
    OPENVINO_ASSERT(it != m_inputs.end(), "MetalTensorMap: input host not found");
    return it->second.host;
}

void MetalTensorMap::reset_inference(MetalAllocatorCore* core) {
    if (core) {
        for (auto& kv : m_inputs) {
            if (kv.second.dev.buf.external) {
                metal_release_external_buffer(kv.second.dev.buf);
            }
        }
        for (auto& kv : m_outputs) {
            if (kv.second.dev.buf.external) {
                metal_release_external_buffer(kv.second.dev.buf);
            }
        }
    }
    m_inputs.clear();
    for (auto& kv : m_outputs) {
        kv.second.host = {};
    }
}

namespace {
thread_local MetalAllocator* tls_alloc = nullptr;
thread_local MetalMemorySession* tls_session = nullptr;
}  // namespace

MetalBufferManager::MetalBufferManager(MetalAllocatorCore& core, MetalConstCache* const_cache)
    : m_core(core), m_const_cache(const_cache) {}

void MetalBufferManager::set_current_allocator(MetalAllocator* alloc) {
    tls_alloc = alloc;
    tls_session = nullptr;
}

void MetalBufferManager::set_current_session(MetalMemorySession* session) {
    tls_session = session;
    tls_alloc = session ? &session->allocator() : nullptr;
}

MetalBuffer MetalBufferManager::allocate(size_t size,
                                         ov::element::Type type,
                                         bool persistent,
                                         bool storageModePrivate,
                                         bool from_handle) {
    MetalAllocator* alloc = tls_alloc;
    BufferDesc desc;
    desc.bytes = size;
    desc.type = type;
    desc.storage = storageModePrivate ? MetalStorage::Private : MetalStorage::Shared;
    desc.usage = BufferUsage::Intermediate;
    MetalBuffer buf;
    if (alloc) {
        buf = alloc->allocate(desc, persistent);
    } else {
        buf = m_core.create_buffer(desc);
    }
    buf.from_handle = from_handle;
    return buf;
}

MetalBuffer MetalBufferManager::allocate_dynamic(size_t requested,
                                                 ov::element::Type type,
                                                 BufferHandle& handle,
                                                 bool persistent,
                                                 bool storageModePrivate) {
    MetalAllocator* alloc = tls_alloc;
    BufferDesc desc;
    desc.bytes = requested;
    desc.type = type;
    desc.storage = storageModePrivate ? MetalStorage::Private : MetalStorage::Shared;
    desc.usage = BufferUsage::Intermediate;
    MetalBuffer buf;
    if (alloc) {
        buf = alloc->ensure_handle(handle, desc, persistent);
    } else {
        buf = m_core.create_buffer(desc);
        handle.buf = buf;
        handle.capacity = buf.size;
    }
    buf.from_handle = true;
    return buf;
}

void MetalBufferManager::release(MetalBuffer&& buf) {
    MetalAllocator* alloc = tls_alloc;
    if (!alloc)
        return;
    alloc->release(std::move(buf));
}

void MetalBufferManager::reset_stats() {
    MetalAllocator* alloc = tls_alloc;
    if (!alloc)
        return;
    alloc->reset_stats();
}

const MetalMemoryStats& MetalBufferManager::stats() const {
    MetalAllocator* alloc = tls_alloc;
    if (!alloc)
        return m_dummy_stats;
    return alloc->stats();
}

MetalBuffer MetalBufferManager::wrap_shared(void* ptr, size_t bytes, ov::element::Type type) {
    return m_core.wrap_shared(ptr, bytes, type);
}

MetalBuffer MetalBufferManager::wrap_const(const std::string& key,
                                           const void* data,
                                           size_t bytes,
                                           ov::element::Type type,
                                           MetalStorage storage) {
    BufferDesc desc;
    desc.bytes = bytes;
    desc.type = type;
    desc.storage = storage;
    desc.usage = BufferUsage::Const;
    if (m_const_cache) {
        return m_const_cache->get_or_create(ConstKey{key}, data, bytes, desc);
    }
    return m_core.wrap_shared(const_cast<void*>(data), bytes, type);
}

}  // namespace gfx_plugin
}  // namespace ov
