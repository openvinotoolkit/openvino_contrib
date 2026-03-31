// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/metal_memory.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
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

struct MetalQueueEntry {
    id<MTLCommandQueue> queue = nil;
    size_t refs = 0;
};

std::mutex& metal_queue_mutex() {
    static std::mutex m;
    return m;
}

std::unordered_map<id<MTLDevice>, MetalQueueEntry>& metal_queue_cache() {
    static std::unordered_map<id<MTLDevice>, MetalQueueEntry> cache;
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
        std::lock_guard<std::mutex> lock(metal_queue_mutex());
        auto& cache = metal_queue_cache();
        auto& entry = cache[dev];
        if (!entry.queue) {
            entry.queue = [dev newCommandQueue];
            if (!entry.queue) {
                cache.erase(dev);
                return nullptr;
            }
        }
        entry.refs += 1;
        return entry.queue;
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
    std::lock_guard<std::mutex> lock(metal_queue_mutex());
    auto& cache = metal_queue_cache();
    for (auto it = cache.begin(); it != cache.end(); ++it) {
        if (it->second.queue == q) {
            if (it->second.refs > 0) {
                it->second.refs -= 1;
            }
            return;
        }
    }
    // Keep command queues alive for the process lifetime to avoid device reloads.
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

namespace {
#ifdef __OBJC__
id<MTLCommandBuffer> resolve_command_buffer_from_handle(MetalCommandQueueHandle execution_context, bool* owns_command_buffer) {
    if (owns_command_buffer) {
        *owns_command_buffer = false;
    }
    if (!execution_context) {
        return nil;
    }
    id object = static_cast<id>(execution_context);
    if ([object conformsToProtocol:@protocol(MTLCommandBuffer)]) {
        return static_cast<id<MTLCommandBuffer>>(object);
    }
    if ([object conformsToProtocol:@protocol(MTLCommandQueue)]) {
        if (owns_command_buffer) {
            *owns_command_buffer = true;
        }
        return [static_cast<id<MTLCommandQueue>>(object) commandBuffer];
    }
    return nil;
}
#endif
}  // namespace

void metal_copy_buffer(MetalCommandQueueHandle queue,
                       const MetalBuffer& src,
                       const MetalBuffer& dst,
                       size_t bytes) {
    GpuBufferCopyRegion region{};
    region.bytes = bytes;
    metal_copy_buffer_regions(queue, src, dst, &region, 1);
}

void metal_copy_buffer_regions(MetalCommandQueueHandle execution_context,
                               const MetalBuffer& src,
                               const MetalBuffer& dst,
                               const GpuBufferCopyRegion* regions,
                               size_t region_count) {
#ifdef __OBJC__
    if (!execution_context || !src.buffer || !dst.buffer || !regions || region_count == 0) {
        return;
    }
    bool owns_command_buffer = false;
    id<MTLCommandBuffer> cb = resolve_command_buffer_from_handle(execution_context, &owns_command_buffer);
    if (!cb) {
        return;
    }
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    for (size_t i = 0; i < region_count; ++i) {
        const auto& region = regions[i];
        if (region.bytes == 0) {
            continue;
        }
        [blit copyFromBuffer:static_cast<id<MTLBuffer>>(src.buffer)
               sourceOffset:src.offset + region.src_offset
                   toBuffer:static_cast<id<MTLBuffer>>(dst.buffer)
          destinationOffset:dst.offset + region.dst_offset
                        size:region.bytes];
    }
    [blit endEncoding];
    if (owns_command_buffer) {
        [cb commit];
        [cb waitUntilCompleted];
    }
#else
    (void)execution_context;
    (void)src;
    (void)dst;
    (void)regions;
    (void)region_count;
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
    GpuBufferDesc desc{};
    desc.bytes = size;
    desc.type = type;
    desc.usage = BufferUsage::Intermediate;
    desc.prefer_device_local = storageModePrivate;
    return allocate(desc, persistent, from_handle);
}

MetalBuffer MetalBufferManager::allocate(const GpuBufferDesc& desc, bool persistent, bool from_handle) {
    MetalAllocator* alloc = tls_alloc;
    validate_gpu_buffer_desc(desc, "GFX Metal");
    BufferDesc mdesc;
    mdesc.bytes = desc.bytes;
    mdesc.type = desc.type;
    mdesc.usage = desc.usage;
    const bool host_visible = desc.cpu_read || desc.cpu_write || !desc.prefer_device_local;
    mdesc.storage = host_visible ? MetalStorage::Shared : MetalStorage::Private;
    mdesc.cpu_read = desc.cpu_read;
    mdesc.cpu_write = desc.cpu_write;
    mdesc.label = desc.label;
    MetalBuffer buf;
    if (alloc) {
        buf = alloc->allocate(mdesc, persistent);
    } else {
        buf = m_core.create_buffer(mdesc);
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

std::optional<GpuExecutionDeviceInfo> MetalBufferManager::query_execution_device_info() const {
    GpuExecutionDeviceInfo info{};
    const auto caps = query_metal_device_caps(m_core.device());
    info.backend = GpuBackend::Metal;
    info.device_family = GpuDeviceFamily::Apple;
    info.preferred_simd_width = std::max<uint32_t>(caps.preferred_simd_width, 1u);
    info.subgroup_size = info.preferred_simd_width;
    info.max_total_threads_per_group = std::max<uint32_t>(caps.max_total_threads_per_threadgroup, 1u);
    info.max_threads_per_group = {std::max<uint32_t>(caps.max_threads_per_threadgroup_x, 1u),
                                  std::max<uint32_t>(caps.max_threads_per_threadgroup_y, 1u),
                                  std::max<uint32_t>(caps.max_threads_per_threadgroup_z, 1u)};

    std::ostringstream os;
    os << "metal:" << gpu_device_family_name(info.device_family) << ':' << m_core.device() << ':'
       << info.preferred_simd_width << ':' << info.max_total_threads_per_group
       << ':' << info.max_threads_per_group[0] << ':' << info.max_threads_per_group[1] << ':'
       << info.max_threads_per_group[2];
    info.device_key = os.str();
    return info;
}

MetalBuffer MetalBufferManager::wrap_const(const std::string& key,
                                           const void* data,
                                           size_t bytes,
                                           ov::element::Type type) {
    if (bytes == 0) {
        return {};
    }
    const size_t aligned = (bytes + 3u) & ~size_t{3u};
    BufferDesc desc;
    desc.bytes = aligned;
    desc.type = type;
    desc.storage = MetalStorage::Private;
    desc.usage = BufferUsage::Const;
    OPENVINO_ASSERT(m_const_cache, "GFX: const cache is required for Metal constants");
    if (aligned == bytes) {
        return m_const_cache->get_or_create(key, data, bytes, desc);
    }
    std::vector<uint8_t> padded(aligned, 0);
    std::memcpy(padded.data(), data, bytes);
    return m_const_cache->get_or_create(key, padded.data(), aligned, desc);
}

}  // namespace gfx_plugin
}  // namespace ov
