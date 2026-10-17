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
#include "backends/metal/runtime/metal_command_encoder.hpp"

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
      NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
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

  const std::vector<std::string> &names() const { return m_names; }

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

MetalDeviceCache &metal_device_cache() {
  static MetalDeviceCache cache;
  return cache;
}

struct MetalQueueEntry {
  id<MTLCommandQueue> queue = nil;
  size_t refs = 0;
};

std::mutex &metal_queue_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<id<MTLDevice>, MetalQueueEntry> &metal_queue_cache() {
  static std::unordered_map<id<MTLDevice>, MetalQueueEntry> cache;
  return cache;
}
#endif
} // namespace

bool metal_safe_debug_enabled() {
  static bool cached = false;
  static bool inited = false;
  if (!inited) {
    const char *env = std::getenv("OV_GFX_SAFE_DEBUG");
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
    auto &cache = metal_queue_cache();
    auto &entry = cache[dev];
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
  auto &cache = metal_queue_cache();
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

void metal_release_external_buffer(MetalBuffer &buf) {
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

void *metal_map_buffer(const MetalBuffer &buf) {
#ifdef __OBJC__
  if (!buf.buffer) {
    return nullptr;
  }
  if (!buf.host_visible &&
      buf.storage_mode == static_cast<uint32_t>(MTLStorageModePrivate)) {
    return nullptr;
  }
  id<MTLBuffer> mtl_buf = static_cast<id<MTLBuffer>>(buf.buffer);
  return mtl_buf ? [mtl_buf contents] : nullptr;
#else
  (void)buf;
  return nullptr;
#endif
}

void metal_unmap_buffer(const MetalBuffer & /*buf*/) {
  // No-op for Metal; buffer contents are persistently mapped.
}

namespace {
#ifdef __OBJC__
id<MTLCommandBuffer>
resolve_command_buffer_from_handle(MetalCommandQueueHandle execution_context,
                                   bool *owns_command_buffer) {
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
} // namespace

void metal_copy_buffer(MetalCommandQueueHandle queue, const MetalBuffer &src,
                       const MetalBuffer &dst, size_t bytes) {
  GpuBufferCopyRegion region{};
  region.bytes = bytes;
  metal_copy_buffer_regions(queue, src, dst, &region, 1);
}

void metal_copy_buffer_regions(MetalCommandQueueHandle execution_context,
                               const MetalBuffer &src, const MetalBuffer &dst,
                               const GpuBufferCopyRegion *regions,
                               size_t region_count) {
#ifdef __OBJC__
  if (!execution_context || !src.buffer || !dst.buffer || !regions ||
      region_count == 0) {
    return;
  }
  bool owns_command_buffer = false;
  id<MTLCommandBuffer> cb = resolve_command_buffer_from_handle(
      execution_context, &owns_command_buffer);
  if (!cb) {
    return;
  }
  metal_end_compute_encoder(reinterpret_cast<GpuCommandBufferHandle>(cb));
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  for (size_t i = 0; i < region_count; ++i) {
    const auto &region = regions[i];
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
    metal_end_compute_encoder(reinterpret_cast<GpuCommandBufferHandle>(cb));
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

namespace {
thread_local MetalAllocator *tls_alloc = nullptr;
thread_local MetalMemorySession *tls_session = nullptr;
} // namespace

MetalBufferManager::MetalBufferManager(MetalAllocatorCore &core,
                                       MetalConstCache *const_cache)
    : m_core(core), m_const_cache(const_cache) {}

void MetalBufferManager::set_current_allocator(MetalAllocator *alloc) {
  tls_alloc = alloc;
  tls_session = nullptr;
}

void MetalBufferManager::set_current_session(MetalMemorySession *session) {
  tls_session = session;
  tls_alloc = session ? &session->allocator() : nullptr;
}

MetalBuffer MetalBufferManager::allocate(size_t size, ov::element::Type type,
                                         bool persistent,
                                         bool storageModePrivate,
                                         bool from_handle) {
  MetalAllocator *alloc = tls_alloc;
  GpuBufferDesc desc{};
  desc.bytes = size;
  desc.type = type;
  desc.usage = BufferUsage::Intermediate;
  desc.prefer_device_local = storageModePrivate;
  return allocate(desc, persistent, from_handle);
}

MetalBuffer MetalBufferManager::allocate(const GpuBufferDesc &desc,
                                         bool persistent, bool from_handle) {
  MetalAllocator *alloc = tls_alloc;
  validate_gpu_buffer_desc(desc, "GFX Metal");
  BufferDesc mdesc;
  mdesc.bytes = desc.bytes;
  mdesc.type = desc.type;
  mdesc.usage = desc.usage;
  const bool host_visible =
      desc.cpu_read || desc.cpu_write || !desc.prefer_device_local;
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
                                                 BufferHandle &handle,
                                                 bool persistent,
                                                 bool storageModePrivate) {
  MetalAllocator *alloc = tls_alloc;
  BufferDesc desc;
  desc.bytes = requested;
  desc.type = type;
  desc.storage =
      storageModePrivate ? MetalStorage::Private : MetalStorage::Shared;
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

void MetalBufferManager::release(MetalBuffer &&buf) {
  MetalAllocator *alloc = tls_alloc;
  if (!alloc)
    return;
  alloc->release(std::move(buf));
}

void MetalBufferManager::reset_stats() {
  MetalAllocator *alloc = tls_alloc;
  if (!alloc)
    return;
  alloc->reset_stats();
}

const MetalMemoryStats &MetalBufferManager::stats() const {
  MetalAllocator *alloc = tls_alloc;
  if (!alloc)
    return m_dummy_stats;
  return alloc->stats();
}

MetalBuffer MetalBufferManager::wrap_shared(void *ptr, size_t bytes,
                                            ov::element::Type type) {
  return m_core.wrap_shared(ptr, bytes, type);
}

std::optional<GpuExecutionDeviceInfo>
MetalBufferManager::query_execution_device_info() const {
  GpuExecutionDeviceInfo info{};
  const auto caps = query_metal_device_caps(m_core.device());
  info.backend = GpuBackend::Metal;
  info.device_family = GpuDeviceFamily::Apple;
#ifdef __OBJC__
  if (auto device = static_cast<id<MTLDevice>>(m_core.device())) {
    NSString *name = [device name];
    if (name) {
      info.device_name = [name UTF8String];
    }
  }
#endif
  if (info.device_name.empty()) {
    info.device_name = "apple_metal";
  }
  info.preferred_simd_width = std::max<uint32_t>(caps.preferred_simd_width, 1u);
  info.subgroup_size = info.preferred_simd_width;
  info.max_total_threads_per_group =
      std::max<uint32_t>(caps.max_total_threads_per_threadgroup, 1u);
  info.max_threads_per_group = {
      std::max<uint32_t>(caps.max_threads_per_threadgroup_x, 1u),
      std::max<uint32_t>(caps.max_threads_per_threadgroup_y, 1u),
      std::max<uint32_t>(caps.max_threads_per_threadgroup_z, 1u)};
  info.min_storage_buffer_offset_alignment = 16;
  info.non_coherent_atom_size = 1;
  info.supports_storage_buffer_16bit = true;
  info.supports_shader_float16 = true;
  info.supports_conv_output_channel_blocking = true;
  info.supports_conv_channel_block_spatial_tiling = true;
  info.parallelism_profile.sort_matmul_tiles_by_shape = false;
  info.parallelism_profile.supports_conv_output_channel_blocking = true;
  info.parallelism_profile.supports_conv_channel_block_spatial_tiling = true;
  info.parallelism_profile.chunk_dispatch = make_metal_chunk_dispatch_profile();

  std::ostringstream os;
  os << "metal:" << gpu_device_family_name(info.device_family) << ':'
     << info.device_name << ':' << m_core.device() << ':'
     << info.preferred_simd_width << ':'
     << info.max_total_threads_per_group << ':' << info.max_threads_per_group[0]
     << ':' << info.max_threads_per_group[1] << ':'
     << info.max_threads_per_group[2];
  info.device_key = os.str();
  info.parallelism_profile.profile_key = info.device_key + ":parallelism";
  return info;
}

MetalBuffer MetalBufferManager::wrap_const(const std::string &key,
                                           const void *data, size_t bytes,
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
  OPENVINO_ASSERT(m_const_cache,
                  "GFX: const cache is required for Metal constants");
  if (aligned == bytes) {
    return m_const_cache->get_or_create(key, data, bytes, desc);
  }
  std::vector<uint8_t> padded(aligned, 0);
  std::memcpy(padded.data(), data, bytes);
  return m_const_cache->get_or_create(key, padded.data(), aligned, desc);
}

} // namespace gfx_plugin
} // namespace ov
