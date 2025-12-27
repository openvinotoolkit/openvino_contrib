// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/memory.hpp"

#include "openvino/core/except.hpp"
#include "backends/metal/memory/allocator.hpp"
#include "backends/metal/memory/const_cache.hpp"
#include "backends/metal/memory/device_caps.hpp"
#include "backends/metal/memory/heap_pool.hpp"
#include "backends/metal/memory/staging_pool.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
[[noreturn]] void throw_metal_unavailable() {
    OPENVINO_THROW("GFX Metal backend is not available in this build");
}
}  // namespace

bool metal_safe_debug_enabled() {
    return false;
}

std::vector<std::string> metal_get_device_names() {
    return {};
}

MetalDeviceHandle metal_get_device_by_id(int /*index*/) {
    return nullptr;
}

MetalCommandQueueHandle metal_create_command_queue(MetalDeviceHandle /*device*/) {
    return nullptr;
}

void metal_release_command_queue(MetalCommandQueueHandle /*queue*/) {}

void metal_release_external_buffer(MetalBuffer& /*buf*/) {}

void* metal_map_buffer(const MetalBuffer& /*buf*/) {
    throw_metal_unavailable();
}

void metal_unmap_buffer(const MetalBuffer& /*buf*/) {}

void metal_copy_buffer(MetalCommandQueueHandle /*queue*/,
                       const MetalBuffer& /*src*/,
                       const MetalBuffer& /*dst*/,
                       size_t /*bytes*/) {
    throw_metal_unavailable();
}

MetalDeviceCaps query_metal_device_caps(MetalDeviceHandle /*device*/) {
    return {};
}

MetalAllocatorCore::MetalAllocatorCore(MetalDeviceHandle /*device*/, MetalDeviceCaps /*caps*/) {
    throw_metal_unavailable();
}

MetalBuffer MetalAllocatorCore::create_buffer(const BufferDesc& /*desc*/) {
    throw_metal_unavailable();
}

MetalBuffer MetalAllocatorCore::create_buffer_from_heap(MetalHeapHandle /*heap*/, const BufferDesc& /*desc*/) {
    throw_metal_unavailable();
}

MetalHeapHandle MetalAllocatorCore::create_heap(MetalStorage /*storage*/, size_t /*heap_bytes*/, uint32_t /*options*/) {
    throw_metal_unavailable();
}

MetalBuffer MetalAllocatorCore::wrap_shared(void* /*ptr*/, size_t /*bytes*/, ov::element::Type /*type*/) {
    throw_metal_unavailable();
}

void MetalAllocatorCore::release_buffer(MetalBuffer& /*buf*/) {
    throw_metal_unavailable();
}

MetalHeapPool::MetalHeapPool(MetalAllocatorCore& core) : m_core(core) {
    throw_metal_unavailable();
}

MetalHeapPool::~MetalHeapPool() = default;

MetalBuffer MetalHeapPool::alloc_private_from_heap(const BufferDesc& /*desc*/) {
    throw_metal_unavailable();
}

void MetalHeapPool::on_release(MetalBuffer& /*buf*/) {}

MetalStagingPool::MetalStagingPool(MetalAllocatorCore& core) : m_core(core) {
    throw_metal_unavailable();
}

MetalBuffer MetalStagingPool::allocate(size_t /*bytes*/, const char* /*label*/) {
    throw_metal_unavailable();
}

void MetalStagingPool::release(MetalBuffer&& /*buf*/) {}

MetalAllocator::MetalAllocator(MetalAllocatorCore& core,
                               MetalHeapPool& heaps,
                               MetalFreeList& freelist,
                               MetalStagingPool& staging,
                               MetalDeviceCaps caps)
    : m_core(core),
      m_heaps(heaps),
      m_freelist(freelist),
      m_staging(staging),
      m_caps(caps) {
    throw_metal_unavailable();
}

MetalBuffer MetalAllocator::allocate(const BufferDesc& /*desc*/, bool /*persistent*/) {
    throw_metal_unavailable();
}

MetalBuffer MetalAllocator::allocate_staging(size_t /*bytes*/, const char* /*label*/) {
    throw_metal_unavailable();
}

MetalBuffer MetalAllocator::ensure_handle(BufferHandle& /*handle*/, const BufferDesc& /*desc*/, bool /*persistent*/) {
    throw_metal_unavailable();
}

void MetalAllocator::release(MetalBuffer&& /*buf*/) {}

void MetalAllocator::set_profiler(MetalProfiler* /*profiler*/, bool /*detailed*/) {}

MetalConstCache::MetalConstCache(MetalAllocator& persistent_alloc) : m_alloc(persistent_alloc) {
    throw_metal_unavailable();
}

MetalConstCache::~MetalConstCache() = default;

const MetalBuffer& MetalConstCache::get_or_create(const ConstKey& /*key*/,
                                                  const void* /*data*/,
                                                  size_t /*bytes*/,
                                                  const BufferDesc& /*desc*/) {
    throw_metal_unavailable();
}

MetalBufferManager::MetalBufferManager(MetalAllocatorCore& core, MetalConstCache* const_cache)
    : m_core(core), m_const_cache(const_cache) {
    throw_metal_unavailable();
}

MetalBuffer MetalBufferManager::allocate(size_t /*size*/,
                                         ov::element::Type /*type*/,
                                         bool /*persistent*/,
                                         bool /*storageModePrivate*/,
                                         bool /*from_handle*/) {
    throw_metal_unavailable();
}

MetalBuffer MetalBufferManager::allocate_dynamic(size_t /*requested*/,
                                                 ov::element::Type /*type*/,
                                                 BufferHandle& /*handle*/,
                                                 bool /*persistent*/,
                                                 bool /*storageModePrivate*/) {
    throw_metal_unavailable();
}

void MetalBufferManager::release(MetalBuffer&& /*buf*/) {}

void MetalBufferManager::reset_stats() {}

const MetalMemoryStats& MetalBufferManager::stats() const {
    static MetalMemoryStats stats{};
    return stats;
}

MetalBuffer MetalBufferManager::wrap_shared(void* /*ptr*/, size_t /*bytes*/, ov::element::Type /*type*/) {
    throw_metal_unavailable();
}

MetalBuffer MetalBufferManager::wrap_const(const std::string& /*key*/,
                                           const void* /*data*/,
                                           size_t /*bytes*/,
                                           ov::element::Type /*type*/,
                                           MetalStorage /*storage*/) {
    throw_metal_unavailable();
}

void MetalBufferManager::set_current_allocator(MetalAllocator* /*alloc*/) {}

void MetalBufferManager::set_current_session(MetalMemorySession* /*session*/) {}

}  // namespace gfx_plugin
}  // namespace ov
