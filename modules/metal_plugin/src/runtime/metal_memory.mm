// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/metal_memory.hpp"

#include <cstring>
#include <utility>
#include <unordered_set>
#include <algorithm>

#include "openvino/core/except.hpp"

namespace ov {
namespace metal_plugin {

namespace {
constexpr size_t kAlignment = 256;  // good default for MTLBuffer alignment
constexpr size_t kDefaultHeapSize = 64 * 1024 * 1024;  // 64MB per heap

MTLResourceOptions choose_options(bool storageModePrivate) {
#ifdef __OBJC__
    return storageModePrivate ? MTLResourceStorageModePrivate : MTLResourceStorageModeShared;
#else
    (void)storageModePrivate;
    return 0;
#endif
}
}  // namespace

MetalBufferManager::MetalBufferManager(MetalDeviceHandle device) : m_device(device) {
#ifdef __OBJC__
    if (auto dev = static_cast<id<MTLDevice>>(m_device)) {
        m_copy_queue = [dev newCommandQueue];
    }
#endif
}

MetalBufferManager::~MetalBufferManager() {
#ifdef __OBJC__
    std::unordered_set<void*> released;
    auto release_buf = [&](const MetalBuffer& b) {
        if (!b.buffer) return;
        void* key = (__bridge void*)static_cast<id<MTLBuffer>>(b.buffer);
        if (released.count(key)) return;
        [static_cast<id<MTLBuffer>>(b.buffer) release];
        released.insert(key);
    };
    for (const auto& b : m_live_persistent) release_buf(b);
    for (const auto& b : m_live_handle) release_buf(b);
    for (const auto& b : m_live_inference) release_buf(b);
    for (const auto& kv : m_free_inference) {
        for (const auto& b : kv.second) release_buf(b);
    }
    for (auto h : m_heaps) {
        if (h) [static_cast<id<MTLHeap>>(h) release];
    }
    if (auto q = static_cast<id<MTLCommandQueue>>(m_copy_queue)) {
        [q release];
        m_copy_queue = nullptr;
    }
#endif
}

size_t MetalBufferManager::align_size(size_t size) const {
    const size_t mask = kAlignment - 1;
    if (size == 0)
        return kAlignment;
    return (size + mask) & ~mask;
}

size_t MetalBufferManager::bucket_size(size_t size) const {
    size = align_size(size);
    // Round up to nearest power of two to improve reuse and reduce fragmentation.
    size_t b = kAlignment;
    while (b < size) {
        b <<= 1;
    }
    return b;
}

MetalBuffer MetalBufferManager::allocate(size_t size,
                                         ov::element::Type type,
                                         bool persistent,
                                         bool storageModePrivate,
                                         bool from_handle) {
    const size_t bucket = bucket_size(size);
#ifdef __OBJC__
    const uint32_t desired_mode = storageModePrivate ? static_cast<uint32_t>(MTLStorageModePrivate)
                                                     : static_cast<uint32_t>(MTLStorageModeShared);
#endif
    FreeKey key{bucket, desired_mode};
    auto& free_list = m_free_inference[key];
    if (!persistent) {
        for (auto it = free_list.begin(); it != free_list.end(); ++it) {
#ifdef __OBJC__
            if (it->storage_mode != desired_mode)
                continue;
#endif
            MetalBuffer buf = *it;
            free_list.erase(it);
            buf.type = type;
            buf.persistent = false;
            buf.from_handle = from_handle;
            if (from_handle) {
                m_live_handle.push_back(buf);
            } else {
                m_live_inference.push_back(buf);
            }
            m_stats.reused_bytes += bucket;
            return buf;
        }
    }

    MetalBuffer out;
#ifdef __OBJC__
    auto dev = static_cast<id<MTLDevice>>(m_device);
    if (!dev) {
        OPENVINO_THROW("MetalBufferManager: device is null");
    }
    // For non-shared, non-persistent temporaries try to suballocate from a private heap.
    auto try_heap = [&]() -> id<MTLBuffer> {
        if (!storageModePrivate || persistent)
            return nil;
        for (auto heap : m_heaps) {
            if (!heap) continue;
            if ([heap maxAvailableSizeWithAlignment:kAlignment] >= bucket) {
                id<MTLBuffer> hbuf = [heap newBufferWithLength:bucket options:MTLResourceStorageModePrivate];
                if (hbuf) {
                    out.heap = heap;
                    return hbuf;
                }
            }
        }
        // create new heap if needed
        MTLHeapDescriptor* desc = [MTLHeapDescriptor new];
        desc.storageMode = MTLStorageModePrivate;
        desc.cpuCacheMode = MTLCPUCacheModeDefaultCache;
        desc.hazardTrackingMode = MTLHazardTrackingModeTracked;
        desc.size = std::max(kDefaultHeapSize, bucket * 2);
        id<MTLHeap> heap = [dev newHeapWithDescriptor:desc];
        [desc release];
        if (!heap)
            return nil;
        m_heaps.push_back(heap);
        id<MTLBuffer> hbuf = [heap newBufferWithLength:bucket options:MTLResourceStorageModePrivate];
        if (hbuf)
            out.heap = heap;
        return hbuf;
    };

    id<MTLBuffer> buf = try_heap();
    if (!buf) {
        MTLResourceOptions opts = choose_options(storageModePrivate);
        buf = [dev newBufferWithLength:bucket options:opts];
    }
    OPENVINO_ASSERT(buf, "MetalBufferManager: newBufferWithLength failed");
    out.buffer = buf;
    out.storage_mode = static_cast<uint32_t>(buf.storageMode);
#else
    (void)shared;
    OPENVINO_THROW("MetalBufferManager::allocate requires Objective-C++ (Metal)");
#endif
    out.size = bucket;
    out.type = type;
    out.persistent = persistent;
    out.from_handle = from_handle;
    m_stats.alloc_bytes += bucket;

    if (persistent) {
        m_live_persistent.push_back(out);
    } else if (from_handle) {
        m_live_handle.push_back(out);
    } else {
        m_live_inference.push_back(out);
    }
    return out;
}

MetalBuffer MetalBufferManager::allocate_dynamic(size_t requested,
                                                 ov::element::Type type,
                                                 BufferHandle& handle,
                                                 bool persistent,
                                                 bool storageModePrivate) {
    const size_t target = bucket_size(requested);
    auto storage_matches = [&](const MetalBuffer& buf) {
#ifdef __OBJC__
        const uint32_t desired = storageModePrivate ? static_cast<uint32_t>(MTLStorageModePrivate)
                                                    : static_cast<uint32_t>(MTLStorageModeShared);
        return buf.storage_mode == desired;
#else
        (void)buf;
        return true;
#endif
    };
    const bool persistent_matches = handle.buf.persistent == persistent;
    if (handle.capacity >= target && handle.buf.valid() && storage_matches(handle.buf) && persistent_matches) {
        handle.buf.from_handle = true;
        m_stats.reused_bytes += target;
        return handle.buf;
    }
    if (handle.buf.valid()) {
        auto it = std::remove_if(m_live_handle.begin(),
                                 m_live_handle.end(),
                                 [&](const MetalBuffer& b) { return b.buffer == handle.buf.buffer; });
        if (it != m_live_handle.end())
            m_live_handle.erase(it, m_live_handle.end());
        MetalBuffer old = handle.buf;
        old.from_handle = false;
        release(old);
    }
    // Growth with 1.5x headroom to reduce reallocations on dynamic shapes.
    size_t grow = handle.capacity == 0 ? target : std::max(target, static_cast<size_t>(handle.capacity * 3 / 2));
    // Old buffer stays in free list/pool if it existed; a larger dedicated buffer is allocated.
    handle.buf = allocate(grow, type, persistent, storageModePrivate, /*from_handle=*/true);
    handle.capacity = handle.buf.size;
    return handle.buf;
}

void MetalBufferManager::release(const MetalBuffer& buf) {
    if (!buf.valid() || buf.persistent || buf.from_handle)
        return;
#ifdef __OBJC__
    if (buf.heap) {
        auto mb = static_cast<id<MTLBuffer>>(buf.buffer);
        if (mb && [mb respondsToSelector:@selector(makeAliasable)]) {
            [mb makeAliasable];
        }
    }
#endif
    FreeKey key{bucket_size(buf.size), buf.storage_mode};
    m_free_inference[key].push_back(buf);
}

void MetalBufferManager::reset_inference_pool() {
    for (auto& buf : m_live_inference) {
        FreeKey key{bucket_size(buf.size), buf.storage_mode};
        m_free_inference[key].push_back(buf);
    }
    m_live_inference.clear();
}

void MetalBufferManager::reset_stats() {
    m_stats = {};
}

void MetalBufferManager::upload(const MetalBuffer& dst, const void* src, size_t bytes) {
#ifdef __OBJC__
    if (!dst.valid() || !src || bytes == 0)
        return;
    auto dev_buf = static_cast<id<MTLBuffer>>(dst.buffer);
    const size_t copy_bytes = std::min(bytes, dst.size);
    // Treat any non-shared storage as GPU-only; use a staging blit for safety.
    if (dst.storage_mode != static_cast<uint32_t>(MTLStorageModeShared)) {
        auto dev = static_cast<id<MTLDevice>>(m_device);
        auto queue = static_cast<id<MTLCommandQueue>>(m_copy_queue);
        OPENVINO_ASSERT(dev && queue, "MetalBufferManager: copy queue not initialized");
        id<MTLBuffer> staging = [dev newBufferWithLength:copy_bytes options:MTLResourceStorageModeShared];
        OPENVINO_ASSERT(staging, "MetalBufferManager: staging buffer alloc failed");
        std::memcpy([staging contents], src, copy_bytes);
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit copyFromBuffer:staging sourceOffset:0 toBuffer:dev_buf destinationOffset:0 size:copy_bytes];
        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        [staging release];
        add_h2d(copy_bytes);
    } else {
        std::memcpy([dev_buf contents], src, copy_bytes);
        add_h2d(copy_bytes);
    }
#else
    OPENVINO_THROW("MetalBufferManager::upload requires Objective-C++ (Metal)");
#endif
}

ov::Tensor MetalBufferManager::copy_to_host(const MetalTensor& tensor) const {
    ov::element::Type dst_type = tensor.expected_type == ov::element::dynamic ? tensor.buf.type : tensor.expected_type;
    ov::Tensor host{dst_type, tensor.shape};
    if (!tensor.buf.valid()) {
        return host;
    }
#ifdef __OBJC__
    auto buf = static_cast<id<MTLBuffer>>(tensor.buf.buffer);
    if (!buf) {
        return host;
    }
    const size_t buf_len = buf ? static_cast<size_t>([buf length]) : tensor.buf.size;
    auto src_type = tensor.buf.type;
    // Heuristic: if destination is f16 but the buffer is larger than the f16 footprint, assume storage is f32.
    const size_t dst_bytes = host.get_byte_size();
    if (dst_type == ov::element::f16 && buf_len >= dst_bytes * 2)
        src_type = ov::element::f32;

    auto copy_bytes = [&](void* dst, size_t bytes) {
        if (tensor.buf.storage_mode == static_cast<uint32_t>(MTLStorageModePrivate)) {
            auto dev = static_cast<id<MTLDevice>>(m_device);
            auto queue = static_cast<id<MTLCommandQueue>>(m_copy_queue);
            OPENVINO_ASSERT(dev && queue, "MetalBufferManager: copy queue not initialized");
            id<MTLBuffer> staging = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            OPENVINO_ASSERT(staging, "MetalBufferManager: staging buffer alloc failed");
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:buf sourceOffset:0 toBuffer:staging destinationOffset:0 size:bytes];
            [blit endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            std::memcpy(dst, [staging contents], bytes);
            [staging release];
        } else {
            std::memcpy(dst, [buf contents], bytes);
        }
    };

    if (src_type == dst_type) {
        const size_t bytes = std::min<size_t>(buf_len, host.get_byte_size());
        copy_bytes(host.data(), bytes);
    } else if (src_type == ov::element::f32 && dst_type == ov::element::f16) {
        const size_t elems = ov::shape_size(tensor.shape);
        std::vector<float> tmp(elems, 0.f);
        const size_t bytes = std::min<size_t>(buf_len, elems * sizeof(float));
        copy_bytes(tmp.data(), bytes);
        host = ov::Tensor{ov::element::f16, tensor.shape};
        auto* dst = host.data<ov::float16>();
        for (size_t i = 0; i < elems; ++i) dst[i] = ov::float16{tmp[i]};
    } else if (src_type == ov::element::f16 && dst_type == ov::element::f32) {
        const size_t elems = ov::shape_size(tensor.shape);
        std::vector<ov::float16> tmp(elems);
        const size_t bytes = std::min<size_t>(buf_len, elems * sizeof(ov::float16));
        copy_bytes(tmp.data(), bytes);
        host = ov::Tensor{ov::element::f32, tensor.shape};
        auto* dst = host.data<float>();
        for (size_t i = 0; i < elems; ++i) dst[i] = static_cast<float>(tmp[i]);
    } else {
        const size_t bytes = std::min<size_t>(buf_len, host.get_byte_size());
        copy_bytes(host.data(), bytes);
    }
#else
    OPENVINO_THROW("MetalBufferManager::copy_to_host requires Objective-C++ (Metal)");
#endif
    if (std::getenv("METAL_F16_DBG")) {
        if (host.get_size() > 0) {
            fprintf(stderr, "[dbg] copy_to_host buf=%p buf_type=%s dst_type=%s elems=%zu bytes=%zu buf.size=%zu\n",
                    static_cast<void*>(tensor.buf.buffer),
                    tensor.buf.type.get_type_name().c_str(),
                    dst_type.get_type_name().c_str(),
                    host.get_size(),
                    host.get_byte_size(),
                    tensor.buf.size);
            if (host.get_element_type() == ov::element::f16) {
                auto* p = host.data<const ov::float16>();
                fprintf(stderr, "[dbg] copy_to_host first=%f\n", static_cast<float>(p[0]));
            } else if (host.get_element_type() == ov::element::f32) {
                auto* p = host.data<const float>();
                fprintf(stderr, "[dbg] copy_to_host first=%f\n", p[0]);
            }
        }
    }
    const_cast<MetalBufferManager*>(this)->add_d2h(host.get_byte_size());
    return host;
}

MetalTensor& MetalTensorMap::bind_input(size_t index, const ov::Tensor& host, MetalBufferManager& mgr, bool shared) {
    auto& binding = m_inputs[index];
    binding.host = host;
    const bool promote = (host.get_element_type() == ov::element::f16);
    ov::element::Type storage_type = promote ? ov::element::f32 : host.get_element_type();
    size_t bytes = host.get_size() * storage_type.size();
    binding.dev = MetalTensor{mgr.allocate(bytes, storage_type, /*persistent=*/false, /*storageModePrivate=*/!shared),
                              host.get_shape(),
                              host.get_element_type()};
#ifdef __OBJC__
    if (auto buf = static_cast<id<MTLBuffer>>(binding.dev.buf.buffer)) {
        if (!shared) {
            // Private buffer: go through manager upload to avoid CPU writes into private memory.
            mgr.upload(binding.dev.buf, host.data(), host.get_byte_size());
        } else if (promote) {
            std::vector<float> tmp(host.get_size());
            const ov::float16* src = host.data<const ov::float16>();
            for (size_t i = 0; i < host.get_size(); ++i) tmp[i] = static_cast<float>(src[i]);
            std::memcpy([buf contents], tmp.data(), bytes);
        } else {
            std::memcpy([buf contents], host.data(), host.get_byte_size());
        }
    }
#endif
    mgr.add_h2d(bytes);
    return binding.dev;
}

MetalTensor& MetalTensorMap::ensure_output_device(size_t index,
                                                  const ov::Shape& shape,
                                                  ov::element::Type type,
                                                  MetalBufferManager& mgr,
                                                  bool shared) {
    (void)shared;
    auto& binding = m_outputs[index];
    bool promote = (type == ov::element::f16);
    ov::element::Type storage_type = promote ? ov::element::f32 : type;
    const size_t bytes = ov::shape_size(shape) * storage_type.size();
    // Keep outputs host-visible for tests and on-demand host copies; private buffers are bound later via handles.
    const bool use_private = false;
    binding.dev.buf = mgr.allocate_dynamic(bytes, storage_type, binding.handle, /*persistent=*/false, use_private);
    if (!binding.dev.buf.buffer) {
        // Fallback allocation to avoid null buffer (tests rely on non-null).
        binding.dev.buf = mgr.allocate(bytes,
                                       storage_type,
                                       /*persistent=*/false,
                                       /*storageModePrivate=*/false,
                                       /*from_handle=*/true);
#ifdef __OBJC__
        if (!binding.dev.buf.buffer) {
            id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
            id<MTLBuffer> buf = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            binding.dev.buf.buffer = buf;
            binding.dev.buf.size = bytes;
            binding.dev.buf.type = storage_type;
            binding.dev.buf.storage_mode = static_cast<uint32_t>(MTLStorageModeShared);
            binding.dev.buf.from_handle = true;
        }
#endif
    }
    binding.dev.shape = shape;
    binding.dev.expected_type = type;
    if (std::getenv("METAL_F16_DBG")) {
        fprintf(stderr,
                "[dbg] ensure_output_device idx=%zu promote=%d type=%s storage=%s bytes=%zu buf.size=%zu buf.type=%s\n",
                index,
                promote ? 1 : 0,
                type.get_type_name().c_str(),
                storage_type.get_type_name().c_str(),
                bytes,
                binding.dev.buf.size,
                binding.dev.buf.type.get_type_name().c_str());
    }
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

ov::Tensor& MetalTensorMap::get_or_create_host_for_output(size_t index, const MetalBufferManager& mgr) {
    auto it = m_outputs.find(index);
    OPENVINO_ASSERT(it != m_outputs.end(), "MetalTensorMap: output binding missing");
    if (!it->second.host) {
        it->second.host = mgr.copy_to_host(it->second.dev);
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

void MetalTensorMap::reset_inference() {
    m_inputs.clear();
    for (auto& kv : m_outputs) {
        // Keep buffer handles to allow growth reuse across inferences
        kv.second.host = {};
    }
}

}  // namespace metal_plugin
}  // namespace ov
