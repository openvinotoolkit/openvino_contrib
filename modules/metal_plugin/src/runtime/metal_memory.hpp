// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <map>
#include <unordered_map>
#include <vector>
#include <string>

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace metal_plugin {

#ifdef __OBJC__
using MetalDeviceHandle = id<MTLDevice>;
using MetalBufferHandle = id<MTLBuffer>;
using MetalHeapHandle = id<MTLHeap>;
using MetalCommandQueueHandle = id<MTLCommandQueue>;
#else
using MetalDeviceHandle = void*;
using MetalBufferHandle = void*;
using MetalHeapHandle = void*;
using MetalCommandQueueHandle = void*;
#endif

// Returns true when OV_METAL_SAFE_DEBUG=1 is set in the environment.
bool metal_safe_debug_enabled();

struct MetalBuffer {
    MetalBufferHandle buffer = nullptr;
    size_t size = 0;                  // bytes (aligned)
    ov::element::Type type = ov::element::dynamic;
    MetalHeapHandle heap = nullptr;   // optional owning heap
    size_t offset = 0;                // offset inside heap
    bool persistent = false;          // model lifetime vs per-inference
    uint32_t storage_mode = 0;        // MTLStorageMode (kept as integer to stay C++-only friendly)
    bool from_handle = false;         // reserved for a specific logical tensor (BufferHandle-backed)

    bool valid() const { return buffer != nullptr; }
};

struct MetalTensor {
    MetalBuffer buf;
    ov::Shape shape;
    ov::element::Type expected_type = ov::element::dynamic;  // desired logical type (model); buf.type is storage
    // TODO: strides/layout tags if needed
};

struct MetalMemoryStats {
    size_t h2d_bytes = 0;
    size_t d2h_bytes = 0;
    size_t alloc_bytes = 0;   // bytes allocated from device
    size_t reused_bytes = 0;  // bytes served from free list
};

struct BufferHandle {
    MetalBuffer buf;
    size_t capacity = 0;  // max bytes currently allocated (aligned)
};

struct FreeKey {
    size_t bucket = 0;
    uint32_t storage_mode = 0;  // MTLStorageMode

    bool operator<(const FreeKey& other) const {
        if (bucket != other.bucket)
            return bucket < other.bucket;
        return storage_mode < other.storage_mode;
    }
};

class MetalBufferManager {
public:
    explicit MetalBufferManager(MetalDeviceHandle device);
    ~MetalBufferManager();

    MetalBuffer allocate(size_t size,
                         ov::element::Type type,
                         bool persistent = false,
                         bool storageModePrivate = true,
                         bool from_handle = false);
    MetalBuffer allocate_dynamic(size_t requested,
                                 ov::element::Type type,
                                 BufferHandle& handle,
                                 bool persistent = false,
                                 bool storageModePrivate = true);
    void release(const MetalBuffer& buf);
    void reset_inference_pool();
    void reset_stats();
    void add_h2d(size_t bytes) { m_stats.h2d_bytes += bytes; }
    void add_d2h(size_t bytes) { m_stats.d2h_bytes += bytes; }
    const MetalMemoryStats& stats() const { return m_stats; }
    // Upload host data into a device buffer (handles Private/Shared seamlessly).
    void upload(const MetalBuffer& dst, const void* src, size_t bytes);

    ov::Tensor copy_to_host(const MetalTensor& tensor) const;
    MetalBufferHandle device_buffer_handle(const MetalBuffer& buf) const { return buf.buffer; }
    MetalDeviceHandle device() const { return m_device; }

private:
    size_t align_size(size_t size) const;
    size_t bucket_size(size_t size) const;

    MetalDeviceHandle m_device = nullptr;
    MetalCommandQueueHandle m_copy_queue = nullptr;
    std::vector<MetalHeapHandle> m_heaps;  // per-inference private heaps
    std::map<FreeKey, std::vector<MetalBuffer>> m_free_inference;  // keyed by bucket + storage mode
    std::vector<MetalBuffer> m_live_inference;
    std::vector<MetalBuffer> m_live_handle;      // buffers dedicated to BufferHandle (not put in free list)
    std::vector<MetalBuffer> m_live_persistent;
    MetalMemoryStats m_stats;
};

class MetalTensorMap {
public:
    MetalTensor& bind_input(size_t index, const ov::Tensor& host, MetalBufferManager& mgr, bool shared = true);

    MetalTensor& ensure_output_device(size_t index,
                                      const ov::Shape& shape,
                                      ov::element::Type type,
                                      MetalBufferManager& mgr,
                                      bool shared = false);

    bool has_output_device(size_t index) const;
    const MetalTensor& get_output_device(size_t index) const;

    bool has_host_for_output(size_t index) const;
    ov::Tensor& get_or_create_host_for_output(size_t index, const MetalBufferManager& mgr);
    void bind_host_for_output(size_t index, ov::Tensor host);
    MetalTensor& bind_output_device(size_t index, const MetalTensor& dev);

    bool has_input_device(size_t index) const;
    MetalTensor& get_input_device(size_t index);
    const MetalTensor& get_input_device(size_t index) const;
    bool has_input_host(size_t index) const;
    ov::Tensor& get_input_host(size_t index);

    void reset_inference();

private:
    struct Binding {
        ov::Tensor host;
        MetalTensor dev;
        BufferHandle handle;  // used for dynamic grow on outputs
    };

    std::unordered_map<size_t, Binding> m_inputs;
    std::unordered_map<size_t, Binding> m_outputs;
};

}  // namespace metal_plugin
}  // namespace ov
