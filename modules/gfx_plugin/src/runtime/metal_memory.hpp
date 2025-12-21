// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/tensor.hpp"

#include "runtime/memory/metal_allocator.hpp"
#include "runtime/memory/metal_buffer.hpp"
#include "runtime/memory/metal_const_cache.hpp"
#include "runtime/memory/metal_device_caps.hpp"
#include "runtime/memory/metal_memory_session.hpp"
#include "runtime/memory/metal_memory_stats.hpp"

namespace ov {
namespace gfx_plugin {

// Returns true when OV_GFX_SAFE_DEBUG=1 is set in the environment.
bool metal_safe_debug_enabled();

// Enumerate Metal devices (names) available on the system.
std::vector<std::string> metal_get_device_names();
// Return Metal device handle by index (0-based). Returns nullptr if out of range.
MetalDeviceHandle metal_get_device_by_id(int index);
// Create/release a command queue for a device (kept in C++ to avoid Objective-C++ in callers).
MetalCommandQueueHandle metal_create_command_queue(MetalDeviceHandle device);
void metal_release_command_queue(MetalCommandQueueHandle queue);

// Release external MetalBuffer wrapper (used for bytesNoCopy buffers).
void metal_release_external_buffer(MetalBuffer& buf);

struct MetalTensor {
    MetalBuffer buf;
    ov::Shape shape;
    ov::element::Type expected_type = ov::element::dynamic;  // desired logical type
    bool prefer_private = true;
};

class MetalTensorMap {
public:
    MetalTensor& bind_input(size_t index, const ov::Tensor& host, MetalAllocatorCore& core);
    MetalTensor& bind_input_device(size_t index, const MetalTensor& dev);

    MetalTensor& ensure_output_device(size_t index,
                                      const ov::Shape& shape,
                                      ov::element::Type type,
                                      MetalAllocator& alloc,
                                      const MetalDeviceCaps& caps,
                                      bool prefer_private);

    bool has_output_device(size_t index) const;
    const MetalTensor& get_output_device(size_t index) const;

    bool has_host_for_output(size_t index) const;
    ov::Tensor& get_or_create_host_for_output(size_t index);
    void bind_host_for_output(size_t index, ov::Tensor host);
    MetalTensor& bind_output_device(size_t index, const MetalTensor& dev);

    bool has_input_device(size_t index) const;
    MetalTensor& get_input_device(size_t index);
    const MetalTensor& get_input_device(size_t index) const;
    bool has_input_host(size_t index) const;
    ov::Tensor& get_input_host(size_t index);

    void reset_inference(MetalAllocatorCore* core = nullptr);

private:
    struct Binding {
        ov::Tensor host;
        MetalTensor dev;
        BufferHandle handle;
    };

    std::unordered_map<size_t, Binding> m_inputs;
    std::unordered_map<size_t, Binding> m_outputs;
};

class MetalBufferManager {
public:
    explicit MetalBufferManager(MetalAllocatorCore& core, MetalConstCache* const_cache = nullptr);

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

    void release(MetalBuffer&& buf);
    void reset_stats();
    const MetalMemoryStats& stats() const;

    MetalBuffer wrap_shared(void* ptr, size_t bytes, ov::element::Type type);
    MetalBuffer wrap_shared(const void* ptr, size_t bytes, ov::element::Type type) {
        return wrap_shared(const_cast<void*>(ptr), bytes, type);
    }
    MetalBuffer wrap_const(const std::string& key,
                           const void* data,
                           size_t bytes,
                           ov::element::Type type,
                           MetalStorage storage = MetalStorage::Shared);

    MetalDeviceHandle device() const { return m_core.device(); }

    static void set_current_allocator(MetalAllocator* alloc);
    static void set_current_session(MetalMemorySession* session);

private:
    MetalAllocatorCore& m_core;
    MetalConstCache* m_const_cache = nullptr;
    MetalMemoryStats m_dummy_stats{};
};

}  // namespace gfx_plugin
}  // namespace ov
