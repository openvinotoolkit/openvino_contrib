// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

#include <vulkan/vulkan.h>

#include "openvino/core/type/element_type.hpp"
#include "runtime/gpu_types.hpp"
#include "backends/vulkan/runtime/memory_api.hpp"

namespace ov {
namespace gfx_plugin {

GpuBuffer vulkan_allocate_buffer(size_t bytes,
                                 ov::element::Type type,
                                 VkBufferUsageFlags usage,
                                 VkMemoryPropertyFlags properties);
GpuBuffer vulkan_upload_device_buffer(const void* src,
                                      size_t bytes,
                                      ov::element::Type type,
                                      VkBufferUsageFlags usage);

inline VkBuffer vk_buffer_from_gpu(const GpuBuffer& buf) {
    return reinterpret_cast<VkBuffer>(buf.buffer);
}

inline VkDeviceMemory vk_memory_from_gpu(const GpuBuffer& buf) {
    return reinterpret_cast<VkDeviceMemory>(buf.heap);
}

}  // namespace gfx_plugin
}  // namespace ov
