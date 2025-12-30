// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <string>

#include <vulkan/vulkan.h>

namespace ov {
namespace gfx_plugin {

using VulkanInstanceHandle = VkInstance;
using VulkanPhysicalDeviceHandle = VkPhysicalDevice;
using VulkanDeviceHandle = VkDevice;
using VulkanQueueHandle = VkQueue;

class VulkanContext {
public:
    VulkanContext();
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    static VulkanContext& instance();

    VulkanInstanceHandle instance_handle() const { return m_instance; }
    VulkanPhysicalDeviceHandle physical_device() const { return m_physical_device; }
    VulkanDeviceHandle device() const { return m_device; }
    VulkanQueueHandle queue() const { return m_queue; }
    uint32_t queue_family_index() const { return m_queue_family_index; }
    const std::string& device_name() const { return m_device_name; }
    size_t noncoherent_atom_size() const { return m_noncoherent_atom_size; }

private:
    VulkanInstanceHandle m_instance = VK_NULL_HANDLE;
    VulkanPhysicalDeviceHandle m_physical_device = VK_NULL_HANDLE;
    VulkanDeviceHandle m_device = VK_NULL_HANDLE;
    VulkanQueueHandle m_queue = VK_NULL_HANDLE;
    uint32_t m_queue_family_index = 0;
    std::string m_device_name;
    size_t m_noncoherent_atom_size = 1;
};

}  // namespace gfx_plugin
}  // namespace ov
