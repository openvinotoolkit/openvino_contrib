// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/runtime/vulkan_backend.hpp"

#include <cstring>
#include <vector>

#include "openvino/core/except.hpp"

#include "runtime/gfx_logger.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

std::string vk_result_to_string(VkResult res) {
    switch (res) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_EVENT_SET: return "VK_EVENT_SET";
    case VK_EVENT_RESET: return "VK_EVENT_RESET";
    case VK_INCOMPLETE: return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    default:
        break;
    }
    return "VK_ERROR_UNKNOWN";
}

std::vector<const char*> collect_instance_extensions() {
    uint32_t count = 0;
    std::vector<const char*> extensions;
    if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) != VK_SUCCESS || count == 0) {
        return extensions;
    }
    std::vector<VkExtensionProperties> props(count);
    if (vkEnumerateInstanceExtensionProperties(nullptr, &count, props.data()) != VK_SUCCESS) {
        return extensions;
    }
    for (const auto& p : props) {
        if (std::strcmp(p.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0) {
            extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        }
    }
    return extensions;
}

}  // namespace

VulkanContext::VulkanContext() {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "GFXPlugin";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "GFX";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;

    auto extensions = collect_instance_extensions();
    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();
    if (!extensions.empty()) {
        create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }

    VkResult res = vkCreateInstance(&create_info, nullptr, &m_instance);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkCreateInstance failed: ", vk_result_to_string(res));
    }

    uint32_t phys_count = 0;
    res = vkEnumeratePhysicalDevices(m_instance, &phys_count, nullptr);
    if (res != VK_SUCCESS || phys_count == 0) {
        OPENVINO_THROW("GFX Vulkan: no physical devices found");
    }
    std::vector<VkPhysicalDevice> devices(phys_count);
    res = vkEnumeratePhysicalDevices(m_instance, &phys_count, devices.data());
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkEnumeratePhysicalDevices failed: ", vk_result_to_string(res));
    }

    uint32_t selected_queue = 0;
    bool found = false;
    for (auto dev : devices) {
        uint32_t queue_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queue_count, nullptr);
        if (!queue_count)
            continue;
        std::vector<VkQueueFamilyProperties> props(queue_count);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queue_count, props.data());
        for (uint32_t i = 0; i < queue_count; ++i) {
            if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                m_physical_device = dev;
                selected_queue = i;
                found = true;
                break;
            }
        }
        if (found)
            break;
    }

    if (!found) {
        OPENVINO_THROW("GFX Vulkan: no compute queue found");
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(m_physical_device, &props);
    m_device_name = props.deviceName;
    m_noncoherent_atom_size = static_cast<size_t>(props.limits.nonCoherentAtomSize);

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = selected_queue;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;

    res = vkCreateDevice(m_physical_device, &device_info, nullptr, &m_device);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkCreateDevice failed: ", vk_result_to_string(res));
    }

    m_queue_family_index = selected_queue;
    vkGetDeviceQueue(m_device, m_queue_family_index, 0, &m_queue);

    GFX_LOG_INFO("Vulkan", "Initialized Vulkan device: " << m_device_name);
}

VulkanContext::~VulkanContext() {
    if (m_device) {
        vkDestroyDevice(m_device, nullptr);
        m_device = VK_NULL_HANDLE;
    }
    if (m_instance) {
        vkDestroyInstance(m_instance, nullptr);
        m_instance = VK_NULL_HANDLE;
    }
}

VulkanContext& VulkanContext::instance() {
    static VulkanContext ctx;
    return ctx;
}


}  // namespace gfx_plugin
}  // namespace ov
