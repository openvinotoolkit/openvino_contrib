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
        if (std::strcmp(p.extensionName, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) == 0) {
            extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        }
        if (std::strcmp(p.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0) {
            extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        }
    }
    return extensions;
}

bool has_extension(const std::vector<VkExtensionProperties>& exts, const char* name) {
    for (const auto& ext : exts) {
        if (std::strcmp(ext.extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

template <typename T>
T* find_in_pnext_chain(void* pnext) {
    auto* base = reinterpret_cast<VkBaseOutStructure*>(pnext);
    while (base) {
        if (base->sType == T::structureType) {
            return reinterpret_cast<T*>(base);
        }
        base = base->pNext;
    }
    return nullptr;
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
    m_max_compute_workgroup_invocations = props.limits.maxComputeWorkGroupInvocations;
    m_max_compute_workgroup_size = {props.limits.maxComputeWorkGroupSize[0],
                                    props.limits.maxComputeWorkGroupSize[1],
                                    props.limits.maxComputeWorkGroupSize[2]};

    std::vector<VkExtensionProperties> dev_exts;
    uint32_t dev_ext_count = 0;
    if (vkEnumerateDeviceExtensionProperties(m_physical_device, nullptr, &dev_ext_count, nullptr) == VK_SUCCESS &&
        dev_ext_count > 0) {
        dev_exts.resize(dev_ext_count);
        vkEnumerateDeviceExtensionProperties(m_physical_device, nullptr, &dev_ext_count, dev_exts.data());
    }

    std::vector<const char*> device_extensions;
    if (has_extension(dev_exts, VK_KHR_8BIT_STORAGE_EXTENSION_NAME)) {
        device_extensions.push_back(VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
    }
    if (has_extension(dev_exts, VK_KHR_16BIT_STORAGE_EXTENSION_NAME)) {
        device_extensions.push_back(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
    }
    if (has_extension(dev_exts, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)) {
        device_extensions.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
    }
    if (has_extension(dev_exts, VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME)) {
        device_extensions.push_back(VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME);
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = selected_queue;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    VkPhysicalDevice16BitStorageFeatures storage16{};
    storage16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    VkPhysicalDevice8BitStorageFeatures storage8{};
    storage8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
    VkPhysicalDeviceShaderFloat16Int8Features float16_int8{};
    float16_int8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    features2.pNext = &storage16;
    storage16.pNext = &storage8;
    storage8.pNext = &float16_int8;

    bool enable_storage16 = false;
    bool enable_storage8 = false;
    bool enable_shader_f16 = false;
    bool enable_shader_int8 = false;
    PFN_vkGetPhysicalDeviceFeatures2 get_features2 =
        reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(vkGetInstanceProcAddr(m_instance,
                                                                                 "vkGetPhysicalDeviceFeatures2"));
    if (!get_features2) {
        auto get_features2_khr =
            reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2KHR>(vkGetInstanceProcAddr(
                m_instance,
                "vkGetPhysicalDeviceFeatures2KHR"));
        get_features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(get_features2_khr);
    }
    if (get_features2) {
        get_features2(m_physical_device, &features2);
        enable_storage16 = (storage16.storageBuffer16BitAccess == VK_TRUE);
        enable_storage8 = (storage8.storageBuffer8BitAccess == VK_TRUE);
        enable_shader_f16 = (float16_int8.shaderFloat16 == VK_TRUE);
        enable_shader_int8 = (float16_int8.shaderInt8 == VK_TRUE);
        storage16.storageBuffer16BitAccess = enable_storage16 ? VK_TRUE : VK_FALSE;
        storage16.uniformAndStorageBuffer16BitAccess =
            (storage16.uniformAndStorageBuffer16BitAccess == VK_TRUE) ? VK_TRUE : VK_FALSE;
        storage16.storagePushConstant16 = VK_FALSE;
        storage16.storageInputOutput16 = VK_FALSE;
        storage8.storageBuffer8BitAccess = enable_storage8 ? VK_TRUE : VK_FALSE;
        storage8.uniformAndStorageBuffer8BitAccess =
            (storage8.uniformAndStorageBuffer8BitAccess == VK_TRUE) ? VK_TRUE : VK_FALSE;
        storage8.storagePushConstant8 = VK_FALSE;
        float16_int8.shaderInt8 = enable_shader_int8 ? VK_TRUE : VK_FALSE;
        float16_int8.shaderFloat16 = enable_shader_f16 ? VK_TRUE : VK_FALSE;
    } else {
        features2.pNext = nullptr;
        storage16.pNext = nullptr;
        storage8.pNext = nullptr;
    }

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    VkPhysicalDeviceSubgroupProperties subgroup_props{};
    subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    props2.pNext = &subgroup_props;
    PFN_vkGetPhysicalDeviceProperties2 get_props2 =
        reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(vkGetInstanceProcAddr(m_instance,
                                                                                   "vkGetPhysicalDeviceProperties2"));
    if (!get_props2) {
        auto get_props2_khr =
            reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2KHR>(vkGetInstanceProcAddr(
                m_instance,
                "vkGetPhysicalDeviceProperties2KHR"));
        get_props2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(get_props2_khr);
    }
    if (get_props2) {
        get_props2(m_physical_device, &props2);
        m_subgroup_size = subgroup_props.subgroupSize ? subgroup_props.subgroupSize : 1u;
    }

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
    device_info.ppEnabledExtensionNames = device_extensions.empty() ? nullptr : device_extensions.data();
    device_info.pNext = get_features2 ? &features2 : nullptr;

    res = vkCreateDevice(m_physical_device, &device_info, nullptr, &m_device);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkCreateDevice failed: ", vk_result_to_string(res));
    }

    m_queue_family_index = selected_queue;
    vkGetDeviceQueue(m_device, m_queue_family_index, 0, &m_queue);

    gfx_log_info("Vulkan") << "Initialized Vulkan device: " << m_device_name;
    gfx_log_info("Vulkan") << "Features2: " << (get_features2 ? "available" : "unavailable");
    gfx_log_info("Vulkan") << "16-bit storage: " << (enable_storage16 ? "on" : "off")
                                   << ", 8-bit storage: " << (enable_storage8 ? "on" : "off")
                                   << ", shader f16: " << (enable_shader_f16 ? "on" : "off")
                                   << ", shader int8: " << (enable_shader_int8 ? "on" : "off");
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
    static auto* ctx = new VulkanContext();
    return *ctx;
}


}  // namespace gfx_plugin
}  // namespace ov
