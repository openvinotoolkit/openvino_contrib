// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/runtime/backend.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>

#include "openvino/core/except.hpp"

#include "spirv/mlir_spirv.hpp"

#include "runtime/gfx_logger.hpp"
#include "backends/vulkan/runtime/memory.hpp"

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

VulkanCompiledKernel::VulkanCompiledKernel(std::vector<uint32_t> spirv,
                                           std::string entry_point,
                                           uint32_t arg_count)
    : m_spirv(std::move(spirv)), m_entry_point(std::move(entry_point)), m_args_count(arg_count) {
    auto& ctx = VulkanContext::instance();
    m_device = ctx.device();
    m_queue = ctx.queue();
    m_queue_family = ctx.queue_family_index();

    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = m_spirv.size() * sizeof(uint32_t);
    shader_info.pCode = m_spirv.data();
    VkResult res = vkCreateShaderModule(m_device, &shader_info, nullptr, &m_shader_module);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkCreateShaderModule failed: ", vk_result_to_string(res));
    }
}

VulkanCompiledKernel::~VulkanCompiledKernel() {
    destroy_pipeline();
    if (m_shader_module) {
        vkDestroyShaderModule(m_device, m_shader_module, nullptr);
        m_shader_module = VK_NULL_HANDLE;
    }
}

void VulkanCompiledKernel::set_args_count(uint32_t count) {
    if (count == 0) {
        return;
    }
    if (m_args_count == 0) {
        m_args_count = count;
        return;
    }
    OPENVINO_ASSERT(m_args_count == count,
                    "GFX Vulkan: arg count mismatch (expected ",
                    m_args_count,
                    ", got ",
                    count,
                    ")");
}

size_t VulkanCompiledKernel::clamp_threadgroup_size(size_t desired) const {
    return desired == 0 ? 1 : desired;
}

void VulkanCompiledKernel::execute(GpuCommandBufferHandle /*command_buffer*/,
                                   const KernelDispatch& dispatch,
                                   const std::vector<KernelArg>& args,
                                   const KernelExecutionHooks* hooks) {
    uint32_t runtime_count = 0;
    OPENVINO_ASSERT(kernel_args_dense(args, &runtime_count),
                    "GFX Vulkan: kernel args must be densely indexed from 0");
    set_args_count(runtime_count);
    const uint32_t binding_count = runtime_count;
    ensure_pipeline(binding_count);

    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> buffer_infos;
    writes.reserve(args.size());
    buffer_infos.reserve(args.size());

    for (const auto& arg : args) {
        OPENVINO_ASSERT(arg.kind == KernelArg::Kind::Buffer,
                        "GFX Vulkan: bytes arguments must be materialized into buffers");
        VkBuffer buffer = VK_NULL_HANDLE;
        size_t size = 0;
        size_t offset = arg.offset;

        buffer = vk_buffer_from_gpu(arg.buffer);
        size = arg.buffer.size;

        if (!buffer) {
            continue;
        }

        VkDescriptorBufferInfo info{};
        info.buffer = buffer;
        info.offset = offset;
        info.range = size ? size - offset : VK_WHOLE_SIZE;
        buffer_infos.push_back(info);

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = m_desc_set;
        write.dstBinding = arg.index;
        write.dstArrayElement = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &buffer_infos.back();
        writes.push_back(write);
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(m_device,
                               static_cast<uint32_t>(writes.size()),
                               writes.data(),
                               0,
                               nullptr);
    }

    VkCommandBuffer cmd = begin_commands();
    if (hooks && hooks->on_begin) {
        hooks->on_begin(reinterpret_cast<GpuCommandEncoderHandle>(cmd));
    }
    {
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             1,
                             &barrier,
                             0,
                             nullptr,
                             0,
                             nullptr);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    if (binding_count > 0) {
        vkCmdBindDescriptorSets(cmd,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_pipeline_layout,
                                0,
                                1,
                                &m_desc_set,
                                0,
                                nullptr);
    }

    const uint32_t tpg_x = dispatch.threads_per_group[0] ? static_cast<uint32_t>(dispatch.threads_per_group[0]) : 1u;
    const uint32_t tpg_y = dispatch.threads_per_group[1] ? static_cast<uint32_t>(dispatch.threads_per_group[1]) : 1u;
    const uint32_t tpg_z = dispatch.threads_per_group[2] ? static_cast<uint32_t>(dispatch.threads_per_group[2]) : 1u;
    const uint32_t grid_x = dispatch.grid[0] ? static_cast<uint32_t>(dispatch.grid[0]) : 1u;
    const uint32_t grid_y = dispatch.grid[1] ? static_cast<uint32_t>(dispatch.grid[1]) : 1u;
    const uint32_t grid_z = dispatch.grid[2] ? static_cast<uint32_t>(dispatch.grid[2]) : 1u;

    const uint32_t groups_x = (grid_x + tpg_x - 1) / tpg_x;
    const uint32_t groups_y = (grid_y + tpg_y - 1) / tpg_y;
    const uint32_t groups_z = (grid_z + tpg_z - 1) / tpg_z;

    vkCmdDispatch(cmd, groups_x ? groups_x : 1u, groups_y ? groups_y : 1u, groups_z ? groups_z : 1u);
    {
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             1,
                             &barrier,
                             0,
                             nullptr,
                             0,
                             nullptr);
    }
    if (hooks && hooks->on_end) {
        hooks->on_end(reinterpret_cast<GpuCommandEncoderHandle>(cmd));
    }
    end_commands(cmd);
    if (hooks && hooks->on_complete) {
        hooks->on_complete();
    }

}

void VulkanCompiledKernel::ensure_pipeline(uint32_t binding_count) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_pipeline && m_binding_count == binding_count) {
        return;
    }
    destroy_pipeline();
    m_binding_count = binding_count;

    if (binding_count > 0) {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.reserve(binding_count);
        for (uint32_t i = 0; i < binding_count; ++i) {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = i;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings.push_back(binding);
        }
        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();
        VkResult res = vkCreateDescriptorSetLayout(m_device, &layout_info, nullptr, &m_desc_layout);
        if (res != VK_SUCCESS) {
            OPENVINO_THROW("GFX Vulkan: vkCreateDescriptorSetLayout failed: ", vk_result_to_string(res));
        }

        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = binding_count;

        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        res = vkCreateDescriptorPool(m_device, &pool_info, nullptr, &m_desc_pool);
        if (res != VK_SUCCESS) {
            OPENVINO_THROW("GFX Vulkan: vkCreateDescriptorPool failed: ", vk_result_to_string(res));
        }

        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = m_desc_pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &m_desc_layout;
        res = vkAllocateDescriptorSets(m_device, &alloc_info, &m_desc_set);
        if (res != VK_SUCCESS) {
            OPENVINO_THROW("GFX Vulkan: vkAllocateDescriptorSets failed: ", vk_result_to_string(res));
        }
    }

    VkPipelineLayoutCreateInfo layout{};
    layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    if (m_desc_layout) {
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &m_desc_layout;
    }
    VkResult res = vkCreatePipelineLayout(m_device, &layout, nullptr, &m_pipeline_layout);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkCreatePipelineLayout failed: ", vk_result_to_string(res));
    }

    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.layout = m_pipeline_layout;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = m_shader_module;
    pipeline_info.stage.pName = m_entry_point.c_str();
    res = vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &m_pipeline);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkCreateComputePipelines failed: ", vk_result_to_string(res));
    }

    if (!m_command_pool) {
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = m_queue_family;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        res = vkCreateCommandPool(m_device, &pool_info, nullptr, &m_command_pool);
        if (res != VK_SUCCESS) {
            OPENVINO_THROW("GFX Vulkan: vkCreateCommandPool failed: ", vk_result_to_string(res));
        }
    }
}

void VulkanCompiledKernel::destroy_pipeline() {
    if (m_pipeline) {
        vkDestroyPipeline(m_device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    if (m_pipeline_layout) {
        vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
        m_pipeline_layout = VK_NULL_HANDLE;
    }
    if (m_desc_pool) {
        vkDestroyDescriptorPool(m_device, m_desc_pool, nullptr);
        m_desc_pool = VK_NULL_HANDLE;
        m_desc_set = VK_NULL_HANDLE;
    }
    if (m_desc_layout) {
        vkDestroyDescriptorSetLayout(m_device, m_desc_layout, nullptr);
        m_desc_layout = VK_NULL_HANDLE;
    }
}

VkCommandBuffer VulkanCompiledKernel::begin_commands() {
    VkCommandBufferAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = m_command_pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkResult res = vkAllocateCommandBuffers(m_device, &alloc, &cmd);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkAllocateCommandBuffers failed: ", vk_result_to_string(res));
    }

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    res = vkBeginCommandBuffer(cmd, &begin);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkBeginCommandBuffer failed: ", vk_result_to_string(res));
    }
    return cmd;
}

void VulkanCompiledKernel::end_commands(VkCommandBuffer cmd) {
    VkResult res = vkEndCommandBuffer(cmd);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkEndCommandBuffer failed: ", vk_result_to_string(res));
    }
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    res = vkCreateFence(m_device, &fence_info, nullptr, &fence);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkCreateFence failed: ", vk_result_to_string(res));
    }
    res = vkQueueSubmit(m_queue, 1, &submit, fence);
    if (res != VK_SUCCESS) {
        vkDestroyFence(m_device, fence, nullptr);
        OPENVINO_THROW("GFX Vulkan: vkQueueSubmit failed: ", vk_result_to_string(res));
    }
    res = vkWaitForFences(m_device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(m_device, fence, nullptr);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkWaitForFences failed: ", vk_result_to_string(res));
    }
    vkFreeCommandBuffers(m_device, m_command_pool, 1, &cmd);
}

VulkanCodegenBackend::VulkanCodegenBackend(VulkanDeviceHandle device) : m_device(device) {
    if (!m_device) {
        auto& ctx = VulkanContext::instance();
        m_device = ctx.device();
        m_physical_device = ctx.physical_device();
    }
}

std::shared_ptr<ICompiledKernel> VulkanCodegenBackend::compile(const KernelSource& source, std::string* log) {
    mlir::ModuleOp module = source.module;
    std::string entry = source.entry_point.empty() ? "gfx_kernel" : source.entry_point;

    std::string local_log;
    std::string* log_ptr = log ? log : &local_log;
    std::vector<uint32_t> spirv_binary = source.spirv_binary;
    if (spirv_binary.empty() && source.spirv_generator) {
        spirv_binary = source.spirv_generator(module);
        if (spirv_binary.empty()) {
            *log_ptr = "SPIR-V generator returned empty output";
        }
    }
    if (spirv_binary.empty()) {
        spirv_binary = lower_to_spirv(module, entry, log_ptr);
    }
    if (spirv_binary.empty()) {
        OPENVINO_THROW("GFX Vulkan: failed to lower MLIR to SPIR-V for entry point ",
                       entry,
                       ". ",
                       *log_ptr);
    }

    return std::make_shared<VulkanCompiledKernel>(std::move(spirv_binary),
                                                  std::move(entry),
                                                  source.signature.arg_count);
}

}  // namespace gfx_plugin
}  // namespace ov
