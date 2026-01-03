// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/codegen/vulkan_codegen_backend.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>

#include "openvino/core/except.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/spirv_codegen.hpp"

#include "backends/vulkan/runtime/vulkan_memory.hpp"
#include "kernel_ir/gfx_kernel_cache.hpp"

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

}  // namespace

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
    const uint32_t runtime_count = ensure_kernel_args_dense(args, "GFX Vulkan");
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

        OPENVINO_ASSERT(buffer,
                        "GFX Vulkan: missing VkBuffer for arg ",
                        arg.index,
                        " (kernel ",
                        m_entry_point,
                        ")");

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
    VkCommandBuffer used_cmd = cmd;
    if (hooks && hooks->on_begin) {
        hooks->on_begin(reinterpret_cast<GpuCommandEncoderHandle>(cmd));
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_pipeline_layout,
                            0,
                            1,
                            &m_desc_set,
                            0,
                            nullptr);

    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         1,
                         &barrier,
                         0,
                         nullptr,
                         0,
                         nullptr);

    const size_t grid_x = dispatch.grid[0];
    const size_t grid_y = dispatch.grid[1];
    const size_t grid_z = dispatch.grid[2];
    if (grid_x == 0 || grid_y == 0 || grid_z == 0) {
        if (hooks && hooks->on_end) {
            hooks->on_end(reinterpret_cast<GpuCommandEncoderHandle>(cmd));
        }
        end_commands(used_cmd);
        if (hooks && hooks->on_complete) {
            hooks->on_complete();
        }
        return;
    }

    const size_t tg_x = std::max<size_t>(dispatch.threads_per_group[0], 1);
    const size_t tg_y = std::max<size_t>(dispatch.threads_per_group[1], 1);
    const size_t tg_z = std::max<size_t>(dispatch.threads_per_group[2], 1);

    const uint32_t groups_x = static_cast<uint32_t>((grid_x + tg_x - 1) / tg_x);
    const uint32_t groups_y = static_cast<uint32_t>((grid_y + tg_y - 1) / tg_y);
    const uint32_t groups_z = static_cast<uint32_t>((grid_z + tg_z - 1) / tg_z);
    vkCmdDispatch(cmd, std::max<uint32_t>(groups_x, 1), std::max<uint32_t>(groups_y, 1), std::max<uint32_t>(groups_z, 1));

    if (hooks && hooks->on_end) {
        hooks->on_end(reinterpret_cast<GpuCommandEncoderHandle>(cmd));
    }
    end_commands(used_cmd);
    if (hooks && hooks->on_complete) {
        hooks->on_complete();
    }
}

void VulkanCompiledKernel::ensure_pipeline(uint32_t binding_count) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_pipeline && binding_count == m_binding_count) {
        return;
    }

    destroy_pipeline_locked();

    m_binding_count = binding_count;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.reserve(binding_count);
    for (uint32_t i = 0; i < binding_count; ++i) {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = i;
        binding.descriptorCount = 1;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(binding);
    }

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.empty() ? nullptr : bindings.data();
    VkResult res = vkCreateDescriptorSetLayout(m_device, &layout_info, nullptr, &m_desc_layout);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkCreateDescriptorSetLayout failed: ", vk_result_to_string(res));
    }

    VkPipelineLayoutCreateInfo layout_create{};
    layout_create.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_create.setLayoutCount = 1;
    layout_create.pSetLayouts = &m_desc_layout;
    res = vkCreatePipelineLayout(m_device, &layout_create, nullptr, &m_pipeline_layout);
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

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = std::max<uint32_t>(1, binding_count);

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.maxSets = 1;
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

    VkCommandPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.queueFamilyIndex = m_queue_family;
    pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    res = vkCreateCommandPool(m_device, &pool_ci, nullptr, &m_command_pool);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkCreateCommandPool failed: ", vk_result_to_string(res));
    }
}

void VulkanCompiledKernel::destroy_pipeline() {
    std::lock_guard<std::mutex> lock(m_mutex);
    destroy_pipeline_locked();
}

void VulkanCompiledKernel::destroy_pipeline_locked() {
    if (m_command_pool) {
        vkDestroyCommandPool(m_device, m_command_pool, nullptr);
        m_command_pool = VK_NULL_HANDLE;
    }
    if (m_desc_pool) {
        vkDestroyDescriptorPool(m_device, m_desc_pool, nullptr);
        m_desc_pool = VK_NULL_HANDLE;
    }
    if (m_desc_layout) {
        vkDestroyDescriptorSetLayout(m_device, m_desc_layout, nullptr);
        m_desc_layout = VK_NULL_HANDLE;
    }
    if (m_pipeline) {
        vkDestroyPipeline(m_device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    if (m_pipeline_layout) {
        vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
        m_pipeline_layout = VK_NULL_HANDLE;
    }
    m_desc_set = VK_NULL_HANDLE;
    m_binding_count = 0;
}

VkCommandBuffer VulkanCompiledKernel::begin_commands() {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = m_command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkResult res = vkAllocateCommandBuffers(m_device, &alloc_info, &cmd);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkAllocateCommandBuffers failed: ", vk_result_to_string(res));
    }

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    res = vkBeginCommandBuffer(cmd, &begin_info);
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

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    res = vkQueueSubmit(m_queue, 1, &submit_info, VK_NULL_HANDLE);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkQueueSubmit failed: ", vk_result_to_string(res));
    }
    res = vkQueueWaitIdle(m_queue);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkQueueWaitIdle failed: ", vk_result_to_string(res));
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
    std::string entry = resolve_entry_point(source, "gfx_kernel");

    std::string local_log;
    std::string* log_ptr = log ? log : &local_log;
    std::vector<uint32_t> spirv_binary = resolve_spirv_binary_from_source(source, log_ptr);
    if (spirv_binary.empty()) {
        spirv_binary = lower_to_spirv(module, entry, log_ptr);
    }
    if (spirv_binary.empty()) {
        OPENVINO_THROW("GFX Vulkan: failed to lower MLIR to SPIR-V for entry point ",
                       entry,
                       ". ",
                       *log_ptr);
    }

    uint32_t arg_count = source.signature.arg_count;
    if (module) {
        if (auto kinds = module->getAttrOfType<mlir::ArrayAttr>("gfx.kernel_operand_kinds")) {
            arg_count = static_cast<uint32_t>(kinds.size());
        } else if (auto scalars = module->getAttrOfType<mlir::ArrayAttr>("gfx.kernel_scalar_args")) {
            arg_count = static_cast<uint32_t>(arg_count + scalars.size());
        }
    }
    const uintptr_t device_key = reinterpret_cast<uintptr_t>(m_device);
    const void* code_ptr = spirv_binary.empty() ? nullptr : spirv_binary.data();
    const size_t code_bytes = spirv_binary.size() * sizeof(uint32_t);
    return lookup_or_compile_kernel(GpuBackend::Vulkan,
                                    device_key,
                                    code_ptr,
                                    code_bytes,
                                    entry,
                                    arg_count,
                                    [&]() -> std::shared_ptr<ICompiledKernel> {
                                        return std::make_shared<VulkanCompiledKernel>(std::move(spirv_binary),
                                                                                     std::move(entry),
                                                                                     arg_count);
                                    });
}

}  // namespace gfx_plugin
}  // namespace ov
