// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "kernel_ir/gfx_codegen_backend.hpp"
#include "backends/vulkan/runtime/vulkan_backend.hpp"

namespace ov {
namespace gfx_plugin {

class VulkanCompiledKernel final : public ICompiledKernel {
public:
    VulkanCompiledKernel(std::vector<uint32_t> spirv, std::string entry_point, uint32_t arg_count = 0);
    ~VulkanCompiledKernel() override;

    const std::vector<uint32_t>& spirv() const { return m_spirv; }
    const std::string& entry_point() const { return m_entry_point; }

    uint32_t args_count() const override { return m_args_count; }
    void set_args_count(uint32_t count) override;
    size_t clamp_threadgroup_size(size_t desired) const override;
    void execute(GpuCommandBufferHandle command_buffer,
                 const KernelDispatch& dispatch,
                 const std::vector<KernelArg>& args,
                 const KernelExecutionHooks* hooks = nullptr) override;

private:
    void ensure_pipeline(uint32_t binding_count);
    void destroy_pipeline();
    void destroy_pipeline_locked();
    VkCommandBuffer begin_commands();
    void end_commands(VkCommandBuffer cmd);

    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    uint32_t m_queue_family = 0;
    VkShaderModule m_shader_module = VK_NULL_HANDLE;
    VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_desc_layout = VK_NULL_HANDLE;
    VkDescriptorPool m_desc_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_desc_set = VK_NULL_HANDLE;
    VkCommandPool m_command_pool = VK_NULL_HANDLE;
    uint32_t m_binding_count = 0;
    std::mutex m_mutex;

    std::vector<uint32_t> m_spirv;
    std::string m_entry_point;
    uint32_t m_args_count = 0;
};

class VulkanCodegenBackend final : public ICodegenBackend {
public:
    explicit VulkanCodegenBackend(VulkanDeviceHandle device = VK_NULL_HANDLE);

    VulkanDeviceHandle device() const { return m_device; }
    VulkanPhysicalDeviceHandle physical_device() const { return m_physical_device; }

    std::shared_ptr<ICompiledKernel> compile(const KernelSource& source,
                                             std::string* log = nullptr) override;

private:
    VulkanDeviceHandle m_device = VK_NULL_HANDLE;
    VulkanPhysicalDeviceHandle m_physical_device = VK_NULL_HANDLE;
};

}  // namespace gfx_plugin
}  // namespace ov
