// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

#include "kernel_ir/gfx_codegen_backend.hpp"
#include "backends/vulkan/runtime/vulkan_backend.hpp"

namespace ov {
namespace gfx_plugin {

class VulkanKernelProgram;

class VulkanCompiledKernel final : public CompiledKernelBase {
public:
    explicit VulkanCompiledKernel(std::shared_ptr<VulkanKernelProgram> program, uint32_t arg_count = 0);
    VulkanCompiledKernel(std::shared_ptr<VulkanKernelProgram> program,
                         std::shared_ptr<const KernelBindingPlan> binding_plan);
    VulkanCompiledKernel(std::shared_ptr<VulkanKernelProgram> program,
                         std::shared_ptr<const KernelBindingPlan> binding_plan,
                         std::shared_ptr<void> prepared_binding_cache);
    ~VulkanCompiledKernel() override;

    size_t clamp_threadgroup_size(size_t desired) const override;
    std::shared_ptr<ICompiledKernel> fork() const override;
    void on_submission_complete() override;
    GpuCommandBufferHandle begin_external_commands();
    void end_external_commands(GpuCommandBufferHandle command_buffer);
    void execute(GpuCommandBufferHandle command_buffer,
                 const KernelDispatch& dispatch,
                 const std::vector<KernelArg>& args,
                 const KernelExecutionHooks* hooks = nullptr) override;
    const void* shared_program_identity() const { return m_program.get(); }
    const void* shared_binding_schema_identity() const;
    size_t cached_descriptor_set_count();

private:
    void destroy_execution_state();
    void destroy_execution_state_locked();
    VkCommandBuffer begin_commands();
    void end_commands(VkCommandBuffer cmd);

    std::shared_ptr<VulkanKernelProgram> m_program;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    uint32_t m_queue_family = 0;
    VkCommandPool m_command_pool = VK_NULL_HANDLE;
    std::mutex m_mutex;
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
    std::shared_ptr<void> m_reuse_context;
};

}  // namespace gfx_plugin
}  // namespace ov
