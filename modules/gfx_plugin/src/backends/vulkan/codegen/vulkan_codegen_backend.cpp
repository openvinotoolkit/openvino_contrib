// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/codegen/vulkan_codegen_backend.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <iostream>
#include <unordered_map>

#include "openvino/core/except.hpp"

#include "mlir/gfx_mlir_kernel_metadata.hpp"
#include "mlir/spirv_codegen.hpp"

#include "backends/vulkan/runtime/vulkan_memory.hpp"
#include "kernel_ir/gfx_kernel_cache.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

constexpr uint32_t kCachedDescriptorSetsPerPool = 32;

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

// Quick scan of SPIR-V to infer max binding used in set 0.
static uint32_t count_bindings_in_spirv(const std::vector<uint32_t>& words) {
    if (words.size() < 5) {
        return 0;
    }
    struct DecorationState {
        bool has_binding = false;
        bool has_set = false;
        uint32_t binding = 0;
        uint32_t set = 0;
    };
    std::unordered_map<uint32_t, DecorationState> decorations;
    // SPIR-V header is 5 words; instructions start at word 5.
    for (size_t i = 5; i < words.size();) {
        uint32_t word = words[i];
        uint16_t wcount = word >> 16;
        uint16_t opcode = word & 0xFFFF;
        if (wcount == 0) {
            break;
        }
        if (opcode == 71 /*OpDecorate*/ && i + 3 < words.size()) {
            uint32_t target = words[i + 1];
            uint32_t decoration = words[i + 2];
            uint32_t value = words[i + 3];
            auto& state = decorations[target];
            if (decoration == 33 /*Binding*/) {
                state.has_binding = true;
                state.binding = value;
            } else if (decoration == 34 /*DescriptorSet*/) {
                state.has_set = true;
                state.set = value;
            }
        }
        i += wcount;
    }
    uint32_t max_binding = 0;
    bool found = false;
    for (const auto& [_, state] : decorations) {
        if (!state.has_binding) {
            continue;
        }
        if (state.has_set && state.set != 0) {
            continue;
        }
        found = true;
        max_binding = std::max(max_binding, state.binding);
    }
    return found ? (max_binding + 1) : 0;
}

class VulkanBindingSchema final {
public:
    struct PipelineLayoutHandles {
        VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    };

    VulkanBindingSchema(VkDevice device, uint32_t binding_count)
        : m_device(device), m_binding_count(binding_count) {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.reserve(m_binding_count);
        for (uint32_t i = 0; i < m_binding_count; ++i) {
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
            OPENVINO_THROW("GFX Vulkan: vkCreatePipelineLayout failed (bindings=",
                           m_binding_count,
                           "): ",
                           vk_result_to_string(res));
        }
    }

    ~VulkanBindingSchema() {
        std::lock_guard<std::mutex> lock(m_descriptor_mutex);
        for (auto& pool : m_cached_desc_pools) {
            if (pool.handle) {
                vkDestroyDescriptorPool(m_device, pool.handle, nullptr);
            }
        }
        m_cached_desc_pools.clear();
        m_descriptor_set_cache.clear();
        if (m_pipeline_layout) {
            vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
            m_pipeline_layout = VK_NULL_HANDLE;
        }
        if (m_desc_layout) {
            vkDestroyDescriptorSetLayout(m_device, m_desc_layout, nullptr);
            m_desc_layout = VK_NULL_HANDLE;
        }
    }

    PipelineLayoutHandles pipeline_layout_handles() const {
        return {m_desc_layout, m_pipeline_layout};
    }

    size_t cached_descriptor_set_count() const {
        std::lock_guard<std::mutex> lock(m_descriptor_mutex);
        return m_descriptor_set_cache.size();
    }

    VkDescriptorSet acquire_descriptor_set(const KernelBindingTable& bindings,
                                           const std::vector<VkDescriptorBufferInfo>& buffer_infos) const {
        std::lock_guard<std::mutex> lock(m_descriptor_mutex);
        if (auto it = m_descriptor_set_cache.find(bindings); it != m_descriptor_set_cache.end()) {
            return it->second;
        }

        if (m_cached_desc_pools.empty() ||
            m_cached_desc_pools.back().used_sets >= kCachedDescriptorSetsPerPool) {
            CachedDescriptorPool block;
            block.handle = create_descriptor_pool_locked(kCachedDescriptorSetsPerPool);
            m_cached_desc_pools.push_back(block);
        }

        auto& pool = m_cached_desc_pools.back();
        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = pool.handle;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &m_desc_layout;
        VkDescriptorSet desc_set = VK_NULL_HANDLE;
        VkResult res = vkAllocateDescriptorSets(m_device, &alloc_info, &desc_set);
        if (res != VK_SUCCESS) {
            OPENVINO_THROW("GFX Vulkan: vkAllocateDescriptorSets failed: ", vk_result_to_string(res));
        }

        if (!buffer_infos.empty()) {
            std::vector<VkWriteDescriptorSet> writes;
            writes.reserve(buffer_infos.size());
            for (size_t index = 0; index < buffer_infos.size(); ++index) {
                VkWriteDescriptorSet write{};
                write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write.dstSet = desc_set;
                write.dstBinding = static_cast<uint32_t>(index);
                write.dstArrayElement = 0;
                write.descriptorCount = 1;
                write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                write.pBufferInfo = &buffer_infos[index];
                writes.push_back(write);
            }
            vkUpdateDescriptorSets(m_device,
                                   static_cast<uint32_t>(writes.size()),
                                   writes.data(),
                                   0,
                                   nullptr);
        }

        ++pool.used_sets;
        m_descriptor_set_cache.emplace(bindings, desc_set);
        return desc_set;
    }

private:
    struct CachedDescriptorPool {
        VkDescriptorPool handle = VK_NULL_HANDLE;
        uint32_t used_sets = 0;
    };

    VkDescriptorPool create_descriptor_pool_locked(uint32_t max_sets) const {
        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = std::max<uint32_t>(1, m_binding_count) * std::max<uint32_t>(1, max_sets);

        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        pool_info.maxSets = std::max<uint32_t>(1, max_sets);
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkResult res = vkCreateDescriptorPool(m_device, &pool_info, nullptr, &pool);
        if (res != VK_SUCCESS) {
            OPENVINO_THROW("GFX Vulkan: vkCreateDescriptorPool failed: ", vk_result_to_string(res));
        }
        return pool;
    }

    VkDevice m_device = VK_NULL_HANDLE;
    const uint32_t m_binding_count = 0;
    VkDescriptorSetLayout m_desc_layout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
    mutable std::vector<CachedDescriptorPool> m_cached_desc_pools;
    mutable std::unordered_map<KernelBindingTable, VkDescriptorSet, KernelBindingTableHash> m_descriptor_set_cache;
    mutable std::mutex m_descriptor_mutex;
};

class VulkanDeviceReuseContext final {
public:
    explicit VulkanDeviceReuseContext(VkDevice device) : m_device(device) {
        VkPipelineCacheCreateInfo cache_info{};
        cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        VkResult res = vkCreatePipelineCache(m_device, &cache_info, nullptr, &m_pipeline_cache);
        if (res != VK_SUCCESS) {
            OPENVINO_THROW("GFX Vulkan: vkCreatePipelineCache failed: ", vk_result_to_string(res));
        }
    }

    ~VulkanDeviceReuseContext() {
        if (m_pipeline_cache) {
            vkDestroyPipelineCache(m_device, m_pipeline_cache, nullptr);
            m_pipeline_cache = VK_NULL_HANDLE;
        }
    }

    VkPipelineCache pipeline_cache() const {
        return m_pipeline_cache;
    }

    VkPipeline create_compute_pipeline(const VkComputePipelineCreateInfo& pipeline_info) {
        std::lock_guard<std::mutex> lock(m_pipeline_mutex);
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkResult res = vkCreateComputePipelines(m_device, m_pipeline_cache, 1, &pipeline_info, nullptr, &pipeline);
        if (res != VK_SUCCESS) {
            OPENVINO_THROW("GFX Vulkan: vkCreateComputePipelines failed: ", vk_result_to_string(res));
        }
        return pipeline;
    }

    std::shared_ptr<VulkanBindingSchema> acquire_binding_schema(uint32_t binding_count) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (auto it = m_binding_schemas.find(binding_count); it != m_binding_schemas.end()) {
            if (auto schema = it->second.lock()) {
                return schema;
            }
        }
        auto schema = std::make_shared<VulkanBindingSchema>(m_device, binding_count);
        m_binding_schemas[binding_count] = schema;
        return schema;
    }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkPipelineCache m_pipeline_cache = VK_NULL_HANDLE;
    std::mutex m_mutex;
    std::mutex m_pipeline_mutex;
    std::unordered_map<uint32_t, std::weak_ptr<VulkanBindingSchema>> m_binding_schemas;
};

class VulkanDeviceReuseRegistry final {
public:
    static VulkanDeviceReuseRegistry& instance() {
        static VulkanDeviceReuseRegistry registry;
        return registry;
    }

    std::shared_ptr<VulkanDeviceReuseContext> acquire(VkDevice device) {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto key = reinterpret_cast<uintptr_t>(device);
        if (auto it = m_contexts.find(key); it != m_contexts.end()) {
            if (auto context = it->second.lock()) {
                return context;
            }
        }
        auto context = std::make_shared<VulkanDeviceReuseContext>(device);
        m_contexts[key] = context;
        return context;
    }

private:
    std::mutex m_mutex;
    std::unordered_map<uintptr_t, std::weak_ptr<VulkanDeviceReuseContext>> m_contexts;
};

}  // namespace

class VulkanKernelProgram final {
public:
    struct PipelineHandles {
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    };

    VulkanKernelProgram(std::vector<uint32_t> spirv,
                        std::string entry_point,
                        uint32_t binding_count,
                        std::shared_ptr<VulkanDeviceReuseContext> reuse_context)
        : m_spirv(std::move(spirv)),
          m_entry_point(std::move(entry_point)),
          m_binding_count(binding_count),
          m_reuse_context(std::move(reuse_context)) {
        auto& ctx = VulkanContext::instance();
        m_device = ctx.device();
        m_queue = ctx.queue();
        m_queue_family = ctx.queue_family_index();
        OPENVINO_ASSERT(m_reuse_context, "GFX Vulkan: device reuse context is null");
        m_binding_schema = m_reuse_context->acquire_binding_schema(m_binding_count);

        VkShaderModuleCreateInfo shader_info{};
        shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shader_info.codeSize = m_spirv.size() * sizeof(uint32_t);
        shader_info.pCode = m_spirv.data();
        VkResult res = vkCreateShaderModule(m_device, &shader_info, nullptr, &m_shader_module);
        if (res != VK_SUCCESS) {
            OPENVINO_THROW("GFX Vulkan: vkCreateShaderModule failed: ", vk_result_to_string(res));
        }
    }

    ~VulkanKernelProgram() {
        std::lock_guard<std::mutex> lock(m_mutex);
        destroy_pipeline_locked();
        if (m_shader_module) {
            vkDestroyShaderModule(m_device, m_shader_module, nullptr);
            m_shader_module = VK_NULL_HANDLE;
        }
    }

    VkDevice device() const {
        return m_device;
    }

    VkQueue queue() const {
        return m_queue;
    }

    uint32_t queue_family() const {
        return m_queue_family;
    }

    const std::string& entry_point() const {
        return m_entry_point;
    }

    PipelineHandles pipeline_handles() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_pipeline || !m_pipeline_layout || !m_desc_layout) {
            create_pipeline_locked();
        }
        return {m_pipeline_layout, m_pipeline, m_desc_layout};
    }

    uint32_t binding_count() const {
        return m_binding_count;
    }

    size_t cached_descriptor_set_count() const {
        return m_binding_schema->cached_descriptor_set_count();
    }

    VkDescriptorSet acquire_descriptor_set(const KernelBindingTable& bindings,
                                           const std::vector<VkDescriptorBufferInfo>& buffer_infos) const {
        return m_binding_schema->acquire_descriptor_set(bindings, buffer_infos);
    }

    std::shared_ptr<VulkanBindingSchema> binding_schema() const {
        return m_binding_schema;
    }

    const void* binding_schema_identity() const {
        return m_binding_schema.get();
    }

private:
    void create_pipeline_locked() const {
        const auto shared_layout = m_binding_schema->pipeline_layout_handles();
        m_desc_layout = shared_layout.descriptor_set_layout;
        m_pipeline_layout = shared_layout.pipeline_layout;

        VkComputePipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.layout = m_pipeline_layout;
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = m_shader_module;
        pipeline_info.stage.pName = m_entry_point.c_str();
        m_pipeline = m_reuse_context->create_compute_pipeline(pipeline_info);
    }

    void destroy_pipeline_locked() const {
        if (m_pipeline) {
            vkDestroyPipeline(m_device, m_pipeline, nullptr);
            m_pipeline = VK_NULL_HANDLE;
        }
        m_desc_layout = VK_NULL_HANDLE;
        m_pipeline_layout = VK_NULL_HANDLE;
    }

    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    uint32_t m_queue_family = 0;
    std::vector<uint32_t> m_spirv;
    std::string m_entry_point;
    VkShaderModule m_shader_module = VK_NULL_HANDLE;
    mutable VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
    mutable VkPipeline m_pipeline = VK_NULL_HANDLE;
    mutable VkDescriptorSetLayout m_desc_layout = VK_NULL_HANDLE;
    const uint32_t m_binding_count = 0;
    std::shared_ptr<VulkanDeviceReuseContext> m_reuse_context;
    std::shared_ptr<VulkanBindingSchema> m_binding_schema;
    mutable std::mutex m_mutex;
};

class VulkanPreparedState final {
public:
    VulkanPreparedState(const KernelBindingTable& table, std::shared_ptr<VulkanBindingSchema> binding_schema)
        : m_binding_schema(std::move(binding_schema)) {
        OPENVINO_ASSERT(m_binding_schema, "GFX Vulkan: prepared state binding schema is null");
        const auto& bindings = table.buffers;
        buffer_infos.reserve(bindings.size());
        for (size_t index = 0; index < bindings.size(); ++index) {
            const auto& binding = bindings[index];
            VkBuffer buffer = vk_buffer_from_gpu(binding.buffer);
            OPENVINO_ASSERT(buffer, "GFX Vulkan: missing VkBuffer for prepared arg ", index);

            VkDescriptorBufferInfo info{};
            info.buffer = buffer;
            info.offset = binding.offset;
            info.range = binding.buffer.size ? binding.buffer.size - binding.offset : VK_WHOLE_SIZE;
            buffer_infos.push_back(info);
        }
        descriptor_set = m_binding_schema->acquire_descriptor_set(table, buffer_infos);
    }

    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    std::vector<VkDescriptorBufferInfo> buffer_infos;

private:
    std::shared_ptr<VulkanBindingSchema> m_binding_schema;
};

VulkanCompiledKernel::VulkanCompiledKernel(std::shared_ptr<VulkanKernelProgram> program, uint32_t arg_count)
    : CompiledKernelBase(arg_count), m_program(std::move(program)) {
    OPENVINO_ASSERT(m_program, "GFX Vulkan: compiled kernel program is null");
    m_device = m_program->device();
    m_queue = m_program->queue();
    m_queue_family = m_program->queue_family();
}

VulkanCompiledKernel::VulkanCompiledKernel(std::shared_ptr<VulkanKernelProgram> program,
                                           std::shared_ptr<const KernelBindingPlan> binding_plan)
    : CompiledKernelBase(std::move(binding_plan)), m_program(std::move(program)) {
    OPENVINO_ASSERT(m_program, "GFX Vulkan: compiled kernel program is null");
    m_device = m_program->device();
    m_queue = m_program->queue();
    m_queue_family = m_program->queue_family();
}

VulkanCompiledKernel::VulkanCompiledKernel(std::shared_ptr<VulkanKernelProgram> program,
                                           std::shared_ptr<const KernelBindingPlan> binding_plan,
                                           std::shared_ptr<void> prepared_binding_cache)
    : CompiledKernelBase(std::move(binding_plan), std::move(prepared_binding_cache)), m_program(std::move(program)) {
    OPENVINO_ASSERT(m_program, "GFX Vulkan: compiled kernel program is null");
    m_device = m_program->device();
    m_queue = m_program->queue();
    m_queue_family = m_program->queue_family();
}

VulkanCompiledKernel::~VulkanCompiledKernel() {
    destroy_execution_state();
}

size_t VulkanCompiledKernel::clamp_threadgroup_size(size_t desired) const {
    return desired == 0 ? 1 : desired;
}

void VulkanCompiledKernel::prepare_runtime_artifacts() {
    try {
        (void)m_program->pipeline_handles();
    } catch (const std::exception&) {
        // Some pipelines still require lazy materialization on-device.
        // Keep compile-time prewarm best-effort and preserve the proven runtime path.
    }
}

std::shared_ptr<ICompiledKernel> VulkanCompiledKernel::fork() const {
    return std::make_shared<VulkanCompiledKernel>(m_program, binding_plan(), prepared_binding_cache());
}

size_t VulkanCompiledKernel::cached_descriptor_set_count() {
    return m_program->cached_descriptor_set_count();
}

const void* VulkanCompiledKernel::shared_binding_schema_identity() const {
    return m_program->binding_schema_identity();
}

void VulkanCompiledKernel::on_submission_complete() {
    // Descriptor sets are cached across submissions using immutable binding keys.
}

GpuCommandBufferHandle VulkanCompiledKernel::begin_external_commands() {
    return reinterpret_cast<GpuCommandBufferHandle>(begin_commands());
}

void VulkanCompiledKernel::end_external_commands(GpuCommandBufferHandle command_buffer) {
    OPENVINO_ASSERT(command_buffer, "GFX Vulkan: external command buffer is null");
    end_commands(reinterpret_cast<VkCommandBuffer>(command_buffer));
}

void VulkanCompiledKernel::execute(GpuCommandBufferHandle command_buffer,
                                   const KernelDispatch& dispatch,
                                   const std::vector<KernelArg>& args,
                                   const KernelExecutionHooks* hooks) {
    auto prepared_base = get_or_create_prepared_bindings(args, "GFX Vulkan");
    const auto pipeline = m_program->pipeline_handles();
    const bool owns_command_buffer = (command_buffer == nullptr);
    auto prepared = prepared_base->get_or_create_backend_state<VulkanPreparedState>(
        reinterpret_cast<uintptr_t>(m_program->binding_schema_identity()),
        [&]() {
            return std::make_shared<VulkanPreparedState>(prepared_base->binding_table(), m_program->binding_schema());
        });
    const VkDescriptorSet desc_set = prepared->descriptor_set;

    VkCommandBuffer cmd = owns_command_buffer ? begin_commands()
                                              : reinterpret_cast<VkCommandBuffer>(command_buffer);
    VkCommandBuffer used_cmd = cmd;
    if (hooks && hooks->on_begin) {
        hooks->on_begin(reinterpret_cast<GpuCommandEncoderHandle>(cmd));
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline.pipeline_layout,
                            0,
                            1,
                            &desc_set,
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
        if (owns_command_buffer) {
            end_commands(used_cmd);
        }
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
    if (owns_command_buffer) {
        end_commands(used_cmd);
    }
    if (hooks && hooks->on_complete) {
        hooks->on_complete();
    }
}

void VulkanCompiledKernel::destroy_execution_state() {
    std::lock_guard<std::mutex> lock(m_mutex);
    destroy_execution_state_locked();
}

void VulkanCompiledKernel::destroy_execution_state_locked() {
    if (m_command_pool) {
        vkDestroyCommandPool(m_device, m_command_pool, nullptr);
        m_command_pool = VK_NULL_HANDLE;
    }
}

VkCommandBuffer VulkanCompiledKernel::begin_commands() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_command_pool) {
            VkCommandPoolCreateInfo pool_ci{};
            pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pool_ci.queueFamilyIndex = m_queue_family;
            pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            VkResult create_res = vkCreateCommandPool(m_device, &pool_ci, nullptr, &m_command_pool);
            if (create_res != VK_SUCCESS) {
                OPENVINO_THROW("GFX Vulkan: vkCreateCommandPool failed: ",
                               vk_result_to_string(create_res));
            }
        }
    }

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
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    res = vkCreateFence(m_device, &fence_info, nullptr, &fence);
    if (res != VK_SUCCESS) {
        vkFreeCommandBuffers(m_device, m_command_pool, 1, &cmd);
        OPENVINO_THROW("GFX Vulkan: vkCreateFence failed: ", vk_result_to_string(res));
    }
    res = vkQueueSubmit(m_queue, 1, &submit_info, fence);
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
    m_reuse_context = VulkanDeviceReuseRegistry::instance().acquire(static_cast<VkDevice>(m_device));
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

    uint32_t arg_count = static_cast<uint32_t>(
        infer_kernel_arg_count_from_module(module, source.signature.arg_count));
    const uint32_t spirv_binding_count = count_bindings_in_spirv(spirv_binary);
    const uint32_t binding_count = arg_count != 0 ? arg_count : spirv_binding_count;
    if (const char* dump_env = std::getenv("OV_GFX_DUMP_SPIRV_BINDINGS")) {
        if (spirv_binding_count && spirv_binding_count != arg_count) {
            std::cerr << "[GFX][Vulkan] entry=" << entry
                      << " arg_count=" << arg_count
                      << " spirv_bindings=" << spirv_binding_count << std::endl;
        }
    }
    const uintptr_t device_key = reinterpret_cast<uintptr_t>(m_device);
    auto shared_prepared_cache = acquire_shared_prepared_binding_cache(GpuBackend::Vulkan, device_key, arg_count);
    return lookup_or_compile_kernel(GpuBackend::Vulkan,
                                    device_key,
                                    spirv_binary.data(),
                                    spirv_binary.size() * sizeof(uint32_t),
                                    entry,
                                    arg_count,
                                    [&]() -> std::shared_ptr<ICompiledKernel> {
                                        auto program =
                                            std::make_shared<VulkanKernelProgram>(std::move(spirv_binary),
                                                                                  entry,
                                                                                  binding_count,
                                                                                  std::static_pointer_cast<VulkanDeviceReuseContext>(m_reuse_context));
                                        auto binding_plan = std::make_shared<KernelBindingPlan>(arg_count);
                                        return std::make_shared<VulkanCompiledKernel>(std::move(program),
                                                                                      std::move(binding_plan),
                                                                                      shared_prepared_cache);
                                    });
}

}  // namespace gfx_plugin
}  // namespace ov
