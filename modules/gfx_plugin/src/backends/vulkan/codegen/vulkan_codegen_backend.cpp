// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/codegen/vulkan_codegen_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "openvino/core/except.hpp"

#include "mlir/gfx_mlir_kernel_metadata.hpp"
#include "mlir/spirv_codegen.hpp"

#include "backends/vulkan/runtime/vulkan_memory.hpp"
#include "kernel_ir/gfx_kernel_cache.hpp"
#include "runtime/gfx_compile_profiling.hpp"
#include "runtime/gfx_vulkan_pipeline_cache_scope.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

constexpr uint32_t kCachedDescriptorSetsPerPool = 32;

std::string sanitize_cache_path_component(std::string value) {
    for (char& ch : value) {
        const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') ||
                        ch == '-' || ch == '_' || ch == '.';
        if (!ok) {
            ch = '_';
        }
    }
    if (value.empty()) {
        return "unknown";
    }
    return value;
}

std::string bytes_to_hex(const uint8_t* bytes, size_t size) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.resize(size * 2);
    for (size_t i = 0; i < size; ++i) {
        out[i * 2] = kHex[(bytes[i] >> 4) & 0xF];
        out[i * 2 + 1] = kHex[bytes[i] & 0xF];
    }
    return out;
}

std::filesystem::path make_vulkan_pipeline_cache_path(const std::string& cache_dir, VkPhysicalDevice physical_device) {
    if (cache_dir.empty() || physical_device == VK_NULL_HANDLE) {
        return {};
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical_device, &props);
    std::ostringstream file_name;
    file_name << "pipeline_cache_vulkan_"
              << sanitize_cache_path_component(props.deviceName)
              << "_vendor_" << props.vendorID
              << "_device_" << props.deviceID
              << "_driver_" << props.driverVersion
              << "_" << bytes_to_hex(props.pipelineCacheUUID, VK_UUID_SIZE)
              << ".bin";
    return std::filesystem::path(cache_dir) / "gfx_plugin" / "vulkan" / file_name.str();
}

std::vector<uint8_t> read_binary_file_best_effort(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return {};
    }
    input.seekg(0, std::ios::end);
    const auto end = input.tellg();
    if (end <= 0) {
        return {};
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(end));
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input) {
        return {};
    }
    return bytes;
}

bool write_binary_file_atomically(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
        return false;
    }
    auto tmp_path = path;
    tmp_path += ".tmp";
    {
        std::ofstream output(tmp_path, std::ios::binary | std::ios::trunc);
        if (!output) {
            return false;
        }
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!output) {
            return false;
        }
    }
    std::filesystem::rename(tmp_path, path, ec);
    if (!ec) {
        return true;
    }
    ec.clear();
    std::filesystem::remove(path, ec);
    ec.clear();
    std::filesystem::rename(tmp_path, path, ec);
    if (!ec) {
        return true;
    }
    ec.clear();
    std::filesystem::remove(tmp_path, ec);
    return false;
}

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

uint32_t record_compute_buffer_barriers(VkCommandBuffer command_buffer,
                                        const std::vector<VkDescriptorBufferInfo>& buffer_infos) {
    std::vector<VkBufferMemoryBarrier> barriers;
    barriers.reserve(buffer_infos.size());
    for (const auto& info : buffer_infos) {
        if (info.buffer == VK_NULL_HANDLE) {
            continue;
        }
        const bool duplicate = std::any_of(barriers.begin(), barriers.end(), [&](const VkBufferMemoryBarrier& barrier) {
            return barrier.buffer == info.buffer && barrier.offset == info.offset && barrier.size == info.range;
        });
        if (duplicate) {
            continue;
        }

        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = info.buffer;
        barrier.offset = info.offset;
        barrier.size = info.range;
        barriers.push_back(barrier);
    }

    if (barriers.empty()) {
        return 0;
    }

    vkCmdPipelineBarrier(command_buffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,
                         0,
                         nullptr,
                         static_cast<uint32_t>(barriers.size()),
                         barriers.data(),
                         0,
                         nullptr);
    return static_cast<uint32_t>(barriers.size());
}

struct CommandBufferWriteRange {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
};

VkDeviceSize normalize_barrier_size(VkDeviceSize size) {
    return size == VK_WHOLE_SIZE ? std::numeric_limits<VkDeviceSize>::max() : size;
}

bool ranges_overlap(VkDeviceSize lhs_offset, VkDeviceSize lhs_size, VkDeviceSize rhs_offset, VkDeviceSize rhs_size) {
    const VkDeviceSize lhs_norm = normalize_barrier_size(lhs_size);
    const VkDeviceSize rhs_norm = normalize_barrier_size(rhs_size);
    const VkDeviceSize lhs_end =
        lhs_norm == std::numeric_limits<VkDeviceSize>::max() ? lhs_norm : lhs_offset + lhs_norm;
    const VkDeviceSize rhs_end =
        rhs_norm == std::numeric_limits<VkDeviceSize>::max() ? rhs_norm : rhs_offset + rhs_norm;
    return lhs_offset < rhs_end && rhs_offset < lhs_end;
}

bool descriptor_ranges_overlap(const VkDescriptorBufferInfo& lhs, const VkDescriptorBufferInfo& rhs) {
    if (lhs.buffer == VK_NULL_HANDLE || rhs.buffer == VK_NULL_HANDLE || lhs.buffer != rhs.buffer) {
        return false;
    }
    return ranges_overlap(lhs.offset, lhs.range, rhs.offset, rhs.range);
}

bool descriptor_is_written(const VkDescriptorBufferInfo& info,
                           const std::vector<VkDescriptorBufferInfo>& current_writes) {
    return std::any_of(current_writes.begin(), current_writes.end(), [&](const VkDescriptorBufferInfo& write) {
        return descriptor_ranges_overlap(info, write);
    });
}

bool descriptor_is_read(const VkDescriptorBufferInfo& info,
                        const std::vector<VkDescriptorBufferInfo>& current_accesses,
                        const std::vector<VkDescriptorBufferInfo>& current_writes) {
    return std::any_of(current_accesses.begin(), current_accesses.end(), [&](const VkDescriptorBufferInfo& access) {
        return descriptor_ranges_overlap(info, access) && !descriptor_is_written(access, current_writes);
    });
}

VkAccessFlags descriptor_dst_access_mask(const VkDescriptorBufferInfo& info,
                                         const std::vector<VkDescriptorBufferInfo>& current_accesses,
                                         const std::vector<VkDescriptorBufferInfo>& current_writes) {
    VkAccessFlags mask = 0;
    if (descriptor_is_read(info, current_accesses, current_writes)) {
        mask |= VK_ACCESS_SHADER_READ_BIT;
    }
    if (descriptor_is_written(info, current_writes)) {
        mask |= VK_ACCESS_SHADER_WRITE_BIT;
    }
    return mask != 0 ? mask : (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

class VulkanCommandBufferAccessTracker final {
public:
    static VulkanCommandBufferAccessTracker& instance() {
        static auto* tracker = new VulkanCommandBufferAccessTracker();
        return *tracker;
    }

    void reset(VkCommandBuffer command_buffer) {
        if (command_buffer == VK_NULL_HANDLE) {
            return;
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        m_pending_writes[reinterpret_cast<uintptr_t>(command_buffer)].clear();
    }

    uint32_t record_required_barriers(VkCommandBuffer command_buffer,
                                      const std::vector<VkDescriptorBufferInfo>& current_accesses,
                                      const std::vector<VkDescriptorBufferInfo>& current_writes) {
        if (command_buffer == VK_NULL_HANDLE) {
            return 0;
        }

        std::vector<VkBufferMemoryBarrier> barriers;
        std::vector<CommandBufferWriteRange> next_pending_writes;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            auto& pending_writes = m_pending_writes[reinterpret_cast<uintptr_t>(command_buffer)];

            auto append_unique_barrier = [&](const VkDescriptorBufferInfo& info) {
                const bool duplicate = std::any_of(barriers.begin(), barriers.end(), [&](const VkBufferMemoryBarrier& barrier) {
                    return barrier.buffer == info.buffer && barrier.offset == info.offset && barrier.size == info.range;
                });
                if (duplicate) {
                    return;
                }
                VkBufferMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barrier.dstAccessMask = descriptor_dst_access_mask(info, current_accesses, current_writes);
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.buffer = info.buffer;
                barrier.offset = info.offset;
                barrier.size = info.range;
                barriers.push_back(barrier);
            };

            auto is_accessed_by_current = [&](const CommandBufferWriteRange& write) {
                return std::any_of(current_accesses.begin(), current_accesses.end(), [&](const VkDescriptorBufferInfo& info) {
                    return info.buffer != VK_NULL_HANDLE &&
                           info.buffer == write.buffer &&
                           ranges_overlap(info.offset, info.range, write.offset, write.size);
                });
            };

            next_pending_writes.reserve(pending_writes.size() + current_writes.size());
            for (const auto& write : pending_writes) {
                if (!is_accessed_by_current(write)) {
                    next_pending_writes.push_back(write);
                    continue;
                }
                for (const auto& info : current_accesses) {
                    if (info.buffer == VK_NULL_HANDLE || info.buffer != write.buffer ||
                        !ranges_overlap(info.offset, info.range, write.offset, write.size)) {
                        continue;
                    }
                    append_unique_barrier(info);
                }
            }

            for (const auto& info : current_writes) {
                if (info.buffer == VK_NULL_HANDLE) {
                    continue;
                }
                const bool duplicate = std::any_of(next_pending_writes.begin(),
                                                   next_pending_writes.end(),
                                                   [&](const CommandBufferWriteRange& write) {
                                                       return write.buffer == info.buffer && write.offset == info.offset &&
                                                              write.size == info.range;
                                                   });
                if (!duplicate) {
                    next_pending_writes.push_back({info.buffer, info.offset, info.range});
                }
            }
            pending_writes = next_pending_writes;
        }

        if (barriers.empty()) {
            return 0;
        }

        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             0,
                             nullptr,
                             static_cast<uint32_t>(barriers.size()),
                             barriers.data(),
                             0,
                             nullptr);
        return static_cast<uint32_t>(barriers.size());
    }

private:
    std::mutex m_mutex;
    std::unordered_map<uintptr_t, std::vector<CommandBufferWriteRange>> m_pending_writes;
};

uint32_t resolve_kernel_output_arg_count(const KernelSource& source) {
    if (source.signature.output_arg_count != 0) {
        return source.signature.output_arg_count;
    }
    if (!source.module) {
        return 0;
    }
    if (auto attr = source.module->getAttrOfType<mlir::IntegerAttr>("gfx.kernel_output_arg_count")) {
        return static_cast<uint32_t>(std::max<int64_t>(attr.getInt(), 0));
    }
    return 0;
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

    struct DescriptorAcquireResult {
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        uint64_t cpu_us = 0;
        uint64_t lookup_cpu_us = 0;
        uint64_t allocate_cpu_us = 0;
        uint64_t write_cpu_us = 0;
        uint64_t pool_create_cpu_us = 0;
        uint32_t write_count = 0;
        bool cache_hit = false;
        bool pool_created = false;
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

    DescriptorAcquireResult acquire_descriptor_set(const KernelBindingTable& bindings,
                                                   const std::vector<VkDescriptorBufferInfo>& buffer_infos) const {
        const auto start = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(m_descriptor_mutex);
        const auto lookup_start = std::chrono::steady_clock::now();
        if (auto it = m_descriptor_set_cache.find(bindings); it != m_descriptor_set_cache.end()) {
            DescriptorAcquireResult result;
            result.descriptor_set = it->second;
            result.cache_hit = true;
            result.lookup_cpu_us = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - lookup_start).count());
            result.cpu_us = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
            return result;
        }
        const uint64_t lookup_cpu_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - lookup_start).count());

        bool pool_created = false;
        uint64_t pool_create_cpu_us = 0;
        if (m_cached_desc_pools.empty() ||
            m_cached_desc_pools.back().used_sets >= kCachedDescriptorSetsPerPool) {
            const auto pool_create_start = std::chrono::steady_clock::now();
            CachedDescriptorPool block;
            block.handle = create_descriptor_pool_locked(kCachedDescriptorSetsPerPool);
            m_cached_desc_pools.push_back(block);
            pool_created = true;
            pool_create_cpu_us = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - pool_create_start).count());
        }

        auto& pool = m_cached_desc_pools.back();
        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = pool.handle;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &m_desc_layout;
        VkDescriptorSet desc_set = VK_NULL_HANDLE;
        const auto allocate_start = std::chrono::steady_clock::now();
        VkResult res = vkAllocateDescriptorSets(m_device, &alloc_info, &desc_set);
        if (res != VK_SUCCESS) {
            OPENVINO_THROW("GFX Vulkan: vkAllocateDescriptorSets failed: ", vk_result_to_string(res));
        }
        const uint64_t allocate_cpu_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - allocate_start).count());

        uint32_t write_count = 0;
        uint64_t write_cpu_us = 0;
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
            write_count = static_cast<uint32_t>(writes.size());
            const auto write_start = std::chrono::steady_clock::now();
            vkUpdateDescriptorSets(m_device,
                                   write_count,
                                   writes.data(),
                                   0,
                                   nullptr);
            write_cpu_us = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - write_start).count());
        }

        ++pool.used_sets;
        m_descriptor_set_cache.emplace(bindings, desc_set);
        DescriptorAcquireResult result;
        result.descriptor_set = desc_set;
        result.lookup_cpu_us = lookup_cpu_us;
        result.allocate_cpu_us = allocate_cpu_us;
        result.write_cpu_us = write_cpu_us;
        result.pool_create_cpu_us = pool_create_cpu_us;
        result.write_count = write_count;
        result.pool_created = pool_created;
        result.cpu_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
        return result;
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
    explicit VulkanDeviceReuseContext(VkDevice device, VkPhysicalDevice physical_device)
        : m_device(device),
          m_physical_device(physical_device) {}

    ~VulkanDeviceReuseContext() {
        if (m_pipeline_cache) {
            vkDestroyPipelineCache(m_device, m_pipeline_cache, nullptr);
            m_pipeline_cache = VK_NULL_HANDLE;
        }
    }

    void configure_cache_dir(const std::string& cache_dir) {
        if (cache_dir.empty()) {
            return;
        }
        std::lock_guard<std::mutex> lock(m_pipeline_mutex);
        const auto path = make_vulkan_pipeline_cache_path(cache_dir, m_physical_device);
        if (path.empty()) {
            return;
        }
        if (!m_pipeline_cache_path.empty() && m_pipeline_cache_path != path) {
            return;
        }
        m_pipeline_cache_path = path;
        if (m_pipeline_cache != VK_NULL_HANDLE && m_cache_loaded) {
            return;
        }
        if (m_pipeline_cache == VK_NULL_HANDLE) {
            return;
        }
        save_pipeline_cache_locked();
    }

    VkPipelineCache pipeline_cache() const {
        std::lock_guard<std::mutex> lock(m_pipeline_mutex);
        ensure_pipeline_cache_locked();
        return m_pipeline_cache;
    }

    VkPipeline create_compute_pipeline(const VkComputePipelineCreateInfo& pipeline_info) {
        std::lock_guard<std::mutex> lock(m_pipeline_mutex);
        ensure_pipeline_cache_locked();
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkResult res = vkCreateComputePipelines(m_device, m_pipeline_cache, 1, &pipeline_info, nullptr, &pipeline);
        if (res != VK_SUCCESS) {
            OPENVINO_THROW("GFX Vulkan: vkCreateComputePipelines failed for entry ",
                           pipeline_info.stage.pName ? pipeline_info.stage.pName : "<unknown>",
                           " with ",
                           pipeline_info.layout ? "valid" : "null",
                           " pipeline layout, error=",
                           vk_result_to_string(res));
        }
        if (!current_vulkan_pipeline_cache_dir().empty()) {
            save_pipeline_cache_locked();
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
    void ensure_pipeline_cache_locked() const {
        if (m_pipeline_cache != VK_NULL_HANDLE) {
            return;
        }
        std::vector<uint8_t> initial_data;
        if (!m_pipeline_cache_path.empty()) {
            initial_data = read_binary_file_best_effort(m_pipeline_cache_path);
        }

        VkPipelineCacheCreateInfo cache_info{};
        cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        cache_info.initialDataSize = initial_data.size();
        cache_info.pInitialData = initial_data.empty() ? nullptr : initial_data.data();
        VkResult res = vkCreatePipelineCache(m_device, &cache_info, nullptr, &m_pipeline_cache);
        if (res != VK_SUCCESS && !initial_data.empty()) {
            cache_info.initialDataSize = 0;
            cache_info.pInitialData = nullptr;
            res = vkCreatePipelineCache(m_device, &cache_info, nullptr, &m_pipeline_cache);
            if (current_compile_trace()) {
                increment_compile_counter("vulkan_pipeline_cache_load_fallback_count");
            }
        }
        if (res != VK_SUCCESS) {
            OPENVINO_THROW("GFX Vulkan: vkCreatePipelineCache failed: ", vk_result_to_string(res));
        }
        if (current_compile_trace()) {
            increment_compile_counter(initial_data.empty() ? "vulkan_pipeline_cache_miss_count"
                                                           : "vulkan_pipeline_cache_hit_count");
        }
        m_cache_loaded = !initial_data.empty();
    }

    void save_pipeline_cache_locked() const {
        if (m_pipeline_cache == VK_NULL_HANDLE || m_pipeline_cache_path.empty()) {
            return;
        }
        size_t size = 0;
        VkResult res = vkGetPipelineCacheData(m_device, m_pipeline_cache, &size, nullptr);
        if (res != VK_SUCCESS || size == 0) {
            return;
        }
        std::vector<uint8_t> data(size);
        res = vkGetPipelineCacheData(m_device, m_pipeline_cache, &size, data.data());
        if (res != VK_SUCCESS || size == 0) {
            return;
        }
        data.resize(size);
        if (!write_binary_file_atomically(m_pipeline_cache_path, data)) {
            return;
        }
        if (current_compile_trace()) {
            increment_compile_counter("vulkan_pipeline_cache_store_count");
        }
    }

    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
    mutable VkPipelineCache m_pipeline_cache = VK_NULL_HANDLE;
    std::mutex m_mutex;
    mutable std::mutex m_pipeline_mutex;
    std::unordered_map<uint32_t, std::weak_ptr<VulkanBindingSchema>> m_binding_schemas;
    mutable std::filesystem::path m_pipeline_cache_path;
    mutable bool m_cache_loaded = false;
};

class VulkanDeviceReuseRegistry final {
public:
    static VulkanDeviceReuseRegistry& instance() {
        static auto* registry = new VulkanDeviceReuseRegistry();
        return *registry;
    }

    std::shared_ptr<VulkanDeviceReuseContext> acquire(VkDevice device, VkPhysicalDevice physical_device) {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto key = reinterpret_cast<uintptr_t>(device);
        if (auto it = m_contexts.find(key); it != m_contexts.end()) {
            if (auto context = it->second.lock()) {
                context->configure_cache_dir(current_vulkan_pipeline_cache_dir());
                return context;
            }
        }
        auto context = std::make_shared<VulkanDeviceReuseContext>(device, physical_device);
        context->configure_cache_dir(current_vulkan_pipeline_cache_dir());
        m_contexts[key] = context;
        return context;
    }

private:
    std::mutex m_mutex;
    std::unordered_map<uintptr_t, std::weak_ptr<VulkanDeviceReuseContext>> m_contexts;
};

}  // namespace

void vulkan_reset_command_buffer_access_tracker(VkCommandBuffer command_buffer) {
    VulkanCommandBufferAccessTracker::instance().reset(command_buffer);
}

class VulkanKernelProgram final {
public:
    struct PipelineHandles {
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    };

    struct PipelineAccessResult {
        PipelineHandles handles;
        uint64_t creation_cpu_us = 0;
        bool created = false;
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
        const auto shader_module_start =
            current_compile_trace() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        VkResult res = vkCreateShaderModule(m_device, &shader_info, nullptr, &m_shader_module);
        if (current_compile_trace()) {
            increment_compile_counter("vulkan_shader_module_create_count");
            add_compile_segment(
                "vulkan_shader_module_create",
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                          std::chrono::steady_clock::now() - shader_module_start)
                                          .count()));
        }
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

    PipelineAccessResult pipeline_handles() const {
        PipelineAccessResult result;
        std::lock_guard<std::mutex> lock(m_mutex);
        const bool missing_pipeline = (!m_pipeline || !m_pipeline_layout || !m_desc_layout);
        const auto start = missing_pipeline ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (!m_pipeline || !m_pipeline_layout || !m_desc_layout) {
            create_pipeline_locked();
            result.created = true;
            result.creation_cpu_us = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
        }
        result.handles = {m_pipeline_layout, m_pipeline, m_desc_layout};
        return result;
    }

    uint32_t binding_count() const {
        return m_binding_count;
    }

    size_t cached_descriptor_set_count() const {
        return m_binding_schema->cached_descriptor_set_count();
    }

    VulkanBindingSchema::DescriptorAcquireResult acquire_descriptor_set(
        const KernelBindingTable& bindings,
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
    VulkanPreparedState(const KernelBindingTable& table,
                        std::shared_ptr<VulkanBindingSchema> binding_schema,
                        uint32_t output_arg_count)
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
            info.range = binding.buffer.size ? static_cast<VkDeviceSize>(binding.buffer.size) : VK_WHOLE_SIZE;
            buffer_infos.push_back(info);
        }
        const size_t writable_count = std::min<size_t>(output_arg_count, buffer_infos.size());
        if (writable_count != 0) {
            writable_buffer_infos.assign(buffer_infos.end() - static_cast<std::ptrdiff_t>(writable_count), buffer_infos.end());
        }
        const auto acquire = m_binding_schema->acquire_descriptor_set(table, buffer_infos);
        descriptor_set = acquire.descriptor_set;
        descriptor_cpu_us = acquire.cpu_us;
        descriptor_lookup_cpu_us = acquire.lookup_cpu_us;
        descriptor_allocate_cpu_us = acquire.allocate_cpu_us;
        descriptor_write_cpu_us = acquire.write_cpu_us;
        descriptor_pool_create_cpu_us = acquire.pool_create_cpu_us;
        descriptor_write_count = acquire.write_count;
        descriptor_cache_hit = acquire.cache_hit;
        descriptor_pool_created = acquire.pool_created;
    }

    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    std::vector<VkDescriptorBufferInfo> buffer_infos;
    std::vector<VkDescriptorBufferInfo> writable_buffer_infos;
    uint64_t descriptor_cpu_us = 0;
    uint64_t descriptor_lookup_cpu_us = 0;
    uint64_t descriptor_allocate_cpu_us = 0;
    uint64_t descriptor_write_cpu_us = 0;
    uint64_t descriptor_pool_create_cpu_us = 0;
    uint32_t descriptor_write_count = 0;
    bool descriptor_cache_hit = false;
    bool descriptor_pool_created = false;

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
        // Prewarm the shared pipeline objects during compile so the first infer
        // does not pay for lazy vkCreateComputePipelines on mobile Vulkan.
        (void)m_program->pipeline_handles();
    } catch (const std::exception&) {
        // Keep compile-time prewarm best-effort and preserve the lazy runtime path.
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

void VulkanCompiledKernel::prewarm_bindings(const std::vector<KernelArg>& args) {
    (void)m_program->pipeline_handles();
    auto prepared_base = get_or_create_prepared_bindings(args, "GFX Vulkan prewarm");
    (void)prepared_base->get_or_create_backend_state<VulkanPreparedState>(
        reinterpret_cast<uintptr_t>(m_program->binding_schema_identity()),
        [&]() {
            return std::make_shared<VulkanPreparedState>(prepared_base->binding_table(),
                                                         m_program->binding_schema(),
                                                         binding_plan()->output_arg_count());
        });
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
    const auto pipeline_access = m_program->pipeline_handles();
    const auto& pipeline = pipeline_access.handles;
    const bool owns_command_buffer = (command_buffer == nullptr);
    if (owns_command_buffer && hooks && hooks->on_event) {
        hooks->on_event("vulkan_owns_command_buffer");
    }
    if (pipeline_access.created) {
        if (hooks && hooks->on_counter) {
            hooks->on_counter("pipeline_creation_count", 1);
        }
        if (hooks && hooks->on_segment) {
            hooks->on_segment("compile",
                              "pipeline_creation",
                              std::chrono::microseconds{static_cast<int64_t>(pipeline_access.creation_cpu_us)},
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              -1,
                              reinterpret_cast<uint64_t>(m_queue),
                              0);
        }
    }
    bool prepared_state_created = false;
    auto prepared = prepared_base->get_or_create_backend_state<VulkanPreparedState>(
        reinterpret_cast<uintptr_t>(m_program->binding_schema_identity()),
        [&]() {
            prepared_state_created = true;
            return std::make_shared<VulkanPreparedState>(prepared_base->binding_table(),
                                                         m_program->binding_schema(),
                                                         binding_plan()->output_arg_count());
        });
    if (prepared_state_created) {
        if (hooks && hooks->on_counter) {
            hooks->on_counter(prepared->descriptor_cache_hit ? "descriptor_cache_hit_count" : "descriptor_cache_miss_count", 1);
            if (prepared->descriptor_write_count != 0) {
                hooks->on_counter("descriptor_update_count", 1);
                hooks->on_counter("descriptor_write_count", prepared->descriptor_write_count);
            }
            if (prepared->descriptor_pool_created) {
                hooks->on_counter("descriptor_pool_create_count", 1);
            }
        }
        if (hooks && hooks->on_segment) {
            hooks->on_segment("descriptor_update",
                              prepared->descriptor_cache_hit ? "descriptor_cache_hit" : "descriptor_cache_miss",
                              std::chrono::microseconds{static_cast<int64_t>(prepared->descriptor_lookup_cpu_us != 0
                                                                                 ? prepared->descriptor_lookup_cpu_us
                                                                                 : prepared->descriptor_cpu_us)},
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              -1,
                              reinterpret_cast<uint64_t>(m_queue),
                              0);
            if (prepared->descriptor_pool_create_cpu_us != 0) {
                hooks->on_segment("descriptor_update",
                                  "descriptor_pool_create",
                                  std::chrono::microseconds{
                                      static_cast<int64_t>(prepared->descriptor_pool_create_cpu_us)},
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  -1,
                                  reinterpret_cast<uint64_t>(m_queue),
                                  0);
            }
            if (prepared->descriptor_allocate_cpu_us != 0) {
                hooks->on_segment("descriptor_update",
                                  "descriptor_set_allocate",
                                  std::chrono::microseconds{
                                      static_cast<int64_t>(prepared->descriptor_allocate_cpu_us)},
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  -1,
                                  reinterpret_cast<uint64_t>(m_queue),
                                  0);
            }
            if (prepared->descriptor_write_cpu_us != 0) {
                hooks->on_segment("descriptor_update",
                                  "descriptor_write",
                                  std::chrono::microseconds{
                                      static_cast<int64_t>(prepared->descriptor_write_cpu_us)},
                                  0,
                                  prepared->descriptor_write_count,
                                  0,
                                  0,
                                  0,
                                  0,
                                  -1,
                                  reinterpret_cast<uint64_t>(m_queue),
                                  0);
            }
        }
    } else if (hooks && hooks->on_counter) {
        hooks->on_counter("prepared_binding_cache_hit_count", 1);
    }
    const VkDescriptorSet desc_set = prepared->descriptor_set;

    VkCommandBuffer cmd = owns_command_buffer ? begin_commands()
                                              : reinterpret_cast<VkCommandBuffer>(command_buffer);
    VkCommandBuffer used_cmd = cmd;
    if (hooks && hooks->on_begin) {
        hooks->on_begin(reinterpret_cast<GpuCommandEncoderHandle>(cmd));
    }

    const bool trace_bind_commands = hooks && (hooks->on_segment || hooks->on_counter);
    const auto bind_pipeline_start =
        trace_bind_commands ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    const auto after_bind_pipeline =
        trace_bind_commands ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline.pipeline_layout,
                            0,
                            1,
                            &desc_set,
                            0,
                            nullptr);
    if (trace_bind_commands) {
        const auto bind_pipeline_cpu_us =
            std::chrono::duration_cast<std::chrono::microseconds>(after_bind_pipeline - bind_pipeline_start);
        const auto bind_descriptor_cpu_us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - after_bind_pipeline);
        if (hooks->on_counter) {
            hooks->on_counter("pipeline_bind_count", 1);
            hooks->on_counter("descriptor_bind_count", 1);
        }
        if (hooks->on_segment) {
            hooks->on_segment("descriptor_update",
                              "vk_bind_pipeline",
                              bind_pipeline_cpu_us,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              -1,
                              reinterpret_cast<uint64_t>(m_queue),
                              reinterpret_cast<uint64_t>(cmd));
            hooks->on_segment("descriptor_update",
                              "vk_bind_descriptors",
                              bind_descriptor_cpu_us,
                              0,
                              1,
                              0,
                              0,
                              0,
                              0,
                              -1,
                              reinterpret_cast<uint64_t>(m_queue),
                              reinterpret_cast<uint64_t>(cmd));
        }
    }

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
    const auto& writable_infos =
        prepared->writable_buffer_infos.empty() ? prepared->buffer_infos : prepared->writable_buffer_infos;
    const uint32_t barrier_buffer_count =
        VulkanCommandBufferAccessTracker::instance().record_required_barriers(cmd, prepared->buffer_infos, writable_infos);
    if (hooks && hooks->on_counter && barrier_buffer_count != 0) {
        hooks->on_counter("barrier_count", 1);
        hooks->on_counter("barrier_buffer_count", barrier_buffer_count);
    }
    if (hooks && hooks->on_segment && barrier_buffer_count != 0) {
        hooks->on_segment("barrier",
                          "compute_buffer_barrier",
                          std::chrono::microseconds{0},
                          0,
                          0,
                          0,
                          0,
                          0,
                          0,
                          -1,
                          reinterpret_cast<uint64_t>(m_queue),
                          reinterpret_cast<uint64_t>(cmd));
    }
    vkCmdDispatch(cmd, std::max<uint32_t>(groups_x, 1), std::max<uint32_t>(groups_y, 1), std::max<uint32_t>(groups_z, 1));

    if (hooks && hooks->on_end) {
        hooks->on_end(reinterpret_cast<GpuCommandEncoderHandle>(cmd));
    }
    if (owns_command_buffer) {
        if (hooks && hooks->on_event) {
            hooks->on_event("vulkan_internal_submit_wait");
        }
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
    if (m_submission_fence) {
        vkDestroyFence(m_device, m_submission_fence, nullptr);
        m_submission_fence = VK_NULL_HANDLE;
    }
    m_command_buffer = VK_NULL_HANDLE;
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
        if (!m_command_buffer) {
            VkCommandBufferAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            alloc_info.commandPool = m_command_pool;
            alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            alloc_info.commandBufferCount = 1;

            VkResult alloc_res = vkAllocateCommandBuffers(m_device, &alloc_info, &m_command_buffer);
            if (alloc_res != VK_SUCCESS) {
                OPENVINO_THROW("GFX Vulkan: vkAllocateCommandBuffers failed: ", vk_result_to_string(alloc_res));
            }
        }
        if (!m_submission_fence) {
            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            VkResult fence_res = vkCreateFence(m_device, &fence_info, nullptr, &m_submission_fence);
            if (fence_res != VK_SUCCESS) {
                OPENVINO_THROW("GFX Vulkan: vkCreateFence failed: ", vk_result_to_string(fence_res));
            }
        }
    }

    VkResult res = vkResetCommandPool(m_device, m_command_pool, 0);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkResetCommandPool failed: ", vk_result_to_string(res));
    }

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    res = vkBeginCommandBuffer(m_command_buffer, &begin_info);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkBeginCommandBuffer failed: ", vk_result_to_string(res));
    }
    vulkan_reset_command_buffer_access_tracker(m_command_buffer);
    return m_command_buffer;
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
    OPENVINO_ASSERT(m_submission_fence, "GFX Vulkan: submission fence is not initialized");
    res = vkResetFences(m_device, 1, &m_submission_fence);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkResetFences failed: ", vk_result_to_string(res));
    }
    res = vkQueueSubmit(m_queue, 1, &submit_info, m_submission_fence);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkQueueSubmit failed: ", vk_result_to_string(res));
    }
    res = vkWaitForFences(m_device, 1, &m_submission_fence, VK_TRUE, UINT64_MAX);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkWaitForFences failed: ", vk_result_to_string(res));
    }
}

VulkanCodegenBackend::VulkanCodegenBackend(VulkanDeviceHandle device) : m_device(device) {
    if (!m_device) {
        auto& ctx = VulkanContext::instance();
        m_device = ctx.device();
        m_physical_device = ctx.physical_device();
    }
    if (!m_physical_device) {
        m_physical_device = VulkanContext::instance().physical_device();
    }
    m_reuse_context =
        VulkanDeviceReuseRegistry::instance().acquire(static_cast<VkDevice>(m_device), static_cast<VkPhysicalDevice>(m_physical_device));
}

std::shared_ptr<ICompiledKernel> VulkanCodegenBackend::compile(const KernelSource& source, std::string* log) {
    mlir::ModuleOp module = source.module;
    std::string entry = resolve_entry_point(source, "gfx_kernel");

    std::string local_log;
    std::string* log_ptr = log ? log : &local_log;
    const auto resolve_spirv_start =
        current_compile_trace() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::vector<uint32_t> spirv_binary = resolve_spirv_binary_from_source(source, log_ptr);
    if (current_compile_trace()) {
        increment_compile_counter("vulkan_resolve_spirv_count");
        add_compile_segment(
            "vulkan_resolve_spirv",
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::steady_clock::now() - resolve_spirv_start)
                                      .count()));
    }
    if (spirv_binary.empty()) {
        const auto lower_spirv_start =
            current_compile_trace() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        spirv_binary = lower_to_spirv(module, entry, log_ptr);
        if (current_compile_trace()) {
            increment_compile_counter("vulkan_lower_to_spirv_count");
            add_compile_segment(
                "vulkan_lower_to_spirv",
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                          std::chrono::steady_clock::now() - lower_spirv_start)
                                          .count()));
        }
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
                                        const auto program_create_start = current_compile_trace()
                                                                              ? std::chrono::steady_clock::now()
                                                                              : std::chrono::steady_clock::time_point{};
                                        auto program =
                                            std::make_shared<VulkanKernelProgram>(std::move(spirv_binary),
                                                                                  entry,
                                                                                  binding_count,
                                                                                  std::static_pointer_cast<VulkanDeviceReuseContext>(m_reuse_context));
                                        if (current_compile_trace()) {
                                            increment_compile_counter("vulkan_program_create_count");
                                            add_compile_segment(
                                                "vulkan_program_create",
                                                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                                          std::chrono::steady_clock::now() -
                                                                          program_create_start)
                                                                          .count()));
                                        }
                                        auto binding_plan = std::make_shared<KernelBindingPlan>(
                                            arg_count,
                                            resolve_kernel_output_arg_count(source));
                                        return std::make_shared<VulkanCompiledKernel>(std::move(program),
                                                                                      std::move(binding_plan),
                                                                                      shared_prepared_cache);
                                    });
}

}  // namespace gfx_plugin
}  // namespace ov
