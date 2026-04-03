// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/runtime/vulkan_buffer_manager.hpp"

#include "backends/vulkan/runtime/vulkan_backend.hpp"
#include "openvino/core/except.hpp"
#include "runtime/gfx_profiler.hpp"
#include "runtime/memory_manager.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace ov {
namespace gfx_plugin {

namespace {

constexpr size_t kDirectMappedConstBytes = 4096;

class VulkanConstBufferReuseContext {
public:
    VulkanConstBufferReuseContext()
        : m_device_alloc(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT),
          m_staging_alloc(VK_BUFFER_USAGE_TRANSFER_SRC_BIT),
          m_const_cache(std::make_shared<ImmutableGpuBufferCache>([this](GpuBuffer&& buf) {
              m_device_alloc.release(std::move(buf));
          })) {}

    void begin_upload_batch() {
        auto& state = batch_state();
        state.depth += 1;
    }

    void flush_upload_batch(GpuCommandBufferHandle command_buffer, GfxProfiler* profiler) {
        auto& state = batch_state();
        flush_upload_batch_state(state, command_buffer, profiler);
    }

    void end_upload_batch() {
        auto& state = batch_state();
        if (state.depth == 0) {
            return;
        }
        state.depth -= 1;
        if (state.depth == 0 && !state.pending_uploads.empty()) {
            flush_upload_batch_state(state, nullptr, nullptr);
        }
    }

    GpuBuffer wrap_const(const std::string& key,
                         const void* data,
                         size_t bytes,
                         ov::element::Type type) {
        if (bytes == 0) {
            return {};
        }
        OPENVINO_ASSERT(data, "GFX Vulkan: const buffer data is null");
        return m_const_cache->get_or_create(key, data, bytes, type, [&]() {
            GpuBuffer staging;
            GpuBufferDesc device_desc{};
            device_desc.bytes = bytes;
            device_desc.type = type;
            device_desc.usage = BufferUsage::Const;
            device_desc.cpu_read = false;
            device_desc.cpu_write = false;
            device_desc.prefer_device_local = true;

            GpuBuffer device_buf;
            if (bytes <= kDirectMappedConstBytes) {
                // Small immutable metadata buffers are cheaper and more robust as direct
                // host-visible storage buffers than as one-off staging uploads.
                device_desc.cpu_write = true;
                device_desc.prefer_device_local = false;
                device_buf = m_device_alloc.allocate(device_desc);
                if (device_buf.valid()) {
                    gpu_copy_from_host(device_buf, data, bytes);
                }
                return device_buf;
            }

            GpuBufferDesc staging_desc{};
            staging_desc.bytes = bytes;
            staging_desc.type = type;
            staging_desc.usage = BufferUsage::Staging;
            staging_desc.cpu_write = true;
            staging_desc.prefer_device_local = false;
            staging = m_staging_alloc.allocate(staging_desc);
            if (bytes) {
                gpu_copy_from_host(staging, data, bytes);
            }
            device_buf = m_device_alloc.allocate(device_desc);
            if (bytes && staging.valid() && device_buf.valid()) {
                auto& state = batch_state();
                if (state.depth > 0) {
                    state.pending_uploads.push_back(PendingUpload{std::move(staging), device_buf, bytes});
                } else {
                    gpu_copy_buffer(nullptr, staging, device_buf, bytes);
                    m_staging_alloc.release(std::move(staging));
                }
            }
            return device_buf;
        });
    }

private:
    struct PendingUpload {
        GpuBuffer staging;
        GpuBuffer device;
        size_t bytes = 0;
    };

    struct UploadBatchState {
        size_t depth = 0;
        std::vector<PendingUpload> pending_uploads;
    };

    struct BatchSubmitContext {
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;

        void reset() {
            auto& vk = VulkanContext::instance();
            if (pool != VK_NULL_HANDLE) {
                vkResetCommandPool(vk.device(), pool, 0);
            }
        }

        void ensure() {
            auto& vk = VulkanContext::instance();
            if (pool != VK_NULL_HANDLE && command_buffer != VK_NULL_HANDLE && fence != VK_NULL_HANDLE) {
                return;
            }

            VkCommandPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pool_info.queueFamilyIndex = vk.queue_family_index();
            pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            VkResult res = vkCreateCommandPool(vk.device(), &pool_info, nullptr, &pool);
            if (res != VK_SUCCESS) {
                OPENVINO_THROW("GFX Vulkan: vkCreateCommandPool failed for const upload batch");
            }

            VkCommandBufferAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            alloc_info.commandPool = pool;
            alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            alloc_info.commandBufferCount = 1;
            res = vkAllocateCommandBuffers(vk.device(), &alloc_info, &command_buffer);
            if (res != VK_SUCCESS) {
                OPENVINO_THROW("GFX Vulkan: vkAllocateCommandBuffers failed for const upload batch");
            }

            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            res = vkCreateFence(vk.device(), &fence_info, nullptr, &fence);
            if (res != VK_SUCCESS) {
                OPENVINO_THROW("GFX Vulkan: vkCreateFence failed for const upload batch");
            }
        }

        ~BatchSubmitContext() {
            auto& vk = VulkanContext::instance();
            if (fence != VK_NULL_HANDLE) {
                vkDestroyFence(vk.device(), fence, nullptr);
            }
            if (command_buffer != VK_NULL_HANDLE && pool != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(vk.device(), pool, 1, &command_buffer);
            }
            if (pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(vk.device(), pool, nullptr);
            }
        }
    };

    static std::unordered_map<const VulkanConstBufferReuseContext*, UploadBatchState>& upload_batch_states() {
        thread_local std::unordered_map<const VulkanConstBufferReuseContext*, UploadBatchState> states;
        return states;
    }

    static BatchSubmitContext& batch_submit_context() {
        thread_local BatchSubmitContext context;
        return context;
    }

    UploadBatchState& batch_state() {
        return upload_batch_states()[this];
    }

    static void release_pending_uploads(std::vector<PendingUpload>& uploads, VulkanGpuAllocator& staging_alloc) {
        for (auto& upload : uploads) {
            if (upload.staging.valid()) {
                staging_alloc.release(std::move(upload.staging));
            }
        }
        uploads.clear();
    }

    void flush_upload_batch_state(UploadBatchState& state,
                                  GpuCommandBufferHandle command_buffer,
                                  GfxProfiler* profiler) {
        if (state.pending_uploads.empty()) {
            return;
        }

        const auto start = std::chrono::steady_clock::now();
        size_t total_bytes = 0;
        for (const auto& upload : state.pending_uploads) {
            total_bytes += upload.bytes;
        }

        if (command_buffer) {
            for (const auto& upload : state.pending_uploads) {
                gpu_copy_buffer(command_buffer, upload.staging, upload.device, upload.bytes);
            }
        } else {
            auto& submit_ctx = batch_submit_context();
            auto& vk = VulkanContext::instance();
            submit_ctx.ensure();

            VkResult res = vkWaitForFences(vk.device(), 1, &submit_ctx.fence, VK_TRUE, UINT64_MAX);
            if (res != VK_SUCCESS) {
                OPENVINO_THROW("GFX Vulkan: vkWaitForFences failed for const upload batch");
            }
            res = vkResetFences(vk.device(), 1, &submit_ctx.fence);
            if (res != VK_SUCCESS) {
                OPENVINO_THROW("GFX Vulkan: vkResetFences failed for const upload batch");
            }

            submit_ctx.reset();
            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            res = vkBeginCommandBuffer(submit_ctx.command_buffer, &begin_info);
            if (res != VK_SUCCESS) {
                OPENVINO_THROW("GFX Vulkan: vkBeginCommandBuffer failed for const upload batch");
            }
            for (const auto& upload : state.pending_uploads) {
                gpu_copy_buffer(reinterpret_cast<GpuCommandBufferHandle>(submit_ctx.command_buffer),
                                upload.staging,
                                upload.device,
                                upload.bytes);
            }
            res = vkEndCommandBuffer(submit_ctx.command_buffer);
            if (res != VK_SUCCESS) {
                OPENVINO_THROW("GFX Vulkan: vkEndCommandBuffer failed for const upload batch");
            }

            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &submit_ctx.command_buffer;
            res = vkQueueSubmit(vk.queue(), 1, &submit_info, submit_ctx.fence);
            if (res != VK_SUCCESS) {
                OPENVINO_THROW("GFX Vulkan: vkQueueSubmit failed for const upload batch");
            }
            res = vkWaitForFences(vk.device(), 1, &submit_ctx.fence, VK_TRUE, UINT64_MAX);
            if (res != VK_SUCCESS) {
                OPENVINO_THROW("GFX Vulkan: vkWaitForFences failed for const upload batch completion");
            }
        }

        const auto cpu_us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
        if (profiler) {
            profiler->record_segment("upload",
                                     "flush_const_upload_batch",
                                     cpu_us,
                                     0,
                                     0,
                                     static_cast<uint64_t>(total_bytes),
                                     0,
                                     0,
                                     0,
                                     -1,
                                     0,
                                     reinterpret_cast<uint64_t>(command_buffer));
            profiler->record_transfer("const_h2d",
                                      static_cast<uint64_t>(total_bytes),
                                      true,
                                      cpu_us);
            profiler->increment_counter("const_upload_batch_count");
            profiler->increment_counter("const_upload_buffer_count",
                                        static_cast<uint64_t>(state.pending_uploads.size()));
        }

        release_pending_uploads(state.pending_uploads, m_staging_alloc);
    }

    VulkanGpuAllocator m_device_alloc;
    VulkanGpuAllocator m_staging_alloc;
    std::shared_ptr<ImmutableGpuBufferCache> m_const_cache;
};

class VulkanConstBufferReuseRegistry {
public:
    static VulkanConstBufferReuseRegistry& instance() {
        static auto* registry = new VulkanConstBufferReuseRegistry();
        return *registry;
    }

    std::shared_ptr<VulkanConstBufferReuseContext> acquire(VkDevice device) {
        const uintptr_t key = reinterpret_cast<uintptr_t>(device);
        std::lock_guard<std::mutex> lock(m_mutex);
        if (auto it = m_entries.find(key); it != m_entries.end()) {
            if (auto cached = it->second.lock()) {
                return cached;
            }
            m_entries.erase(it);
        }

        auto created = std::make_shared<VulkanConstBufferReuseContext>();
        m_entries[key] = created;
        return created;
    }

private:
    std::mutex m_mutex;
    std::unordered_map<uintptr_t, std::weak_ptr<VulkanConstBufferReuseContext>> m_entries;
};

}  // namespace

VulkanBufferManager::VulkanBufferManager()
    : m_reuse_context(VulkanConstBufferReuseRegistry::instance().acquire(VulkanContext::instance().device())) {}

VulkanBufferManager::~VulkanBufferManager() = default;

std::optional<GpuExecutionDeviceInfo> VulkanBufferManager::query_execution_device_info() const {
    GpuExecutionDeviceInfo info{};
    const auto& vk = VulkanContext::instance();
    info.backend = GpuBackend::Vulkan;
    info.device_family = vk.device_family();
    info.preferred_simd_width = std::max<uint32_t>(vk.subgroup_size(), 1u);
    info.subgroup_size = std::max<uint32_t>(vk.subgroup_size(), 1u);
    info.max_total_threads_per_group = std::max<uint32_t>(vk.max_compute_workgroup_invocations(), 1u);
    info.max_threads_per_group = vk.max_compute_workgroup_size();

    std::ostringstream os;
    os << "vulkan:" << gpu_device_family_name(info.device_family) << ':'
       << vk.vendor_id() << ':' << vk.device_id() << ':'
       << vk.device_name() << ':' << info.subgroup_size << ':' << info.max_total_threads_per_group << ':'
       << info.max_threads_per_group[0] << ':' << info.max_threads_per_group[1] << ':' << info.max_threads_per_group[2];
    info.device_key = os.str();
    return info;
}

GpuBuffer VulkanBufferManager::wrap_const(const std::string& key,
                                          const void* data,
                                          size_t bytes,
                                          ov::element::Type type) {
    auto context = std::static_pointer_cast<VulkanConstBufferReuseContext>(m_reuse_context);
    OPENVINO_ASSERT(context, "GFX Vulkan: missing const buffer reuse context");
    return context->wrap_const(key, data, bytes, type);
}

void VulkanBufferManager::begin_const_upload_batch() {
    auto context = std::static_pointer_cast<VulkanConstBufferReuseContext>(m_reuse_context);
    OPENVINO_ASSERT(context, "GFX Vulkan: missing const buffer reuse context");
    context->begin_upload_batch();
}

void VulkanBufferManager::flush_const_upload_batch(GpuCommandBufferHandle command_buffer,
                                                   GfxProfiler* profiler) {
    auto context = std::static_pointer_cast<VulkanConstBufferReuseContext>(m_reuse_context);
    OPENVINO_ASSERT(context, "GFX Vulkan: missing const buffer reuse context");
    context->flush_upload_batch(command_buffer, profiler);
}

void VulkanBufferManager::end_const_upload_batch() {
    auto context = std::static_pointer_cast<VulkanConstBufferReuseContext>(m_reuse_context);
    OPENVINO_ASSERT(context, "GFX Vulkan: missing const buffer reuse context");
    context->end_upload_batch();
}

const void* VulkanBufferManager::shared_const_cache_identity() const {
    return m_reuse_context.get();
}

}  // namespace gfx_plugin
}  // namespace ov
