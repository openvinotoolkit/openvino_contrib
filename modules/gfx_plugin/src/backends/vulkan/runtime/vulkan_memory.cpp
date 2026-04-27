// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/runtime/vulkan_memory.hpp"

#include <cstring>
#include <vector>

#include "openvino/core/except.hpp"
#include "backends/vulkan/runtime/vulkan_backend.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

uint32_t find_memory_type(VkPhysicalDevice phys,
                          uint32_t type_bits,
                          VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_props{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

struct UploadContext {
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    ~UploadContext() {
        reset();
    }

    void reset() {
        if (cmd && pool && device) {
            vkFreeCommandBuffers(device, pool, 1, &cmd);
        }
        cmd = VK_NULL_HANDLE;
        if (fence && device) {
            vkDestroyFence(device, fence, nullptr);
        }
        fence = VK_NULL_HANDLE;
        if (pool && device) {
            vkDestroyCommandPool(device, pool, nullptr);
        }
        pool = VK_NULL_HANDLE;
        device = VK_NULL_HANDLE;
        queue_family = 0;
    }

    void ensure(VkDevice target_device, uint32_t target_queue_family) {
        if (device == target_device && pool && cmd && queue_family == target_queue_family) {
            return;
        }
        reset();
        device = target_device;
        queue_family = target_queue_family;

        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = queue_family;
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VkResult res = vkCreateCommandPool(device, &pool_info, nullptr, &pool);
        if (res != VK_SUCCESS) {
            OPENVINO_THROW("GFX Vulkan: vkCreateCommandPool failed");
        }

        VkCommandBufferAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc.commandPool = pool;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1;
        res = vkAllocateCommandBuffers(device, &alloc, &cmd);
        if (res != VK_SUCCESS) {
            reset();
            OPENVINO_THROW("GFX Vulkan: vkAllocateCommandBuffers failed");
        }

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        res = vkCreateFence(device, &fence_info, nullptr, &fence);
        if (res != VK_SUCCESS) {
            reset();
            OPENVINO_THROW("GFX Vulkan: vkCreateFence failed");
        }
    }
};

UploadContext& upload_context() {
    thread_local UploadContext ctx;
    return ctx;
}

void record_vulkan_copy_regions(VkCommandBuffer command_buffer,
                                const GpuBuffer& src,
                                const GpuBuffer& dst,
                                const GpuBufferCopyRegion* regions,
                                size_t region_count);

void record_vulkan_copy_buffer(VkCommandBuffer command_buffer,
                               const GpuBuffer& src,
                               const GpuBuffer& dst,
                               size_t bytes) {
    GpuBufferCopyRegion region{};
    region.bytes = bytes;
    const GpuBufferCopyRegion regions[] = {region};
    record_vulkan_copy_regions(command_buffer, src, dst, regions, 1);
}

void record_vulkan_copy_regions(VkCommandBuffer command_buffer,
                                const GpuBuffer& src,
                                const GpuBuffer& dst,
                                const GpuBufferCopyRegion* regions,
                                size_t region_count) {
    OPENVINO_ASSERT(command_buffer != VK_NULL_HANDLE, "GFX Vulkan: command buffer is null");
    OPENVINO_ASSERT(regions && region_count > 0, "GFX Vulkan: copy regions are empty");

    std::vector<VkBufferCopy> vk_regions;
    vk_regions.reserve(region_count);
    VkDeviceSize dst_begin = 0;
    VkDeviceSize dst_end = 0;
    bool have_range = false;
    for (size_t i = 0; i < region_count; ++i) {
        const auto& region = regions[i];
        if (region.bytes == 0) {
            continue;
        }
        VkBufferCopy vk_region{};
        vk_region.srcOffset = static_cast<VkDeviceSize>(src.offset + region.src_offset);
        vk_region.dstOffset = static_cast<VkDeviceSize>(dst.offset + region.dst_offset);
        vk_region.size = static_cast<VkDeviceSize>(region.bytes);
        vk_regions.push_back(vk_region);
        if (!have_range) {
            dst_begin = vk_region.dstOffset;
            dst_end = vk_region.dstOffset + vk_region.size;
            have_range = true;
        } else {
            dst_begin = std::min(dst_begin, vk_region.dstOffset);
            dst_end = std::max(dst_end, vk_region.dstOffset + vk_region.size);
        }
    }
    if (vk_regions.empty()) {
        return;
    }
    vkCmdCopyBuffer(command_buffer,
                    vk_buffer_from_gpu(src),
                    vk_buffer_from_gpu(dst),
                    static_cast<uint32_t>(vk_regions.size()),
                    vk_regions.data());

    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = vk_buffer_from_gpu(dst);
    barrier.offset = dst_begin;
    barrier.size = dst_end - dst_begin;
    vkCmdPipelineBarrier(command_buffer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,
                         0,
                         nullptr,
                         1,
                         &barrier,
                         0,
                         nullptr);
}

}  // namespace

GpuBuffer vulkan_allocate_buffer(size_t bytes,
                                 ov::element::Type type,
                                 VkBufferUsageFlags usage,
                                 VkMemoryPropertyFlags properties) {
    GpuBuffer buf{};
    if (bytes == 0) {
        return buf;
    }

    auto& ctx = VulkanContext::instance();
    VkDevice device = ctx.device();
    VkPhysicalDevice phys = ctx.physical_device();

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = bytes;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    VkResult res = vkCreateBuffer(device, &buffer_info, nullptr, &buffer);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkCreateBuffer failed");
    }

    VkMemoryRequirements mem_req{};
    vkGetBufferMemoryRequirements(device, buffer, &mem_req);
    uint32_t mem_type = find_memory_type(phys, mem_req.memoryTypeBits, properties);
    if (mem_type == UINT32_MAX) {
        vkDestroyBuffer(device, buffer, nullptr);
        OPENVINO_THROW("GFX Vulkan: suitable memory type not found");
    }

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_req.size;
    alloc_info.memoryTypeIndex = mem_type;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    res = vkAllocateMemory(device, &alloc_info, nullptr, &memory);
    if (res != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
        OPENVINO_THROW("GFX Vulkan: vkAllocateMemory failed");
    }

    res = vkBindBufferMemory(device, buffer, memory, 0);
    if (res != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
        vkFreeMemory(device, memory, nullptr);
        OPENVINO_THROW("GFX Vulkan: vkBindBufferMemory failed");
    }

    buf.buffer = reinterpret_cast<GpuBufferHandle>(buffer);
    buf.heap = reinterpret_cast<GpuHeapHandle>(memory);
    buf.size = bytes;
    buf.type = type;
    buf.offset = 0;
    buf.persistent = false;
    buf.from_handle = false;
    buf.external = false;
    buf.backend = GpuBackend::Vulkan;
    buf.host_visible = (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    buf.allocation_uid = allocate_gpu_buffer_uid();
    return buf;
}

void vulkan_free_buffer(GpuBuffer& buf) {
    if (!buf.buffer || buf.external || !buf.owned) {
        return;
    }
    auto& ctx = VulkanContext::instance();
    VkDevice device = ctx.device();
    VkBuffer buffer = vk_buffer_from_gpu(buf);
    VkDeviceMemory memory = vk_memory_from_gpu(buf);
    vkDestroyBuffer(device, buffer, nullptr);
    if (memory) {
        vkFreeMemory(device, memory, nullptr);
    }
    buf = GpuBuffer{};
}

void* vulkan_map_buffer(const GpuBuffer& buf) {
    if (!buf.buffer) {
        return nullptr;
    }
    auto& ctx = VulkanContext::instance();
    VkDevice device = ctx.device();
    VkDeviceMemory memory = vk_memory_from_gpu(buf);
    void* mapped = nullptr;
    VkResult res = vkMapMemory(device, memory, buf.offset, buf.size, 0, &mapped);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkMapMemory failed");
    }
    return mapped;
}

void vulkan_unmap_buffer(const GpuBuffer& buf) {
    if (!buf.buffer) {
        return;
    }
    auto& ctx = VulkanContext::instance();
    VkDevice device = ctx.device();
    VkDeviceMemory memory = vk_memory_from_gpu(buf);
    vkUnmapMemory(device, memory);
}

namespace {
VkDeviceSize align_down(VkDeviceSize value, VkDeviceSize align) {
    if (align == 0) {
        return value;
    }
    return value & ~(align - 1);
}

VkDeviceSize align_up(VkDeviceSize value, VkDeviceSize align) {
    if (align == 0) {
        return value;
    }
    return (value + align - 1) & ~(align - 1);
}
}  // namespace

void vulkan_flush_buffer(const GpuBuffer& buf, size_t bytes, size_t offset) {
    if (!buf.buffer || bytes == 0) {
        return;
    }
    auto& ctx = VulkanContext::instance();
    VkDevice device = ctx.device();
    VkDeviceMemory memory = vk_memory_from_gpu(buf);
    VkDeviceSize atom = static_cast<VkDeviceSize>(ctx.noncoherent_atom_size());
    VkDeviceSize start = static_cast<VkDeviceSize>(buf.offset + offset);
    VkDeviceSize end = start + static_cast<VkDeviceSize>(bytes);
    VkDeviceSize aligned_start = align_down(start, atom);
    VkDeviceSize aligned_end = align_up(end, atom);

    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory;
    range.offset = aligned_start;
    range.size = aligned_end - aligned_start;
    vkFlushMappedMemoryRanges(device, 1, &range);
}

void vulkan_invalidate_buffer(const GpuBuffer& buf, size_t bytes, size_t offset) {
    if (!buf.buffer || bytes == 0) {
        return;
    }
    auto& ctx = VulkanContext::instance();
    VkDevice device = ctx.device();
    VkDeviceMemory memory = vk_memory_from_gpu(buf);
    VkDeviceSize atom = static_cast<VkDeviceSize>(ctx.noncoherent_atom_size());
    VkDeviceSize start = static_cast<VkDeviceSize>(buf.offset + offset);
    VkDeviceSize end = start + static_cast<VkDeviceSize>(bytes);
    VkDeviceSize aligned_start = align_down(start, atom);
    VkDeviceSize aligned_end = align_up(end, atom);

    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory;
    range.offset = aligned_start;
    range.size = aligned_end - aligned_start;
    vkInvalidateMappedMemoryRanges(device, 1, &range);
}

void vulkan_copy_buffer(GpuCommandQueueHandle execution_context,
                        const GpuBuffer& src,
                        const GpuBuffer& dst,
                        size_t bytes) {
    GpuBufferCopyRegion region{};
    region.bytes = bytes;
    vulkan_copy_buffer_regions(execution_context, src, dst, &region, 1);
}

void vulkan_copy_buffer_regions(GpuCommandQueueHandle execution_context,
                                const GpuBuffer& src,
                                const GpuBuffer& dst,
                                const GpuBufferCopyRegion* regions,
                                size_t region_count) {
    if (!src.buffer || !dst.buffer || !regions || region_count == 0) {
        return;
    }
    if (execution_context) {
        record_vulkan_copy_regions(reinterpret_cast<VkCommandBuffer>(execution_context), src, dst, regions, region_count);
        return;
    }
    auto& ctx = VulkanContext::instance();
    VkDevice device = ctx.device();
    auto& upload = upload_context();
    upload.ensure(device, ctx.queue_family_index());

    VkResult res = vkResetCommandPool(device, upload.pool, 0);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkResetCommandPool failed");
    }

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    res = vkBeginCommandBuffer(upload.cmd, &begin);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkBeginCommandBuffer failed");
    }

    record_vulkan_copy_regions(upload.cmd, src, dst, regions, region_count);

    res = vkEndCommandBuffer(upload.cmd);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkEndCommandBuffer failed");
    }

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &upload.cmd;
    res = vkResetFences(device, 1, &upload.fence);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkResetFences failed");
    }
    res = vkQueueSubmit(ctx.queue(), 1, &submit, upload.fence);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkQueueSubmit failed");
    }
    res = vkWaitForFences(device, 1, &upload.fence, VK_TRUE, UINT64_MAX);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkWaitForFences failed");
    }
}

GpuBuffer vulkan_upload_device_buffer(const void* src,
                                      size_t bytes,
                                      ov::element::Type type,
                                      VkBufferUsageFlags usage) {
    if (bytes == 0) {
        return GpuBuffer{};
    }
    OPENVINO_ASSERT(src, "GFX Vulkan: upload source is null");
    GpuBuffer staging = vulkan_allocate_buffer(bytes,
                                               type,
                                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* mapped = vulkan_map_buffer(staging);
    std::memcpy(mapped, src, bytes);
    vulkan_unmap_buffer(staging);

    GpuBuffer device_buf = vulkan_allocate_buffer(bytes,
                                                  type,
                                                  usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vulkan_copy_buffer(nullptr, staging, device_buf, bytes);
    vulkan_free_buffer(staging);
    device_buf.host_visible = false;
    return device_buf;
}

}  // namespace gfx_plugin
}  // namespace ov
