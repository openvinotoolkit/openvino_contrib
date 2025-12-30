// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstring>

#include "openvino/core/except.hpp"
#include "plugin/gfx_backend_config.hpp"
#include "runtime/memory_manager.hpp"
#include "backends/metal/runtime/gpu_memory.hpp"
#include "backends/metal/runtime/metal_memory.hpp"
#include "backends/metal/runtime/memory/device_caps.hpp"
#include "backends/metal/runtime/memory/heap_pool.hpp"
#include "backends/metal/runtime/memory/staging_pool.hpp"
#include "backends/metal/runtime/memory/freelist.hpp"
#if GFX_BACKEND_VULKAN_AVAILABLE
#include "backends/vulkan/runtime/gpu_memory.hpp"
#endif

namespace {

ov::gfx_plugin::GpuBufferDesc make_desc(size_t bytes) {
    ov::gfx_plugin::GpuBufferDesc desc;
    desc.bytes = bytes;
    desc.type = ov::element::u8;
    desc.usage = ov::gfx_plugin::BufferUsage::IO;
    desc.cpu_read = true;
    desc.cpu_write = true;
    desc.prefer_device_local = false;
    desc.label = "gpu_memory_test";
    return desc;
}

}  // namespace

TEST(GfxMemoryUnified, CopyRoundtrip) {
#if GFX_BACKEND_METAL_AVAILABLE
    try {
        auto device = ov::gfx_plugin::metal_get_device_by_id(0);
        ASSERT_NE(device, nullptr);
        auto caps = ov::gfx_plugin::query_metal_device_caps(device);
        ov::gfx_plugin::MetalAllocatorCore core(device, caps);
        ov::gfx_plugin::MetalHeapPool heaps(core);
        ov::gfx_plugin::MetalFreeList freelist;
        ov::gfx_plugin::MetalStagingPool staging(core);
        ov::gfx_plugin::MetalAllocator alloc(core, heaps, freelist, staging, caps);
        ov::gfx_plugin::MetalGpuAllocator gpu_alloc(alloc, core, caps);

        auto desc = make_desc(256);
        auto buf = gpu_alloc.allocate(desc);
        ASSERT_TRUE(buf.valid());

        std::array<uint8_t, 256> src{};
        std::array<uint8_t, 256> dst{};
        for (size_t i = 0; i < src.size(); ++i) {
            src[i] = static_cast<uint8_t>(i & 0xFF);
        }

        ov::gfx_plugin::gpu_copy_from_host(buf, src.data(), src.size());
        ov::gfx_plugin::gpu_copy_to_host(buf, dst.data(), dst.size());

        EXPECT_EQ(std::memcmp(src.data(), dst.data(), src.size()), 0);
        gpu_alloc.release(std::move(buf));
    } catch (const ov::Exception& e) {
        FAIL() << "Metal allocator unavailable: " << e.what();
    }
#elif GFX_BACKEND_VULKAN_AVAILABLE
    try {
        ov::gfx_plugin::VulkanGpuAllocator alloc;
        auto desc = make_desc(256);
        auto buf = alloc.allocate(desc);
        ASSERT_TRUE(buf.valid());

        std::array<uint8_t, 256> src{};
        std::array<uint8_t, 256> dst{};
        for (size_t i = 0; i < src.size(); ++i) {
            src[i] = static_cast<uint8_t>((i * 3) & 0xFF);
        }

        ov::gfx_plugin::gpu_copy_from_host(buf, src.data(), src.size());
        ov::gfx_plugin::gpu_copy_to_host(buf, dst.data(), dst.size());

        EXPECT_EQ(std::memcmp(src.data(), dst.data(), src.size()), 0);
        alloc.release(std::move(buf));
    } catch (const ov::Exception& e) {
        FAIL() << "Vulkan allocator unavailable: " << e.what();
    }
#else
    FAIL() << "No GFX backend available for GPU memory test";
#endif
}
