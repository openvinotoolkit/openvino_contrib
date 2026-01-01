// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstring>

#include "openvino/core/except.hpp"
#include "runtime/memory_manager.hpp"
#include "backends/vulkan/runtime/gpu_memory.hpp"

namespace {

ov::gfx_plugin::GpuBufferDesc make_desc(size_t bytes) {
    ov::gfx_plugin::GpuBufferDesc desc;
    desc.bytes = bytes;
    desc.type = ov::element::u8;
    desc.usage = ov::gfx_plugin::BufferUsage::IO;
    desc.cpu_read = true;
    desc.cpu_write = true;
    desc.prefer_device_local = false;
    desc.label = "gfx_vulkan_memory_copy_test";
    return desc;
}

}  // namespace

TEST(GfxVulkanMemory, CopyRoundtrip) {
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
}
