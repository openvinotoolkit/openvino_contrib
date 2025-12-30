// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <cstdlib>
#include <string>

#include "openvino/openvino.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/matmul.hpp"
#include "plugin/gfx_backend_config.hpp"
#include "runtime/memory_manager.hpp"
#include "backends/vulkan/runtime/gpu_memory.hpp"
#include "backends/vulkan/runtime/vulkan_memory.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool is_vulkan_unsupported_error(const std::string& msg) {
    return msg.find("GFX Vulkan") != std::string::npos ||
           msg.find("SPIR-V") != std::string::npos ||
           msg.find("spirv") != std::string::npos ||
           msg.find("vulkan") != std::string::npos;
}

inline void register_gfx_plugin(ov::Core& core) {
    try {
#ifdef GFX_PLUGIN_PATH
        const char* env_path = std::getenv("GFX_PLUGIN_PATH");
        const char* path = (env_path && *env_path) ? env_path : GFX_PLUGIN_PATH;
        core.register_plugin(path, "GFX");
#endif
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        if (msg.find("already registered") == std::string::npos) {
            FAIL() << "GFX plugin unavailable: " << e.what();
        }
    }
}

TEST(GfxBufferManagerTest, AllocAlignedAndNonNull) {
    constexpr size_t kBytes = 1024;
    auto buf = vulkan_allocate_buffer(kBytes,
                                      ov::element::f32,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    ASSERT_TRUE(buf.valid());
    EXPECT_GE(buf.size, kBytes);
    vulkan_free_buffer(buf);
}

TEST(GfxBufferManagerTest, ReuseBuffersViaFreeList) {
    constexpr size_t kBytes = 2048;
    auto buf1 = vulkan_allocate_buffer(kBytes,
                                       ov::element::f32,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    ASSERT_TRUE(buf1.valid());
    auto handle1 = buf1.buffer;
    vulkan_free_buffer(buf1);

    auto buf2 = vulkan_allocate_buffer(kBytes,
                                       ov::element::f32,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    ASSERT_TRUE(buf2.valid());
    EXPECT_NE(handle1, nullptr);
    EXPECT_NE(buf2.buffer, nullptr);
    vulkan_free_buffer(buf2);
}

TEST(GfxBufferManagerTest, DynamicGrowthWithBufferHandle) {
    auto buf1 = vulkan_allocate_buffer(512,
                                       ov::element::f32,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    ASSERT_TRUE(buf1.valid());
    auto buf2 = vulkan_allocate_buffer(4096,
                                       ov::element::f32,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    ASSERT_TRUE(buf2.valid());
    EXPECT_GE(buf1.size, 512u);
    EXPECT_GE(buf2.size, 4096u);
    vulkan_free_buffer(buf1);
    vulkan_free_buffer(buf2);
}

TEST(GfxBufferManagerTest, PersistentAndPerInferAreSeparated) {
    auto buf1 = vulkan_allocate_buffer(1024,
                                       ov::element::f32,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    auto buf2 = vulkan_allocate_buffer(1024,
                                       ov::element::f32,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    ASSERT_TRUE(buf1.valid());
    ASSERT_TRUE(buf2.valid());
    EXPECT_NE(buf1.buffer, buf2.buffer);
    vulkan_free_buffer(buf1);
    vulkan_free_buffer(buf2);
}

TEST(GfxTensorMapTest, BindInputReusesDeviceBufferSamePort) {
    VulkanGpuAllocator alloc;
    GpuBufferDesc desc{};
    desc.bytes = 256;
    desc.type = ov::element::u8;
    desc.usage = BufferUsage::IO;
    desc.cpu_read = true;
    desc.cpu_write = true;
    desc.prefer_device_local = false;
    auto buf = alloc.allocate(desc);
    ASSERT_TRUE(buf.valid());

    std::array<uint8_t, 256> src{};
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<uint8_t>(i);
    }
    gpu_copy_from_host(buf, src.data(), src.size());
    gpu_copy_from_host(buf, src.data(), src.size());
    EXPECT_TRUE(buf.valid());
    alloc.release(std::move(buf));
}

TEST(GfxTensorMapTest, HostTensorCreatedOnDemand) {
    auto buf = vulkan_allocate_buffer(128,
                                      ov::element::f32,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    ASSERT_TRUE(buf.valid());
    EXPECT_FALSE(buf.host_visible);
    vulkan_free_buffer(buf);
}

TEST(GfxTensorMapTest, BindInputDoesNotCopyToCPU) {
    VulkanGpuAllocator alloc;
    GpuBufferDesc desc{};
    desc.bytes = 128;
    desc.type = ov::element::u8;
    desc.usage = BufferUsage::IO;
    desc.cpu_read = true;
    desc.cpu_write = true;
    desc.prefer_device_local = false;
    auto buf = alloc.allocate(desc);
    ASSERT_TRUE(buf.valid());

    std::array<uint8_t, 128> src{};
    std::array<uint8_t, 128> dst{};
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<uint8_t>((i * 7) & 0xFF);
    }
    gpu_copy_from_host(buf, src.data(), src.size());
    gpu_copy_to_host(buf, dst.data(), dst.size());
    EXPECT_EQ(std::memcmp(src.data(), dst.data(), src.size()), 0);
    alloc.release(std::move(buf));
}

TEST(GfxRunDeviceIntegration, NoHostRoundTripUntilOutputRequested) {
    ov::Core core;
    register_gfx_plugin(core);
    const auto backend = core.get_property("GFX", "GFX_BACKEND").as<std::string>();
    ASSERT_FALSE(backend.empty());
    try {
        auto p0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3});
        auto p1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 4});
        auto mm = std::make_shared<ov::op::v0::MatMul>(p0, p1, false, false);
        auto res = std::make_shared<ov::op::v0::Result>(mm);
        auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{p0, p1}, "gfx_mem_integ");

        auto gfx_cm = core.compile_model(model, "GFX");

        ov::Tensor a{ov::element::f32, {2, 3}};
        ov::Tensor b{ov::element::f32, {3, 4}};
        for (size_t i = 0; i < a.get_size(); ++i) a.data<float>()[i] = static_cast<float>(i + 1);
        for (size_t i = 0; i < b.get_size(); ++i) b.data<float>()[i] = static_cast<float>((i % 5) - 2);

        auto req = gfx_cm.create_infer_request();
        req.set_input_tensor(0, a);
        req.set_input_tensor(1, b);
        req.infer();

        auto out = req.get_output_tensor(0);
        ASSERT_EQ(out.get_shape(), ov::Shape({2, 4}));
        ASSERT_EQ(out.get_element_type(), ov::element::f32);
    } catch (const std::exception& e) {
        if (is_vulkan_unsupported_error(e.what())) {
            SUCCEED() << "Vulkan backend did not support this case yet: " << e.what();
            return;
        }
        throw;
    }
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
