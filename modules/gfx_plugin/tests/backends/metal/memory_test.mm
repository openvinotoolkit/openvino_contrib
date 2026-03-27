// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Metal/Metal.h>
#ifdef NO
#undef NO
#endif
#ifdef YES
#undef YES
#endif
#include <cstdlib>
#include <gtest/gtest.h>

#include "openvino/openvino.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/matmul.hpp"

#include "backends/metal/runtime/metal_memory.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

// Local helper to register GFX plugin in tests (mirrors metal/basic_ops_test.cpp).
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

constexpr size_t kAlign = 256;

class GfxBufferManagerTest : public ::testing::Test {
protected:
    id<MTLDevice> device = nil;
    MetalDeviceCaps caps{};
    std::unique_ptr<MetalAllocatorCore> core;
    std::unique_ptr<MetalHeapPool> heaps;
    std::unique_ptr<MetalFreeList> freelist;
    std::unique_ptr<MetalStagingPool> staging;
    std::unique_ptr<MetalAllocator> allocator;
    std::unique_ptr<MetalConstCache> const_cache;
    std::unique_ptr<MetalBufferManager> mgr;
    MetalCommandQueueHandle queue = nullptr;

    void SetUp() override {
        device = MTLCreateSystemDefaultDevice();
        ASSERT_NE(device, nil);
        caps = query_metal_device_caps(device);
        core = std::make_unique<MetalAllocatorCore>(device, caps);
        queue = metal_create_command_queue(device);
        heaps = std::make_unique<MetalHeapPool>(*core);
        freelist = std::make_unique<MetalFreeList>();
        staging = std::make_unique<MetalStagingPool>(*core);
        allocator = std::make_unique<MetalAllocator>(*core, *heaps, *freelist, *staging, caps);
        const_cache = std::make_unique<MetalConstCache>(*allocator, queue);
        mgr = std::make_unique<MetalBufferManager>(*core, const_cache.get());
        MetalBufferManager::set_current_allocator(allocator.get());
    }

    void TearDown() override {
        MetalBufferManager::set_current_allocator(nullptr);
        if (queue) {
            metal_release_command_queue(queue);
            queue = nullptr;
        }
    }
};

TEST_F(GfxBufferManagerTest, AllocAlignedAndNonNull) {
    auto buf = mgr->allocate(1024, ov::element::f32, /*persistent=*/false, /*storageModePrivate=*/true);
    ASSERT_NE(static_cast<id<MTLBuffer>>(buf.buffer), nil);
    EXPECT_GE(buf.size, 1024u);
    EXPECT_EQ(buf.size % kAlign, 0u);
}

TEST_F(GfxBufferManagerTest, ReuseBuffersViaFreeList) {
    if (ov::gfx_plugin::metal_safe_debug_enabled()) {
        SUCCEED() << "Reuse is disabled in SAFE_DEBUG mode";
        return;
    }
    mgr->reset_stats();
    auto buf1 = mgr->allocate(2048, ov::element::f32, /*persistent=*/false, /*storageModePrivate=*/true);
    auto ptr1 = static_cast<id<MTLBuffer>>(buf1.buffer);
    mgr->release(std::move(buf1));
    auto buf2 = mgr->allocate(2048, ov::element::f32, /*persistent=*/false, /*storageModePrivate=*/true);
    auto ptr2 = static_cast<id<MTLBuffer>>(buf2.buffer);
    EXPECT_EQ(ptr1, ptr2);
    auto stats = mgr->stats();
    EXPECT_GE(stats.num_reuse_hits, 1u);
}

TEST_F(GfxBufferManagerTest, DynamicGrowthWithBufferHandle) {
    BufferHandle handle;
    auto b1 = mgr->allocate_dynamic(512, ov::element::f32, handle, /*persistent=*/false, /*shared=*/false);
    auto ptr1 = static_cast<id<MTLBuffer>>(b1.buffer);
    auto b2 = mgr->allocate_dynamic(256, ov::element::f32, handle, /*persistent=*/false, /*shared=*/false);
    EXPECT_EQ(static_cast<id<MTLBuffer>>(b2.buffer), ptr1);
    EXPECT_GE(b2.size, b1.size);
    auto b3 = mgr->allocate_dynamic(4096, ov::element::f32, handle, /*persistent=*/false, /*shared=*/false);
    EXPECT_NE(static_cast<id<MTLBuffer>>(b3.buffer), ptr1);
    EXPECT_GE(b3.size, 4096u);
}

TEST_F(GfxBufferManagerTest, PersistentAndPerInferAreSeparated) {
    auto persist = mgr->allocate(1024, ov::element::f32, /*persistent=*/true, /*storageModePrivate=*/true);
    auto p_ptr = static_cast<id<MTLBuffer>>(persist.buffer);
    auto tmp = mgr->allocate(1024, ov::element::f32, /*persistent=*/false, /*storageModePrivate=*/true);
    auto t_ptr = static_cast<id<MTLBuffer>>(tmp.buffer);
    mgr->release(std::move(tmp));
    auto tmp2 = mgr->allocate(1024, ov::element::f32, /*persistent=*/false, /*storageModePrivate=*/true);
    EXPECT_EQ(static_cast<id<MTLBuffer>>(tmp2.buffer), t_ptr);
    EXPECT_EQ(p_ptr, static_cast<id<MTLBuffer>>(persist.buffer));
}

TEST_F(GfxBufferManagerTest, ConstCacheContextIsSharedAcrossCacheInstances) {
    MetalConstCache second_cache(*allocator, queue);
    EXPECT_EQ(const_cache->shared_cache_identity(), second_cache.shared_cache_identity());

    const uint32_t values[4] = {1u, 2u, 3u, 4u};
    BufferDesc desc;
    desc.bytes = sizeof(values);
    desc.type = ov::element::u32;
    desc.storage = MetalStorage::Private;
    desc.usage = BufferUsage::Const;

    auto first = const_cache->get_or_create("shared_weights", values, sizeof(values), desc);
    auto second = second_cache.get_or_create("shared_weights", values, sizeof(values), desc);

    ASSERT_TRUE(first.valid());
    ASSERT_TRUE(second.valid());
    EXPECT_EQ(first.buffer, second.buffer);
}

class GfxTensorMapTest : public ::testing::Test {
protected:
    id<MTLDevice> device = nil;
    MetalDeviceCaps caps{};
    std::unique_ptr<MetalAllocatorCore> core;
    std::unique_ptr<MetalHeapPool> heaps;
    std::unique_ptr<MetalFreeList> freelist;
    std::unique_ptr<MetalStagingPool> staging;
    std::unique_ptr<MetalAllocator> allocator;
    std::unique_ptr<MetalConstCache> const_cache;
    std::unique_ptr<MetalBufferManager> mgr;
    MetalTensorMap tensor_map;
    MetalCommandQueueHandle queue = nullptr;

    void SetUp() override {
        device = MTLCreateSystemDefaultDevice();
        ASSERT_NE(device, nil);
        caps = query_metal_device_caps(device);
        core = std::make_unique<MetalAllocatorCore>(device, caps);
        queue = metal_create_command_queue(device);
        heaps = std::make_unique<MetalHeapPool>(*core);
        freelist = std::make_unique<MetalFreeList>();
        staging = std::make_unique<MetalStagingPool>(*core);
        allocator = std::make_unique<MetalAllocator>(*core, *heaps, *freelist, *staging, caps);
        const_cache = std::make_unique<MetalConstCache>(*allocator, queue);
        mgr = std::make_unique<MetalBufferManager>(*core, const_cache.get());
        MetalBufferManager::set_current_allocator(allocator.get());
    }

    void TearDown() override {
        MetalBufferManager::set_current_allocator(nullptr);
        if (queue) {
            metal_release_command_queue(queue);
            queue = nullptr;
        }
    }
};

TEST_F(GfxTensorMapTest, BindInputReusesDeviceBufferSamePort) {
    ov::Tensor host{ov::element::f32, {1, 4}};
    std::fill(host.data<float>(), host.data<float>() + host.get_size(), 1.f);
    auto dev1 = tensor_map.bind_input(0, host, *core);
    auto dev1b = tensor_map.bind_input(0, host, *core);
    EXPECT_EQ(dev1.buf.size, dev1b.buf.size);
    EXPECT_TRUE(tensor_map.has_input_device(0));
}

TEST_F(GfxTensorMapTest, HostTensorCreatedOnDemand) {
    mgr->reset_stats();
    ov::Shape shape{1, 2, 2};
    auto& dev = tensor_map.ensure_output_device(0, shape, ov::element::f32, *allocator, caps, /*prefer_private=*/false);
    (void)dev;
    EXPECT_FALSE(tensor_map.has_host_for_output(0));
    EXPECT_THROW(tensor_map.get_or_create_host_for_output(0), ov::Exception);
    EXPECT_FALSE(tensor_map.has_host_for_output(0));
    EXPECT_EQ(mgr->stats().d2h_bytes, 0u);
}

TEST_F(GfxTensorMapTest, BindInputDoesNotCopyToCPU) {
    mgr->reset_stats();
    ov::Tensor host{ov::element::f32, {2, 2}};
    std::fill(host.data<float>(), host.data<float>() + host.get_size(), 1.f);
    tensor_map.bind_input(0, host, *core);
    size_t h2d_first = mgr->stats().h2d_bytes;
    tensor_map.bind_input(0, host, *core);
    size_t h2d_second = mgr->stats().h2d_bytes;
    EXPECT_EQ(h2d_first, 0u);
    EXPECT_EQ(h2d_second, 0u);
}


}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
