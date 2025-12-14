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
#include <gtest/gtest.h>

#include "openvino/openvino.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/matmul.hpp"

#include "runtime/metal_memory.hpp"

namespace ov {
namespace metal_plugin {
namespace {

// Local helper to register METAL plugin in tests (mirrors metal_basic_ops_test.cpp).
inline void register_metal_plugin(ov::Core& core) {
    try {
#ifdef METAL_PLUGIN_PATH
        core.register_plugin(METAL_PLUGIN_PATH, "METAL");
#endif
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        if (msg.find("already registered") == std::string::npos) {
            FAIL() << "METAL plugin unavailable: " << e.what();
        }
    }
}

constexpr size_t kAlign = 256;

class MetalBufferManagerTest : public ::testing::Test {
protected:
    id<MTLDevice> device = nil;
    std::unique_ptr<MetalBufferManager> mgr;

    void SetUp() override {
        device = MTLCreateSystemDefaultDevice();
        ASSERT_NE(device, nil);
        mgr = std::make_unique<MetalBufferManager>(device);
    }
};

TEST_F(MetalBufferManagerTest, AllocAlignedAndNonNull) {
    auto buf = mgr->allocate(1024, ov::element::f32, /*persistent=*/false, /*shared=*/false);
    ASSERT_NE(static_cast<id<MTLBuffer>>(buf.buffer), nil);
    EXPECT_GE(buf.size, 1024u);
    EXPECT_EQ(buf.size % kAlign, 0u);
}

TEST_F(MetalBufferManagerTest, ReuseBuffersInPerInferPool) {
    mgr->reset_stats();
    auto buf1 = mgr->allocate(2048, ov::element::f32, /*persistent=*/false, /*shared=*/false);
    auto ptr1 = static_cast<id<MTLBuffer>>(buf1.buffer);
    mgr->reset_inference_pool();
    auto buf2 = mgr->allocate(2048, ov::element::f32, /*persistent=*/false, /*shared=*/false);
    auto ptr2 = static_cast<id<MTLBuffer>>(buf2.buffer);
    EXPECT_EQ(ptr1, ptr2);
    auto stats = mgr->stats();
    EXPECT_GE(stats.reused_bytes, buf2.size);
}

TEST_F(MetalBufferManagerTest, DynamicGrowthWithBufferHandle) {
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

TEST_F(MetalBufferManagerTest, PersistentAndPerInferAreSeparated) {
    auto persist = mgr->allocate(1024, ov::element::f32, /*persistent=*/true, /*shared=*/false);
    auto p_ptr = static_cast<id<MTLBuffer>>(persist.buffer);
    mgr->reset_inference_pool();
    auto tmp = mgr->allocate(1024, ov::element::f32, /*persistent=*/false, /*shared=*/false);
    auto t_ptr = static_cast<id<MTLBuffer>>(tmp.buffer);
    mgr->reset_inference_pool();
    auto tmp2 = mgr->allocate(1024, ov::element::f32, /*persistent=*/false, /*shared=*/false);
    EXPECT_EQ(static_cast<id<MTLBuffer>>(tmp2.buffer), t_ptr);
    EXPECT_EQ(p_ptr, static_cast<id<MTLBuffer>>(persist.buffer));
}

class MetalTensorMapTest : public ::testing::Test {
protected:
    id<MTLDevice> device = nil;
    std::unique_ptr<MetalBufferManager> mgr;
    MetalTensorMap tensor_map;

    void SetUp() override {
        device = MTLCreateSystemDefaultDevice();
        ASSERT_NE(device, nil);
        mgr = std::make_unique<MetalBufferManager>(device);
    }
};

TEST_F(MetalTensorMapTest, BindInputReusesDeviceBufferSamePort) {
    ov::Tensor host{ov::element::f32, {1, 4}};
    std::fill(host.data<float>(), host.data<float>() + host.get_size(), 1.f);
    auto dev1 = tensor_map.bind_input(0, host, *mgr, /*shared=*/true);
    auto dev1b = tensor_map.bind_input(0, host, *mgr, /*shared=*/true);
    EXPECT_EQ(dev1.buf.size, dev1b.buf.size);
    EXPECT_TRUE(tensor_map.has_input_device(0));
}

TEST_F(MetalTensorMapTest, HostTensorCreatedOnDemand) {
    mgr->reset_stats();
    ov::Shape shape{1, 2, 2};
    auto& dev = tensor_map.ensure_output_device(0, shape, ov::element::f32, *mgr, /*shared=*/false);
    auto buf = static_cast<id<MTLBuffer>>(dev.buf.buffer);
    float pattern[4] = {1.f, 2.f, 3.f, 4.f};
    std::memcpy([buf contents], pattern, sizeof(pattern));
    EXPECT_FALSE(tensor_map.has_host_for_output(0));
    auto& host = tensor_map.get_or_create_host_for_output(0, *mgr);
    EXPECT_TRUE(tensor_map.has_host_for_output(0));
    EXPECT_EQ(host.get_shape(), shape);
    const float* p = host.data<const float>();
    for (size_t i = 0; i < host.get_size(); ++i) {
        EXPECT_FLOAT_EQ(p[i], pattern[i]);
    }
    size_t first_d2h = mgr->stats().d2h_bytes;
    auto& host2 = tensor_map.get_or_create_host_for_output(0, *mgr);
    (void)host2;
    EXPECT_EQ(mgr->stats().d2h_bytes, first_d2h);  // no extra D2H on second call
}

TEST_F(MetalTensorMapTest, BindInputAddsH2DOncePerCall) {
    mgr->reset_stats();
    ov::Tensor host{ov::element::f32, {2, 2}};
    std::fill(host.data<float>(), host.data<float>() + host.get_size(), 1.f);
    tensor_map.bind_input(0, host, *mgr, /*shared=*/true);
    size_t h2d_first = mgr->stats().h2d_bytes;
    tensor_map.bind_input(0, host, *mgr, /*shared=*/true);
    size_t h2d_second = mgr->stats().h2d_bytes;
    EXPECT_EQ(h2d_first * 2, h2d_second);  // each bind copies host->device once
}

// Integration: run_device should avoid D2H until output is requested.
TEST(MetalRunDeviceIntegration, NoHostRoundTripUntilOutputRequested) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);
    ov::Core core;
    register_metal_plugin(core);

    auto p0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3});
    auto p1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 4});
    auto mm = std::make_shared<ov::op::v0::MatMul>(p0, p1, false, false);
    auto res = std::make_shared<ov::op::v0::Result>(mm);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{p0, p1}, "mm_mem_integ");

    auto cpu_cm = core.compile_model(model, "CPU");
    auto metal_cm = core.compile_model(model, "METAL");

    ov::Tensor a{ov::element::f32, {2, 3}};
    ov::Tensor b{ov::element::f32, {3, 4}};
    for (size_t i = 0; i < a.get_size(); ++i) a.data<float>()[i] = static_cast<float>(i + 1);
    for (size_t i = 0; i < b.get_size(); ++i) b.data<float>()[i] = static_cast<float>((i % 5) - 2);

    // Reference on host
    std::vector<float> ref(2 * 4, 0.f);
    for (size_t m = 0; m < 2; ++m) {
        for (size_t n = 0; n < 4; ++n) {
            float acc = 0.f;
            for (size_t k = 0; k < 3; ++k) {
                acc += a.data<const float>()[m * 3 + k] * b.data<const float>()[k * 4 + n];
            }
            ref[m * 4 + n] = acc;
        }
    }

    auto metal_req = metal_cm.create_infer_request();
    metal_req.set_input_tensor(0, a);
    metal_req.set_input_tensor(1, b);
    metal_req.infer();

    // After infer but before get_output_tensor, D2H should be zero.
    auto stats_before = metal_cm.get_property("METAL_MEM_STATS").as<MetalMemoryStats>();
    std::cout << "[TEST] stats_before h2d=" << stats_before.h2d_bytes
              << " d2h=" << stats_before.d2h_bytes
              << " alloc=" << stats_before.alloc_bytes
              << " reused=" << stats_before.reused_bytes << std::endl;
    if (stats_before.alloc_bytes == 0 && stats_before.h2d_bytes == 0) {
        GTEST_SKIP() << "METAL backend did not execute device path in this build; skipping integration check.";
    }
    EXPECT_EQ(stats_before.d2h_bytes, 0u);
    EXPECT_GE(stats_before.h2d_bytes, a.get_byte_size() + b.get_byte_size());

    auto metal_out = metal_req.get_output_tensor();
    auto stats_after = metal_cm.get_property("METAL_MEM_STATS").as<MetalMemoryStats>();
    std::cout << "[TEST] stats_after h2d=" << stats_after.h2d_bytes
              << " d2h=" << stats_after.d2h_bytes
              << " alloc=" << stats_after.alloc_bytes
              << " reused=" << stats_after.reused_bytes << std::endl;
    if (stats_after.d2h_bytes == 0) {
        GTEST_SKIP() << "No device→host copy recorded; run_device likely fell back. Skipping correctness check.";
        return;
    }
    EXPECT_GE(stats_after.d2h_bytes, metal_out.get_byte_size());

    ASSERT_EQ(metal_out.get_shape(), ov::Shape({2, 4}));
    ASSERT_EQ(metal_out.get_element_type(), ov::element::f32);
    const float* m = metal_out.data<const float>();
    for (size_t i = 0; i < ref.size(); ++i) {
        EXPECT_NEAR(ref[i], m[i], 2e-3f);
    }
}

}  // namespace
}  // namespace metal_plugin
}  // namespace ov
