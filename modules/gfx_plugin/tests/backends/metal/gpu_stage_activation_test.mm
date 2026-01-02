// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#import <Metal/Metal.h>

#include <cmath>

#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/tanh.hpp"
#include "backends/metal/runtime/metal_memory.hpp"
#include "backends/metal/runtime/stage_factory.hpp"
#include "runtime/execution_dispatcher.hpp"

using namespace ov::gfx_plugin;

namespace {

struct ActivationCase {
    std::vector<float> in;
    std::vector<float> expected;
};

template <typename OpFactory>
void run_activation(const ActivationCase& tc) {
    ensure_metal_stage_factory_registered();
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);
    MetalCommandQueueHandle gfx_queue = metal_create_command_queue(device);
    ASSERT_NE(gfx_queue, nullptr);
    id<MTLCommandQueue> queue = static_cast<id<MTLCommandQueue>>(gfx_queue);

    MetalDeviceCaps caps = query_metal_device_caps(device);
    MetalAllocatorCore core(device, caps);
    MetalHeapPool heaps(core);
    MetalFreeList freelist;
    MetalStagingPool staging(core);
    MetalAllocator allocator(core, heaps, freelist, staging, caps);
    MetalConstCache const_cache(allocator, gfx_queue);
    MetalBufferManager mgr(core, &const_cache);
    MetalBufferManager::set_current_allocator(&allocator);

    // Prepare tensors
    MetalTensor input{};
    input.shape = {tc.in.size()};
    input.expected_type = ov::element::f32;
    input.buf = mgr.wrap_shared(tc.in.data(), tc.in.size() * sizeof(float), ov::element::f32);

    MetalTensor output{};
    output.shape = {tc.in.size()};
    output.expected_type = ov::element::f32;
    output.prefer_private = false;

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{tc.in.size()});
    auto node = OpFactory::make_node(param);
    auto stage = GpuStageFactory::create(node, GpuBackend::Metal, device, queue);
    ASSERT_NE(stage, nullptr);
    stage->set_inputs({&input});
    stage->set_output(&output);
    stage->init(&mgr);
    stage->compile(&mgr);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    stage->execute(cb);
    [cb commit];
    [cb waitUntilCompleted];

    MetalBufferManager::set_current_allocator(nullptr);
    metal_release_command_queue(gfx_queue);

    id<MTLBuffer> out_buf = static_cast<id<MTLBuffer>>(output.buf.buffer);
    ASSERT_NE(out_buf, nil);
    auto* data = static_cast<const float*>([out_buf contents]);
    ASSERT_NE(data, nullptr);
    ASSERT_EQ(output.shape.size(), 1u);
    ASSERT_EQ(output.shape[0], tc.expected.size());
    for (size_t i = 0; i < tc.expected.size(); ++i) {
        EXPECT_NEAR(data[i], tc.expected[i], 1e-4f) << "idx=" << i;
    }
}

struct ReluFactory {
    static std::shared_ptr<ov::Node> make_node(const std::shared_ptr<ov::Node>& p) {
        return std::make_shared<ov::op::v0::Relu>(p);
    }
};
struct SigmoidFactory {
    static std::shared_ptr<ov::Node> make_node(const std::shared_ptr<ov::Node>& p) {
        return std::make_shared<ov::op::v0::Sigmoid>(p);
    }
};
struct TanhFactory {
    static std::shared_ptr<ov::Node> make_node(const std::shared_ptr<ov::Node>& p) {
        return std::make_shared<ov::op::v0::Tanh>(p);
    }
};

}  // namespace

TEST(GpuStageActivation, Relu) {
    ActivationCase tc{{-1.f, 0.f, 2.f}, {0.f, 0.f, 2.f}};
    run_activation<ReluFactory>(tc);
}

TEST(GpuStageActivation, Sigmoid) {
    ActivationCase tc{{0.f, 2.f, -2.f}, {0.5f, 1.f / (1.f + std::exp(-2.f)), 1.f / (1.f + std::exp(2.f))}};
    run_activation<SigmoidFactory>(tc);
}

TEST(GpuStageActivation, Tanh) {
    ActivationCase tc{{0.f, 1.f, -1.f}, {0.f, std::tanh(1.f), std::tanh(-1.f)}};
    run_activation<TanhFactory>(tc);
}
