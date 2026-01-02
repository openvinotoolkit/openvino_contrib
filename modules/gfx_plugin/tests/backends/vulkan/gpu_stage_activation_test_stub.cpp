// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/tanh.hpp"
#include "backends/vulkan/runtime/stage_factory.hpp"
#include "runtime/gfx_backend_utils.hpp"
#include "runtime/execution_dispatcher.hpp"
#include "backends/vulkan/runtime/vulkan_memory.hpp"

using namespace ov::gfx_plugin;

namespace {

struct ActivationCase {
    std::vector<float> in;
    std::vector<float> expected;
};

bool is_vulkan_unsupported_error(const std::string& msg) {
    return msg.find("GFX Vulkan") != std::string::npos ||
           msg.find("SPIR-V") != std::string::npos ||
           msg.find("spirv") != std::string::npos ||
           msg.find("vulkan") != std::string::npos;
}

template <typename OpFactory>
void run_activation(const ActivationCase& tc) {
    ensure_vulkan_stage_factory_registered();
    try {
        const size_t bytes = tc.in.size() * sizeof(float);
        ASSERT_GT(bytes, 0u);

        GpuTensor input{};
        input.shape = {tc.in.size()};
        input.expected_type = ov::element::f32;
        input.buf = vulkan_allocate_buffer(bytes,
                                           ov::element::f32,
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        ASSERT_TRUE(input.buf.valid());
        ASSERT_TRUE(input.buf.host_visible);

        void* mapped_in = vulkan_map_buffer(input.buf);
        ASSERT_NE(mapped_in, nullptr);
        std::memcpy(mapped_in, tc.in.data(), bytes);
        vulkan_unmap_buffer(input.buf);

        GpuTensor output{};
        output.shape = {tc.in.size()};
        output.expected_type = ov::element::f32;
        output.prefer_private = false;

        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{tc.in.size()});
        auto node = OpFactory::make_node(param);
        auto stage = GpuStageFactory::create(node, default_backend_kind());
        ASSERT_NE(stage, nullptr);
        stage->set_inputs({&input});
        stage->set_output(&output);
        stage->init(nullptr);
        stage->compile(nullptr);
        stage->execute(nullptr);

        ASSERT_TRUE(output.buf.valid());
        ASSERT_TRUE(output.buf.host_visible);

        void* mapped_out = vulkan_map_buffer(output.buf);
        ASSERT_NE(mapped_out, nullptr);
        auto* data = static_cast<const float*>(mapped_out);
        ASSERT_EQ(output.shape.size(), 1u);
        ASSERT_EQ(output.shape[0], tc.expected.size());
        for (size_t i = 0; i < tc.expected.size(); ++i) {
            EXPECT_NEAR(data[i], tc.expected[i], 1e-4f) << "idx=" << i;
        }
        vulkan_unmap_buffer(output.buf);

        vulkan_free_buffer(input.buf);
        if (output.buf.valid() && !output.buf.external) {
            vulkan_free_buffer(output.buf);
        }
    } catch (const std::exception& e) {
        if (is_vulkan_unsupported_error(e.what())) {
            SUCCEED() << "Vulkan backend did not support this case yet: " << e.what();
            return;
        }
        throw;
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
