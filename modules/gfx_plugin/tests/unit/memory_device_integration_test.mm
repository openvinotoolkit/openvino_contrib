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
// Integration: run_device should avoid D2H until output is requested.
TEST(GfxRunDeviceIntegration, NoHostRoundTripUntilOutputRequested) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);
    ov::Core core;
    register_gfx_plugin(core);

    auto p0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3});
    auto p1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 4});
    auto mm = std::make_shared<ov::op::v0::MatMul>(p0, p1, false, false);
    auto res = std::make_shared<ov::op::v0::Result>(mm);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{p0, p1}, "mm_mem_integ");

    auto metal_cm = core.compile_model(model, "GFX");

    ov::Tensor a{ov::element::f32, {2, 3}};
    ov::Tensor b{ov::element::f32, {3, 4}};
    for (size_t i = 0; i < a.get_size(); ++i) a.data<float>()[i] = static_cast<float>(i + 1);
    for (size_t i = 0; i < b.get_size(); ++i) b.data<float>()[i] = static_cast<float>((i % 5) - 2);

    auto metal_req = metal_cm.create_infer_request();
    metal_req.set_input_tensor(0, a);
    metal_req.set_input_tensor(1, b);
    auto ctx = core.get_default_context("GFX");
    auto out_port = model->output(0);
    ov::Shape out_shape = out_port.get_partial_shape().is_static() ? out_port.get_shape() : ov::Shape{1};
    auto remote_out = ctx.create_tensor(out_port.get_element_type(), out_shape);
    metal_req.set_tensor(out_port, remote_out);
    metal_req.infer();

    // After infer but before get_output_tensor, D2H should be zero.
    auto stats_before = metal_cm.get_property("GFX_MEM_STATS").as<MetalMemoryStats>();
    std::cout << "[TEST] stats_before h2d=" << stats_before.h2d_bytes
              << " d2h=" << stats_before.d2h_bytes
              << " alloc_total=" << stats_before.bytes_allocated_total
              << " reuse_hits=" << stats_before.num_reuse_hits << std::endl;
    EXPECT_GT(stats_before.bytes_allocated_total, 0u);
    EXPECT_EQ(stats_before.h2d_bytes, 0u);
    EXPECT_EQ(stats_before.d2h_bytes, 0u);

    auto stats_after = metal_cm.get_property("GFX_MEM_STATS").as<MetalMemoryStats>();
    std::cout << "[TEST] stats_after h2d=" << stats_after.h2d_bytes
              << " d2h=" << stats_after.d2h_bytes
              << " alloc_total=" << stats_after.bytes_allocated_total
              << " reuse_hits=" << stats_after.num_reuse_hits << std::endl;
    // No CPU copies in GFX backend; output should be shared-memory view.
    EXPECT_EQ(stats_after.d2h_bytes, 0u);

    ASSERT_EQ(remote_out.get_shape(), ov::Shape({2, 4}));
    ASSERT_EQ(remote_out.get_element_type(), ov::element::f32);
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
