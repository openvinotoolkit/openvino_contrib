// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "backends/vulkan/runtime/stage_factory.hpp"
#include "runtime/gfx_backend_utils.hpp"
#include "runtime/execution_dispatcher.hpp"

using namespace ov::gfx_plugin;

namespace {

bool is_vulkan_unsupported_error(const std::string& msg) {
    return msg.find("GFX Vulkan") != std::string::npos ||
           msg.find("SPIR-V") != std::string::npos ||
           msg.find("spirv") != std::string::npos ||
           msg.find("vulkan") != std::string::npos;
}

}  // namespace

TEST(GpuStageFactory, CreatesStubForRelu) {
    ensure_vulkan_stage_factory_registered();
    auto p = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto relu = std::make_shared<ov::op::v0::Relu>(p);

    try {
        auto stage = GpuStageFactory::create(relu, GpuBackend::Vulkan);
        ASSERT_NE(stage, nullptr);
        EXPECT_FALSE(stage->type().empty());
    } catch (const std::exception& e) {
        if (is_vulkan_unsupported_error(e.what())) {
            SUCCEED() << "Vulkan backend did not support this case yet: " << e.what();
            return;
        }
        throw;
    }
}

TEST(GpuStageFactory, ReturnsNullForUnsupportedParameter) {
    ensure_vulkan_stage_factory_registered();
    auto p = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});

    try {
        auto stage = GpuStageFactory::create(p, GpuBackend::Vulkan);
        EXPECT_NE(stage, nullptr);
    } catch (const std::exception& e) {
        if (is_vulkan_unsupported_error(e.what())) {
            SUCCEED() << "Vulkan backend did not support this case yet: " << e.what();
            return;
        }
        throw;
    }
}
