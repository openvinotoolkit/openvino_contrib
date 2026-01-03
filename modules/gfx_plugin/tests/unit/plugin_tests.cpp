// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "openvino/openvino.hpp"
#include "plugin/gfx_backend_config.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"

namespace {

void register_gfx_plugin(ov::Core& core) {
#ifdef GFX_PLUGIN_PATH
    try {
        const char* env_path = std::getenv("GFX_PLUGIN_PATH");
        const char* path = (env_path && *env_path) ? env_path : GFX_PLUGIN_PATH;
        core.register_plugin(path, "GFX");
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        if (msg.find("already registered") == std::string::npos) {
            FAIL() << "GFX plugin unavailable: " << e.what();
        }
    }
#else
    (void)core;
#endif
}

}  // namespace

TEST(GfxBackendProperty, DefaultAndExplicitSelection) {
    ov::Core core;
    register_gfx_plugin(core);

    const auto metal_available = ov::gfx_plugin::kGfxBackendMetalAvailable;
    const auto vulkan_available = ov::gfx_plugin::kGfxBackendVulkanAvailable;

    if (!metal_available && !vulkan_available) {
        EXPECT_THROW(core.get_property("GFX", "GFX_BACKEND"), ov::Exception);
        return;
    }

    const auto default_backend = core.get_property("GFX", "GFX_BACKEND").as<std::string>();
    EXPECT_EQ(default_backend, std::string(ov::gfx_plugin::kGfxDefaultBackend));

    if (metal_available) {
        core.set_property("GFX", {{"GFX_BACKEND", "metal"}});
        EXPECT_EQ(core.get_property("GFX", "GFX_BACKEND").as<std::string>(), "metal");
    } else {
        EXPECT_THROW(core.set_property("GFX", {{"GFX_BACKEND", "metal"}}), ov::Exception);
    }

    if (vulkan_available) {
        core.set_property("GFX", {{"GFX_BACKEND", "VULKAN"}});
        EXPECT_EQ(core.get_property("GFX", "GFX_BACKEND").as<std::string>(), "vulkan");
    } else {
        EXPECT_THROW(core.set_property("GFX", {{"GFX_BACKEND", "VULKAN"}}), ov::Exception);
    }
}

TEST(GfxBackendProperty, CompileModelHonorsBackend) {
    ov::Core core;
    register_gfx_plugin(core);

    const auto metal_available = ov::gfx_plugin::kGfxBackendMetalAvailable;
    const auto vulkan_available = ov::gfx_plugin::kGfxBackendVulkanAvailable;

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    auto res = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "backend_model");

    if (metal_available) {
        auto cm_metal = core.compile_model(model, "GFX", {{"GFX_BACKEND", "metal"}});
        EXPECT_EQ(cm_metal.get_property("GFX_BACKEND").as<std::string>(), "metal");
    } else {
        EXPECT_THROW(core.compile_model(model, "GFX", {{"GFX_BACKEND", "metal"}}), ov::Exception);
    }

    if (vulkan_available) {
        auto cm_vulkan = core.compile_model(model, "GFX", {{"GFX_BACKEND", "vulkan"}});
        EXPECT_EQ(cm_vulkan.get_property("GFX_BACKEND").as<std::string>(), "vulkan");
    } else {
        EXPECT_THROW(core.compile_model(model, "GFX", {{"GFX_BACKEND", "vulkan"}}), ov::Exception);
    }
}

TEST(GfxBackendProperty, InferenceWithSelectedBackend) {
    ov::Core core;
    register_gfx_plugin(core);

    const auto metal_available = ov::gfx_plugin::kGfxBackendMetalAvailable;
    const auto vulkan_available = ov::gfx_plugin::kGfxBackendVulkanAvailable;

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    auto res = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "backend_infer_model");

    if (metal_available) {
        auto cm = core.compile_model(model, "GFX", {{"GFX_BACKEND", "metal"}});
        auto req = cm.create_infer_request();
        ov::Tensor input{ov::element::f32, {1}};
        input.data<float>()[0] = 1.0f;
        req.set_input_tensor(input);
        req.infer();
    } else {
        EXPECT_THROW(core.compile_model(model, "GFX", {{"GFX_BACKEND", "metal"}}), ov::Exception);
    }

    if (vulkan_available) {
        auto cm = core.compile_model(model, "GFX", {{"GFX_BACKEND", "vulkan"}});
        auto req = cm.create_infer_request();
        ov::Tensor input{ov::element::f32, {1}};
        input.data<float>()[0] = 1.0f;
        req.set_input_tensor(input);
        req.infer();
    } else {
        EXPECT_THROW(core.compile_model(model, "GFX", {{"GFX_BACKEND", "vulkan"}}), ov::Exception);
    }
}

TEST(GfxBackendProperty, LogsBackendSelection) {
    const char* old_trace = std::getenv("OV_GFX_TRACE");
    ::setenv("OV_GFX_TRACE", "1", 1);

    ov::Core core;
    register_gfx_plugin(core);

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    auto res = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "backend_log_model");

    testing::internal::CaptureStderr();
    if (ov::gfx_plugin::kGfxBackendMetalAvailable) {
        (void)core.compile_model(model, "GFX", {{"GFX_BACKEND", "metal"}});
        auto log_metal = testing::internal::GetCapturedStderr();
        EXPECT_NE(log_metal.find("Selected GFX backend: metal"), std::string::npos);
    } else {
        EXPECT_THROW(core.compile_model(model, "GFX", {{"GFX_BACKEND", "metal"}}), ov::Exception);
        (void)testing::internal::GetCapturedStderr();
    }

    testing::internal::CaptureStderr();
    if (ov::gfx_plugin::kGfxBackendVulkanAvailable) {
        (void)core.compile_model(model, "GFX", {{"GFX_BACKEND", "vulkan"}});
        auto log_vulkan = testing::internal::GetCapturedStderr();
        EXPECT_NE(log_vulkan.find("Selected GFX backend: vulkan"), std::string::npos);
    } else {
        EXPECT_THROW(core.compile_model(model, "GFX", {{"GFX_BACKEND", "vulkan"}}), ov::Exception);
        (void)testing::internal::GetCapturedStderr();
    }

    if (old_trace) {
        ::setenv("OV_GFX_TRACE", old_trace, 1);
    } else {
        ::unsetenv("OV_GFX_TRACE");
    }
}
