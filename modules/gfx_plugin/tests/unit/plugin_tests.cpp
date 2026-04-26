// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdlib>
#include <cctype>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"
#include "plugin/gfx_backend_config.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/strided_slice.hpp"
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

TEST(GfxDeviceProperties, AvailableDevicesExposeIds) {
    ov::Core core;
    register_gfx_plugin(core);

    const auto available = core.get_property("GFX", ov::available_devices);
    ASSERT_FALSE(available.empty());
    for (const auto& entry : available) {
        EXPECT_FALSE(entry.empty());
        EXPECT_TRUE(std::all_of(entry.begin(), entry.end(), [](unsigned char ch) { return std::isdigit(ch) != 0; }))
            << "available_devices must expose numeric device ids, got '" << entry << "'";
    }

    const auto default_device_id = core.get_property("GFX", ov::device::id);
    EXPECT_EQ(default_device_id, "0");
    EXPECT_EQ(available.front(), "0");
}

TEST(GfxDynamicShapeSupport, CompileModelWithDynamicShapeOf) {
    ov::Core core;
    register_gfx_plugin(core);

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, 64});
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(param, ov::element::i64);
    auto res = std::make_shared<ov::op::v0::Result>(shape_of);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "dynamic_shapeof");

    try {
        auto cm = core.compile_model(model, "GFX");
        EXPECT_EQ(cm.get_runtime_model()->get_results().size(), 1);
    } catch (const std::exception& e) {
        FAIL() << "compile_model(dynamic ShapeOf) failed: " << e.what();
    }
}

TEST(GfxDynamicShapeSupport, QueryModelWithDynamicShapeOf) {
    ov::Core core;
    register_gfx_plugin(core);

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, 64});
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(param, ov::element::i64);
    shape_of->set_friendly_name("shapeof_dyn");
    auto res = std::make_shared<ov::op::v0::Result>(shape_of);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "dynamic_shapeof");

    EXPECT_NO_THROW({
        const auto supported = core.query_model(model, "GFX");
        EXPECT_TRUE(supported.count("shapeof_dyn") != 0);
    });
}

TEST(GfxDynamicShapeSupport, QueryModelWithDynamicShapeDataMovementOps) {
    ov::Core core;
    register_gfx_plugin(core);

    auto data0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, 4});
    auto data1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, 4});
    auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::PartialShape{1, -1, 4});
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3});
    auto slice_end = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3});
    auto range_stop = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{});

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{data0, data1}, 1);
    concat->set_friendly_name("dyn_concat");
    auto broadcast = std::make_shared<ov::op::v3::Broadcast>(data0, target_shape, ov::op::BroadcastType::BIDIRECTIONAL);
    broadcast->set_friendly_name("dyn_broadcast");
    auto select = std::make_shared<ov::op::v1::Select>(cond, data0, data1);
    select->set_friendly_name("dyn_select");

    auto begin = ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 0, 0});
    auto strides = ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 1, 1});
    auto slice = std::make_shared<ov::op::v1::StridedSlice>(data0,
                                                            begin,
                                                            slice_end,
                                                            strides,
                                                            std::vector<int64_t>{0, 0, 0},
                                                            std::vector<int64_t>{0, 0, 0});
    slice->set_friendly_name("dyn_slice");

    auto range_start = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    auto range_step = ov::op::v0::Constant::create(ov::element::i64, {}, {1});
    auto range = std::make_shared<ov::op::v4::Range>(range_start, range_stop, range_step, ov::element::i64);
    range->set_friendly_name("dyn_range");

    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(concat),
                         std::make_shared<ov::op::v0::Result>(broadcast),
                         std::make_shared<ov::op::v0::Result>(select),
                         std::make_shared<ov::op::v0::Result>(slice),
                         std::make_shared<ov::op::v0::Result>(range)},
        ov::ParameterVector{data0, data1, cond, target_shape, slice_end, range_stop},
        "dynamic_data_movement");

    EXPECT_NO_THROW({
        const auto supported = core.query_model(model, "GFX");
        EXPECT_TRUE(supported.count("dyn_concat") != 0);
        EXPECT_TRUE(supported.count("dyn_broadcast") != 0);
        EXPECT_TRUE(supported.count("dyn_select") != 0);
        EXPECT_TRUE(supported.count("dyn_slice") != 0);
        EXPECT_TRUE(supported.count("dyn_range") != 0);
    });
}
