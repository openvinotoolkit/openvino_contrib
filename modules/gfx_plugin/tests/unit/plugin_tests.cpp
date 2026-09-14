// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <vector>

#include "../gfx_plugin_runtime_path.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/openvino.hpp"
#include "compiler/backend_config.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"

namespace {

void register_gfx_plugin(ov::Core& core) {
    std::string error;
    if (!ov::test::utils::register_gfx_plugin_runtime_path(core, &error)) {
        FAIL() << error;
    }
}

ov::Tensor make_f16_tensor(const ov::Shape& shape, const std::vector<float>& values) {
    ov::Tensor tensor(ov::element::f16, shape);
    EXPECT_EQ(tensor.get_size(), values.size());
    auto* data = tensor.data<ov::float16>();
    for (size_t i = 0; i < values.size(); ++i) {
        data[i] = ov::float16(values[i]);
    }
    return tensor;
}

ov::Tensor make_f32_tensor(const ov::Shape& shape, const std::vector<float>& values) {
    ov::Tensor tensor(ov::element::f32, shape);
    EXPECT_EQ(tensor.get_size(), values.size());
    auto* data = tensor.data<float>();
    std::copy(values.begin(), values.end(), data);
    return tensor;
}

ov::Tensor make_i32_tensor(const ov::Shape& shape, const std::vector<int32_t>& values) {
    ov::Tensor tensor(ov::element::i32, shape);
    EXPECT_EQ(tensor.get_size(), values.size());
    auto* data = tensor.data<int32_t>();
    std::copy(values.begin(), values.end(), data);
    return tensor;
}

ov::Tensor make_i64_tensor(const ov::Shape& shape, const std::vector<int64_t>& values) {
    ov::Tensor tensor(ov::element::i64, shape);
    EXPECT_EQ(tensor.get_size(), values.size());
    auto* data = tensor.data<int64_t>();
    std::copy(values.begin(), values.end(), data);
    return tensor;
}

ov::Tensor make_bool_tensor(const ov::Shape& shape, const std::vector<uint8_t>& values) {
    ov::Tensor tensor(ov::element::boolean, shape);
    EXPECT_EQ(tensor.get_size(), values.size());
    auto* data = tensor.data<uint8_t>();
    std::copy(values.begin(), values.end(), data);
    return tensor;
}

void expect_f16_tensor(const ov::Tensor& tensor,
                       const ov::Shape& shape,
                       const std::vector<float>& expected) {
    ASSERT_EQ(tensor.get_element_type(), ov::element::f16);
    ASSERT_EQ(tensor.get_shape(), shape);
    ASSERT_EQ(tensor.get_size(), expected.size());
    const auto* data = tensor.data<const ov::float16>();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(static_cast<float>(data[i]), expected[i]) << "at index " << i;
    }
}

void expect_f32_tensor(const ov::Tensor& tensor,
                       const ov::Shape& shape,
                       const std::vector<float>& expected) {
    ASSERT_EQ(tensor.get_element_type(), ov::element::f32);
    ASSERT_EQ(tensor.get_shape(), shape);
    ASSERT_EQ(tensor.get_size(), expected.size());
    const auto* data = tensor.data<const float>();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(data[i], expected[i]) << "at index " << i;
    }
}

void expect_i32_tensor(const ov::Tensor& tensor,
                       const ov::Shape& shape,
                       const std::vector<int32_t>& expected) {
    ASSERT_EQ(tensor.get_element_type(), ov::element::i32);
    ASSERT_EQ(tensor.get_shape(), shape);
    ASSERT_EQ(tensor.get_size(), expected.size());
    const auto* data = tensor.data<const int32_t>();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(data[i], expected[i]) << "at index " << i;
    }
}

void expect_i64_tensor(const ov::Tensor& tensor,
                       const ov::Shape& shape,
                       const std::vector<int64_t>& expected) {
    ASSERT_EQ(tensor.get_element_type(), ov::element::i64);
    ASSERT_EQ(tensor.get_shape(), shape);
    ASSERT_EQ(tensor.get_size(), expected.size());
    const auto* data = tensor.data<const int64_t>();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(data[i], expected[i]) << "at index " << i;
    }
}

}  // namespace

TEST(GfxBackendProperty, DefaultAndExplicitSelection) {
    ov::Core core;
    register_gfx_plugin(core);

    const auto metal_available = ov::gfx_plugin::kGfxBackendMetalAvailable;
    const auto opencl_available = ov::gfx_plugin::kGfxBackendOpenCLAvailable;

    if (!metal_available && !opencl_available) {
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

    if (opencl_available) {
        core.set_property("GFX", {{"GFX_BACKEND", "OPENCL"}});
        EXPECT_EQ(core.get_property("GFX", "GFX_BACKEND").as<std::string>(), "opencl");
    } else {
        EXPECT_THROW(core.set_property("GFX", {{"GFX_BACKEND", "OPENCL"}}), ov::Exception);
    }
    EXPECT_THROW(core.set_property("GFX", {{"GFX_BACKEND", "invalid_backend"}}), ov::Exception);
}

TEST(GfxBackendProperty, CompileModelHonorsBackend) {
    ov::Core core;
    register_gfx_plugin(core);

    const auto metal_available = ov::gfx_plugin::kGfxBackendMetalAvailable;
    const auto opencl_available = ov::gfx_plugin::kGfxBackendOpenCLAvailable;

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

    if (opencl_available) {
        auto cm_opencl = core.compile_model(model, "GFX", {{"GFX_BACKEND", "opencl"}});
        EXPECT_EQ(cm_opencl.get_property("GFX_BACKEND").as<std::string>(), "opencl");
    } else {
        EXPECT_THROW(core.compile_model(model, "GFX", {{"GFX_BACKEND", "opencl"}}), ov::Exception);
    }
}

TEST(GfxPrecisionProperty, PluginAndCompiledModelHonorExplicitPrecision) {
    ov::Core core;
    register_gfx_plugin(core);

    if (!ov::gfx_plugin::kGfxBackendMetalAvailable && !ov::gfx_plugin::kGfxBackendOpenCLAvailable) {
        FAIL() << "GFX backend unavailable";
    }

    core.set_property("GFX", {{ov::hint::inference_precision.name(), ov::element::f32}});
    EXPECT_EQ(core.get_property("GFX", ov::hint::inference_precision), ov::element::f32);
    EXPECT_THROW(core.set_property("GFX", {{ov::hint::inference_precision.name(), std::string("i8")}}),
                 ov::Exception);

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    auto res = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param},
                                             "precision_property_model");

    auto cm = core.compile_model(model, "GFX",
                                 {{ov::hint::inference_precision.name(), std::string("f32")}});
    EXPECT_EQ(cm.get_property(ov::hint::inference_precision.name()).as<ov::element::Type>(),
              ov::element::f32);
    cm.set_property({{ov::hint::inference_precision.name(), std::string("f16")}});
    EXPECT_EQ(cm.get_property(ov::hint::inference_precision.name()).as<ov::element::Type>(),
              ov::element::f16);
}

TEST(GfxElementwiseSupport, CompileAndInferStaticF16BinaryOps) {
    ov::Core core;
    register_gfx_plugin(core);

    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{2, 3});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{2, 3});
    auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);
    add->set_friendly_name("static_f16_add");
    auto sqdiff = std::make_shared<ov::op::v0::SquaredDifference>(lhs, rhs);
    sqdiff->set_friendly_name("static_f16_squared_difference");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(add),
                         std::make_shared<ov::op::v0::Result>(sqdiff)},
        ov::ParameterVector{lhs, rhs},
        "static_f16_binary_elementwise_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, make_f16_tensor({2, 3}, {1, 2, 3, 4, 5, 6}));
    request.set_input_tensor(1, make_f16_tensor({2, 3}, {0.5f, 1.5f, 2, 2.5f, 1, 3}));

    request.infer();

    expect_f16_tensor(request.get_output_tensor(0), {2, 3},
                      {1.5f, 3.5f, 5, 6.5f, 6, 9});
    expect_f16_tensor(request.get_output_tensor(1), {2, 3},
                      {0.25f, 0.25f, 1, 2.25f, 16, 9});
}

TEST(GfxElementwiseSupport, CompileAndInferStaticI32ModFloorMod) {
    ov::Core core;
    register_gfx_plugin(core);

    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{6});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{6});
    auto mod = std::make_shared<ov::op::v1::Mod>(lhs, rhs);
    mod->set_friendly_name("static_i32_mod");
    auto floor_mod = std::make_shared<ov::op::v1::FloorMod>(lhs, rhs);
    floor_mod->set_friendly_name("static_i32_floor_mod");
    auto sqdiff = std::make_shared<ov::op::v0::SquaredDifference>(lhs, rhs);
    sqdiff->set_friendly_name("static_i32_squared_difference");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(mod),
                         std::make_shared<ov::op::v0::Result>(floor_mod),
                         std::make_shared<ov::op::v0::Result>(sqdiff)},
        ov::ParameterVector{lhs, rhs},
        "static_i32_mod_floor_mod_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, make_i32_tensor({6}, {5, 7, -5, -7, 9, 10}));
    request.set_input_tensor(1, make_i32_tensor({6}, {2, 3, 2, 3, 4, 5}));

    request.infer();

    expect_i32_tensor(request.get_output_tensor(0), {6}, {1, 1, -1, -1, 1, 0});
    expect_i32_tensor(request.get_output_tensor(1), {6}, {1, 1, 1, 2, 1, 0});
    expect_i32_tensor(request.get_output_tensor(2), {6}, {9, 16, 49, 100, 25, 25});
}

TEST(GfxBackendProperty, InferenceWithSelectedBackend) {
    ov::Core core;
    register_gfx_plugin(core);

    const auto metal_available = ov::gfx_plugin::kGfxBackendMetalAvailable;
    const auto opencl_available = ov::gfx_plugin::kGfxBackendOpenCLAvailable;

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

    if (opencl_available) {
        auto cm = core.compile_model(model, "GFX", {{"GFX_BACKEND", "opencl"}});
        auto req = cm.create_infer_request();
        ov::Tensor input{ov::element::f32, {1}};
        input.data<float>()[0] = 1.0f;
        req.set_input_tensor(input);
        req.infer();
    } else {
        EXPECT_THROW(core.compile_model(model, "GFX", {{"GFX_BACKEND", "opencl"}}), ov::Exception);
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
    if (ov::gfx_plugin::kGfxBackendOpenCLAvailable) {
        (void)core.compile_model(model, "GFX", {{"GFX_BACKEND", "opencl"}});
        auto log_opencl = testing::internal::GetCapturedStderr();
        EXPECT_NE(log_opencl.find("Selected GFX backend: opencl"), std::string::npos);
    } else {
        EXPECT_THROW(core.compile_model(model, "GFX", {{"GFX_BACKEND", "opencl"}}), ov::Exception);
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
    auto data2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, 4});
    auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::PartialShape{1, -1, 4});
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3});
    auto slice_end = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3});
    auto slice_v8_start = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3});
    auto slice_v8_end = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3});
    auto slice_v8_step = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3});
    auto range_stop = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{});

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{data0, data1, data2}, 1);
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
    auto slice_v8 = std::make_shared<ov::op::v8::Slice>(data0, slice_v8_start, slice_v8_end, slice_v8_step);
    slice_v8->set_friendly_name("dyn_slice_v8");

    auto range_start = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    auto range_step = ov::op::v0::Constant::create(ov::element::i64, {}, {1});
    auto range = std::make_shared<ov::op::v4::Range>(range_start, range_stop, range_step, ov::element::i64);
    range->set_friendly_name("dyn_range");

    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(concat),
                         std::make_shared<ov::op::v0::Result>(broadcast),
                         std::make_shared<ov::op::v0::Result>(select),
                         std::make_shared<ov::op::v0::Result>(slice),
                         std::make_shared<ov::op::v0::Result>(slice_v8),
                         std::make_shared<ov::op::v0::Result>(range)},
        ov::ParameterVector{data0, data1, data2, cond, target_shape, slice_end, slice_v8_start, slice_v8_end, slice_v8_step, range_stop},
        "dynamic_data_movement");

    EXPECT_NO_THROW({
        const auto supported = core.query_model(model, "GFX");
        EXPECT_TRUE(supported.count("dyn_concat") != 0);
        EXPECT_TRUE(supported.count("dyn_broadcast") != 0);
        EXPECT_TRUE(supported.count("dyn_select") != 0);
        EXPECT_TRUE(supported.count("dyn_slice") != 0);
        EXPECT_TRUE(supported.count("dyn_slice_v8") != 0);
        EXPECT_TRUE(supported.count("dyn_range") != 0);
    });
}

TEST(GfxDynamicShapeSupport, CompileAndInferDynamicShapeDataMovementOps) {
    ov::Core core;
    register_gfx_plugin(core);

    auto data0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, 4});
    auto data1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, 4});
    auto data2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, 4});
    auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::PartialShape{1, -1, 4});
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3});
    auto slice_end = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3});
    auto slice_v8_start = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3});
    auto slice_v8_end = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3});
    auto slice_v8_step = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3});
    auto range_stop = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{});

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{data0, data1, data2}, 1);
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
    auto slice_v8 = std::make_shared<ov::op::v8::Slice>(data0, slice_v8_start, slice_v8_end, slice_v8_step);
    slice_v8->set_friendly_name("dyn_slice_v8");

    auto range_start = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    auto range_step = ov::op::v0::Constant::create(ov::element::i64, {}, {1});
    auto range = std::make_shared<ov::op::v4::Range>(range_start, range_stop, range_step, ov::element::i64);
    range->set_friendly_name("dyn_range");

    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(concat),
                         std::make_shared<ov::op::v0::Result>(broadcast),
                         std::make_shared<ov::op::v0::Result>(select),
                         std::make_shared<ov::op::v0::Result>(slice),
                         std::make_shared<ov::op::v0::Result>(slice_v8),
                         std::make_shared<ov::op::v0::Result>(range)},
        ov::ParameterVector{data0, data1, data2, cond, target_shape, slice_end, slice_v8_start, slice_v8_end, slice_v8_step, range_stop},
        "dynamic_data_movement_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, make_f16_tensor({1, 2, 4}, {0, 1, 2, 3, 4, 5, 6, 7}));
    request.set_input_tensor(1, make_f16_tensor({1, 2, 4}, {100, 101, 102, 103, 104, 105, 106, 107}));
    request.set_input_tensor(2, make_f16_tensor({1, 2, 4}, {200, 201, 202, 203, 204, 205, 206, 207}));
    request.set_input_tensor(3, make_bool_tensor({1, 2, 4}, {1, 0, 1, 0, 1, 0, 1, 0}));
    request.set_input_tensor(4, make_i64_tensor({3}, {1, 2, 4}));
    request.set_input_tensor(5, make_i64_tensor({3}, {1, 1, 4}));
    request.set_input_tensor(6, make_i64_tensor({3}, {0, 1, 0}));
    request.set_input_tensor(7, make_i64_tensor({3}, {1, 2, 4}));
    request.set_input_tensor(8, make_i64_tensor({3}, {1, 1, 2}));
    request.set_input_tensor(9, make_i64_tensor({}, {5}));

    request.infer();

    expect_f16_tensor(request.get_output_tensor(0),
                      {1, 6, 4},
                      {0, 1, 2, 3, 4, 5, 6, 7,
                       100, 101, 102, 103, 104, 105, 106, 107,
                       200, 201, 202, 203, 204, 205, 206, 207});
    expect_f16_tensor(request.get_output_tensor(1),
                      {1, 2, 4},
                      {0, 1, 2, 3, 4, 5, 6, 7});
    expect_f16_tensor(request.get_output_tensor(2),
                      {1, 2, 4},
                      {0, 101, 2, 103, 4, 105, 6, 107});
    expect_f16_tensor(request.get_output_tensor(3),
                      {1, 1, 4},
                      {0, 1, 2, 3});
    expect_f16_tensor(request.get_output_tensor(4),
                      {1, 1, 2},
                      {4, 6});
    expect_i64_tensor(request.get_output_tensor(5), {5}, {0, 1, 2, 3, 4});
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF16Tile) {
    ov::Core core;
    register_gfx_plugin(core);

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{2, 1, 3});
    auto repeats = ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 3, 2});
    auto tile = std::make_shared<ov::op::v0::Tile>(data, repeats);
    tile->set_friendly_name("static_f16_tile");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(tile)},
        ov::ParameterVector{data},
        "static_f16_tile_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, make_f16_tensor({2, 1, 3}, {0, 1, 2, 3, 4, 5}));

    request.infer();

    expect_f16_tensor(request.get_output_tensor(0),
                      {2, 3, 6},
                      {0, 1, 2, 0, 1, 2,
                       0, 1, 2, 0, 1, 2,
                       0, 1, 2, 0, 1, 2,
                       3, 4, 5, 3, 4, 5,
                       3, 4, 5, 3, 4, 5,
                       3, 4, 5, 3, 4, 5});
}

TEST(GfxDataMovementSupport, CompileAndInferDynamicF16Tile) {
    ov::Core core;
    register_gfx_plugin(core);

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, 3});
    auto repeats = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3});
    auto tile = std::make_shared<ov::op::v0::Tile>(data, repeats);
    tile->set_friendly_name("dynamic_f16_tile");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(tile)},
        ov::ParameterVector{data, repeats},
        "dynamic_f16_tile_infer");

    EXPECT_NO_THROW({
        const auto supported = core.query_model(model, "GFX");
        EXPECT_TRUE(supported.count("dynamic_f16_tile") != 0);
    });

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, make_f16_tensor({1, 2, 3}, {0, 1, 2, 3, 4, 5}));
    request.set_input_tensor(1, make_i64_tensor({3}, {1, 3, 2}));

    request.infer();

    expect_f16_tensor(request.get_output_tensor(0),
                      {1, 6, 6},
                      {0, 1, 2, 0, 1, 2,
                       3, 4, 5, 3, 4, 5,
                       0, 1, 2, 0, 1, 2,
                       3, 4, 5, 3, 4, 5,
                       0, 1, 2, 0, 1, 2,
                       3, 4, 5, 3, 4, 5});
}

TEST(GfxDataMovementSupport, CompileAndInferDynamicF32Tile) {
    ov::Core core;
    register_gfx_plugin(core);

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 3});
    auto repeats = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3});
    auto tile = std::make_shared<ov::op::v0::Tile>(data, repeats);
    tile->set_friendly_name("dynamic_f32_tile");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(tile)},
        ov::ParameterVector{data, repeats},
        "dynamic_f32_tile_infer");

    EXPECT_NO_THROW({
        const auto supported = core.query_model(model, "GFX");
        EXPECT_TRUE(supported.count("dynamic_f32_tile") != 0);
    });

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, make_f32_tensor({1, 2, 3}, {0, 1, 2, 3, 4, 5}));
    request.set_input_tensor(1, make_i64_tensor({3}, {1, 3, 2}));

    request.infer();

    expect_f32_tensor(request.get_output_tensor(0),
                      {1, 6, 6},
                      {0, 1, 2, 0, 1, 2,
                       3, 4, 5, 3, 4, 5,
                       0, 1, 2, 0, 1, 2,
                       3, 4, 5, 3, 4, 5,
                       0, 1, 2, 0, 1, 2,
                       3, 4, 5, 3, 4, 5});
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF32Range) {
    ov::Core core;
    register_gfx_plugin(core);

    auto start = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.5f});
    auto stop = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {6.5f});
    auto step = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
    auto range = std::make_shared<ov::op::v4::Range>(start, stop, step, ov::element::f32);
    range->set_friendly_name("static_f32_range");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(range)},
        ov::ParameterVector{},
        "static_f32_range_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();

    request.infer();

    expect_f32_tensor(request.get_output_tensor(0),
                      {5},
                      {1.5f, 2.5f, 3.5f, 4.5f, 5.5f});
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF32Transpose) {
    ov::Core core;
    register_gfx_plugin(core);

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 4});
    auto order = ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 2, 0});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(data, order);
    transpose->set_friendly_name("static_f32_transpose");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(transpose)},
        ov::ParameterVector{data},
        "static_f32_transpose_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, make_f32_tensor({2, 3, 4},
                                                {0, 1, 2, 3, 4, 5, 6, 7,
                                                 8, 9, 10, 11, 12, 13, 14, 15,
                                                 16, 17, 18, 19, 20, 21, 22, 23}));

    request.infer();

    expect_f32_tensor(request.get_output_tensor(0),
                      {3, 4, 2},
                      {0, 12, 1, 13, 2, 14, 3, 15,
                       4, 16, 5, 17, 6, 18, 7, 19,
                       8, 20, 9, 21, 10, 22, 11, 23});
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF32Slice) {
    ov::Core core;
    register_gfx_plugin(core);

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 4});
    auto slice = std::make_shared<ov::op::v8::Slice>(
        data,
        ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 0}),
        ov::op::v0::Constant::create(ov::element::i64, {3}, {2, 3, 4}),
        ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 1, 2}),
        ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
    slice->set_friendly_name("static_f32_slice");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(slice)},
        ov::ParameterVector{data},
        "static_f32_slice_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, make_f32_tensor({2, 3, 4},
                                                {0, 1, 2, 3, 4, 5, 6, 7,
                                                 8, 9, 10, 11, 12, 13, 14, 15,
                                                 16, 17, 18, 19, 20, 21, 22, 23}));

    request.infer();

    expect_f32_tensor(request.get_output_tensor(0),
                      {2, 2, 2},
                      {4, 6, 8, 10, 16, 18, 20, 22});
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF32Concat) {
    ov::Core core;
    register_gfx_plugin(core);

    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 3});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 3});
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{lhs, rhs}, 1);
    concat->set_friendly_name("static_f32_concat");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(concat)},
        ov::ParameterVector{lhs, rhs},
        "static_f32_concat_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, make_f32_tensor({1, 2, 3}, {0, 1, 2, 3, 4, 5}));
    request.set_input_tensor(1, make_f32_tensor({1, 4, 3},
                                                {10, 11, 12, 13, 14, 15,
                                                 16, 17, 18, 19, 20, 21}));

    request.infer();

    expect_f32_tensor(request.get_output_tensor(0),
                      {1, 6, 3},
                      {0, 1, 2, 3, 4, 5,
                       10, 11, 12, 13, 14, 15,
                       16, 17, 18, 19, 20, 21});
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF32ConcatFiveInputs) {
    ov::Core core;
    register_gfx_plugin(core);

    ov::ParameterVector params;
    ov::OutputVector inputs;
    const std::vector<ov::Shape> shapes = {
        {1, 1, 2}, {1, 2, 2}, {1, 3, 2}, {1, 4, 2}, {1, 5, 2}};
    for (const auto& shape : shapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape));
        inputs.push_back(params.back());
    }
    auto concat = std::make_shared<ov::op::v0::Concat>(inputs, 1);
    concat->set_friendly_name("static_f32_concat5");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(concat)},
        params,
        "static_f32_concat5_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    float next_value = 0.0f;
    std::vector<float> expected;
    for (size_t input_idx = 0; input_idx < shapes.size(); ++input_idx) {
        std::vector<float> values(shapes[input_idx][1] * shapes[input_idx][2]);
        for (auto& value : values) {
            value = next_value++;
            expected.push_back(value);
        }
        request.set_input_tensor(input_idx, make_f32_tensor(shapes[input_idx], values));
    }

    request.infer();

    expect_f32_tensor(request.get_output_tensor(0), {1, 15, 2}, expected);
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF16ConcatFiveInputs) {
    ov::Core core;
    register_gfx_plugin(core);

    ov::ParameterVector params;
    ov::OutputVector inputs;
    const std::vector<ov::Shape> shapes = {
        {1, 1, 2}, {1, 2, 2}, {1, 3, 2}, {1, 4, 2}, {1, 5, 2}};
    for (const auto& shape : shapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f16, shape));
        inputs.push_back(params.back());
    }
    auto concat = std::make_shared<ov::op::v0::Concat>(inputs, 1);
    concat->set_friendly_name("static_f16_concat5");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(concat)},
        params,
        "static_f16_concat5_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    float next_value = 0.0f;
    std::vector<float> expected;
    for (size_t input_idx = 0; input_idx < shapes.size(); ++input_idx) {
        std::vector<float> values(shapes[input_idx][1] * shapes[input_idx][2]);
        for (auto& value : values) {
            value = next_value++;
            expected.push_back(value);
        }
        request.set_input_tensor(input_idx, make_f16_tensor(shapes[input_idx], values));
    }

    request.infer();

    expect_f16_tensor(request.get_output_tensor(0), {1, 15, 2}, expected);
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF32ConcatThirtyInputs) {
    ov::Core core;
    register_gfx_plugin(core);

    ov::ParameterVector params;
    ov::OutputVector inputs;
    const ov::Shape input_shape{1, 1, 2};
    constexpr size_t kInputCount = 30;
    for (size_t input_idx = 0; input_idx < kInputCount; ++input_idx) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape));
        inputs.push_back(params.back());
    }
    auto concat = std::make_shared<ov::op::v0::Concat>(inputs, 1);
    concat->set_friendly_name("static_f32_concat30");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(concat)},
        params,
        "static_f32_concat30_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    std::vector<float> expected;
    expected.reserve(kInputCount * 2);
    for (size_t input_idx = 0; input_idx < kInputCount; ++input_idx) {
        const float base = static_cast<float>(input_idx * 2);
        std::vector<float> values{base, base + 1.0f};
        expected.insert(expected.end(), values.begin(), values.end());
        request.set_input_tensor(input_idx, make_f32_tensor(input_shape, values));
    }

    request.infer();

    expect_f32_tensor(request.get_output_tensor(0), {1, kInputCount, 2}, expected);
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF16ConcatThirtyInputs) {
    ov::Core core;
    register_gfx_plugin(core);

    ov::ParameterVector params;
    ov::OutputVector inputs;
    const ov::Shape input_shape{1, 1, 2};
    constexpr size_t kInputCount = 30;
    for (size_t input_idx = 0; input_idx < kInputCount; ++input_idx) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape));
        inputs.push_back(params.back());
    }
    auto concat = std::make_shared<ov::op::v0::Concat>(inputs, 1);
    concat->set_friendly_name("static_f16_concat30");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(concat)},
        params,
        "static_f16_concat30_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    std::vector<float> expected;
    expected.reserve(kInputCount * 2);
    for (size_t input_idx = 0; input_idx < kInputCount; ++input_idx) {
        const float base = static_cast<float>(input_idx * 2);
        std::vector<float> values{base, base + 1.0f};
        expected.insert(expected.end(), values.begin(), values.end());
        request.set_input_tensor(input_idx, make_f16_tensor(input_shape, values));
    }

    request.infer();

    expect_f16_tensor(request.get_output_tensor(0), {1, kInputCount, 2}, expected);
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF32Split) {
    ov::Core core;
    register_gfx_plugin(core);

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 6, 2});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto split = std::make_shared<ov::op::v1::Split>(data, axis, 3);
    split->set_friendly_name("static_f32_split");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(split->output(0)),
                         std::make_shared<ov::op::v0::Result>(split->output(1)),
                         std::make_shared<ov::op::v0::Result>(split->output(2))},
        ov::ParameterVector{data},
        "static_f32_split_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, make_f32_tensor({1, 6, 2},
                                                {0, 1, 2, 3, 4, 5,
                                                 6, 7, 8, 9, 10, 11}));

    request.infer();

    expect_f32_tensor(request.get_output_tensor(0), {1, 2, 2}, {0, 1, 2, 3});
    expect_f32_tensor(request.get_output_tensor(1), {1, 2, 2}, {4, 5, 6, 7});
    expect_f32_tensor(request.get_output_tensor(2), {1, 2, 2}, {8, 9, 10, 11});
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF16Split) {
    ov::Core core;
    register_gfx_plugin(core);

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 6, 2});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto split = std::make_shared<ov::op::v1::Split>(data, axis, 3);
    split->set_friendly_name("static_f16_split");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(split->output(0)),
                         std::make_shared<ov::op::v0::Result>(split->output(1)),
                         std::make_shared<ov::op::v0::Result>(split->output(2))},
        ov::ParameterVector{data},
        "static_f16_split_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, make_f16_tensor({1, 6, 2},
                                                {0, 1, 2, 3, 4, 5,
                                                 6, 7, 8, 9, 10, 11}));

    request.infer();

    expect_f16_tensor(request.get_output_tensor(0), {1, 2, 2}, {0, 1, 2, 3});
    expect_f16_tensor(request.get_output_tensor(1), {1, 2, 2}, {4, 5, 6, 7});
    expect_f16_tensor(request.get_output_tensor(2), {1, 2, 2}, {8, 9, 10, 11});
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF32SplitThirtyOutputs) {
    ov::Core core;
    register_gfx_plugin(core);

    constexpr size_t kOutputCount = 30;
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, kOutputCount, 2});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto split = std::make_shared<ov::op::v1::Split>(data, axis, kOutputCount);
    split->set_friendly_name("static_f32_split30");
    ov::ResultVector results;
    for (size_t output_idx = 0; output_idx < split->get_output_size(); ++output_idx) {
        results.push_back(std::make_shared<ov::op::v0::Result>(split->output(output_idx)));
    }
    auto model = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{data},
        "static_f32_split30_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    std::vector<float> values(kOutputCount * 2);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i);
    }
    request.set_input_tensor(0, make_f32_tensor({1, kOutputCount, 2}, values));

    request.infer();

    for (size_t output_idx = 0; output_idx < kOutputCount; ++output_idx) {
        const float base = static_cast<float>(output_idx * 2);
        expect_f32_tensor(request.get_output_tensor(output_idx),
                          {1, 1, 2},
                          {base, base + 1.0f});
    }
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF16SplitThirtyOutputs) {
    ov::Core core;
    register_gfx_plugin(core);

    constexpr size_t kOutputCount = 30;
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, kOutputCount, 2});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto split = std::make_shared<ov::op::v1::Split>(data, axis, kOutputCount);
    split->set_friendly_name("static_f16_split30");
    ov::ResultVector results;
    for (size_t output_idx = 0; output_idx < split->get_output_size(); ++output_idx) {
        results.push_back(std::make_shared<ov::op::v0::Result>(split->output(output_idx)));
    }
    auto model = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{data},
        "static_f16_split30_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    std::vector<float> values(kOutputCount * 2);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i);
    }
    request.set_input_tensor(0, make_f16_tensor({1, kOutputCount, 2}, values));

    request.infer();

    for (size_t output_idx = 0; output_idx < kOutputCount; ++output_idx) {
        const float base = static_cast<float>(output_idx * 2);
        expect_f16_tensor(request.get_output_tensor(output_idx),
                          {1, 1, 2},
                          {base, base + 1.0f});
    }
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF16SplitFiveOutputs) {
    ov::Core core;
    register_gfx_plugin(core);

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 10, 1});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto split = std::make_shared<ov::op::v1::Split>(data, axis, 5);
    split->set_friendly_name("static_f16_split5");
    ov::ResultVector results;
    for (size_t output_idx = 0; output_idx < split->get_output_size(); ++output_idx) {
        results.push_back(std::make_shared<ov::op::v0::Result>(split->output(output_idx)));
    }
    auto model = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{data},
        "static_f16_split5_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, make_f16_tensor({1, 10, 1},
                                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));

    request.infer();

    for (size_t output_idx = 0; output_idx < 5; ++output_idx) {
        const float base = static_cast<float>(output_idx * 2);
        expect_f16_tensor(request.get_output_tensor(output_idx),
                          {1, 2, 1},
                          {base, base + 1.0f});
    }
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF32VariadicSplit) {
    ov::Core core;
    register_gfx_plugin(core);

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 7, 2});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto lengths = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {2, 3, 2});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(data, axis, lengths);
    split->set_friendly_name("static_f32_variadic_split");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(split->output(0)),
                         std::make_shared<ov::op::v0::Result>(split->output(1)),
                         std::make_shared<ov::op::v0::Result>(split->output(2))},
        ov::ParameterVector{data},
        "static_f32_variadic_split_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, make_f32_tensor({1, 7, 2},
                                                {0, 1, 2, 3, 4, 5, 6,
                                                 7, 8, 9, 10, 11, 12, 13}));

    request.infer();

    expect_f32_tensor(request.get_output_tensor(0), {1, 2, 2}, {0, 1, 2, 3});
    expect_f32_tensor(request.get_output_tensor(1), {1, 3, 2}, {4, 5, 6, 7, 8, 9});
    expect_f32_tensor(request.get_output_tensor(2), {1, 2, 2}, {10, 11, 12, 13});
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF16VariadicSplit) {
    ov::Core core;
    register_gfx_plugin(core);

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 7, 2});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto lengths = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {2, 3, 2});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(data, axis, lengths);
    split->set_friendly_name("static_f16_variadic_split");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(split->output(0)),
                         std::make_shared<ov::op::v0::Result>(split->output(1)),
                         std::make_shared<ov::op::v0::Result>(split->output(2))},
        ov::ParameterVector{data},
        "static_f16_variadic_split_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, make_f16_tensor({1, 7, 2},
                                                {0, 1, 2, 3, 4, 5, 6,
                                                 7, 8, 9, 10, 11, 12, 13}));

    request.infer();

    expect_f16_tensor(request.get_output_tensor(0), {1, 2, 2}, {0, 1, 2, 3});
    expect_f16_tensor(request.get_output_tensor(1), {1, 3, 2}, {4, 5, 6, 7, 8, 9});
    expect_f16_tensor(request.get_output_tensor(2), {1, 2, 2}, {10, 11, 12, 13});
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF32VariadicSplitThirtyOutputs) {
    ov::Core core;
    register_gfx_plugin(core);

    constexpr size_t kOutputCount = 30;
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, kOutputCount, 2});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    std::vector<int64_t> lengths(kOutputCount, 1);
    auto lengths_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{kOutputCount}, lengths);
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(data, axis, lengths_const);
    split->set_friendly_name("static_f32_variadic_split30");
    ov::ResultVector results;
    for (size_t output_idx = 0; output_idx < split->get_output_size(); ++output_idx) {
        results.push_back(std::make_shared<ov::op::v0::Result>(split->output(output_idx)));
    }
    auto model = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{data},
        "static_f32_variadic_split30_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    std::vector<float> values(kOutputCount * 2);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i);
    }
    request.set_input_tensor(0, make_f32_tensor({1, kOutputCount, 2}, values));

    request.infer();

    for (size_t output_idx = 0; output_idx < kOutputCount; ++output_idx) {
        const float base = static_cast<float>(output_idx * 2);
        expect_f32_tensor(request.get_output_tensor(output_idx),
                          {1, 1, 2},
                          {base, base + 1.0f});
    }
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF16VariadicSplitThirtyOutputs) {
    ov::Core core;
    register_gfx_plugin(core);

    constexpr size_t kOutputCount = 30;
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, kOutputCount, 2});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    std::vector<int64_t> lengths(kOutputCount, 1);
    auto lengths_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{kOutputCount}, lengths);
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(data, axis, lengths_const);
    split->set_friendly_name("static_f16_variadic_split30");
    ov::ResultVector results;
    for (size_t output_idx = 0; output_idx < split->get_output_size(); ++output_idx) {
        results.push_back(std::make_shared<ov::op::v0::Result>(split->output(output_idx)));
    }
    auto model = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{data},
        "static_f16_variadic_split30_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    std::vector<float> values(kOutputCount * 2);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i);
    }
    request.set_input_tensor(0, make_f16_tensor({1, kOutputCount, 2}, values));

    request.infer();

    for (size_t output_idx = 0; output_idx < kOutputCount; ++output_idx) {
        const float base = static_cast<float>(output_idx * 2);
        expect_f16_tensor(request.get_output_tensor(output_idx),
                          {1, 1, 2},
                          {base, base + 1.0f});
    }
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF16VariadicSplitFiveOutputs) {
    ov::Core core;
    register_gfx_plugin(core);

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 15, 1});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto lengths = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {1, 2, 3, 4, 5});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(data, axis, lengths);
    split->set_friendly_name("static_f16_variadic_split5");
    ov::ResultVector results;
    for (size_t output_idx = 0; output_idx < split->get_output_size(); ++output_idx) {
        results.push_back(std::make_shared<ov::op::v0::Result>(split->output(output_idx)));
    }
    auto model = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{data},
        "static_f16_variadic_split5_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, make_f16_tensor({1, 15, 1},
                                                {0, 1, 2, 3, 4, 5, 6, 7,
                                                 8, 9, 10, 11, 12, 13, 14}));

    request.infer();

    expect_f16_tensor(request.get_output_tensor(0), {1, 1, 1}, {0});
    expect_f16_tensor(request.get_output_tensor(1), {1, 2, 1}, {1, 2});
    expect_f16_tensor(request.get_output_tensor(2), {1, 3, 1}, {3, 4, 5});
    expect_f16_tensor(request.get_output_tensor(3), {1, 4, 1}, {6, 7, 8, 9});
    expect_f16_tensor(request.get_output_tensor(4), {1, 5, 1}, {10, 11, 12, 13, 14});
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF32GatherI32) {
    ov::Core core;
    register_gfx_plugin(core);

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3});
    auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto gather = std::make_shared<ov::op::v8::Gather>(data, indices, axis);
    gather->set_friendly_name("static_f32_gather_i32");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(gather)},
        ov::ParameterVector{data, indices},
        "static_f32_gather_i32_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, make_f32_tensor({2, 3}, {0, 1, 2, 10, 11, 12}));
    request.set_input_tensor(1, make_i32_tensor({2}, {2, 0}));

    request.infer();

    expect_f32_tensor(request.get_output_tensor(0), {2, 2}, {2, 0, 12, 10});
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF32GatherElementsI32) {
    ov::Core core;
    register_gfx_plugin(core);

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3});
    auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2, 2});
    auto gather = std::make_shared<ov::op::v6::GatherElements>(data, indices, 1);
    gather->set_friendly_name("static_f32_gather_elements_i32");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(gather)},
        ov::ParameterVector{data, indices},
        "static_f32_gather_elements_i32_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, make_f32_tensor({2, 3}, {0, 1, 2, 10, 11, 12}));
    request.set_input_tensor(1, make_i32_tensor({2, 2}, {2, 0, 1, 2}));

    request.infer();

    expect_f32_tensor(request.get_output_tensor(0), {2, 2}, {2, 0, 11, 12});
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF32GatherNDI32) {
    ov::Core core;
    register_gfx_plugin(core);

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2, 2});
    auto gather = std::make_shared<ov::op::v8::GatherND>(data, indices);
    gather->set_friendly_name("static_f32_gather_nd_i32");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(gather)},
        ov::ParameterVector{data, indices},
        "static_f32_gather_nd_i32_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, make_f32_tensor({2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7}));
    request.set_input_tensor(1, make_i32_tensor({2, 2}, {0, 1, 1, 0}));

    request.infer();

    expect_f32_tensor(request.get_output_tensor(0), {2, 2}, {2, 3, 4, 5});
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF32ScatterUpdateI32) {
    ov::Core core;
    register_gfx_plugin(core);

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3});
    auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
    auto updates = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto scatter = std::make_shared<ov::op::v3::ScatterUpdate>(data, indices, updates, axis);
    scatter->set_friendly_name("static_f32_scatter_update_i32");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(scatter)},
        ov::ParameterVector{data, indices, updates},
        "static_f32_scatter_update_i32_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, make_f32_tensor({2, 3}, {0, 1, 2, 10, 11, 12}));
    request.set_input_tensor(1, make_i32_tensor({2}, {2, 0}));
    request.set_input_tensor(2, make_f32_tensor({2, 2}, {20, 21, 30, 31}));

    request.infer();

    expect_f32_tensor(request.get_output_tensor(0), {2, 3}, {21, 1, 20, 31, 11, 30});
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF32ScatterElementsI32) {
    ov::Core core;
    register_gfx_plugin(core);

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3});
    auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2, 2});
    auto updates = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto scatter = std::make_shared<ov::op::v3::ScatterElementsUpdate>(data, indices, updates, axis);
    scatter->set_friendly_name("static_f32_scatter_elements_i32");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(scatter)},
        ov::ParameterVector{data, indices, updates},
        "static_f32_scatter_elements_i32_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, make_f32_tensor({2, 3}, {0, 1, 2, 10, 11, 12}));
    request.set_input_tensor(1, make_i32_tensor({2, 2}, {2, 0, 1, 2}));
    request.set_input_tensor(2, make_f32_tensor({2, 2}, {20, 21, 30, 31}));

    request.infer();

    expect_f32_tensor(request.get_output_tensor(0), {2, 3}, {21, 1, 20, 10, 30, 31});
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF32ScatterNDI32) {
    ov::Core core;
    register_gfx_plugin(core);

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2, 2});
    auto updates = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    auto scatter = std::make_shared<ov::op::v3::ScatterNDUpdate>(data, indices, updates);
    scatter->set_friendly_name("static_f32_scatter_nd_i32");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(scatter)},
        ov::ParameterVector{data, indices, updates},
        "static_f32_scatter_nd_i32_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, make_f32_tensor({2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7}));
    request.set_input_tensor(1, make_i32_tensor({2, 2}, {0, 1, 1, 0}));
    request.set_input_tensor(2, make_f32_tensor({2, 2}, {20, 21, 30, 31}));

    request.infer();

    expect_f32_tensor(request.get_output_tensor(0), {2, 2, 2}, {0, 1, 20, 21, 30, 31, 6, 7});
}

TEST(GfxDataMovementSupport, CompileAndInferStaticF16Range) {
    ov::Core core;
    register_gfx_plugin(core);

    auto start = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {1.0f});
    auto stop = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {4.0f});
    auto step = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {0.5f});
    auto range = std::make_shared<ov::op::v4::Range>(start, stop, step, ov::element::f16);
    range->set_friendly_name("static_f16_range");
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(range)},
        ov::ParameterVector{},
        "static_f16_range_infer");

    auto compiled = core.compile_model(model, "GFX");
    auto request = compiled.create_infer_request();

    request.infer();

    expect_f16_tensor(request.get_output_tensor(0),
                      {6},
                      {1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f});
}
