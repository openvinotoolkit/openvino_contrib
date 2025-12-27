// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "runtime/gfx_backend_utils.hpp"
#include "runtime/gpu_stage_factory.hpp"

using namespace ov::gfx_plugin;

TEST(GpuStageFactory, CreatesStubForRelu) {
    auto p = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto relu = std::make_shared<ov::op::v0::Relu>(p);

    auto stage = GpuStageFactory::create(relu, default_backend_kind());

    ASSERT_NE(stage, nullptr);
    const std::string backend = kGfxDefaultBackend ? std::string(kGfxDefaultBackend) : std::string{};
    if (backend == "vulkan") {
        EXPECT_FALSE(stage->type().empty());
    } else {
        EXPECT_EQ(stage->type(), std::string("Activation"));
        EXPECT_EQ(stage->name(), relu->get_friendly_name());
    }
}

TEST(GpuStageFactory, ReturnsNullForUnsupportedParameter) {
    auto p = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});

    auto stage = GpuStageFactory::create(p, default_backend_kind());

    const std::string backend = kGfxDefaultBackend ? std::string(kGfxDefaultBackend) : std::string{};
    if (backend == "vulkan") {
        EXPECT_NE(stage, nullptr);
    } else {
        EXPECT_EQ(stage, nullptr);
    }
}
