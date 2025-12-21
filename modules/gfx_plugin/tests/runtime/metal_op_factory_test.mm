// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "runtime/metal_op_factory.hpp"
#include "runtime/metal_op_activations.hpp"

using namespace ov::gfx_plugin;

TEST(MetalOpFactory, CreatesStubForRelu) {
    auto p = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto relu = std::make_shared<ov::op::v0::Relu>(p);

    auto op = MetalOpFactory::create(relu);

    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->type(), std::string("Activation"));
    EXPECT_NE(dynamic_cast<MetalActivationOp*>(op.get()), nullptr);
    EXPECT_EQ(op->name(), relu->get_friendly_name());
}

TEST(MetalOpFactory, ReturnsNullForUnsupportedParameter) {
    auto p = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});

    auto op = MetalOpFactory::create(p);

    EXPECT_EQ(op, nullptr);
}
