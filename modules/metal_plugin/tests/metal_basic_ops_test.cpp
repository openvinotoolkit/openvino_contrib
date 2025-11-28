// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sigmoid.hpp"

namespace {

void register_metal_plugin(ov::Core& core) {
#ifdef METAL_PLUGIN_PATH
    try {
        core.register_plugin(METAL_PLUGIN_PATH, "METAL");
    } catch (const std::exception& e) {
        GTEST_SKIP() << "METAL plugin unavailable: " << e.what();
        return;
    }
#else
    GTEST_SKIP() << "METAL_PLUGIN_PATH is not defined; METAL plugin is not built";
    return;
#endif
}

void expect_allclose(const ov::Tensor& a, const ov::Tensor& b, float tol = 1e-5f) {
    ASSERT_EQ(a.get_element_type(), ov::element::f32);
    ASSERT_EQ(b.get_element_type(), ov::element::f32);
    ASSERT_EQ(a.get_byte_size(), b.get_byte_size());
    auto* pa = a.data<const float>();
    auto* pb = b.data<const float>();
    size_t count = a.get_size();
    for (size_t i = 0; i < count; ++i) {
        float diff = std::abs(pa[i] - pb[i]);
        ASSERT_LE(diff, tol) << "Mismatch at index " << i << ": " << pa[i] << " vs " << pb[i];
    }
}

}  // namespace

TEST(MetalBasicOps, Add) {
    ov::Core core;
    register_metal_plugin(core);

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto c = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 4}, std::vector<float>{1.f, -2.f, 3.f, 4.f});
    auto add = std::make_shared<ov::op::v1::Add>(param, c);
    auto res = std::make_shared<ov::op::v0::Result>(add);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "add_model");

    auto cpu_cm = core.compile_model(model, "CPU");
    auto metal_cm = core.compile_model(model, "METAL");

    ov::Tensor input{ov::element::f32, {1, 4}};
    float* data = input.data<float>();
    std::vector<float> vals{0.5f, 2.f, -1.f, 0.f};
    std::copy(vals.begin(), vals.end(), data);

    auto cpu_req = cpu_cm.create_infer_request();
    cpu_req.set_input_tensor(input);
    cpu_req.infer();
    auto cpu_out = cpu_req.get_output_tensor();

    auto metal_req = metal_cm.create_infer_request();
    metal_req.set_input_tensor(input);
    metal_req.infer();
    auto metal_out = metal_req.get_output_tensor();

    expect_allclose(cpu_out, metal_out, /*tol=*/1e-5f);
}

TEST(MetalBasicOps, Relu) {
    ov::Core core;
    register_metal_plugin(core);

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    auto res = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "relu_model");

    auto cpu_cm = core.compile_model(model, "CPU");
    auto metal_cm = core.compile_model(model, "METAL");

    ov::Tensor input{ov::element::f32, {1, 4}};
    std::vector<float> vals{-1.f, -0.5f, 0.25f, 2.f};
    std::copy(vals.begin(), vals.end(), input.data<float>());

    auto cpu_req = cpu_cm.create_infer_request();
    cpu_req.set_input_tensor(input);
    cpu_req.infer();
    auto cpu_out = cpu_req.get_output_tensor();

    auto metal_req = metal_cm.create_infer_request();
    metal_req.set_input_tensor(input);
    metal_req.infer();
    auto metal_out = metal_req.get_output_tensor();

    expect_allclose(cpu_out, metal_out, /*tol=*/1e-5f);
}

TEST(MetalBasicOps, MatMul2D) {
    ov::Core core;
    register_metal_plugin(core);

    const size_t K = 3;
    const size_t N = 2;
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, K});
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{K, N},
                                                          std::vector<float>{
                                                              1.f, 2.f,
                                                              3.f, 4.f,
                                                              5.f, 6.f});
    auto mm = std::make_shared<ov::op::v0::MatMul>(param, weights, false, false);
    auto res = std::make_shared<ov::op::v0::Result>(mm);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "matmul_model");

    auto cpu_cm = core.compile_model(model, "CPU");
    auto metal_cm = core.compile_model(model, "METAL");

    ov::Tensor input{ov::element::f32, {1, K}};
    std::vector<float> vals{1.f, -1.f, 0.5f};
    std::copy(vals.begin(), vals.end(), input.data<float>());

    auto cpu_req = cpu_cm.create_infer_request();
    cpu_req.set_input_tensor(input);
    cpu_req.infer();
    auto cpu_out = cpu_req.get_output_tensor();

    auto metal_req = metal_cm.create_infer_request();
    metal_req.set_input_tensor(input);
    metal_req.infer();
    auto metal_out = metal_req.get_output_tensor();

    expect_allclose(cpu_out, metal_out, /*tol=*/1e-5f);
}

TEST(MetalBasicOps, DevicePropertiesAndQuery) {
    ov::Core core;
    register_metal_plugin(core);

    auto full_name = core.get_property("METAL", ov::device::full_name);
    EXPECT_FALSE(full_name.empty());

    auto dtype = core.get_property<ov::device::Type>("METAL", ov::device::type);
    EXPECT_EQ(dtype, ov::device::Type::INTEGRATED);

    // Simple model to check query_model
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    auto res = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "qmodel");

    auto supported = core.query_model(model, "METAL");
    // Expect both Parameter and Relu/Result to be marked; at least Relu by friendly name
    EXPECT_FALSE(supported.empty());
    EXPECT_NE(supported.find(relu->get_friendly_name()), supported.end());
}

TEST(MetalBasicOps, Conv2D) {
    ov::Core core;
    register_metal_plugin(core);

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 4, 4});  // NCHW
    auto weights = std::make_shared<ov::op::v0::Constant>(
        ov::element::f32, ov::Shape{2, 3, 3, 3},
        std::vector<float>{
            // out 0 kernel (3x3x3)
            1.f, 0.f, -1.f, 0.f, 1.f, 0.f, -1.f, 0.f, 1.f,  // ch0
            1.f, 0.f, -1.f, 0.f, 1.f, 0.f, -1.f, 0.f, 1.f,  // ch1
            1.f, 0.f, -1.f, 0.f, 1.f, 0.f, -1.f, 0.f, 1.f,  // ch2
            // out 1 kernel
            -1.f, -1.f, -1.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f,  // ch0
            -1.f, -1.f, -1.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f,  // ch1
            -1.f, -1.f, -1.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f   // ch2
        });

    ov::Strides strides{1, 1};
    ov::CoordinateDiff pads_begin{1, 1};  // spatial pads
    ov::CoordinateDiff pads_end{1, 1};
    ov::Strides dilations{1, 1};

    auto conv = std::make_shared<ov::op::v1::Convolution>(param,
                                                          weights,
                                                          strides,
                                                          pads_begin,
                                                          pads_end,
                                                          dilations);
    auto res = std::make_shared<ov::op::v0::Result>(conv);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "conv2d_model");

    auto cpu_cm = core.compile_model(model, "CPU");
    auto metal_cm = core.compile_model(model, "METAL");

    ov::Tensor input{ov::element::f32, {1, 3, 4, 4}};
    std::vector<float> vals(1 * 3 * 4 * 4);
    for (size_t i = 0; i < vals.size(); ++i)
        vals[i] = static_cast<float>(i % 7) - 3.f;  // some repeating pattern
    std::copy(vals.begin(), vals.end(), input.data<float>());

    auto cpu_req = cpu_cm.create_infer_request();
    cpu_req.set_input_tensor(input);
    cpu_req.infer();
    auto cpu_out = cpu_req.get_output_tensor();

    auto metal_req = metal_cm.create_infer_request();
    metal_req.set_input_tensor(input);
    metal_req.infer();
    auto metal_out = metal_req.get_output_tensor();

    expect_allclose(cpu_out, metal_out, /*tol=*/1e-4f);
}

TEST(MetalBasicOps, MaxPool2D) {
    ov::Core core;
    register_metal_plugin(core);

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 4, 4});  // NCHW
    ov::Strides strides{2, 2};
    ov::Shape kernel{2, 2};
    ov::Shape pads_begin{0, 0};
    ov::Shape pads_end{0, 0};

    auto pool = std::make_shared<ov::op::v1::MaxPool>(param,
                                                      strides,
                                                      pads_begin,
                                                      pads_end,
                                                      kernel,
                                                      ov::op::RoundingType::FLOOR);
    auto res = std::make_shared<ov::op::v0::Result>(pool);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "maxpool2d");

    auto cpu_cm = core.compile_model(model, "CPU");
    auto metal_cm = core.compile_model(model, "METAL");

    ov::Tensor input{ov::element::f32, {1, 1, 4, 4}};
    for (size_t i = 0; i < 16; ++i) {
        input.data<float>()[i] = static_cast<float>(i);
    }

    auto cpu_req = cpu_cm.create_infer_request();
    cpu_req.set_input_tensor(input);
    cpu_req.infer();
    auto cpu_out = cpu_req.get_output_tensor();

    auto metal_req = metal_cm.create_infer_request();
    metal_req.set_input_tensor(input);
    metal_req.infer();
    auto metal_out = metal_req.get_output_tensor();

    expect_allclose(cpu_out, metal_out, /*tol=*/1e-5f);
}

TEST(MetalBasicOps, AvgPool2D) {
    ov::Core core;
    register_metal_plugin(core);

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 4, 4});  // NCHW
    ov::Strides strides{2, 2};
    ov::Shape kernel{2, 2};
    ov::Shape pads_begin{0, 0};
    ov::Shape pads_end{0, 0};

    auto pool = std::make_shared<ov::op::v1::AvgPool>(param,
                                                      strides,
                                                      pads_begin,
                                                      pads_end,
                                                      kernel,
                                                      true,   // exclude_pad
                                                      ov::op::RoundingType::FLOOR);
    auto res = std::make_shared<ov::op::v0::Result>(pool);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "avgpool2d");

    auto cpu_cm = core.compile_model(model, "CPU");
    auto metal_cm = core.compile_model(model, "METAL");

    ov::Tensor input{ov::element::f32, {1, 1, 4, 4}};
    for (size_t i = 0; i < 16; ++i) {
        input.data<float>()[i] = static_cast<float>(i);
    }

    auto cpu_req = cpu_cm.create_infer_request();
    cpu_req.set_input_tensor(input);
    cpu_req.infer();
    auto cpu_out = cpu_req.get_output_tensor();

    auto metal_req = metal_cm.create_infer_request();
    metal_req.set_input_tensor(input);
    metal_req.infer();
    auto metal_out = metal_req.get_output_tensor();

    expect_allclose(cpu_out, metal_out, /*tol=*/1e-5f);
}

// AUTO should split: Relu/Add on METAL, Sigmoid (unsupported) on CPU; final result must match pure CPU.
TEST(MetalBasicOps, AutoMetalCpuHybrid) {
    ov::Core core;
    register_metal_plugin(core);

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto relu = std::make_shared<ov::op::v0::Relu>(param);  // supported by METAL
    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(relu);  // NOT supported by METAL
    auto c = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 4},
                                                   std::vector<float>{0.1f, -0.2f, 0.3f, -0.4f});
    auto add = std::make_shared<ov::op::v1::Add>(sigmoid, c);  // supported by METAL
    auto res = std::make_shared<ov::op::v0::Result>(add);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "auto_metal_cpu");

    auto cpu_cm = core.compile_model(model, "CPU");
    auto auto_cm = core.compile_model(model, "AUTO:METAL,CPU");

    ov::Tensor input{ov::element::f32, {1, 4}};
    std::vector<float> vals{0.5f, -1.f, 2.f, -0.5f};
    std::copy(vals.begin(), vals.end(), input.data<float>());

    auto cpu_req = cpu_cm.create_infer_request();
    cpu_req.set_input_tensor(input);
    cpu_req.infer();
    auto cpu_out = cpu_req.get_output_tensor();

    auto auto_req = auto_cm.create_infer_request();
    auto_req.set_input_tensor(input);
    auto_req.infer();
    auto auto_out = auto_req.get_output_tensor();

    expect_allclose(cpu_out, auto_out, /*tol=*/1e-5f);
}
