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
#include "openvino/op/softmax.hpp"
#include "openvino/op/batch_norm.hpp"
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

void expect_finite(const ov::Tensor& t) {
    const float* p = t.data<const float>();
    for (size_t i = 0; i < t.get_size(); ++i) {
        ASSERT_TRUE(std::isfinite(p[i])) << "Non-finite at " << i << ": " << p[i];
    }
}

void expect_shape_type(const ov::Tensor& t, const ov::Shape& shape, ov::element::Type type = ov::element::f32) {
    ASSERT_EQ(t.get_element_type(), type);
    ASSERT_EQ(t.get_shape(), shape);
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

    std::vector<std::vector<float>> patterns{
        {0.5f, 2.f, -1.f, 0.f},
        {-10.f, 0.f, 10.f, 1e-3f},
        {1e3f, -1e3f, 5.f, -5.f}};

    auto cpu_req = cpu_cm.create_infer_request();
    auto metal_req = metal_cm.create_infer_request();
    for (const auto& vals : patterns) {
        ov::Tensor input{ov::element::f32, {1, 4}};
        std::copy(vals.begin(), vals.end(), input.data<float>());

        cpu_req.set_input_tensor(input);
        cpu_req.infer();
        auto cpu_out = cpu_req.get_output_tensor();

        metal_req.set_input_tensor(input);
        metal_req.infer();
        auto metal_out = metal_req.get_output_tensor();

        expect_shape_type(cpu_out, {1, 4});
        expect_shape_type(metal_out, {1, 4});
        expect_allclose(cpu_out, metal_out, /*tol=*/2e-3f);
    }

    // Negative: incompatible shapes should fail at graph validation
    auto bad_const = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{2, 3}, std::vector<float>(6, 1.f));
    EXPECT_THROW(std::make_shared<ov::op::v1::Add>(param, bad_const)->validate_and_infer_types(), ov::Exception);
}

TEST(MetalBasicOps, AddBroadcastScalar) {
    ov::Core core;
    register_metal_plugin(core);

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto c = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{0.5f});
    auto add = std::make_shared<ov::op::v1::Add>(param, c);
    auto res = std::make_shared<ov::op::v0::Result>(add);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "add_scalar");

    auto cpu_cm = core.compile_model(model, "CPU");
    auto metal_cm = core.compile_model(model, "METAL");

    std::vector<std::vector<float>> inputs{
        {0.5f, 2.f, -1.f, 0.f},
        {-3.f, -2.f, -1.f, 0.f},
        {100.f, -100.f, 50.f, -50.f}};

    auto cpu_req = cpu_cm.create_infer_request();
    auto metal_req = metal_cm.create_infer_request();

    for (const auto& vals : inputs) {
        ov::Tensor input{ov::element::f32, {1, 4}};
        std::copy(vals.begin(), vals.end(), input.data<float>());

        cpu_req.set_input_tensor(input);
        cpu_req.infer();
        auto cpu_out = cpu_req.get_output_tensor();

        metal_req.set_input_tensor(input);
        metal_req.infer();
        auto metal_out = metal_req.get_output_tensor();

        expect_shape_type(cpu_out, {1, 4});
        expect_shape_type(metal_out, {1, 4});
        expect_allclose(cpu_out, metal_out, /*tol=*/2e-3f);
    }
}

TEST(MetalBasicOps, AddBroadcastChannel) {
    ov::Core core;
    register_metal_plugin(core);

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 4, 4});
    auto c = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 3, 1, 1},
                                                    std::vector<float>{0.1f, -0.2f, 0.3f});
    auto add = std::make_shared<ov::op::v1::Add>(param, c);
    auto res = std::make_shared<ov::op::v0::Result>(add);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "add_channel");

    auto cpu_cm = core.compile_model(model, "CPU");
    auto metal_cm = core.compile_model(model, "METAL");

    std::vector<std::vector<float>> patterns;
    patterns.emplace_back(1 * 3 * 4 * 4);
    for (size_t i = 0; i < patterns.back().size(); ++i)
        patterns.back()[i] = static_cast<float>(i % 7) - 2.f;
    patterns.emplace_back(1 * 3 * 4 * 4, 0.f);  // zeros
    patterns.emplace_back(1 * 3 * 4 * 4, 1.f);  // ones

    auto cpu_req = cpu_cm.create_infer_request();
    auto metal_req = metal_cm.create_infer_request();

    for (const auto& vals : patterns) {
        ov::Tensor input{ov::element::f32, {1, 3, 4, 4}};
        std::copy(vals.begin(), vals.end(), input.data<float>());

        cpu_req.set_input_tensor(input);
        cpu_req.infer();
        auto cpu_out = cpu_req.get_output_tensor();

        metal_req.set_input_tensor(input);
        metal_req.infer();
        auto metal_out = metal_req.get_output_tensor();

        expect_shape_type(cpu_out, {1, 3, 4, 4});
        expect_shape_type(metal_out, {1, 3, 4, 4});
        expect_allclose(cpu_out, metal_out, /*tol=*/2e-3f);
    }
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

    std::vector<std::vector<float>> inputs{
        {-1.f, -0.5f, 0.25f, 2.f},
        {0.f, 0.f, 0.f, 0.f},
        {10.f, -10.f, 1e-3f, -1e-3f}};

    auto cpu_req = cpu_cm.create_infer_request();
    auto metal_req = metal_cm.create_infer_request();

    for (const auto& vals : inputs) {
        ov::Tensor input{ov::element::f32, {1, 4}};
        std::copy(vals.begin(), vals.end(), input.data<float>());

        cpu_req.set_input_tensor(input);
        cpu_req.infer();
        auto cpu_out = cpu_req.get_output_tensor();

        metal_req.set_input_tensor(input);
        metal_req.infer();
        auto metal_out = metal_req.get_output_tensor();

        expect_allclose(cpu_out, metal_out, /*tol=*/2e-4f);

        // ReLU invariant: outputs are non-negative
        const float* p = cpu_out.data<const float>();
        for (size_t i = 0; i < cpu_out.get_size(); ++i) {
            EXPECT_GE(p[i], 0.f);
        }
    }
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

    std::vector<std::vector<float>> inputs{
        {1.f, -1.f, 0.5f},
        {0.f, 0.f, 0.f},
        {10.f, 10.f, 10.f}};

    auto cpu_req = cpu_cm.create_infer_request();
    auto metal_req = metal_cm.create_infer_request();

    for (const auto& vals : inputs) {
        ov::Tensor input{ov::element::f32, {1, K}};
        std::copy(vals.begin(), vals.end(), input.data<float>());

        cpu_req.set_input_tensor(input);
        cpu_req.infer();
        auto cpu_out = cpu_req.get_output_tensor();

        metal_req.set_input_tensor(input);
        metal_req.infer();
        auto metal_out = metal_req.get_output_tensor();

        expect_shape_type(cpu_out, {1, N});
        expect_allclose(cpu_out, metal_out, /*tol=*/2e-4f);
    }
}

TEST(MetalBasicOps, MatMulBatchBroadcastLeft) {
    ov::Core core;
    register_metal_plugin(core);

    const size_t B = 2;
    const size_t M = 2;
    const size_t K = 3;
    const size_t N = 2;

    auto a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, M, K});
    auto b = std::make_shared<ov::op::v0::Constant>(
        ov::element::f32, ov::Shape{B, K, N},
        std::vector<float>{
            1.f, 2.f, 3.f, 4.f, 6.f, 7.f,    // batch0 KxN
            -1.f, 0.5f, 2.f, -2.f, 1.f, 3.f  // batch1
        });
    auto mm = std::make_shared<ov::op::v0::MatMul>(a, b, false, false);
    auto res = std::make_shared<ov::op::v0::Result>(mm);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{a}, "matmul_broadcast_left");

    auto cpu_cm = core.compile_model(model, "CPU");
    auto metal_cm = core.compile_model(model, "METAL");

    std::vector<std::vector<float>> inputs{
        {1.f, -1.f, 0.5f, 2.f, 0.f, -0.5f},
        {0.f, 0.f, 0.f, 0.f, 0.f, 0.f}};

    auto cpu_req = cpu_cm.create_infer_request();
    auto metal_req = metal_cm.create_infer_request();

    for (const auto& vals : inputs) {
        ov::Tensor input{ov::element::f32, {1, M, K}};
        std::copy(vals.begin(), vals.end(), input.data<float>());

        cpu_req.set_input_tensor(input);
        cpu_req.infer();
        auto cpu_out = cpu_req.get_output_tensor();

        metal_req.set_input_tensor(input);
        metal_req.infer();
        auto metal_out = metal_req.get_output_tensor();

        expect_shape_type(cpu_out, {B, M, N});
        expect_allclose(cpu_out, metal_out, /*tol=*/2e-4f);
    }
}

TEST(MetalBasicOps, MatMulBatchBroadcastRight) {
    ov::Core core;
    register_metal_plugin(core);

    const size_t B = 2;
    const size_t M = 2;
    const size_t K = 3;
    const size_t N = 2;

    auto a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{B, M, K});
    auto b = std::make_shared<ov::op::v0::Constant>(
        ov::element::f32, ov::Shape{1, K, N},
        std::vector<float>{
            1.f, 2.f,
            3.f, 4.f,
            5.f, 6.f});
    auto mm = std::make_shared<ov::op::v0::MatMul>(a, b, false, false);
    auto res = std::make_shared<ov::op::v0::Result>(mm);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{a}, "matmul_broadcast_right");

    auto cpu_cm = core.compile_model(model, "CPU");
    auto metal_cm = core.compile_model(model, "METAL");

    std::vector<std::vector<float>> inputs{
        {1.f, -1.f, 0.5f, 2.f, 0.f, -0.5f,  -2.f, 1.f, 0.25f, 1.5f, -1.f, 0.1f},
        {0.f, 0.f, 0.f, 0.f, 0.f, 0.f,      1.f, 2.f, 3.f, 4.f, 5.f, 6.f}};

    auto cpu_req = cpu_cm.create_infer_request();
    auto metal_req = metal_cm.create_infer_request();

    for (const auto& vals : inputs) {
        ov::Tensor input{ov::element::f32, {B, M, K}};
        std::copy(vals.begin(), vals.end(), input.data<float>());

        cpu_req.set_input_tensor(input);
        cpu_req.infer();
        auto cpu_out = cpu_req.get_output_tensor();

        metal_req.set_input_tensor(input);
        metal_req.infer();
        auto metal_out = metal_req.get_output_tensor();

        expect_shape_type(cpu_out, {B, M, N});
        expect_allclose(cpu_out, metal_out, /*tol=*/2e-4f);
    }
}

TEST(MetalBasicOps, Softmax) {
    ov::Core core;
    register_metal_plugin(core);

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    auto sm = std::make_shared<ov::op::v1::Softmax>(param, 1);
    auto res = std::make_shared<ov::op::v0::Result>(sm);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "softmax");

    auto cpu_cm = core.compile_model(model, "CPU");
    auto metal_cm = core.compile_model(model, "METAL");

    std::vector<std::vector<float>> inputs{
        {0.0f, 1.0f, 2.0f, 3.0f,
         -1.0f, -2.0f, -3.0f, -4.0f},
        {10.f, 10.f, 10.f, 10.f,
         -10.f, -9.f, -8.f, -7.f},
        {0.f, 0.f, 0.f, 0.f,
         5.f, 4.f, 3.f, 2.f}};

    auto cpu_req = cpu_cm.create_infer_request();
    auto metal_req = metal_cm.create_infer_request();

    for (const auto& vals : inputs) {
        ov::Tensor input{ov::element::f32, {2, 4}};
        std::copy(vals.begin(), vals.end(), input.data<float>());

        cpu_req.set_input_tensor(input);
        cpu_req.infer();
        auto cpu_out = cpu_req.get_output_tensor();

        metal_req.set_input_tensor(input);
        metal_req.infer();
        auto metal_out = metal_req.get_output_tensor();

        expect_allclose(cpu_out, metal_out, /*tol=*/2e-4f);

        // Softmax invariant: sum along axis == 1
        auto check_sum_one = [](const ov::Tensor& t) {
            const float* p = t.data<const float>();
            for (size_t row = 0; row < 2; ++row) {
                float sum = 0.f;
                for (size_t c = 0; c < 4; ++c)
                    sum += p[row * 4 + c];
                EXPECT_NEAR(sum, 1.f, 1e-3f);
            }
        };
        expect_finite(cpu_out);
        expect_finite(metal_out);
        check_sum_one(cpu_out);
        check_sum_one(metal_out);
    }
}

TEST(MetalBasicOps, BatchNormInference) {
    ov::Core core;
    register_metal_plugin(core);

    const ov::Shape shape{1, 3, 4, 4};  // NCHW
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);

    std::vector<float> gamma_vals{1.f, 0.5f, 2.f};
    std::vector<float> beta_vals{0.f, 1.f, -1.f};
    std::vector<float> mean_vals{0.1f, -0.2f, 0.3f};
    std::vector<float> var_vals{1.0f, 0.5f, 2.0f};

    auto gamma = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{3}, gamma_vals);
    auto beta = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{3}, beta_vals);
    auto mean = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{3}, mean_vals);
    auto var = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{3}, var_vals);

    double eps = 1e-5;
    auto bn = std::make_shared<ov::op::v5::BatchNormInference>(param, gamma, beta, mean, var, eps);
    auto res = std::make_shared<ov::op::v0::Result>(bn);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "batchnorm");

    auto cpu_cm = core.compile_model(model, "CPU");
    auto metal_cm = core.compile_model(model, "METAL");

    ov::Tensor input_cpu{ov::element::f32, shape};
    ov::Tensor input_metal{ov::element::f32, shape};
    for (size_t i = 0; i < input_cpu.get_size(); ++i) {
        int v_int = static_cast<int>(i % 11) - 5;
        float v = static_cast<float>(v_int);
        input_cpu.data<float>()[i] = v;
        input_metal.data<float>()[i] = v;
    }

    // Reference on host (use the pristine CPU input)
    ov::Tensor ref{ov::element::f32, shape};
    const float* x_data = input_cpu.data<const float>();
    float* ref_data = ref.data<float>();
    const size_t C = 3;
    const size_t HW = shape[2] * shape[3];
    float eps_f = static_cast<float>(eps);
    for (size_t n = 0; n < shape[0]; ++n) {
        for (size_t c = 0; c < C; ++c) {
            float g = gamma_vals[c];
            float b = beta_vals[c];
            float m = mean_vals[c];
            float v = var_vals[c];
            float denom = std::sqrt(v + eps_f);
            for (size_t hw = 0; hw < HW; ++hw) {
                size_t idx = n * C * HW + c * HW + hw;
                ref_data[idx] = g * (x_data[idx] - m) / denom + b;
            }
        }
    }

    auto cpu_req = cpu_cm.create_infer_request();
    cpu_req.set_input_tensor(input_cpu);
    cpu_req.infer();
    auto cpu_out = cpu_req.get_output_tensor();

    auto metal_req = metal_cm.create_infer_request();
    metal_req.set_input_tensor(input_metal);
    metal_req.infer();
    auto metal_out = metal_req.get_output_tensor();

    auto max_abs = [](const ov::Tensor& t, bool& finite) -> float {
        finite = true;
        const float* ptr = t.data<const float>();
        size_t n = t.get_size();
        float m = 0.f;
        for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(ptr[i])) {
                finite = false;
            }
            m = std::max(m, std::abs(ptr[i]));
        }
        return m;
    };

    bool cpu_finite = true, metal_finite = true, ref_finite = true;
    float cpu_max = max_abs(cpu_out, cpu_finite);
    float metal_max = max_abs(metal_out, metal_finite);
    float ref_max = max_abs(ref, ref_finite);
    ASSERT_TRUE(ref_finite);
    if (!cpu_finite || !metal_finite) {
        const float* rptr = ref.data<const float>();
        std::cerr << "REF first values: ";
        for (size_t i = 0; i < std::min<size_t>(ref.get_size(), 6); ++i) {
            std::cerr << rptr[i] << " ";
        }
        std::cerr << "\n";
        const float* cptr = cpu_out.data<const float>();
        const float* mptr = metal_out.data<const float>();
        std::cerr << "CPU first values: ";
        for (size_t i = 0; i < std::min<size_t>(cpu_out.get_size(), 6); ++i) {
            std::cerr << cptr[i] << " ";
        }
        std::cerr << "\nMETAL first values: ";
        for (size_t i = 0; i < std::min<size_t>(metal_out.get_size(), 6); ++i) {
            std::cerr << mptr[i] << " ";
        }
        std::cerr << std::endl;
    }
    ASSERT_TRUE(cpu_finite);
    ASSERT_TRUE(metal_finite);
    SCOPED_TRACE("max_abs cpu=" + std::to_string(cpu_max) + " metal=" + std::to_string(metal_max) +
                 " ref=" + std::to_string(ref_max));

    expect_allclose(ref, cpu_out, /*tol=*/5e-3f);
    expect_allclose(ref, metal_out, /*tol=*/1e-3f);
}

TEST(MetalBasicOps, DevicePropertiesAndQuery) {
    ov::Core core;
    register_metal_plugin(core);

    auto full_name = core.get_property("METAL", ov::device::full_name);
    EXPECT_FALSE(full_name.empty());
    EXPECT_NE(full_name.find("METAL"), std::string::npos);

    auto dtype = core.get_property<ov::device::Type>("METAL", ov::device::type);
    EXPECT_EQ(dtype, ov::device::Type::INTEGRATED);

    // Capabilities are optional; if provided, they should not be empty.
    try {
        auto caps = core.get_property("METAL", ov::device::capabilities);
        EXPECT_FALSE(caps.empty());
    } catch (const std::exception&) {
        // Ignore if capability property is not implemented yet.
    }

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

    expect_shape_type(cpu_out, {1, 2, 4, 4});  // same spatial because stride=1, pad=1
    expect_allclose(cpu_out, metal_out, /*tol=*/1e-4f);

    // Second pattern: delta input to sanity-check kernel wiring
    ov::Tensor delta{ov::element::f32, {1, 3, 4, 4}};
    std::fill(delta.data<float>(), delta.data<float>() + delta.get_size(), 0.f);
    delta.data<float>()[0] = 1.f;  // single pixel
    cpu_req.set_input_tensor(delta);
    cpu_req.infer();
    auto cpu_out2 = cpu_req.get_output_tensor();
    metal_req.set_input_tensor(delta);
    metal_req.infer();
    auto metal_out2 = metal_req.get_output_tensor();
    expect_allclose(cpu_out2, metal_out2, /*tol=*/1e-4f);
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

    expect_shape_type(cpu_out, {1, 1, 2, 2});
    // Known expected values for sequential input 0..15 with 2x2 stride2 maxpool
    std::vector<float> expected{5.f, 7.f, 13.f, 15.f};
    const float* p = cpu_out.data<const float>();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(p[i], expected[i], 1e-5f);
    }
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

    expect_shape_type(cpu_out, {1, 1, 2, 2});
    // Average of each 2x2 block of 0..15 sequence
    std::vector<float> expected{2.5f, 4.5f, 10.5f, 12.5f};
    const float* p = cpu_out.data<const float>();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(p[i], expected[i], 1e-5f);
    }
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

    std::vector<std::vector<float>> inputs{
        {0.5f, -1.f, 2.f, -0.5f},
        {0.f, 0.f, 0.f, 0.f}};

    auto cpu_req = cpu_cm.create_infer_request();
    auto auto_req = auto_cm.create_infer_request();
    for (const auto& vals : inputs) {
        ov::Tensor input{ov::element::f32, {1, 4}};
        std::copy(vals.begin(), vals.end(), input.data<float>());

        cpu_req.set_input_tensor(input);
        cpu_req.infer();
        auto cpu_out = cpu_req.get_output_tensor();

        auto_req.set_input_tensor(input);
        auto_req.infer();
        auto auto_out = auto_req.get_output_tensor();

        expect_allclose(cpu_out, auto_out, /*tol=*/1e-5f);
    }
}
