// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#    include <signal.h>
#    include <sys/wait.h>
#    include <unistd.h>
#endif

#include "common_test_utils/ov_plugin_cache.hpp"
#include "openvino/openvino.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"

namespace {

std::string gfx_skip_reason;

void gfx_try_catch_fail(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const ov::Exception& e) {
        const std::string msg = e.what();
        if (msg.find("device-only") != std::string::npos ||
            msg.find("output tensors are device-only") != std::string::npos) {
            FAIL() << "GFX outputs are device-only; host readback required for test";
        }
        if (msg.find("GFX Vulkan") != std::string::npos ||
            msg.find("SPIR-V") != std::string::npos ||
            msg.find("spirv") != std::string::npos ||
            msg.find("vulkan") != std::string::npos) {
            FAIL() << "Vulkan backend did not support this case: " << msg;
            return;
        }
        throw;
    }
}

bool register_gfx_plugin(ov::Core& core) {
    gfx_skip_reason.clear();
    try {
#ifdef GFX_PLUGIN_PATH
        const char* env_path = std::getenv("GFX_PLUGIN_PATH");
        const char* path = (env_path && *env_path) ? env_path : GFX_PLUGIN_PATH;
        core.register_plugin(path, "GFX");
#endif
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        if (msg.find("already registered") == std::string::npos) {
            throw std::runtime_error(std::string("GFX plugin unavailable: ") + e.what());
        }
    }
    try {
        const auto backend = core.get_property("GFX", "GFX_BACKEND").as<std::string>();
        if (backend.empty()) {
            gfx_skip_reason = "GFX backend not available";
            return false;
        }
    } catch (const std::exception& e) {
        gfx_skip_reason = std::string("GFX backend property unavailable: ") + e.what();
        return false;
    }
    try {
        ov::test::utils::register_template_plugin(core);
    } catch (const std::exception& e) {
        gfx_skip_reason = std::string("TEMPLATE plugin unavailable: ") + e.what();
        return false;
    }
    return true;
}

std::string reference_device(const ov::Core& core) {
    const auto devices = core.get_available_devices();
    if (std::find(devices.begin(), devices.end(), "TEMPLATE") != devices.end()) {
        return "TEMPLATE";
    }
    throw std::runtime_error("TEMPLATE reference device not available");
}

void expect_allclose(const ov::Tensor& a, const ov::Tensor& b, float atol = 1e-5f, float rtol = 0.f) {
    ASSERT_EQ(a.get_element_type(), ov::element::f32);
    ASSERT_EQ(b.get_element_type(), ov::element::f32);
    ASSERT_EQ(a.get_byte_size(), b.get_byte_size());
    auto* pa = a.data<const float>();
    auto* pb = b.data<const float>();
    const size_t count = a.get_size();
    for (size_t i = 0; i < count; ++i) {
        const float diff = std::abs(pa[i] - pb[i]);
        const float thresh = std::max(atol, rtol * std::abs(pa[i]));
        ASSERT_LE(diff, thresh) << "Mismatch at index " << i << ": " << pa[i] << " vs " << pb[i];
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

ov::AnyMap fp16_compile_config() {
    ov::AnyMap config;
    config[ov::hint::inference_precision.name()] = ov::element::f16;
    return config;
}

void require_allclose(const ov::Tensor& a, const ov::Tensor& b, float atol = 1e-5f, float rtol = 0.f) {
    if (a.get_element_type() != ov::element::f32 || b.get_element_type() != ov::element::f32) {
        throw std::runtime_error("expected f32 tensors");
    }
    if (a.get_byte_size() != b.get_byte_size() || a.get_shape() != b.get_shape()) {
        throw std::runtime_error("tensor shape mismatch");
    }
    auto* pa = a.data<const float>();
    auto* pb = b.data<const float>();
    const size_t count = a.get_size();
    for (size_t i = 0; i < count; ++i) {
        const float diff = std::abs(pa[i] - pb[i]);
        const float thresh = std::max(atol, rtol * std::abs(pa[i]));
        if (diff > thresh) {
            throw std::runtime_error("tensor mismatch at index " + std::to_string(i) +
                                     ": ref=" + std::to_string(pa[i]) +
                                     " gfx=" + std::to_string(pb[i]) +
                                     " diff=" + std::to_string(diff) +
                                     " thresh=" + std::to_string(thresh));
        }
    }
}

#ifndef _WIN32
void run_in_subprocess_with_timeout(const std::function<void()>& fn, int timeout_seconds) {
#ifdef __APPLE__
    // Metal shader compilation goes through Apple compiler services that are not
    // reliable after raw fork-without-exec. Keep the micro-repro coverage on
    // macOS, but run it inline until we add an exec-based watchdog helper.
    (void)timeout_seconds;
    fn();
    return;
#else
    const pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        try {
            fn();
            _exit(0);
        } catch (const std::exception& ex) {
            std::cerr << "child_failure: " << ex.what() << std::endl;
            _exit(1);
        } catch (...) {
            std::cerr << "child_failure: unknown exception" << std::endl;
            _exit(1);
        }
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
    int status = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        const pid_t waited = waitpid(pid, &status, WNOHANG);
        ASSERT_NE(waited, -1);
        if (waited == pid) {
            if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                return;
            }
            FAIL() << "subprocess exited with status " << status;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
    FAIL() << "subprocess timed out after " << timeout_seconds << " seconds";
#endif
}

void run_with_gfx_core_in_subprocess(const std::function<void(ov::Core&)>& fn, int timeout_seconds) {
    run_in_subprocess_with_timeout([&] {
        ov::Core child_core;
        if (!register_gfx_plugin(child_core)) {
            throw std::runtime_error(gfx_skip_reason.empty() ? "GFX backend unavailable" : gfx_skip_reason);
        }
        fn(child_core);
    }, timeout_seconds);
}

void compare_model_in_subprocess(const std::shared_ptr<ov::Model>& model,
                                 const std::vector<ov::Tensor>& inputs,
                                 int timeout_seconds,
                                 float atol = 1e-5f,
                                 float rtol = 0.f) {
    run_in_subprocess_with_timeout([&] {
        ov::Core child_core;
        if (!register_gfx_plugin(child_core)) {
            throw std::runtime_error(gfx_skip_reason.empty() ? "GFX backend unavailable" : gfx_skip_reason);
        }
        const auto ref_dev = reference_device(child_core);
        auto ref_cm = child_core.compile_model(model, ref_dev, fp16_compile_config());
        auto gfx_cm = child_core.compile_model(model, "GFX", fp16_compile_config());
        auto ref_req = ref_cm.create_infer_request();
        auto gfx_req = gfx_cm.create_infer_request();
        for (size_t i = 0; i < inputs.size(); ++i) {
            ref_req.set_input_tensor(i, inputs[i]);
            gfx_req.set_input_tensor(i, inputs[i]);
        }
        ref_req.infer();
        gfx_req.infer();
        for (size_t i = 0; i < ref_cm.outputs().size(); ++i) {
            const auto ref_out = ref_req.get_output_tensor(i);
            const auto gfx_out = gfx_req.get_output_tensor(i);
            try {
                require_allclose(ref_out, gfx_out, atol, rtol);
            } catch (const std::exception& ex) {
                throw std::runtime_error("output[" + std::to_string(i) + "] " + ex.what());
            }
        }
    }, timeout_seconds);
}

void compare_model_repeated_infer_in_subprocess(const std::shared_ptr<ov::Model>& model,
                                                const std::vector<ov::Tensor>& inputs,
                                                size_t infer_count,
                                                int timeout_seconds,
                                                float atol = 1e-5f,
                                                float rtol = 0.f) {
    run_in_subprocess_with_timeout([&] {
        ov::Core child_core;
        if (!register_gfx_plugin(child_core)) {
            throw std::runtime_error(gfx_skip_reason.empty() ? "GFX backend unavailable" : gfx_skip_reason);
        }
        const auto ref_dev = reference_device(child_core);
        auto ref_cm = child_core.compile_model(model, ref_dev, fp16_compile_config());
        auto gfx_cm = child_core.compile_model(model, "GFX", fp16_compile_config());
        auto ref_req = ref_cm.create_infer_request();
        auto gfx_req = gfx_cm.create_infer_request();
        for (size_t i = 0; i < inputs.size(); ++i) {
            ref_req.set_input_tensor(i, inputs[i]);
            gfx_req.set_input_tensor(i, inputs[i]);
        }
        for (size_t infer_index = 0; infer_index < infer_count; ++infer_index) {
            ref_req.infer();
            gfx_req.infer();
            for (size_t output_index = 0; output_index < ref_cm.outputs().size(); ++output_index) {
                const auto ref_out = ref_req.get_output_tensor(output_index);
                const auto gfx_out = gfx_req.get_output_tensor(output_index);
                try {
                    require_allclose(ref_out, gfx_out, atol, rtol);
                } catch (const std::exception& ex) {
                    throw std::runtime_error("infer[" + std::to_string(infer_index) +
                                             "] output[" + std::to_string(output_index) + "] " + ex.what());
                }
            }
        }
    }, timeout_seconds);
}
#endif

ov::Tensor get_output_or_skip(ov::InferRequest& req, size_t idx = 0) {
    return req.get_output_tensor(idx);
}

TEST(GfxRuntime, BroadcastAddBiasSubgraphMatchesTemplate) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 256, 20, 20});
    std::vector<float> bias_vals(256, 0.0f);
    for (size_t i = 0; i < bias_vals.size(); ++i) {
        bias_vals[i] = static_cast<float>((static_cast<int>(i % 13) - 6)) * 0.125f;
    }
    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 256, 1, 1}, bias_vals);
    auto add = std::make_shared<ov::op::v1::Add>(param, bias);
    auto res = std::make_shared<ov::op::v0::Result>(add);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "broadcast_add_bias");

    ov::Tensor input(ov::element::f32, ov::Shape{1, 256, 20, 20});
    auto* data = input.data<float>();
    for (size_t i = 0; i < input.get_size(); ++i) {
        data[i] = static_cast<float>((static_cast<int>(i % 37) - 18)) * 0.0625f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {input}, 15, 1e-5f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, SplitSubgraphMatchesTemplate) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 6});
    auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{2});
    auto split = std::make_shared<ov::op::v1::Split>(param, axis, 3);
    ov::ResultVector results;
    for (size_t i = 0; i < 3; ++i) {
        results.push_back(std::make_shared<ov::op::v0::Result>(split->output(i)));
    }
    auto model = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "split_runtime");

    ov::Tensor input(ov::element::f32, ov::Shape{1, 4, 6});
    auto* data = input.data<float>();
    for (size_t i = 0; i < input.get_size(); ++i) {
        data[i] = static_cast<float>((static_cast<int>(i % 23) - 11)) * 0.125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {input}, 15, 1e-5f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, MatMulSubgraphMatchesTemplate) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, true);
    auto res = std::make_shared<ov::op::v0::Result>(matmul);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{lhs, rhs}, "matmul_runtime");

    ov::Tensor lhs_t(ov::element::f32, ov::Shape{1, 4, 2});
    ov::Tensor rhs_t(ov::element::f32, ov::Shape{1, 4, 2});
    for (size_t i = 0; i < lhs_t.get_size(); ++i) {
        lhs_t.data<float>()[i] = static_cast<float>((static_cast<int>(i % 17) - 8)) * 0.125f;
        rhs_t.data<float>()[i] = static_cast<float>((static_cast<int>((i + 5) % 19) - 9)) * 0.125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {lhs_t, rhs_t}, 15, 1e-5f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, LargeMatMulMatchesTemplate) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, true, false);
    auto res = std::make_shared<ov::op::v0::Result>(matmul);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{lhs, rhs}, "large_matmul_runtime");

    ov::Tensor lhs_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    ov::Tensor rhs_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    for (size_t i = 0; i < lhs_t.get_size(); ++i) {
        lhs_t.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
        rhs_t.data<float>()[i] = static_cast<float>((static_cast<int>((i + 17) % 257) - 128)) * 0.03125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {lhs_t, rhs_t}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, MatMulCompileModelSucceeds) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, true);
    auto res = std::make_shared<ov::op::v0::Result>(matmul);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{lhs, rhs}, "matmul_compile_only");

#ifndef _WIN32
    run_with_gfx_core_in_subprocess([&](ov::Core& child_core) {
        auto gfx_cm = child_core.compile_model(model, "GFX", fp16_compile_config());
        if (!static_cast<bool>(gfx_cm)) {
            throw std::runtime_error("compile_model returned empty compiled model");
        }
    }, 15);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, MatMulCreateInferRequestSucceeds) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, true);
    auto res = std::make_shared<ov::op::v0::Result>(matmul);
    auto model =
        std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{lhs, rhs}, "matmul_create_request");

#ifndef _WIN32
    run_with_gfx_core_in_subprocess([&](ov::Core& child_core) {
        auto gfx_cm = child_core.compile_model(model, "GFX", fp16_compile_config());
        auto gfx_req = gfx_cm.create_infer_request();
        if (!static_cast<bool>(gfx_req)) {
            throw std::runtime_error("create_infer_request returned empty request");
        }
    }, 15);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, SoftmaxSubgraphMatchesTemplate) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4});
    auto softmax = std::make_shared<ov::op::v1::Softmax>(param, 2);
    auto res = std::make_shared<ov::op::v0::Result>(softmax);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "softmax_runtime");

    ov::Tensor input(ov::element::f32, ov::Shape{1, 4, 4});
    for (size_t i = 0; i < input.get_size(); ++i) {
        input.data<float>()[i] = static_cast<float>((static_cast<int>(i % 19) - 9)) * 0.125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {input}, 15, 1e-5f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, MultiplyMatMulSubgraphMatchesTemplate) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{0.5f});
    auto scaled = std::make_shared<ov::op::v1::Multiply>(lhs, scale);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(scaled, rhs, false, true);
    auto res = std::make_shared<ov::op::v0::Result>(matmul);
    auto model =
        std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{lhs, rhs}, "multiply_matmul_runtime");

    ov::Tensor lhs_t(ov::element::f32, ov::Shape{1, 4, 2});
    ov::Tensor rhs_t(ov::element::f32, ov::Shape{1, 4, 2});
    for (size_t i = 0; i < lhs_t.get_size(); ++i) {
        lhs_t.data<float>()[i] = static_cast<float>((static_cast<int>(i % 17) - 8)) * 0.125f;
        rhs_t.data<float>()[i] = static_cast<float>((static_cast<int>((i + 5) % 19) - 9)) * 0.125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {lhs_t, rhs_t}, 15, 1e-5f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, SoftmaxMatMulSubgraphMatchesTemplate) {
    auto probs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4});
    auto values = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto softmax = std::make_shared<ov::op::v1::Softmax>(probs, 2);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(softmax, values, false, false);
    auto res = std::make_shared<ov::op::v0::Result>(matmul);
    auto model =
        std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{probs, values}, "softmax_matmul_runtime");

    ov::Tensor probs_t(ov::element::f32, ov::Shape{1, 4, 4});
    ov::Tensor values_t(ov::element::f32, ov::Shape{1, 4, 2});
    for (size_t i = 0; i < probs_t.get_size(); ++i) {
        probs_t.data<float>()[i] = static_cast<float>((static_cast<int>(i % 19) - 9)) * 0.125f;
    }
    for (size_t i = 0; i < values_t.get_size(); ++i) {
        values_t.data<float>()[i] = static_cast<float>((static_cast<int>((i + 11) % 23) - 11)) * 0.125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {probs_t, values_t}, 15, 1e-5f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, AttentionCoreNoSplitMatchesTemplate) {
    auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{0.5f});
    auto q_scaled = std::make_shared<ov::op::v1::Multiply>(q, scale);
    auto scores = std::make_shared<ov::op::v0::MatMul>(q_scaled, k, false, true);
    auto probs = std::make_shared<ov::op::v1::Softmax>(scores, 2);
    auto attn = std::make_shared<ov::op::v0::MatMul>(probs, v, false, false);
    auto res = std::make_shared<ov::op::v0::Result>(attn);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{q, k, v}, "attn_core_no_split");

    ov::Tensor q_t(ov::element::f32, ov::Shape{1, 4, 2});
    ov::Tensor k_t(ov::element::f32, ov::Shape{1, 4, 2});
    ov::Tensor v_t(ov::element::f32, ov::Shape{1, 4, 2});
    for (size_t i = 0; i < q_t.get_size(); ++i) {
        q_t.data<float>()[i] = static_cast<float>((static_cast<int>(i % 17) - 8)) * 0.125f;
        k_t.data<float>()[i] = static_cast<float>((static_cast<int>((i + 5) % 19) - 9)) * 0.125f;
        v_t.data<float>()[i] = static_cast<float>((static_cast<int>((i + 11) % 23) - 11)) * 0.125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {q_t, k_t, v_t}, 15, 1e-5f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, AttentionPreScaleSubgraphMatchesTemplate) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 6});
    auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{2});
    auto split = std::make_shared<ov::op::v1::Split>(param, axis, 3);
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{0.5f});
    auto q_scaled = std::make_shared<ov::op::v1::Multiply>(split->output(0), scale);
    auto scores = std::make_shared<ov::op::v0::MatMul>(q_scaled, split->output(1), false, true);
    auto probs = std::make_shared<ov::op::v1::Softmax>(scores, 2);
    auto attn = std::make_shared<ov::op::v0::MatMul>(probs, split->output(2), false, false);
    auto res = std::make_shared<ov::op::v0::Result>(attn);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "attn_prescale_runtime");

    ov::Tensor input(ov::element::f32, ov::Shape{1, 4, 6});
    auto* data = input.data<float>();
    for (size_t i = 0; i < input.get_size(); ++i) {
        data[i] = static_cast<float>((static_cast<int>(i % 19) - 9)) * 0.125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {input}, 20, 1e-5f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, MultiplyBroadcast400x400MatchesTemplate) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 400, 400});
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                        ov::Shape{1, 1, 1, 1},
                                                        std::vector<float>{0.176776695f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(param, scale);
    auto res = std::make_shared<ov::op::v0::Result>(mul);
    auto model =
        std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "multiply_broadcast_400x400");

    ov::Tensor input(ov::element::f32, ov::Shape{1, 4, 400, 400});
    auto* data = input.data<float>();
    for (size_t i = 0; i < input.get_size(); ++i) {
        data[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {input}, 15, 1e-5f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, AttentionScoreScaleMatMulMatchesTemplate) {
    auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto scores = std::make_shared<ov::op::v0::MatMul>(q, k, true, false);
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                        ov::Shape{1, 1, 1, 1},
                                                        std::vector<float>{0.176776695f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(scores, scale);
    auto res = std::make_shared<ov::op::v0::Result>(mul);
    auto model =
        std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{q, k}, "attn_score_scale_matmul");

    ov::Tensor q_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    ov::Tensor k_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    for (size_t i = 0; i < q_t.get_size(); ++i) {
        q_t.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
        k_t.data<float>()[i] = static_cast<float>((static_cast<int>((i + 17) % 257) - 128)) * 0.03125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {q_t, k_t}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, AttentionLayoutSplitScaleMatchesTemplate) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 400, 384});
    auto reshape_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                ov::Shape{4},
                                                                std::vector<int64_t>{1, 400, 4, 96});
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(param, reshape_shape, true);
    auto perm = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                       ov::Shape{4},
                                                       std::vector<int64_t>{0, 2, 3, 1});
    auto transposed = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{2});
    auto split_lengths = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                ov::Shape{3},
                                                                std::vector<int64_t>{32, 32, 32});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(transposed, split_axis, split_lengths);
    auto scores = std::make_shared<ov::op::v0::MatMul>(split->output(0), split->output(1), true, false);
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                        ov::Shape{1, 1, 1, 1},
                                                        std::vector<float>{0.176776695f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(scores, scale);
    auto res = std::make_shared<ov::op::v0::Result>(mul);
    auto model =
        std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "attn_layout_split_scale");

    ov::Tensor input(ov::element::f32, ov::Shape{1, 400, 384});
    for (size_t i = 0; i < input.get_size(); ++i) {
        input.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {input}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, AttentionLayoutSplitScaledOperandMatchesTemplate) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 400, 384});
    auto reshape_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                ov::Shape{4},
                                                                std::vector<int64_t>{1, 400, 4, 96});
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(param, reshape_shape, true);
    auto perm = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                       ov::Shape{4},
                                                       std::vector<int64_t>{0, 2, 3, 1});
    auto transposed = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{2});
    auto split_lengths = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                ov::Shape{3},
                                                                std::vector<int64_t>{32, 32, 32});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(transposed, split_axis, split_lengths);
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                        ov::Shape{1, 1, 1, 1},
                                                        std::vector<float>{0.176776695f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(split->output(1), scale);
    auto res = std::make_shared<ov::op::v0::Result>(mul);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "attn_layout_scaled");

    ov::Tensor input(ov::element::f32, ov::Shape{1, 400, 384});
    for (size_t i = 0; i < input.get_size(); ++i) {
        input.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {input}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, AttentionLayoutVariadicSplitOutputsMatchTemplate) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 400, 384});
    auto reshape_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                ov::Shape{4},
                                                                std::vector<int64_t>{1, 400, 4, 96});
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(param, reshape_shape, true);
    auto perm = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                       ov::Shape{4},
                                                       std::vector<int64_t>{0, 2, 3, 1});
    auto transposed = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{2});
    auto split_lengths = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                ov::Shape{3},
                                                                std::vector<int64_t>{32, 32, 32});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(transposed, split_axis, split_lengths);
    ov::ResultVector results{
        std::make_shared<ov::op::v0::Result>(split->output(0)),
        std::make_shared<ov::op::v0::Result>(split->output(1)),
        std::make_shared<ov::op::v0::Result>(split->output(2)),
    };
    auto model = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "attn_layout_split_outputs");

    ov::Tensor input(ov::element::f32, ov::Shape{1, 400, 384});
    for (size_t i = 0; i < input.get_size(); ++i) {
        input.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {input}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, VariadicSplitMatMulMatchesTemplate) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
    auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{2});
    auto split_lengths = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                ov::Shape{3},
                                                                std::vector<int64_t>{32, 32, 32});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(param, split_axis, split_lengths);
    auto scores = std::make_shared<ov::op::v0::MatMul>(split->output(0), split->output(1), true, false);
    auto res = std::make_shared<ov::op::v0::Result>(scores);
    auto model =
        std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "variadic_split_matmul");

    ov::Tensor input(ov::element::f32, ov::Shape{1, 4, 96, 400});
    for (size_t i = 0; i < input.get_size(); ++i) {
        input.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {input}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, VariadicSplitLhsMatMulMatchesTemplate) {
    auto split_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
    auto rhs_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{2});
    auto split_lengths = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                ov::Shape{3},
                                                                std::vector<int64_t>{32, 32, 32});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(split_in, split_axis, split_lengths);
    auto scores = std::make_shared<ov::op::v0::MatMul>(split->output(0), rhs_in, true, false);
    auto res = std::make_shared<ov::op::v0::Result>(scores);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{split_in, rhs_in},
                                             "variadic_split_lhs_matmul");

    ov::Tensor split_t(ov::element::f32, ov::Shape{1, 4, 96, 400});
    ov::Tensor rhs_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    for (size_t i = 0; i < split_t.get_size(); ++i) {
        split_t.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
    }
    for (size_t i = 0; i < rhs_t.get_size(); ++i) {
        rhs_t.data<float>()[i] = static_cast<float>((static_cast<int>((i + 17) % 257) - 128)) * 0.03125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {split_t, rhs_t}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, VariadicSplitRhsMatMulMatchesTemplate) {
    auto lhs_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto split_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
    auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{2});
    auto split_lengths = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                ov::Shape{3},
                                                                std::vector<int64_t>{32, 32, 32});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(split_in, split_axis, split_lengths);
    auto scores = std::make_shared<ov::op::v0::MatMul>(lhs_in, split->output(1), true, false);
    auto res = std::make_shared<ov::op::v0::Result>(scores);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{lhs_in, split_in},
                                             "variadic_split_rhs_matmul");

    ov::Tensor lhs_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    ov::Tensor split_t(ov::element::f32, ov::Shape{1, 4, 96, 400});
    for (size_t i = 0; i < lhs_t.get_size(); ++i) {
        lhs_t.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
    }
    for (size_t i = 0; i < split_t.get_size(); ++i) {
        split_t.data<float>()[i] = static_cast<float>((static_cast<int>((i + 17) % 257) - 128)) * 0.03125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {lhs_t, split_t}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, VariadicSplitScaledMatMulMatchesTemplate) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
    auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{2});
    auto split_lengths = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                ov::Shape{3},
                                                                std::vector<int64_t>{32, 32, 32});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(param, split_axis, split_lengths);
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                        ov::Shape{1, 1, 1, 1},
                                                        std::vector<float>{0.176776695f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(split->output(1), scale);
    auto scores = std::make_shared<ov::op::v0::MatMul>(split->output(0), mul, true, false);
    auto res = std::make_shared<ov::op::v0::Result>(scores);
    auto model =
        std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "variadic_split_scaled_matmul");

    ov::Tensor input(ov::element::f32, ov::Shape{1, 4, 96, 400});
    for (size_t i = 0; i < input.get_size(); ++i) {
        input.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {input}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, LargeBroadcastMultiplyMatchesTemplate) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                        ov::Shape{1, 1, 1, 1},
                                                        std::vector<float>{0.176776695f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(input, scale);
    auto res = std::make_shared<ov::op::v0::Result>(mul);
    auto model =
        std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{input}, "large_broadcast_multiply");

    ov::Tensor input_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    for (size_t i = 0; i < input_t.get_size(); ++i) {
        input_t.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {input_t}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, LargeBroadcastMultiplyMatMulMatchesTemplate) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                        ov::Shape{1, 1, 1, 1},
                                                        std::vector<float>{0.176776695f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(rhs, scale);
    auto scores = std::make_shared<ov::op::v0::MatMul>(lhs, mul, true, false);
    auto res = std::make_shared<ov::op::v0::Result>(scores);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{lhs, rhs},
                                             "large_broadcast_multiply_matmul");

    ov::Tensor lhs_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    ov::Tensor rhs_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    for (size_t i = 0; i < lhs_t.get_size(); ++i) {
        lhs_t.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
        rhs_t.data<float>()[i] = static_cast<float>((static_cast<int>((i + 17) % 257) - 128)) * 0.03125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {lhs_t, rhs_t}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, DualProducedBroadcastMultiplyMatMulMatchesTemplate) {
    auto lhs_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto rhs_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto one = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                      ov::Shape{1, 1, 1, 1},
                                                      std::vector<float>{1.0f});
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                        ov::Shape{1, 1, 1, 1},
                                                        std::vector<float>{0.176776695f});
    auto lhs = std::make_shared<ov::op::v1::Multiply>(lhs_in, one);
    auto rhs = std::make_shared<ov::op::v1::Multiply>(rhs_in, scale);
    auto scores = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, true, false);
    auto res = std::make_shared<ov::op::v0::Result>(scores);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{lhs_in, rhs_in},
                                             "dual_produced_broadcast_multiply_matmul");

    ov::Tensor lhs_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    ov::Tensor rhs_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    for (size_t i = 0; i < lhs_t.get_size(); ++i) {
        lhs_t.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
        rhs_t.data<float>()[i] = static_cast<float>((static_cast<int>((i + 17) % 257) - 128)) * 0.03125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {lhs_t, rhs_t}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, ProducedLhsMatMulMatchesTemplate) {
    auto lhs_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto rhs_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto zero = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto lhs = std::make_shared<ov::op::v1::Add>(lhs_in, zero);
    auto scores = std::make_shared<ov::op::v0::MatMul>(lhs, rhs_in, true, false);
    auto res = std::make_shared<ov::op::v0::Result>(scores);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{lhs_in, rhs_in, zero},
                                             "produced_lhs_matmul");

    ov::Tensor lhs_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    ov::Tensor rhs_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    ov::Tensor zero_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    for (size_t i = 0; i < lhs_t.get_size(); ++i) {
        lhs_t.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
        rhs_t.data<float>()[i] = static_cast<float>((static_cast<int>((i + 17) % 257) - 128)) * 0.03125f;
    }
    std::fill_n(zero_t.data<float>(), zero_t.get_size(), 0.0f);

#ifndef _WIN32
    compare_model_in_subprocess(model, {lhs_t, rhs_t, zero_t}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, ProducedLhsAddMatchesTemplate) {
    auto lhs_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto zero = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto lhs = std::make_shared<ov::op::v1::Add>(lhs_in, zero);
    auto res = std::make_shared<ov::op::v0::Result>(lhs);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{lhs_in, zero},
                                             "produced_lhs_add");

    ov::Tensor lhs_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    ov::Tensor zero_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    for (size_t i = 0; i < lhs_t.get_size(); ++i) {
        lhs_t.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
    }
    std::fill_n(zero_t.data<float>(), zero_t.get_size(), 0.0f);

#ifndef _WIN32
    compare_model_in_subprocess(model, {lhs_t, zero_t}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, ProducedLhsAndMatMulOutputsMatchTemplate) {
    auto lhs_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto rhs_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto zero = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto lhs = std::make_shared<ov::op::v1::Add>(lhs_in, zero);
    auto scores = std::make_shared<ov::op::v0::MatMul>(lhs, rhs_in, true, false);
    ov::ResultVector results{
        std::make_shared<ov::op::v0::Result>(lhs),
        std::make_shared<ov::op::v0::Result>(scores),
    };
    auto model = std::make_shared<ov::Model>(results,
                                             ov::ParameterVector{lhs_in, rhs_in, zero},
                                             "produced_lhs_and_matmul_outputs");

    ov::Tensor lhs_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    ov::Tensor rhs_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    ov::Tensor zero_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    for (size_t i = 0; i < lhs_t.get_size(); ++i) {
        lhs_t.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
        rhs_t.data<float>()[i] = static_cast<float>((static_cast<int>((i + 17) % 257) - 128)) * 0.03125f;
    }
    std::fill_n(zero_t.data<float>(), zero_t.get_size(), 0.0f);

#ifndef _WIN32
    compare_model_in_subprocess(model, {lhs_t, rhs_t, zero_t}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, ProducedLhsAndMatMulOutputsRemainStableAcrossRepeatedInfer) {
    auto lhs_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto rhs_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto zero = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto lhs = std::make_shared<ov::op::v1::Add>(lhs_in, zero);
    auto scores = std::make_shared<ov::op::v0::MatMul>(lhs, rhs_in, true, false);
    ov::ResultVector results{
        std::make_shared<ov::op::v0::Result>(lhs),
        std::make_shared<ov::op::v0::Result>(scores),
    };
    auto model = std::make_shared<ov::Model>(results,
                                             ov::ParameterVector{lhs_in, rhs_in, zero},
                                             "produced_lhs_and_matmul_outputs_repeat");

    ov::Tensor lhs_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    ov::Tensor rhs_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    ov::Tensor zero_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    for (size_t i = 0; i < lhs_t.get_size(); ++i) {
        lhs_t.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
        rhs_t.data<float>()[i] = static_cast<float>((static_cast<int>((i + 17) % 257) - 128)) * 0.03125f;
    }
    std::fill_n(zero_t.data<float>(), zero_t.get_size(), 0.0f);

#ifndef _WIN32
    compare_model_repeated_infer_in_subprocess(model, {lhs_t, rhs_t, zero_t}, 4, 30, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, VariadicSplitScaledOutputsMatchTemplate) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
    auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{2});
    auto split_lengths = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                ov::Shape{3},
                                                                std::vector<int64_t>{32, 32, 32});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(param, split_axis, split_lengths);
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                        ov::Shape{1, 1, 1, 1},
                                                        std::vector<float>{0.176776695f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(split->output(1), scale);
    ov::ResultVector results{
        std::make_shared<ov::op::v0::Result>(split->output(0)),
        std::make_shared<ov::op::v0::Result>(mul),
    };
    auto model =
        std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "variadic_split_scaled_outputs");

    ov::Tensor input(ov::element::f32, ov::Shape{1, 4, 96, 400});
    for (size_t i = 0; i < input.get_size(); ++i) {
        input.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {input}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, VariadicSplitDualScaledMatMulMatchesTemplate) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
    auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{2});
    auto split_lengths = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                ov::Shape{3},
                                                                std::vector<int64_t>{32, 32, 32});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(param, split_axis, split_lengths);
    auto one = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                      ov::Shape{1, 1, 1, 1},
                                                      std::vector<float>{1.0f});
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                        ov::Shape{1, 1, 1, 1},
                                                        std::vector<float>{0.176776695f});
    auto lhs = std::make_shared<ov::op::v1::Multiply>(split->output(0), one);
    auto rhs = std::make_shared<ov::op::v1::Multiply>(split->output(1), scale);
    auto scores = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, true, false);
    auto res = std::make_shared<ov::op::v0::Result>(scores);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{param},
                                             "variadic_split_dual_scaled_matmul");

    ov::Tensor input(ov::element::f32, ov::Shape{1, 4, 96, 400});
    for (size_t i = 0; i < input.get_size(); ++i) {
        input.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {input}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, VariadicSplitAddScaledMatMulMatchesTemplate) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
    auto zero = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{2});
    auto split_lengths = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                ov::Shape{3},
                                                                std::vector<int64_t>{32, 32, 32});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(param, split_axis, split_lengths);
    auto lhs = std::make_shared<ov::op::v1::Add>(split->output(0), zero);
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                        ov::Shape{1, 1, 1, 1},
                                                        std::vector<float>{0.176776695f});
    auto rhs = std::make_shared<ov::op::v1::Multiply>(split->output(1), scale);
    auto scores = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, true, false);
    auto res = std::make_shared<ov::op::v0::Result>(scores);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{param, zero},
                                             "variadic_split_add_scaled_matmul");

    ov::Tensor input(ov::element::f32, ov::Shape{1, 4, 96, 400});
    ov::Tensor zero_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    for (size_t i = 0; i < input.get_size(); ++i) {
        input.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
    }
    std::fill_n(zero_t.data<float>(), zero_t.get_size(), 0.0f);

#ifndef _WIN32
    compare_model_in_subprocess(model, {input, zero_t}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, VariadicSplitAddScaledMatMulRemainsStableAcrossRepeatedInfer) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
    auto zero = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{2});
    auto split_lengths = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                ov::Shape{3},
                                                                std::vector<int64_t>{32, 32, 32});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(param, split_axis, split_lengths);
    auto lhs = std::make_shared<ov::op::v1::Add>(split->output(0), zero);
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                        ov::Shape{1, 1, 1, 1},
                                                        std::vector<float>{0.176776695f});
    auto rhs = std::make_shared<ov::op::v1::Multiply>(split->output(1), scale);
    auto scores = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, true, false);
    auto res = std::make_shared<ov::op::v0::Result>(scores);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{param, zero},
                                             "variadic_split_add_scaled_matmul_repeat");

    ov::Tensor input(ov::element::f32, ov::Shape{1, 4, 96, 400});
    ov::Tensor zero_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    for (size_t i = 0; i < input.get_size(); ++i) {
        input.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
    }
    std::fill_n(zero_t.data<float>(), zero_t.get_size(), 0.0f);

#ifndef _WIN32
    compare_model_repeated_infer_in_subprocess(model, {input, zero_t}, 4, 30, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, VariadicSplitAddOnlyMatchesTemplate) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
    auto zero = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{2});
    auto split_lengths = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                ov::Shape{3},
                                                                std::vector<int64_t>{32, 32, 32});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(param, split_axis, split_lengths);
    auto lhs = std::make_shared<ov::op::v1::Add>(split->output(0), zero);
    auto res = std::make_shared<ov::op::v0::Result>(lhs);
    auto model =
        std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param, zero}, "variadic_split_add_only");

    ov::Tensor input(ov::element::f32, ov::Shape{1, 4, 96, 400});
    ov::Tensor zero_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    for (size_t i = 0; i < input.get_size(); ++i) {
        input.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
    }
    std::fill_n(zero_t.data<float>(), zero_t.get_size(), 0.0f);

#ifndef _WIN32
    compare_model_in_subprocess(model, {input, zero_t}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, VariadicSplitAddAndMatMulOutputsMatchTemplate) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
    auto zero = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{2});
    auto split_lengths = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                ov::Shape{3},
                                                                std::vector<int64_t>{32, 32, 32});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(param, split_axis, split_lengths);
    auto lhs = std::make_shared<ov::op::v1::Add>(split->output(0), zero);
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                        ov::Shape{1, 1, 1, 1},
                                                        std::vector<float>{0.176776695f});
    auto rhs = std::make_shared<ov::op::v1::Multiply>(split->output(1), scale);
    auto scores = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, true, false);
    ov::ResultVector results{
        std::make_shared<ov::op::v0::Result>(lhs),
        std::make_shared<ov::op::v0::Result>(scores),
    };
    auto model = std::make_shared<ov::Model>(results,
                                             ov::ParameterVector{param, zero},
                                             "variadic_split_add_and_matmul_outputs");

    ov::Tensor input(ov::element::f32, ov::Shape{1, 4, 96, 400});
    ov::Tensor zero_t(ov::element::f32, ov::Shape{1, 4, 32, 400});
    for (size_t i = 0; i < input.get_size(); ++i) {
        input.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
    }
    std::fill_n(zero_t.data<float>(), zero_t.get_size(), 0.0f);

#ifndef _WIN32
    compare_model_in_subprocess(model, {input, zero_t}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, VariadicSplitAttentionAndValueLayoutMatchTemplate) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
    auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{2});
    auto split_lengths = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                ov::Shape{3},
                                                                std::vector<int64_t>{32, 32, 32});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(param, split_axis, split_lengths);
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                        ov::Shape{1, 1, 1, 1},
                                                        std::vector<float>{0.176776695f});
    auto scaled_k = std::make_shared<ov::op::v1::Multiply>(split->output(1), scale);
    auto scores = std::make_shared<ov::op::v0::MatMul>(split->output(0), scaled_k, true, false);
    auto probs = std::make_shared<ov::op::v1::Softmax>(scores, 3);
    auto attn = std::make_shared<ov::op::v0::MatMul>(probs, split->output(2), false, true);

    auto value_perm = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                             ov::Shape{4},
                                                             std::vector<int64_t>{0, 3, 1, 2});
    auto value_transpose = std::make_shared<ov::op::v1::Transpose>(split->output(2), value_perm);
    auto value_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                              ov::Shape{3},
                                                              std::vector<int64_t>{1, 400, 128});
    auto value_reshape = std::make_shared<ov::op::v1::Reshape>(value_transpose, value_shape, true);

    ov::ResultVector results{
        std::make_shared<ov::op::v0::Result>(attn),
        std::make_shared<ov::op::v0::Result>(value_reshape),
    };
    auto model = std::make_shared<ov::Model>(results,
                                             ov::ParameterVector{param},
                                             "variadic_split_attention_and_value_layout");

    ov::Tensor input(ov::element::f32, ov::Shape{1, 4, 96, 400});
    for (size_t i = 0; i < input.get_size(); ++i) {
        input.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {input}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, VariadicSplitValueLayoutOnlyMatchesTemplate) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
    auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{2});
    auto split_lengths = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                ov::Shape{3},
                                                                std::vector<int64_t>{32, 32, 32});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(param, split_axis, split_lengths);

    auto value_perm = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                             ov::Shape{4},
                                                             std::vector<int64_t>{0, 3, 1, 2});
    auto value_transpose = std::make_shared<ov::op::v1::Transpose>(split->output(2), value_perm);
    auto value_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                              ov::Shape{3},
                                                              std::vector<int64_t>{1, 400, 128});
    auto value_reshape = std::make_shared<ov::op::v1::Reshape>(value_transpose, value_shape, true);
    auto res = std::make_shared<ov::op::v0::Result>(value_reshape);
    auto model =
        std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "variadic_split_value_layout");

    ov::Tensor input(ov::element::f32, ov::Shape{1, 4, 96, 400});
    for (size_t i = 0; i < input.get_size(); ++i) {
        input.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {input}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, VariadicSplitScaledMatMulAndValueLayoutMatchTemplate) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
    auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{2});
    auto split_lengths = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                ov::Shape{3},
                                                                std::vector<int64_t>{32, 32, 32});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(param, split_axis, split_lengths);
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                        ov::Shape{1, 1, 1, 1},
                                                        std::vector<float>{0.176776695f});
    auto scaled_k = std::make_shared<ov::op::v1::Multiply>(split->output(1), scale);
    auto scores = std::make_shared<ov::op::v0::MatMul>(split->output(0), scaled_k, true, false);

    auto value_perm = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                             ov::Shape{4},
                                                             std::vector<int64_t>{0, 3, 1, 2});
    auto value_transpose = std::make_shared<ov::op::v1::Transpose>(split->output(2), value_perm);
    auto value_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                              ov::Shape{3},
                                                              std::vector<int64_t>{1, 400, 128});
    auto value_reshape = std::make_shared<ov::op::v1::Reshape>(value_transpose, value_shape, true);

    ov::ResultVector results{
        std::make_shared<ov::op::v0::Result>(scores),
        std::make_shared<ov::op::v0::Result>(value_reshape),
    };
    auto model = std::make_shared<ov::Model>(results,
                                             ov::ParameterVector{param},
                                             "variadic_split_scaled_matmul_and_value_layout");

    ov::Tensor input(ov::element::f32, ov::Shape{1, 4, 96, 400});
    for (size_t i = 0; i < input.get_size(); ++i) {
        input.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {input}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, SplitMatMulMatchesTemplate) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
    auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{2});
    auto split = std::make_shared<ov::op::v1::Split>(param, split_axis, 3);
    auto scores = std::make_shared<ov::op::v0::MatMul>(split->output(0), split->output(1), true, false);
    auto res = std::make_shared<ov::op::v0::Result>(scores);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "split_matmul");

    ov::Tensor input(ov::element::f32, ov::Shape{1, 4, 96, 400});
    for (size_t i = 0; i < input.get_size(); ++i) {
        input.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {input}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

TEST(GfxRuntime, AttentionLayoutSplitMatMulMatchesTemplate) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 400, 384});
    auto reshape_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                ov::Shape{4},
                                                                std::vector<int64_t>{1, 400, 4, 96});
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(param, reshape_shape, true);
    auto perm = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                       ov::Shape{4},
                                                       std::vector<int64_t>{0, 2, 3, 1});
    auto transposed = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{2});
    auto split_lengths = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                ov::Shape{3},
                                                                std::vector<int64_t>{32, 32, 32});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(transposed, split_axis, split_lengths);
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                        ov::Shape{1, 1, 1, 1},
                                                        std::vector<float>{0.176776695f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(split->output(1), scale);
    auto scores = std::make_shared<ov::op::v0::MatMul>(split->output(0), mul, true, false);
    auto res = std::make_shared<ov::op::v0::Result>(scores);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "attn_layout_matmul");

    ov::Tensor input(ov::element::f32, ov::Shape{1, 400, 384});
    for (size_t i = 0; i < input.get_size(); ++i) {
        input.data<float>()[i] = static_cast<float>((static_cast<int>(i % 251) - 125)) * 0.03125f;
    }

#ifndef _WIN32
    compare_model_in_subprocess(model, {input}, 20, 1e-4f, 0.f);
#else
    GTEST_SKIP() << "subprocess watchdog requires POSIX";
#endif
}

}  // namespace
