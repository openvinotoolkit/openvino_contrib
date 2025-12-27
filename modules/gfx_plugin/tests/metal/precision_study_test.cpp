// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"
#include "../gfx_test_utils.hpp"
#include "plugin/gfx_backend_config.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"

namespace {

std::string gfx_skip_reason;

bool register_gfx_plugin(ov::Core& core) {
    gfx_skip_reason.clear();
    try {
#ifdef GFX_PLUGIN_PATH
        const char* env_path = std::getenv("GFX_PLUGIN_PATH");
        const char* path = (env_path && *env_path) ? env_path : GFX_PLUGIN_PATH;
        core.register_plugin(path, "GFX");
#else
        // Assume default discovery if path macro is absent.
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

std::vector<float> make_random_data(size_t count, std::mt19937& gen, float lo = -10.f, float hi = 10.f) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(count);
    for (auto& x : v)
        x = dist(gen);
    return v;
}

ov::Tensor make_tensor(const ov::Shape& shape, const std::vector<float>& data) {
    ov::Tensor t{ov::element::f32, shape};
    std::copy(data.begin(), data.end(), t.data<float>());
    return t;
}

struct ErrorStats {
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double p99_abs = 0.0;
    size_t samples = 0;
};

ErrorStats compute_error_stats(const ov::Tensor& ref, const ov::Tensor& test) {
    if (ref.get_byte_size() != test.get_byte_size()) {
        ADD_FAILURE() << "Tensor byte size mismatch";
        return {};
    }
    // Ensure element type matches
    EXPECT_EQ(ref.get_element_type(), ov::element::f32);
    EXPECT_EQ(test.get_element_type(), ov::element::f32);
    auto* a = ref.data<const float>();
    auto* b = test.data<const float>();
    size_t n = ref.get_size();
    std::vector<double> errs;
    errs.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        errs.push_back(std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    ErrorStats s;
    s.samples = n;
    s.max_abs = *std::max_element(errs.begin(), errs.end());
    s.mean_abs = std::accumulate(errs.begin(), errs.end(), 0.0) / static_cast<double>(n);
    if (!errs.empty()) {
        std::nth_element(errs.begin(), errs.begin() + static_cast<long>(0.99 * n), errs.end());
        s.p99_abs = errs[static_cast<size_t>(0.99 * n)];
    }
    return s;
}

void expect_finite(const ov::Tensor& t) {
    const float* p = t.data<const float>();
    for (size_t i = 0; i < t.get_size(); ++i) {
        ASSERT_TRUE(std::isfinite(p[i])) << "Non-finite at " << i << ": " << p[i];
    }
}

ov::Tensor run(ov::CompiledModel& cm, const ov::Tensor& input) {
    auto req = cm.create_infer_request();
    req.set_input_tensor(input);
    req.infer();
    return req.get_output_tensor();
}

std::shared_ptr<ov::Model> make_add_scalar_model(const ov::Shape& shape) {
    auto p = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto c = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{0.5f});
    auto add = std::make_shared<ov::op::v1::Add>(p, c);
    auto r = std::make_shared<ov::op::v0::Result>(add);
    return std::make_shared<ov::Model>(ov::ResultVector{r}, ov::ParameterVector{p}, "add_scalar_model");
}

std::shared_ptr<ov::Model> make_add_channel_model(const ov::Shape& shape, size_t channels) {
    auto p = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    std::vector<float> cvals(channels);
    for (size_t i = 0; i < channels; ++i)
        cvals[i] = static_cast<float>(0.1 * (static_cast<int>(i) - 1));  // simple deterministic values
    auto c = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, channels, 1, 1}, cvals);
    auto add = std::make_shared<ov::op::v1::Add>(p, c);
    auto r = std::make_shared<ov::op::v0::Result>(add);
    return std::make_shared<ov::Model>(ov::ResultVector{r}, ov::ParameterVector{p}, "add_channel_model");
}

std::shared_ptr<ov::Model> make_softmax_model(const ov::Shape& shape, int64_t axis) {
    auto p = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto sm = std::make_shared<ov::op::v1::Softmax>(p, axis);
    auto r = std::make_shared<ov::op::v0::Result>(sm);
    return std::make_shared<ov::Model>(ov::ResultVector{r}, ov::ParameterVector{p}, "softmax_model");
}

void log_stats(const std::string& name, const std::vector<double>& maxima) {
    double max_overall = *std::max_element(maxima.begin(), maxima.end());
    double mean_overall = std::accumulate(maxima.begin(), maxima.end(), 0.0) / static_cast<double>(maxima.size());
    std::vector<double> sorted = maxima;
    std::sort(sorted.begin(), sorted.end());
    double p99 = sorted[static_cast<size_t>(0.99 * sorted.size())];
    std::cout << "[GfxPrecisionStudy] " << name
              << " runs=" << maxima.size()
              << " max=" << max_overall
              << " mean=" << mean_overall
              << " p99=" << p99 << std::endl;
}

}  // namespace

TEST(GfxPrecisionStudy, AddBroadcastScalarRandom) {
    ov::Core core;
    ASSERT_TRUE(register_gfx_plugin(core)) << gfx_skip_reason;
    const ov::Shape shape{1, 4};
    auto model = make_add_scalar_model(shape);
    ov::CompiledModel ref_cm = core.compile_model(model, reference_device(core));
    ov::CompiledModel metal_cm = core.compile_model(model, "GFX");

        std::mt19937 gen(12345);
        std::vector<double> run_max_errors;
        std::vector<double> run_mean_errors;
        constexpr int kRuns = 200;
        for (int i = 0; i < kRuns; ++i) {
            auto data = make_random_data(ov::shape_size(shape), gen);
            auto input = make_tensor(shape, data);

            auto ref_out = run(ref_cm, input);
            auto metal_out = run(metal_cm, input);
            expect_finite(ref_out);
            expect_finite(metal_out);
            auto stats = compute_error_stats(ref_out, metal_out);
            run_max_errors.push_back(stats.max_abs);
            run_mean_errors.push_back(stats.mean_abs);
        }
    log_stats("AddBroadcastScalar", run_max_errors);
    EXPECT_LT(*std::max_element(run_max_errors.begin(), run_max_errors.end()), 1e-2);  // sanity guard
    EXPECT_LT(*std::max_element(run_mean_errors.begin(), run_mean_errors.end()), 1e-2);
}

TEST(GfxPrecisionStudy, AddBroadcastChannelRandom) {
    ov::Core core;
    ASSERT_TRUE(register_gfx_plugin(core)) << gfx_skip_reason;
    const ov::Shape shape{1, 3, 4, 4};
    auto model = make_add_channel_model(shape, /*channels=*/3);
    ov::CompiledModel ref_cm = core.compile_model(model, reference_device(core));
    ov::CompiledModel metal_cm = core.compile_model(model, "GFX");

        std::mt19937 gen(23456);
        std::vector<double> run_max_errors;
        std::vector<double> run_mean_errors;
        constexpr int kRuns = 200;
        for (int i = 0; i < kRuns; ++i) {
            auto data = make_random_data(ov::shape_size(shape), gen);
            auto input = make_tensor(shape, data);

            auto ref_out = run(ref_cm, input);
            auto metal_out = run(metal_cm, input);
            expect_finite(ref_out);
            expect_finite(metal_out);
            auto stats = compute_error_stats(ref_out, metal_out);
            run_max_errors.push_back(stats.max_abs);
            run_mean_errors.push_back(stats.mean_abs);
        }
    log_stats("AddBroadcastChannel", run_max_errors);
    EXPECT_LT(*std::max_element(run_max_errors.begin(), run_max_errors.end()), 2.0);  // diagnostic guard only
    EXPECT_LT(*std::max_element(run_mean_errors.begin(), run_mean_errors.end()), 1e-2);
}

TEST(GfxPrecisionStudy, SoftmaxRandom) {
    ov::Core core;
    ASSERT_TRUE(register_gfx_plugin(core)) << gfx_skip_reason;
    const ov::Shape shape{2, 16};
    auto model = make_softmax_model(shape, 1);
    ov::CompiledModel ref_cm = core.compile_model(model, reference_device(core));
    ov::CompiledModel metal_cm = core.compile_model(model, "GFX");

    std::mt19937 gen(34567);
    std::vector<double> run_max_errors;
    std::vector<double> run_mean_errors;
    constexpr int kRuns = 200;
    for (int i = 0; i < kRuns; ++i) {
        auto data = make_random_data(ov::shape_size(shape), gen, -10.f, 10.f);
        auto input = make_tensor(shape, data);

        auto ref_out = run(ref_cm, input);
        auto metal_out = run(metal_cm, input);
        expect_finite(ref_out);
        expect_finite(metal_out);
        auto stats = compute_error_stats(ref_out, metal_out);
        run_max_errors.push_back(stats.max_abs);
        run_mean_errors.push_back(stats.mean_abs);

        // Softmax invariant: probabilities should lie in [0,1]
        auto check_bounds = [](const ov::Tensor& t) {
            const float* p = t.data<const float>();
            for (size_t i = 0; i < t.get_size(); ++i) {
                EXPECT_GE(p[i], 0.f);
                EXPECT_LE(p[i], 1.f + 1e-3f);
            }
        };
        check_bounds(ref_out);
        check_bounds(metal_out);
    }
    log_stats("Softmax", run_max_errors);
    EXPECT_LT(*std::max_element(run_max_errors.begin(), run_max_errors.end()), 2.0);  // diagnostic guard only
    EXPECT_LT(*std::max_element(run_mean_errors.begin(), run_mean_errors.end()), 0.1);
}
