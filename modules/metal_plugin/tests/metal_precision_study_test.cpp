// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
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
    std::cout << "[MetalPrecisionStudy] " << name
              << " runs=" << maxima.size()
              << " max=" << max_overall
              << " mean=" << mean_overall
              << " p99=" << p99 << std::endl;
}

}  // namespace

TEST(MetalPrecisionStudy, AddBroadcastScalarRandom) {
    ov::Core core;
    register_metal_plugin(core);

    const ov::Shape shape{1, 4};
    auto model = make_add_scalar_model(shape);
    ov::CompiledModel cpu_cm = core.compile_model(model, "CPU");
    ov::CompiledModel metal_cm = core.compile_model(model, "METAL");

    std::mt19937 gen(12345);
    std::vector<double> run_max_errors;
    std::vector<double> run_mean_errors;
    constexpr int kRuns = 200;
    for (int i = 0; i < kRuns; ++i) {
        auto data = make_random_data(ov::shape_size(shape), gen);
        auto input = make_tensor(shape, data);

        auto cpu_out = run(cpu_cm, input);
        auto metal_out = run(metal_cm, input);
        expect_finite(cpu_out);
        expect_finite(metal_out);
        auto stats = compute_error_stats(cpu_out, metal_out);
        run_max_errors.push_back(stats.max_abs);
        run_mean_errors.push_back(stats.mean_abs);
    }
    log_stats("AddBroadcastScalar", run_max_errors);
    EXPECT_LT(*std::max_element(run_max_errors.begin(), run_max_errors.end()), 1e-2);  // sanity guard
    EXPECT_LT(*std::max_element(run_mean_errors.begin(), run_mean_errors.end()), 1e-2);
}

TEST(MetalPrecisionStudy, AddBroadcastChannelRandom) {
    ov::Core core;
    register_metal_plugin(core);

    const ov::Shape shape{1, 3, 4, 4};
    auto model = make_add_channel_model(shape, /*channels=*/3);
    ov::CompiledModel cpu_cm = core.compile_model(model, "CPU");
    ov::CompiledModel metal_cm = core.compile_model(model, "METAL");

    std::mt19937 gen(23456);
    std::vector<double> run_max_errors;
    std::vector<double> run_mean_errors;
    constexpr int kRuns = 200;
    for (int i = 0; i < kRuns; ++i) {
        auto data = make_random_data(ov::shape_size(shape), gen);
        auto input = make_tensor(shape, data);

        auto cpu_out = run(cpu_cm, input);
        auto metal_out = run(metal_cm, input);
        expect_finite(cpu_out);
        expect_finite(metal_out);
        auto stats = compute_error_stats(cpu_out, metal_out);
        run_max_errors.push_back(stats.max_abs);
        run_mean_errors.push_back(stats.mean_abs);
    }
    log_stats("AddBroadcastChannel", run_max_errors);
    EXPECT_LT(*std::max_element(run_max_errors.begin(), run_max_errors.end()), 2.0);  // diagnostic guard only
    EXPECT_LT(*std::max_element(run_mean_errors.begin(), run_mean_errors.end()), 1e-2);
}

TEST(MetalPrecisionStudy, SoftmaxRandom) {
    ov::Core core;
    register_metal_plugin(core);

    const ov::Shape shape{2, 16};
    auto model = make_softmax_model(shape, 1);
    ov::CompiledModel cpu_cm = core.compile_model(model, "CPU");
    ov::CompiledModel metal_cm = core.compile_model(model, "METAL");

    std::mt19937 gen(34567);
    std::vector<double> run_max_errors;
    std::vector<double> run_mean_errors;
    constexpr int kRuns = 200;
    for (int i = 0; i < kRuns; ++i) {
        auto data = make_random_data(ov::shape_size(shape), gen, -10.f, 10.f);
        auto input = make_tensor(shape, data);

        auto cpu_out = run(cpu_cm, input);
        auto metal_out = run(metal_cm, input);
        expect_finite(cpu_out);
        expect_finite(metal_out);
        auto stats = compute_error_stats(cpu_out, metal_out);
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
        check_bounds(cpu_out);
        check_bounds(metal_out);
    }
    log_stats("Softmax", run_max_errors);
    EXPECT_LT(*std::max_element(run_max_errors.begin(), run_max_errors.end()), 2.0);  // diagnostic guard only
    EXPECT_LT(*std::max_element(run_mean_errors.begin(), run_mean_errors.end()), 0.1);
}
