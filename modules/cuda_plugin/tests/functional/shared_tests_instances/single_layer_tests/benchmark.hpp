// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

namespace details {
// A workaround for an MSVC 19 bug, complaining 'fpclassify': ambiguous call
template <typename T>
constexpr bool equal_infs(const T&, const T& b) {
    static_assert(std::is_integral_v<T>, "Default implementation is valid for integer types only");
    return false;
}
template <>
inline bool equal_infs<float>(const float& a, const float& b) {
    return std::isinf(a) && std::isinf(b) && ((a > 0) == (b > 0));
}
template <>
inline bool equal_infs<double>(const double& a, const double& b) {
    return std::isinf(a) && std::isinf(b) && ((a > 0) == (b > 0));
}
template <>
inline bool equal_infs<ngraph::float16>(const ngraph::float16& a, const ngraph::float16& b) {
    return equal_infs<float>(a, b);  // Explicit conversion to floats
}
template <>
inline bool equal_infs<ngraph::bfloat16>(const ngraph::bfloat16& a, const ngraph::bfloat16& b) {
    return equal_infs<float>(a, b);  // Explicit conversion to floats
}
template <typename T>
constexpr T conv_infs(const T& val, const T& threshold, const T& infinityValue) {
    if constexpr (std::is_floating_point_v<T>) {
        if ((val + threshold) >= infinityValue || (val - threshold) <= -infinityValue) {
            return val > 0 ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
        }
    }
    return val;
}
}  // namespace details

template <typename BaseLayerTest>
class BenchmarkLayerTest : public BaseLayerTest, virtual public LayerTestsUtils::LayerTestsCommon {
    static_assert(std::is_base_of<LayerTestsUtils::LayerTestsCommon, BaseLayerTest>::value,
                  "BaseLayerTest should inherit from LayerTestsUtils::LayerTestsCommon");

public:
    void Run(const std::string& name, const std::chrono::milliseconds warmupTime = 2000, const int numAttempts = 100) {
        bench_name_ = name;
        warmup_time_ = warmupTime;
        num_attempts_ = numAttempts;
        LayerTestsUtils::LayerTestsCommon::Run();
    }

    void Validate() override {
        // NOTE: Validation is ignored because we are interested in benchmarks results
    }

protected:
    void Infer() override {
        // Warmup
        auto warmCur = std::chrono::steady_clock::now();
        const auto warmEnd = warmCur + warmup_time_;
        while (warmCur < warmEnd) {
            LayerTestsUtils::LayerTestsCommon::Infer();
            warmCur = std::chrono::steady_clock::now();
        }

        // Benchmark
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < num_attempts_; ++i) {
            LayerTestsUtils::LayerTestsCommon::Infer();
        }
        const auto end = std::chrono::steady_clock::now();

        const auto averageMicroExecTime =
            std::chrono::duration_cast<std::chrono::microseconds>((end - start) / num_attempts_);
        const auto averageMilliExecTime = std::chrono::duration_cast<std::chrono::milliseconds>(averageMicroExecTime);
        std::cout << std::fixed << std::setfill('0') << bench_name_ << ": " << averageMicroExecTime.count() << " us\n";
        std::cout << std::fixed << std::setfill('0') << bench_name_ << ": " << averageMilliExecTime.count() << " ms\n";
    }

private:
    std::string bench_name_;
    std::chrono::milliseconds warmup_time_;
    int num_attempts_;
};

}  // namespace LayerTestsDefinitions
