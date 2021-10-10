// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

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
    void Run(const std::string& name, const std::chrono::milliseconds warmupTime, const int numAttempts) {
        // Warmup
        auto warm_cur = std::chrono::steady_clock::now();
        const auto warm_end = warm_cur + warmupTime;
        while (warm_cur < warm_end) {
            LayerTestsUtils::LayerTestsCommon::Run();
            warm_cur = std::chrono::steady_clock::now();
        }

        // Benchmark
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < numAttempts; ++i) {
            LayerTestsUtils::LayerTestsCommon::Run();
        }
        const auto end = std::chrono::steady_clock::now();

        auto average_us_exec_time = std::chrono::duration_cast<std::chrono::microseconds>((end - start) / numAttempts);
        auto average_ms_exec_time = std::chrono::duration_cast<std::chrono::milliseconds>(average_us_exec_time);
        std::cout << std::fixed << std::setfill('0') << name << ": " << average_us_exec_time.count() << " us\n";
        std::cout << std::fixed << std::setfill('0') << name << ": " << average_ms_exec_time.count() << " ms\n";
    }

    void Validate() override {
        // NOTE: Validation is ignored because we are interested in benchmarks results
    }
};

}  // namespace LayerTestsDefinitions
