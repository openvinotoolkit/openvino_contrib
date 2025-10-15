// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <limits>
#include <optional>
#include <type_traits>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

namespace details {
// A workaround for an MSVC 19 bug, complaining 'fpclassify': ambiguous call
template <typename T>
constexpr bool equal_infs(const T &, const T &b) {
    static_assert(std::is_integral_v<T>, "Default implementation is valid for integer types only");
    return false;
}

template <>
inline bool equal_infs<float>(const float &a, const float &b) {
    return std::isinf(a) && std::isinf(b) && ((a > 0) == (b > 0));
}

template <>
inline bool equal_infs<double>(const double &a, const double &b) {
    return std::isinf(a) && std::isinf(b) && ((a > 0) == (b > 0));
}

template <>
inline bool equal_infs<ov::float16>(const ov::float16 &a, const ov::float16 &b) {
    return equal_infs<float>(a, b);  // Explicit conversion to floats
}

template <>
inline bool equal_infs<ov::bfloat16>(const ov::bfloat16 &a, const ov::bfloat16 &b) {
    return equal_infs<float>(a, b);  // Explicit conversion to floats
}

template <typename T1, typename T2>
inline bool equal_infs(T1 a, T2 b) {
    // If one of the types is intergral it's value couldn't be an inf
    if constexpr (std::is_integral_v<T1> || std::is_integral_v<T2>) {
        return false;
    } else if constexpr (std::is_same_v<T1, double> || std::is_same_v<T2, double>) {
        return equal_infs<double>(static_cast<double>(a), static_cast<double>(b));
    } else if constexpr (std::is_same_v<T1, float> || std::is_same_v<T2, float> || std::is_same_v<T1, ov::float16> ||
                         std::is_same_v<T2, ov::float16> || std::is_same_v<T1, ov::bfloat16> ||
                         std::is_same_v<T2, ov::bfloat16>) {
        return equal_infs<float>(static_cast<float>(a), static_cast<float>(b));
    }
    return equal_infs<double>(static_cast<double>(a), static_cast<double>(b));
}

template <typename T>
constexpr T conv_infs(const T &val, const T &threshold, const T &infinity_value) {
    if constexpr (std::is_floating_point_v<T>) {
        if ((val + threshold) >= infinity_value || (val - threshold) <= -infinity_value) {
            return val > 0 ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
        }
    }
    return val;
}

template <typename T>
constexpr bool equal_nans(T, T) {
    static_assert(std::is_integral_v<T>, "Default implementation is valid for integer types only");
    return false;
}
template <>
inline bool equal_nans<float>(float a, float b) {
    return std::isnan(a) && std::isnan(b);
}
template <>
inline bool equal_nans<double>(double a, double b) {
    return std::isnan(a) && std::isnan(b);
}
template <>
inline bool equal_nans<ov::float16>(ov::float16 a, ov::float16 b) {
    return equal_nans(static_cast<float>(a), static_cast<float>(b));
}
template <>
inline bool equal_nans<ov::bfloat16>(ov::bfloat16 a, ov::bfloat16 b) {
    return equal_nans(static_cast<float>(a), static_cast<float>(b));
}

template <typename T1, typename T2>
inline bool equal_nans(T1 a, T2 b) {
    // If one of the types is intergral it's value couldn't be nan
    if constexpr (std::is_integral_v<T1> || std::is_integral_v<T2>) {
        return false;
    } else if constexpr (std::is_same_v<T1, double> || std::is_same_v<T2, double>) {
        return equal_nans<double>(static_cast<double>(a), static_cast<double>(b));
    } else if constexpr (std::is_same_v<T1, float> || std::is_same_v<T2, float> || std::is_same_v<T1, ov::float16> ||
                         std::is_same_v<T2, ov::float16> || std::is_same_v<T1, ov::bfloat16> ||
                         std::is_same_v<T2, ov::bfloat16>) {
        return equal_nans<float>(static_cast<float>(a), static_cast<float>(b));
    }
    return equal_nans<double>(static_cast<double>(a), static_cast<double>(b));
}

}  // namespace details

using namespace ov::test::utils;
using ov::test::SubgraphBaseTest;

/**
 * @brief This class implements the logics of the correct comparison of infinity and nan values on CUDA
 * for the floating point tensors.
 * It is based on the copies of compare() member functions of SubgraphBaseTest class with slight
 * differences only. This approach had to be taken because of the fact that function  where logic had to be changed and
 * functions calling it are static but not virtual.
 * Please pay attention to the future updates in the base SubgraphBaseTest class and update this class
 * correspondingly.
 */
class FiniteLayerComparer : virtual public SubgraphBaseTest {
public:
    static void compare(const std::vector<ov::Tensor> &expected,
                        const std::vector<ov::Tensor> &actual,
                        float threshold,
                        bool to_check_nans,
                        std::optional<double> infinity_value);

    static void compare(const ov::Tensor &expected,
                        const ov::Tensor &actual,
                        float threshold,
                        bool to_check_nans,
                        std::optional<double> infinity_value);

    void compare(const std::vector<ov::Tensor> &expected_outputs,
                 const std::vector<ov::Tensor> &actual_outputs) override;

    template <class T_IE, class T_NGRAPH>
    static void compare(const T_NGRAPH *expected,
                        const T_IE *actual,
                        std::size_t size,
                        float threshold,
                        bool to_check_nans,
                        std::optional<double> infinity_value) {
        for (std::size_t i = 0; i < size; ++i) {
            const T_NGRAPH &ref =
                infinity_value ? details::conv_infs<T_NGRAPH>(expected[i],
                                                              static_cast<T_NGRAPH>(threshold),
                                                              std::fabs(static_cast<T_NGRAPH>(infinity_value.value())))
                               : expected[i];
            const T_IE &res =
                infinity_value
                    ? details::conv_infs<T_IE>(
                          actual[i], static_cast<T_IE>(threshold), std::fabs(static_cast<T_IE>(infinity_value.value())))
                    : actual[i];
            if (details::equal_infs(ref, res)) {
                continue;
            }
            if (to_check_nans && details::equal_nans(ref, res)) {
                continue;
            }
            const auto absolute_difference = ie_abs(res - ref);
            if (absolute_difference <= threshold) {
                continue;
            }
            double max;
            if (sizeof(T_IE) < sizeof(T_NGRAPH)) {
                max = std::max(ie_abs(T_NGRAPH(res)), ie_abs(ref));
            } else {
                max = std::max(ie_abs(res), ie_abs(T_IE(ref)));
            }
            double diff = static_cast<float>(absolute_difference) / max;
            if (max == 0 || (diff > static_cast<float>(threshold)) || std::isnan(static_cast<float>(res)) ||
                std::isnan(static_cast<float>(ref))) {
                OPENVINO_THROW("Relative comparison of values expected: " + std::to_string(ref) +
                               " and actual: " + std::to_string(res) +
                               " at index " + std::to_string(i) +
                               " with threshold " + std::to_string(threshold) +
                               " failed");
            }
        }
    }

protected:
    std::optional<double> infinity_value;
    bool to_check_nans = false;
};

template <typename BaseLayerTest>
class FiniteComparer : public BaseLayerTest, public FiniteLayerComparer {
    static_assert(std::is_base_of_v<SubgraphBaseTest, BaseLayerTest>,
                  "BaseLayerTest should inherit from ov::test::SubgraphBaseTest");
};
}  // namespace test
}  // namespace ov
