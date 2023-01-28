// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <common_test_utils/common_utils.hpp>
#include <limits>
#include <ngraph/type/bfloat16.hpp>
#include <ngraph/type/float16.hpp>
#include <optional>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <type_traits>

namespace LayerTestsDefinitions {

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
inline bool equal_infs<ngraph::float16>(const ngraph::float16 &a, const ngraph::float16 &b) {
    return equal_infs<float>(a, b);  // Explicit conversion to floats
}

template <>
inline bool equal_infs<ngraph::bfloat16>(const ngraph::bfloat16 &a, const ngraph::bfloat16 &b) {
    return equal_infs<float>(a, b);  // Explicit conversion to floats
}

template <typename T1, typename T2>
inline bool equal_infs(T1 a, T2 b) {
    // If one of the types is intergral it's value couldn't be an inf
    if constexpr (std::is_integral_v<T1> || std::is_integral_v<T2>) {
        return false;
    } else if constexpr (std::is_same_v<T1, double> || std::is_same_v<T2, double>) {
        return equal_infs<double>(static_cast<double>(a), static_cast<double>(b));
    } else if constexpr (std::is_same_v<T1, float> || std::is_same_v<T2, float> ||
                         std::is_same_v<T1, ngraph::float16> || std::is_same_v<T2, ngraph::float16> ||
                         std::is_same_v<T1, ngraph::bfloat16> || std::is_same_v<T2, ngraph::bfloat16>) {
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
inline bool equal_nans<ngraph::float16>(ngraph::float16 a, ngraph::float16 b) {
    return equal_nans(static_cast<float>(a), static_cast<float>(b));
}
template <>
inline bool equal_nans<ngraph::bfloat16>(ngraph::bfloat16 a, ngraph::bfloat16 b) {
    return equal_nans(static_cast<float>(a), static_cast<float>(b));
}

template <typename T1, typename T2>
inline bool equal_nans(T1 a, T2 b) {
    // If one of the types is intergral it's value couldn't be nan
    if constexpr (std::is_integral_v<T1> || std::is_integral_v<T2>) {
        return false;
    } else if constexpr (std::is_same_v<T1, double> || std::is_same_v<T2, double>) {
        return equal_nans<double>(static_cast<double>(a), static_cast<double>(b));
    } else if constexpr (std::is_same_v<T1, float> || std::is_same_v<T2, float> ||
                         std::is_same_v<T1, ngraph::float16> || std::is_same_v<T2, ngraph::float16> ||
                         std::is_same_v<T1, ngraph::bfloat16> || std::is_same_v<T2, ngraph::bfloat16>) {
        return equal_nans<float>(static_cast<float>(a), static_cast<float>(b));
    }
    return equal_nans<double>(static_cast<double>(a), static_cast<double>(b));
}

}  // namespace details

/**
 * @brief This class implements the logics of the correct comparison of infinity and nan values on CUDA
 * for the floating point tensors.
 * It is based on the copies of Compare() member functions of LayerTestsUtils::LayerTestsCommon class with slight
 * differences only. This approach had to be taken because of the fact that function  where logic had to be changed and
 * functions calling it are static but not virtual.
 * Please pay attention to the future updates in the base LayerTestsUtils::LayerTestsCommon class and update this class
 * correspondingly.
 */
class FiniteLayerComparer : virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expected,
                        const std::vector<InferenceEngine::Blob::Ptr> &actual,
                        float threshold,
                        bool to_check_nans,
                        std::optional<double> infinity_value);

    static void Compare(const std::pair<ngraph::element::Type, std::vector<std::uint8_t>> &expected,
                        const InferenceEngine::Blob::Ptr &actual,
                        float threshold,
                        bool to_check_nans,
                        std::optional<double> infinity_value);

    void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
                 const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs) override;

    void Compare(const std::pair<ngraph::element::Type, std::vector<std::uint8_t>> &expected,
                 const InferenceEngine::Blob::Ptr &actual) override;

    void Compare(const InferenceEngine::Blob::Ptr &expected, const InferenceEngine::Blob::Ptr &actual) override;

    void Compare(const InferenceEngine::TensorDesc &actualDesc,
                 const InferenceEngine::TensorDesc &expectedDesc) override;

    template <class T_IE, class T_NGRAPH>
    static void Compare(const T_NGRAPH *expected,
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
            const auto absoluteDifference = CommonTestUtils::ie_abs(res - ref);
            if (absoluteDifference <= threshold) {
                continue;
            }
            double max;
            if (sizeof(T_IE) < sizeof(T_NGRAPH)) {
                max = std::max(CommonTestUtils::ie_abs(T_NGRAPH(res)), CommonTestUtils::ie_abs(ref));
            } else {
                max = std::max(CommonTestUtils::ie_abs(res), CommonTestUtils::ie_abs(T_IE(ref)));
            }
            double diff = static_cast<float>(absoluteDifference) / max;
            if (max == 0 || (diff > static_cast<float>(threshold)) || std::isnan(static_cast<float>(res)) ||
                std::isnan(static_cast<float>(ref))) {
                IE_THROW() << "Relative comparison of values expected: " << ref << " and actual: " << res
                           << " at index " << i << " with threshold " << threshold << " failed";
            }
        }
    }

protected:
    std::optional<double> infinity_value;
    bool to_check_nans = false;
};

template <typename BaseLayerTest>
class FiniteComparer : public BaseLayerTest, public FiniteLayerComparer {
    static_assert(std::is_base_of_v<LayerTestsUtils::LayerTestsCommon, BaseLayerTest>,
                  "BaseLayerTest should inherit from LayerTestsUtils::LayerTestsCommon");
};

}  // namespace LayerTestsDefinitions
