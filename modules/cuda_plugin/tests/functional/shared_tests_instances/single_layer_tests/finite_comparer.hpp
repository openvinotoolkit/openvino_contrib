// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

namespace details {
// A workaround for an MSVC 19 bug, complaining 'fpclassify': ambiguous call
template<typename T>
constexpr bool equal_infs(const T&, const T& b) {
  static_assert(std::is_integral_v<T>, "Default implementation is valid for integer types only");
  return false;
}
template<>
inline
bool equal_infs<float>(const float& a, const float& b) {
  return std::isinf(a) && std::isinf(b) && ((a > 0) == (b > 0));
}
template<>
inline
bool equal_infs<double>(const double& a, const double& b) {
  return std::isinf(a) && std::isinf(b) && ((a > 0) == (b > 0));
}
template<>
inline
bool equal_infs<ngraph::float16>(const ngraph::float16& a, const ngraph::float16& b) {
  return equal_infs<float>(a, b); // Explicit conversion to floats
}
template<>
inline
bool equal_infs<ngraph::bfloat16>(const ngraph::bfloat16& a, const ngraph::bfloat16& b) {
  return equal_infs<float>(a, b); // Explicit conversion to floats
}
template<typename T>
constexpr T conv_infs(const T& val, const T& threshold, const T& infinityValue) {
  if constexpr (std::is_floating_point_v<T>) {
    if ((val+threshold) >= infinityValue || (val-threshold) <= -infinityValue) {
      return val > 0 ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
    }
  }
  return val;
}
} //namespace details

class FiniteLayerComparer : virtual public LayerTestsUtils::LayerTestsCommon {
  template<class T>
  static void Compare(const T *expected, const T *actual, std::size_t size,
                      T threshold, const std::optional<T>& infinityValue) {
    for (std::size_t i = 0; i < size; ++i) {
      const auto &ref = infinityValue
                        ? details::conv_infs<T>(expected[i], threshold, std::fabs(infinityValue.value()))
                        : expected[i];
      const auto &res = infinityValue
                        ? details::conv_infs<T>(actual[i], threshold, std::fabs(infinityValue.value()))
                        : actual[i];
      if (details::equal_infs<T>(ref, res)) {
        continue;
      }
      const auto absoluteDifference = CommonTestUtils::ie_abs(res - ref);
      if (absoluteDifference <= threshold) {
        continue;
      }

      const auto max = std::max(CommonTestUtils::ie_abs(res), CommonTestUtils::ie_abs(ref));
      float diff = static_cast<float>(absoluteDifference) / static_cast<float>(max);
      ASSERT_TRUE(max != 0 && (diff <= static_cast<float>(threshold)))
                    << "Relative comparison of values expected: " << ref << " and actual: " << res
                    << " at index " << i << " with threshold " << threshold
                    << " failed";
    }
  }

  void Compare(const std::vector<std::uint8_t> &expected, const InferenceEngine::Blob::Ptr &actual) override {
    ASSERT_EQ(expected.size(), actual->byteSize());
    const auto &expectedBuffer = expected.data();

    auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
    IE_ASSERT(memory);
    const auto lockedMemory = memory->wmap();
    const auto actualBuffer = lockedMemory.as<const std::uint8_t *>();

    const auto &precision = actual->getTensorDesc().getPrecision();
    const auto &size = actual->size();
    switch (precision) {
      case InferenceEngine::Precision::FP32:
        Compare<float>(reinterpret_cast<const float *>(expectedBuffer),
                       reinterpret_cast<const float *>(actualBuffer), size,
                       this->threshold, this->infinity_value);
        break;
      case InferenceEngine::Precision::I32:
        Compare<int32_t>(reinterpret_cast<const int32_t *>(expectedBuffer),
                         reinterpret_cast<const int32_t *>(actualBuffer), size,
                         0, std::nullopt);
        break;
      case InferenceEngine::Precision::I64:
        Compare<int64_t>(reinterpret_cast<const int64_t *>(expectedBuffer),
                         reinterpret_cast<const int64_t *>(actualBuffer), size,
                         0, std::nullopt);
        break;
      case InferenceEngine::Precision::I8:
        Compare<int8_t>(reinterpret_cast<const int8_t *>(expectedBuffer),
                        reinterpret_cast<const int8_t *>(actualBuffer), size,
                        0, std::nullopt);
        break;
      case InferenceEngine::Precision::U16:
        Compare<uint16_t>(reinterpret_cast<const uint16_t *>(expectedBuffer),
                          reinterpret_cast<const uint16_t *>(actualBuffer), size,
                          0, std::nullopt);
        break;
      case InferenceEngine::Precision::I16:
        Compare<int16_t>(reinterpret_cast<const int16_t *>(expectedBuffer),
                         reinterpret_cast<const int16_t *>(actualBuffer), size,
                         0, std::nullopt);
        break;
      case InferenceEngine::Precision::BOOL:
      case InferenceEngine::Precision::U8:
        Compare<uint8_t>(reinterpret_cast<const uint8_t *>(expectedBuffer),
                         reinterpret_cast<const uint8_t *>(actualBuffer), size,
                         0, std::nullopt);
        break;
      case InferenceEngine::Precision::U64:
        Compare<uint64_t>(reinterpret_cast<const uint64_t *>(expectedBuffer),
                          reinterpret_cast<const uint64_t *>(actualBuffer), size,
                          0, std::nullopt);
        break;
      case InferenceEngine::Precision::BF16:
        Compare(reinterpret_cast<const ngraph::bfloat16 *>(expectedBuffer),
                reinterpret_cast<const ngraph::bfloat16 *>(actualBuffer), size,
                ngraph::bfloat16(this->threshold), std::optional<ngraph::bfloat16>{std::nullopt});
        break;
      case InferenceEngine::Precision::FP16:
        Compare(reinterpret_cast<const ngraph::float16 *>(expectedBuffer),
                reinterpret_cast<const ngraph::float16 *>(actualBuffer), size,
                ngraph::float16(this->threshold), std::optional<ngraph::float16>{std::nullopt});
        break;
      default:
        FAIL() << "Comparator for " << precision << " precision isn't supported";
    }
  }

 protected:
  std::optional<float> infinity_value;
};

template <typename BaseLayerTest>
class FiniteComparer : public BaseLayerTest
                     , public FiniteLayerComparer {
  static_assert(std::is_base_of<LayerTestsUtils::LayerTestsCommon, BaseLayerTest>::value,
                "BaseLayerTest should inherit from LayerTestsUtils::LayerTestsCommon");
};

}