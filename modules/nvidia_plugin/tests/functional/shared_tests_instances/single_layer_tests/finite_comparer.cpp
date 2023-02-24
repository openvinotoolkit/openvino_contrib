// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "finite_comparer.hpp"

#include <ie_blob.h>

#include <ie_precision.hpp>
#include <ngraph/type/element_type.hpp>

namespace LayerTestsDefinitions {

void FiniteLayerComparer::Compare(
    const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
    const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs,
    float threshold,
    bool to_check_nans,
    std::optional<double> infinity_value) {
    for (std::size_t outputIndex = 0; outputIndex < expectedOutputs.size(); ++outputIndex) {
        const auto &expected = expectedOutputs[outputIndex];
        const auto &actual = actualOutputs[outputIndex];
        FiniteLayerComparer::Compare(expected, actual, threshold, to_check_nans, infinity_value);
    }
}

template <typename T_IE>
inline void callCompare(const std::pair<ngraph::element::Type, std::vector<std::uint8_t>> &expected,
                        const T_IE *actualBuffer,
                        size_t size,
                        float threshold,
                        bool to_check_nans,
                        std::optional<double> infinity_value) {
    auto expectedBuffer = expected.second.data();
    switch (expected.first) {
        case ngraph::element::Type_t::i64:
            FiniteLayerComparer::Compare<T_IE, int64_t>(reinterpret_cast<const int64_t *>(expectedBuffer),
                                                        actualBuffer,
                                                        size,
                                                        threshold,
                                                        to_check_nans,
                                                        infinity_value);
            break;
        case ngraph::element::Type_t::i32:
            FiniteLayerComparer::Compare<T_IE, int32_t>(reinterpret_cast<const int32_t *>(expectedBuffer),
                                                        actualBuffer,
                                                        size,
                                                        threshold,
                                                        to_check_nans,
                                                        infinity_value);
            break;
        case ngraph::element::Type_t::i16:
            FiniteLayerComparer::Compare<T_IE, int16_t>(reinterpret_cast<const int16_t *>(expectedBuffer),
                                                        actualBuffer,
                                                        size,
                                                        threshold,
                                                        to_check_nans,
                                                        infinity_value);
            break;
        case ngraph::element::Type_t::i8:
            FiniteLayerComparer::Compare<T_IE, int8_t>(reinterpret_cast<const int8_t *>(expectedBuffer),
                                                       actualBuffer,
                                                       size,
                                                       threshold,
                                                       to_check_nans,
                                                       infinity_value);
            break;
        case ngraph::element::Type_t::u64:
            FiniteLayerComparer::Compare<T_IE, uint64_t>(reinterpret_cast<const uint64_t *>(expectedBuffer),
                                                         actualBuffer,
                                                         size,
                                                         threshold,
                                                         to_check_nans,
                                                         infinity_value);
            break;
        case ngraph::element::Type_t::u32:
            FiniteLayerComparer::Compare<T_IE, uint32_t>(reinterpret_cast<const uint32_t *>(expectedBuffer),
                                                         actualBuffer,
                                                         size,
                                                         threshold,
                                                         to_check_nans,
                                                         infinity_value);
            break;
        case ngraph::element::Type_t::u16:
            FiniteLayerComparer::Compare<T_IE, uint16_t>(reinterpret_cast<const uint16_t *>(expectedBuffer),
                                                         actualBuffer,
                                                         size,
                                                         threshold,
                                                         to_check_nans,
                                                         infinity_value);
            break;
        case ngraph::element::Type_t::boolean:
        case ngraph::element::Type_t::u8:
            FiniteLayerComparer::Compare<T_IE, uint8_t>(reinterpret_cast<const uint8_t *>(expectedBuffer),
                                                        actualBuffer,
                                                        size,
                                                        threshold,
                                                        to_check_nans,
                                                        infinity_value);
            break;
        case ngraph::element::Type_t::f64:
            FiniteLayerComparer::Compare<T_IE, double>(reinterpret_cast<const double *>(expectedBuffer),
                                                       actualBuffer,
                                                       size,
                                                       threshold,
                                                       to_check_nans,
                                                       infinity_value);
            break;
        case ngraph::element::Type_t::f32:
            FiniteLayerComparer::Compare<T_IE, float>(reinterpret_cast<const float *>(expectedBuffer),
                                                      actualBuffer,
                                                      size,
                                                      threshold,
                                                      to_check_nans,
                                                      infinity_value);
            break;
        case ngraph::element::Type_t::f16:
            FiniteLayerComparer::Compare<T_IE, ngraph::float16>(
                reinterpret_cast<const ngraph::float16 *>(expectedBuffer),
                actualBuffer,
                size,
                threshold,
                to_check_nans,
                infinity_value);
            break;
        case ngraph::element::Type_t::bf16:
            FiniteLayerComparer::Compare<T_IE, ngraph::bfloat16>(
                reinterpret_cast<const ngraph::bfloat16 *>(expectedBuffer),
                actualBuffer,
                size,
                threshold,
                to_check_nans,
                infinity_value);
            break;
        case ngraph::element::Type_t::i4: {
            auto expectedOut = ngraph::helpers::convertOutputPrecision(
                expected.second, expected.first, ngraph::element::Type_t::i8, size);
            FiniteLayerComparer::Compare<T_IE, int8_t>(reinterpret_cast<const int8_t *>(expectedOut.data()),
                                                       actualBuffer,
                                                       size,
                                                       threshold,
                                                       to_check_nans,
                                                       infinity_value);
            break;
        }
        case ngraph::element::Type_t::u4: {
            auto expectedOut = ngraph::helpers::convertOutputPrecision(
                expected.second, expected.first, ngraph::element::Type_t::u8, size);
            FiniteLayerComparer::Compare<T_IE, uint8_t>(reinterpret_cast<const uint8_t *>(expectedOut.data()),
                                                        actualBuffer,
                                                        size,
                                                        threshold,
                                                        to_check_nans,
                                                        infinity_value);
            break;
        }
        case ngraph::element::Type_t::dynamic:
        case ngraph::element::Type_t::undefined:
            FiniteLayerComparer::Compare<T_IE, T_IE>(reinterpret_cast<const T_IE *>(expectedBuffer),
                                                     actualBuffer,
                                                     size,
                                                     threshold,
                                                     to_check_nans,
                                                     infinity_value);
            break;
        default:
            FAIL() << "Comparator for " << expected.first << " precision isn't supported";
    }
    return;
}

void FiniteLayerComparer::Compare(const std::pair<ngraph::element::Type, std::vector<std::uint8_t>> &expected,
                                  const InferenceEngine::Blob::Ptr &actual,
                                  float threshold,
                                  bool to_check_nans,
                                  std::optional<double> infinity_value) {
    const auto &precision = actual->getTensorDesc().getPrecision();
    auto k = static_cast<float>(expected.first.size()) / precision.size();
    // W/A for int4, uint4
    if (expected.first == ngraph::element::Type_t::u4 || expected.first == ngraph::element::Type_t::i4) {
        k /= 2;
    } else if (expected.first == ngraph::element::Type_t::undefined ||
               expected.first == ngraph::element::Type_t::dynamic) {
        k = 1;
    }
    ASSERT_EQ(expected.second.size(), actual->byteSize() * k);

    auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
    IE_ASSERT(memory);
    const auto lockedMemory = memory->wmap();
    const auto actualBuffer = lockedMemory.as<const std::uint8_t *>();

    const auto &size = actual->size();
    switch (precision) {
        case InferenceEngine::Precision::FP32:
            callCompare<float>(expected,
                               reinterpret_cast<const float *>(actualBuffer),
                               size,
                               threshold,
                               to_check_nans,
                               infinity_value);
            break;
        case InferenceEngine::Precision::I32:
            callCompare<int32_t>(expected,
                                 reinterpret_cast<const int32_t *>(actualBuffer),
                                 size,
                                 threshold,
                                 to_check_nans,
                                 infinity_value);
            break;
        case InferenceEngine::Precision::U32:
            callCompare<uint32_t>(expected,
                                  reinterpret_cast<const uint32_t *>(actualBuffer),
                                  size,
                                  threshold,
                                  to_check_nans,
                                  infinity_value);
            break;
        case InferenceEngine::Precision::I64:
            callCompare<int64_t>(expected,
                                 reinterpret_cast<const int64_t *>(actualBuffer),
                                 size,
                                 threshold,
                                 to_check_nans,
                                 infinity_value);
            break;
        case InferenceEngine::Precision::I8:
            callCompare<int8_t>(expected,
                                reinterpret_cast<const int8_t *>(actualBuffer),
                                size,
                                threshold,
                                to_check_nans,
                                infinity_value);
            break;
        case InferenceEngine::Precision::U16:
            callCompare<uint16_t>(expected,
                                  reinterpret_cast<const uint16_t *>(actualBuffer),
                                  size,
                                  threshold,
                                  to_check_nans,
                                  infinity_value);
            break;
        case InferenceEngine::Precision::I16:
            callCompare<int16_t>(expected,
                                 reinterpret_cast<const int16_t *>(actualBuffer),
                                 size,
                                 threshold,
                                 to_check_nans,
                                 infinity_value);
            break;
        case InferenceEngine::Precision::BOOL:
        case InferenceEngine::Precision::U8:
            callCompare<uint8_t>(expected,
                                 reinterpret_cast<const uint8_t *>(actualBuffer),
                                 size,
                                 threshold,
                                 to_check_nans,
                                 infinity_value);
            break;
        case InferenceEngine::Precision::U64:
            callCompare<uint64_t>(expected,
                                  reinterpret_cast<const uint64_t *>(actualBuffer),
                                  size,
                                  threshold,
                                  to_check_nans,
                                  infinity_value);
            break;
        case InferenceEngine::Precision::BF16:
            callCompare<ngraph::bfloat16>(expected,
                                          reinterpret_cast<const ngraph::bfloat16 *>(actualBuffer),
                                          size,
                                          threshold,
                                          to_check_nans,
                                          infinity_value);
            break;
        case InferenceEngine::Precision::FP16:
            callCompare<ngraph::float16>(expected,
                                         reinterpret_cast<const ngraph::float16 *>(actualBuffer),
                                         size,
                                         threshold,
                                         to_check_nans,
                                         infinity_value);
            break;
        default:
            FAIL() << "Comparator for " << precision << " precision isn't supported";
    }
}

void FiniteLayerComparer::Compare(const std::pair<ngraph::element::Type, std::vector<std::uint8_t>> &expected,
                                  const InferenceEngine::Blob::Ptr &actual) {
    FiniteLayerComparer::Compare(expected, actual, threshold, this->to_check_nans, this->infinity_value);
}

void FiniteLayerComparer::Compare(const InferenceEngine::Blob::Ptr &expected,
                                  const InferenceEngine::Blob::Ptr &actual) {
    auto get_raw_buffer = [](const InferenceEngine::Blob::Ptr &blob) {
        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
        IE_ASSERT(memory);
        const auto lockedMemory = memory->wmap();
        return lockedMemory.as<const std::uint8_t *>();
    };
    const auto expectedBuffer = get_raw_buffer(expected);
    const auto actualBuffer = get_raw_buffer(actual);

    const auto &precision = actual->getTensorDesc().getPrecision();
    const auto &size = actual->size();
    switch (precision) {
        case InferenceEngine::Precision::FP32:
            FiniteLayerComparer::Compare(reinterpret_cast<const float *>(expectedBuffer),
                                         reinterpret_cast<const float *>(actualBuffer),
                                         size,
                                         threshold,
                                         this->to_check_nans,
                                         this->infinity_value);
            break;
        case InferenceEngine::Precision::I32:
            FiniteLayerComparer::Compare(reinterpret_cast<const std::int32_t *>(expectedBuffer),
                                         reinterpret_cast<const std::int32_t *>(actualBuffer),
                                         size,
                                         0,
                                         this->to_check_nans,
                                         this->infinity_value);
            break;
        default:
            FAIL() << "Comparator for " << precision << " precision isn't supported";
    }
}

void FiniteLayerComparer::Compare(const InferenceEngine::TensorDesc &actualDesc,
                                  const InferenceEngine::TensorDesc &expectedDesc) {
    LayerTestsCommon::Compare(actualDesc, expectedDesc);
}

void FiniteLayerComparer::Compare(
    const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
    const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs) {
    FiniteLayerComparer::Compare(expectedOutputs, actualOutputs, threshold, this->to_check_nans, this->infinity_value);
}

}  // namespace LayerTestsDefinitions
