// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <vector>

#include "single_layer_tests/pooling.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

/* ============= 1D MaxPooling ============= */
const std::vector<std::vector<size_t>> kernels1d   = {{3}, {5}};
const std::vector<std::vector<size_t>> strides1d   = {{1}, {2}};
const std::vector<std::vector<size_t>> padBegins1d = {{0}, {1}, {2}};
const std::vector<std::vector<size_t>> padEnds1d   = {{0}, {1}, {2}};

const auto maxPool1D_FloorRounding_Params = ::testing::Combine(
        ::testing::Values(ngraph::helpers::PoolingTypes::MAX),
        ::testing::ValuesIn(kernels1d),
        ::testing::ValuesIn(strides1d),
        ::testing::ValuesIn(padBegins1d),
        ::testing::ValuesIn(padEnds1d),
        ::testing::Values(ngraph::op::RoundingType::FLOOR),
        ::testing::Values(ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID),
        ::testing::Values(false) // exclude pad not applicable for max pooling
);

INSTANTIATE_TEST_CASE_P(MaxPool1D_FloorRounding, PoolingLayerTest,
                        ::testing::Combine(
                                maxPool1D_FloorRounding_Params,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t>({1, 3, 30})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        PoolingLayerTest::getTestCaseName);

const auto maxPool1D_CeilRounding_Params = ::testing::Combine(
        // Non 1 strides fails in ngraph reference implementation with error "The end corner is out of bounds at axis 3" thrown in the test body.
        ::testing::Values(ngraph::helpers::PoolingTypes::MAX),
        ::testing::ValuesIn(kernels1d),
        ::testing::Values(std::vector<size_t>({1})),
        ::testing::ValuesIn(padBegins1d),
        ::testing::ValuesIn(padEnds1d),
        ::testing::Values(ngraph::op::RoundingType::CEIL),
        ::testing::Values(ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID),
        ::testing::Values(false) // exclude pad not applicable for max pooling
);

INSTANTIATE_TEST_CASE_P(MaxPool1D_CeilRounding, PoolingLayerTest,
                        ::testing::Combine(
                                maxPool1D_CeilRounding_Params,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t>({1, 3, 30})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        PoolingLayerTest::getTestCaseName);

//* ========== 1D AvgPooling ========== */
const auto AvgPool1D_FloorRounding_Params = ::testing::Combine(
        ::testing::Values(ngraph::helpers::PoolingTypes::AVG),
        ::testing::ValuesIn(kernels1d),
        ::testing::ValuesIn(strides1d),
        ::testing::ValuesIn(padBegins1d),
        ::testing::ValuesIn(padEnds1d),
        ::testing::Values(ngraph::op::RoundingType::FLOOR),
        ::testing::Values(ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID),
        ::testing::Values(true, false)
);

INSTANTIATE_TEST_CASE_P(AvgPool1D_FloorRounding, PoolingLayerTest,
                        ::testing::Combine(
                                AvgPool1D_FloorRounding_Params,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({1, 3, 30})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        PoolingLayerTest::getTestCaseName);

const auto avgPool1D_CeilRounding_Params = ::testing::Combine(
        // Non 1 strides fails in ngraph reference implementation with error "The end corner is out of bounds at axis 3" thrown in the test body.
        ::testing::Values(ngraph::helpers::PoolingTypes::AVG),
        ::testing::ValuesIn(kernels1d),
        ::testing::Values(std::vector<size_t>({1})),
        ::testing::ValuesIn(padBegins1d),
        ::testing::ValuesIn(padEnds1d),
        ::testing::Values(ngraph::op::RoundingType::CEIL),
        ::testing::Values(ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID),
        ::testing::Values(true, false)
);

INSTANTIATE_TEST_CASE_P(AvgPool1D_CeilRounding, PoolingLayerTest,
                        ::testing::Combine(
                                avgPool1D_CeilRounding_Params,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({1, 3, 30})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        PoolingLayerTest::getTestCaseName);

/* ============= 2D MaxPooling ============= */
const std::vector<std::vector<size_t>> kernels   = {{3, 3}, {3, 5}};
const std::vector<std::vector<size_t>> strides   = {{1, 1}, {2, 2}};
const std::vector<std::vector<size_t>> padBegins = {{0, 0}, {0, 1}, {1, 1}};
const std::vector<std::vector<size_t>> padEnds   = {{0, 0}, {1, 0}, {1, 1}};

const auto maxPool2D_FloorRounding_Params = ::testing::Combine(
        ::testing::Values(ngraph::helpers::PoolingTypes::MAX),
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::Values(ngraph::op::RoundingType::FLOOR),
        ::testing::Values(ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID),
        ::testing::Values(false) // exclude pad not applicable for max pooling
);

INSTANTIATE_TEST_CASE_P(MaxPool2D_FloorRounding, PoolingLayerTest,
                        ::testing::Combine(
                                maxPool2D_FloorRounding_Params,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        PoolingLayerTest::getTestCaseName);

const auto maxPool2D_CeilRounding_Params = ::testing::Combine(
        // Non 1 strides fails in ngraph reference implementation with error "The end corner is out of bounds at axis 3" thrown in the test body.
        ::testing::Values(ngraph::helpers::PoolingTypes::MAX),
        ::testing::ValuesIn(kernels),
        ::testing::Values(std::vector<size_t>({1, 1})),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::Values(ngraph::op::RoundingType::CEIL),
        ::testing::Values(ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID),
        ::testing::Values(false) // exclude pad not applicable for max pooling
);

INSTANTIATE_TEST_CASE_P(MaxPool2D_CeilRounding, PoolingLayerTest,
                        ::testing::Combine(
                                maxPool2D_CeilRounding_Params,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        PoolingLayerTest::getTestCaseName);

//* ========== 2D AvgPooling ========== */
const auto AvgPool2D_FloorRounding_Params = ::testing::Combine(
        ::testing::Values(ngraph::helpers::PoolingTypes::AVG),
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::Values(ngraph::op::RoundingType::FLOOR),
        ::testing::Values(ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID),
        ::testing::Values(true, false)
);

INSTANTIATE_TEST_CASE_P(AvgPool2D_FloorRounding, PoolingLayerTest,
                        ::testing::Combine(
                                AvgPool2D_FloorRounding_Params,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({1, 3, 30, 30})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        PoolingLayerTest::getTestCaseName);

const auto avgPool2D_CeilRounding_Params = ::testing::Combine(
        // Non 1 strides fails in ngraph reference implementation with error "The end corner is out of bounds at axis 3" thrown in the test body.
        ::testing::Values(ngraph::helpers::PoolingTypes::AVG),
        ::testing::ValuesIn(kernels),
        ::testing::Values(std::vector<size_t>({1, 1})),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::Values(ngraph::op::RoundingType::CEIL),
        ::testing::Values(ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID),
        ::testing::Values(true, false)
);

INSTANTIATE_TEST_CASE_P(AvgPool2D_CeilRounding, PoolingLayerTest,
                        ::testing::Combine(
                                avgPool2D_CeilRounding_Params,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({1, 3, 30, 30})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        PoolingLayerTest::getTestCaseName);
}  // namespace
