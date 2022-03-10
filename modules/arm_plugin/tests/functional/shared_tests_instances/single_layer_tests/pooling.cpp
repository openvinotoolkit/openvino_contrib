// Copyright (C) 2020-2022 Intel Corporation
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

INSTANTIATE_TEST_CASE_P(smoke_MaxPool1D_FloorRounding, PoolingLayerTest,
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

INSTANTIATE_TEST_CASE_P(smoke_MaxPool1D_CeilRounding, PoolingLayerTest,
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

INSTANTIATE_TEST_CASE_P(smoke_AvgPool1D_FloorRounding, PoolingLayerTest,
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

INSTANTIATE_TEST_CASE_P(smoke_AvgPool1D_CeilRounding, PoolingLayerTest,
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
const std::vector<std::vector<size_t>> kernels2d   = {{3, 3}, {3, 5}};
const std::vector<std::vector<size_t>> strides2d   = {{1, 1}, {2, 2}};
const std::vector<std::vector<size_t>> padBegins2d = {{0, 0}, {0, 1}, {1, 1}};
const std::vector<std::vector<size_t>> padEnds2d   = {{0, 0}, {1, 0}, {1, 1}};

const auto maxPool2D_FloorRounding_Params = ::testing::Combine(
        ::testing::Values(ngraph::helpers::PoolingTypes::MAX),
        ::testing::ValuesIn(kernels2d),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::Values(ngraph::op::RoundingType::FLOOR),
        ::testing::Values(ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID),
        ::testing::Values(false) // exclude pad not applicable for max pooling
);

INSTANTIATE_TEST_CASE_P(smoke_MaxPool2D_FloorRounding, PoolingLayerTest,
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
        ::testing::ValuesIn(kernels2d),
        ::testing::Values(std::vector<size_t>({1, 1})),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::Values(ngraph::op::RoundingType::CEIL),
        ::testing::Values(ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID),
        ::testing::Values(false) // exclude pad not applicable for max pooling
);

INSTANTIATE_TEST_CASE_P(smoke_MaxPool2D_CeilRounding, PoolingLayerTest,
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
        ::testing::ValuesIn(kernels2d),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::Values(ngraph::op::RoundingType::FLOOR),
        ::testing::Values(ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID),
        ::testing::Values(true, false)
);

INSTANTIATE_TEST_CASE_P(smoke_AvgPool2D_FloorRounding, PoolingLayerTest,
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
        ::testing::ValuesIn(kernels2d),
        ::testing::Values(std::vector<size_t>({1, 1})),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::Values(ngraph::op::RoundingType::CEIL),
        ::testing::Values(ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID),
        ::testing::Values(true, false)
);

INSTANTIATE_TEST_CASE_P(smoke_AvgPool2D_CeilRounding, PoolingLayerTest,
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

/* ============= 3D MaxPooling ============= */
const std::vector<std::vector<size_t >> kernel3D = {{2, 2, 2}};
const std::vector<std::vector<size_t >> strides3D = {{1, 1, 1},
                                                          {2, 2, 2}};
const std::vector<std::vector<size_t >> stridess3D = {{2, 2, 2}};
const std::vector<std::vector<size_t >> padBegins3D = {{0, 0, 0}};
const std::vector<std::vector<size_t >> padEnds3D = {{0, 0, 0}};

/* ========== Explicit Pad Floor Rounding 5D input========== */
const auto maxPool3D_ExplicitPad_FloorRounding_Params = ::testing::Combine(
        ::testing::Values(ngraph::helpers::PoolingTypes::MAX),
        ::testing::ValuesIn(kernel3D),
        ::testing::ValuesIn(strides3D),
        ::testing::ValuesIn(padBegins3D),
        ::testing::ValuesIn(padEnds3D),
        ::testing::Values(ngraph::op::RoundingType::FLOOR),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::Values(false)  // placeholder value - exclude pad not applicable for max pooling
);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool3D_ExplicitPad_FloorRounding, PoolingLayerTest,
                        ::testing::Combine(
                                maxPool3D_ExplicitPad_FloorRounding_Params,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({32, 32, 2, 2, 2})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        PoolingLayerTest::getTestCaseName);

/* ========== Same Upper Pad Floor Rounding 5D input========== */
const auto maxPool3D_SameUpperPad_FloorRounding_Params = ::testing::Combine(
        ::testing::Values(ngraph::helpers::PoolingTypes::MAX),
        ::testing::ValuesIn(kernel3D),
        ::testing::ValuesIn(strides3D),
        ::testing::ValuesIn(padBegins3D),
        ::testing::ValuesIn(padEnds3D),
        ::testing::Values(ngraph::op::RoundingType::FLOOR),
        ::testing::Values(ngraph::op::PadType::SAME_UPPER),
        ::testing::Values(false)  // placeholder value - exclude pad not applicable for max pooling
);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool3D_SameUpperPad_FloorRounding, PoolingLayerTest,
                        ::testing::Combine(
                                maxPool3D_SameUpperPad_FloorRounding_Params,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({32, 32, 2, 2, 2})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        PoolingLayerTest::getTestCaseName);

/* ========== Same Lower Pad Ceil Rounding 5D input========== */
const auto maxPool3D_SameLowerPad_CeilRounding_Params = ::testing::Combine(
        ::testing::Values(ngraph::helpers::PoolingTypes::MAX),
        ::testing::ValuesIn(kernel3D),
        ::testing::ValuesIn(strides3D),
        ::testing::ValuesIn(padBegins3D),
        ::testing::ValuesIn(padEnds3D),
        ::testing::Values(ngraph::op::RoundingType::CEIL),
        ::testing::Values(ngraph::op::PadType::SAME_LOWER),
        ::testing::Values(false)  // placeholder value - exclude pad not applicable for max pooling
);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool3D_SameLowerPad_CeilRounding, PoolingLayerTest,
                        ::testing::Combine(
                                maxPool3D_SameLowerPad_CeilRounding_Params,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({32, 32, 2, 2, 2})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        PoolingLayerTest::getTestCaseName);

/* ============= 3D AvgPooling ============= */
/* ========== Explicit Pad Floor Rounding 5D input========== */
const auto avgPool_ExplicitPad_FloorRounding_5Dinput_Params = ::testing::Combine(
        ::testing::Values(ngraph::helpers::PoolingTypes::AVG),
        ::testing::ValuesIn(kernel3D),
        ::testing::ValuesIn(strides3D),
        ::testing::ValuesIn(padBegins3D),
        ::testing::ValuesIn(padEnds3D),
        ::testing::Values(ngraph::op::RoundingType::FLOOR),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::Values(true, false)
);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_ExplicitPad_FloorRounding_5Dinput, PoolingLayerTest,
                        ::testing::Combine(
                                avgPool_ExplicitPad_FloorRounding_5Dinput_Params,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({32, 32, 2, 2, 4})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        PoolingLayerTest::getTestCaseName);

/* ========== Same Upper Pad Floor Rounding 5D input========== */
const auto avgPool_SameUpperPad_FloorRounding_5Dinput_Params = ::testing::Combine(
        ::testing::Values(ngraph::helpers::PoolingTypes::AVG),
        ::testing::ValuesIn(kernel3D),
        ::testing::ValuesIn(strides3D),
        ::testing::ValuesIn(padBegins3D),
        ::testing::ValuesIn(padEnds3D),
        ::testing::Values(ngraph::op::RoundingType::FLOOR),
        ::testing::Values(ngraph::op::PadType::SAME_UPPER),
        ::testing::Values(true)
);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_SameUpperPad_FloorRounding_5Dinput, PoolingLayerTest,
                        ::testing::Combine(
                                avgPool_SameUpperPad_FloorRounding_5Dinput_Params,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({32, 32, 2, 2, 4})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        PoolingLayerTest::getTestCaseName);

/* ========== Same Lower Pad Ceil Rounding 5D input========== */
const auto avgPool_SameLowerPad_CeilRounding_5Dinput_Params = ::testing::Combine(
        ::testing::Values(ngraph::helpers::PoolingTypes::AVG),
        ::testing::ValuesIn(kernel3D),
        ::testing::ValuesIn(strides3D),
        ::testing::ValuesIn(padBegins3D),
        ::testing::ValuesIn(padEnds3D),
        ::testing::Values(ngraph::op::RoundingType::CEIL),
        ::testing::Values(ngraph::op::PadType::SAME_LOWER),
        ::testing::Values(true)
);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_SameLowerPad_CeilRounding_5Dinput, PoolingLayerTest,
                        ::testing::Combine(
                                avgPool_SameLowerPad_CeilRounding_5Dinput_Params,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({32, 32, 2, 2, 2})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        PoolingLayerTest::getTestCaseName);

//* ========== Max Pooling V8 ========== */

const std::vector<std::vector<size_t >> kernels = {{3, 3}, {3, 5}, {2, 2}};
const std::vector<std::vector<size_t>> strides = {{1, 1},
                                                  {1, 2},
                                                  {2, 1},
                                                  {2, 2}};
const std::vector<std::vector<size_t >> padBegins = {{0, 0}};
const std::vector<std::vector<size_t >> padEnds = {{0, 0}};
const std::vector<ngraph::op::RoundingType> roundingTypes = {ngraph::op::RoundingType::CEIL,
                                                             ngraph::op::RoundingType::FLOOR};

const std::vector<std::vector<size_t>> dilation = {{1, 1}, {2, 2}};
const std::vector<std::vector<size_t >> dilation3D = {{1, 1, 1}, {2, 2, 2}};

/* ========== Explicit Pad Floor Rounding ========== */
const auto maxPoolv8_ExplicitPad_FloorRounding_Params = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(dilation),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::Values(ngraph::element::Type_t::i32),
        ::testing::Values(0),
        ::testing::Values(ngraph::op::RoundingType::FLOOR),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolv8_ExplicitPad_FloorRounding, MaxPoolingV8LayerTest,
                         ::testing::Combine(
                                 maxPoolv8_ExplicitPad_FloorRounding_Params,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(std::vector<size_t >({1, 3, 30, 30})),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MaxPoolingV8LayerTest::getTestCaseName);

/* ========== Same Upper Pad Floor Rounding ========== */
const auto maxPoolv8_SameUpperPad_FloorRounding_Params = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(dilation),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::Values(ngraph::element::Type_t::i32),
        ::testing::Values(0),
        ::testing::Values(ngraph::op::RoundingType::FLOOR),
        ::testing::Values(ngraph::op::PadType::SAME_UPPER)
);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolv8_SameUpperPad_FloorRounding, MaxPoolingV8LayerTest,
                         ::testing::Combine(
                                 maxPoolv8_SameUpperPad_FloorRounding_Params,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(std::vector<size_t >({1, 3, 30, 30})),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MaxPoolingV8LayerTest::getTestCaseName);

/* ========== Same Lower Pad Floor Rounding ========== */
const auto maxPoolv8_SameLowerPad_FloorRounding_Params = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(dilation),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::Values(ngraph::element::Type_t::i32),
        ::testing::Values(0),
        ::testing::Values(ngraph::op::RoundingType::FLOOR),
        ::testing::Values(ngraph::op::PadType::SAME_LOWER)
);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolv8_SameLowerPad_FloorRounding, MaxPoolingV8LayerTest,
                         ::testing::Combine(
                                 maxPoolv8_SameLowerPad_FloorRounding_Params,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(std::vector<size_t >({1, 3, 30, 30})),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MaxPoolingV8LayerTest::getTestCaseName);

/* ========= Explicit Pad Floor Rounding 5D input========== */
const auto maxPoolv8_ExplicitPad_FloorRounding_5Dinput_Params = ::testing::Combine(
        ::testing::ValuesIn(kernel3D),
        ::testing::ValuesIn(strides3D),
        ::testing::Values(dilation3D[0]),
        ::testing::ValuesIn(padBegins3D),
        ::testing::ValuesIn(padEnds3D),
        ::testing::Values(ngraph::element::Type_t::i32),
        ::testing::Values(0),
        ::testing::Values(ngraph::op::RoundingType::FLOOR),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolv8_ExplicitPad_FloorRounding_5Dinput, MaxPoolingV8LayerTest,
                         ::testing::Combine(
                                 maxPoolv8_ExplicitPad_FloorRounding_5Dinput_Params,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(std::vector<size_t >({32, 32, 2, 2, 2})),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MaxPoolingV8LayerTest::getTestCaseName);

/* ========= Same Upper Pad Floor Rounding 5D input========== */
const auto maxPoolv8_SameUpperPad_FloorRounding_5Dinput_Params = ::testing::Combine(
        ::testing::ValuesIn(kernel3D),
        ::testing::ValuesIn(strides3D),
        ::testing::ValuesIn(dilation3D),
        ::testing::ValuesIn(padBegins3D),
        ::testing::ValuesIn(padEnds3D),
        ::testing::Values(ngraph::element::Type_t::i32),
        ::testing::Values(0),
        ::testing::Values(ngraph::op::RoundingType::FLOOR),
        ::testing::Values(ngraph::op::PadType::SAME_UPPER)
);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolv8_SameUpperPad_FloorRounding_5Dinput, MaxPoolingV8LayerTest,
                         ::testing::Combine(
                                 maxPoolv8_SameUpperPad_FloorRounding_5Dinput_Params,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(std::vector<size_t >({32, 32, 2, 2, 2})),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MaxPoolingV8LayerTest::getTestCaseName);

/* ========= Same Lower Pad Ceil Rounding 5D input========== */
const auto maxPoolv8_SameLowerPad_CeilRounding_5Dinput_Params = ::testing::Combine(
        ::testing::ValuesIn(kernel3D),
        ::testing::ValuesIn(strides3D),
        ::testing::ValuesIn(dilation3D),
        ::testing::ValuesIn(padBegins3D),
        ::testing::ValuesIn(padEnds3D),
        ::testing::Values(ngraph::element::Type_t::i32),
        ::testing::Values(0),
        ::testing::Values(ngraph::op::RoundingType::CEIL),
        ::testing::Values(ngraph::op::PadType::SAME_LOWER)
);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolv8_SameLowerPad_CeilRounding_5Dinput, MaxPoolingV8LayerTest,
                         ::testing::Combine(
                                 maxPoolv8_SameLowerPad_CeilRounding_5Dinput_Params,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(std::vector<size_t >({32, 32, 2, 2, 2})),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MaxPoolingV8LayerTest::getTestCaseName);

/* ========= Explicit Pad Ceil Rounding ========== */
const auto maxPoolv8_ExplicitPad_CeilRounding_Params = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(dilation),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::Values(ngraph::element::Type_t::i32),
        ::testing::Values(0),
        ::testing::Values(ngraph::op::RoundingType::CEIL),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolv8_ExplicitPad_CeilRounding, MaxPoolingV8LayerTest,
                         ::testing::Combine(
                                 maxPoolv8_ExplicitPad_CeilRounding_Params,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(std::vector<size_t >({1, 3, 30, 30})),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MaxPoolingV8LayerTest::getTestCaseName);

/* ========== Valid Pad Rounding Not Applicable ========== */

const auto maxPoolv8_ValidPad_Params = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(dilation),
        ::testing::Values(std::vector<size_t>({0, 0})),
        ::testing::Values(std::vector<size_t>({0, 0})),
        ::testing::Values(ngraph::element::Type_t::i32),
        ::testing::Values(0),
        ::testing::Values(ngraph::op::RoundingType::FLOOR),  // placeholder value - Rounding Type not applicable for Valid pad type
        ::testing::Values(ngraph::op::PadType::VALID)
);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolv8_ValidPad, MaxPoolingV8LayerTest,
                         ::testing::Combine(
                                 maxPoolv8_ValidPad_Params,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(std::vector<size_t >({1, 3, 30, 30})),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MaxPoolingV8LayerTest::getTestCaseName);

}  // namespace
