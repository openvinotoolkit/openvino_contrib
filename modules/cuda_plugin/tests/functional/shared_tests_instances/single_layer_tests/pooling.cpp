// Copyright (C) 2019 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_test_constants.hpp>
#include <single_layer_tests/pooling.hpp>
#include <vector>

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16};

const std::vector<std::vector<size_t>> kernels = {{3, 3}, {3, 5}, {2, 2}};
const std::vector<std::vector<size_t>> strides = {{1, 1}, {1, 2}, {2, 2}};
// TODO: find why paddings >0 for the first spatial axis, and/or paddings
// >1 for the second spatial axis fail.
// Note: asymmetric paddings (begin != end) are not supported in cuDNN.
const std::vector<std::vector<size_t>> padBegins = {{0, 1} /*, {0, 0}*/};
const std::vector<std::vector<size_t>> padEnds = {{0, 1} /*, {0, 0}*/};
const std::vector<ngraph::op::RoundingType> roundingTypes = {
    ngraph::op::RoundingType::CEIL, ngraph::op::RoundingType::FLOOR};

////* ========== Max Poolling ========== */
/* +========== Explicit Pad Floor Rounding ========== */
const auto maxPool_ExplicitPad_FloorRounding_Params = ::testing::Combine(
    ::testing::Values(PoolingTypes::MAX), ::testing::ValuesIn(kernels),
    ::testing::ValuesIn(strides), ::testing::ValuesIn(padBegins),
    ::testing::ValuesIn(padEnds),
    ::testing::Values(ngraph::op::RoundingType::FLOOR),
    ::testing::Values(ngraph::op::PadType::EXPLICIT),
    ::testing::Values(false)  // placeholder value - exclude pad not applicable
                              // for max pooling
);

INSTANTIATE_TEST_CASE_P(
    smoke_MaxPool_ExplicitPad_FloorRounding, PoolingLayerTest,
    ::testing::Combine(
        maxPool_ExplicitPad_FloorRounding_Params,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 50, 50})),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    PoolingLayerTest::getTestCaseName);

/* ========== Explicit Pad Ceil Rounding ========== */
const auto maxPool_ExplicitPad_CeilRounding_Params = ::testing::Combine(
    ::testing::Values(PoolingTypes::MAX), ::testing::ValuesIn(kernels),
    // TODO: Non 1 strides fails in ngraph reference implementation with error
    // "The end corner is out of bounds at axis 3" thrown in the test body.
    ::testing::Values(std::vector<size_t>({1, 1})),
    ::testing::ValuesIn(padBegins), ::testing::ValuesIn(padEnds),
    ::testing::Values(ngraph::op::RoundingType::CEIL),
    ::testing::Values(ngraph::op::PadType::EXPLICIT),
    ::testing::Values(false)  // placeholder value - exclude pad not applicable
                              // for max pooling
);

INSTANTIATE_TEST_CASE_P(
    smoke_MaxPool_ExplicitPad_CeilRounding, PoolingLayerTest,
    ::testing::Combine(
        maxPool_ExplicitPad_CeilRounding_Params,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 50, 50})),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    PoolingLayerTest::getTestCaseName);

////* ========== Avg Pooling ========== */
/* +========== Explicit Pad Ceil Rounding ========== */
const auto avgPoolExplicitPadCeilRoundingParams = ::testing::Combine(
    ::testing::Values(PoolingTypes::AVG), ::testing::ValuesIn(kernels),
    // TODO: Non 1 strides fails in ngraph reference implementation with error
    // "The end corner is out of bounds at axis 3" thrown in the test body.
    ::testing::Values(std::vector<size_t>({1, 1})),
    ::testing::ValuesIn(padBegins), ::testing::ValuesIn(padEnds),
    ::testing::Values(ngraph::op::RoundingType::CEIL),
    ::testing::Values(ngraph::op::PadType::EXPLICIT),
    ::testing::Values(true, false));

INSTANTIATE_TEST_CASE_P(
    smoke_AvgPool_ExplicitPad_CeilRounding, PoolingLayerTest,
    ::testing::Combine(
        avgPoolExplicitPadCeilRoundingParams,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    PoolingLayerTest::getTestCaseName);

/* +========== Explicit Pad Floor Rounding ========== */
const auto avgPoolExplicitPadFloorRoundingParams = ::testing::Combine(
    ::testing::Values(PoolingTypes::AVG), ::testing::ValuesIn(kernels),
    ::testing::ValuesIn(strides), ::testing::ValuesIn(padBegins),
    ::testing::ValuesIn(padEnds),
    ::testing::Values(ngraph::op::RoundingType::FLOOR),
    ::testing::Values(ngraph::op::PadType::EXPLICIT),
    ::testing::Values(true, false));

INSTANTIATE_TEST_CASE_P(
    smoke_AvgPool_ExplicitPad_FloorRounding, PoolingLayerTest,
    ::testing::Combine(
        avgPoolExplicitPadFloorRoundingParams,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    PoolingLayerTest::getTestCaseName);

////* ========== Avg and Max Pooling Cases ========== */
/*    ========== Valid Pad Rounding Not Applicable ========== */
const auto allPools_ValidPad_Params = ::testing::Combine(
    ::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),
    ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
    ::testing::Values(std::vector<size_t>({0, 0})),
    ::testing::ValuesIn(padEnds),
    ::testing::Values(
        ngraph::op::RoundingType::FLOOR),  // placeholder value - Rounding Type
                                           // not applicable for Valid pad type
    // TODO: PadType::VALID seems not to ignore padBegins
    ::testing::Values(ngraph::op::PadType::VALID),
    ::testing::Values(false)  // placeholder value - exclude pad not applicable
                              // for max pooling
);

INSTANTIATE_TEST_CASE_P(
    smoke_MAX_and_AVGPool_ValidPad, PoolingLayerTest,
    ::testing::Combine(
        allPools_ValidPad_Params, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 50, 50})),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    PoolingLayerTest::getTestCaseName);
}  // namespace

////* ========== Resnet50 and VGG16 specific cases ========== */
namespace resnet50_vgg16_maxpooling {
const std::vector<std::vector<size_t>> kernels = {{2, 2}, {3, 3}};
const std::vector<std::vector<size_t>> strides = {{2, 2}};
const std::vector<std::vector<size_t>> padBegins = {{0, 0}};
const std::vector<std::vector<size_t>> padEnds = {{0, 0}, {1, 1}};
const std::vector<ngraph::op::RoundingType> roundingTypes = {
    ngraph::op::RoundingType::CEIL};
const std::vector<std::vector<size_t>> shapes = {
    {1, 512, 28, 28},   {1, 256, 56, 56}, {1, 64, 224, 224},
    {1, 128, 112, 112}, {1, 512, 14, 14},
};

const auto resnet50_vgg16_maxpool_params = ::testing::Combine(
    ::testing::Values(PoolingTypes::MAX), ::testing::ValuesIn(kernels),
    ::testing::ValuesIn(strides), ::testing::ValuesIn(padBegins),
    ::testing::ValuesIn(padEnds),
    ::testing::Values(
        ngraph::op::RoundingType::FLOOR),  // placeholder value - Rounding Type
    ::testing::Values(ngraph::op::PadType::EXPLICIT),
    ::testing::Values(false)  // placeholder value - exclude pad not applicable
                              // for max pooling
);

INSTANTIATE_TEST_CASE_P(
    costly_Maxpool_Phase1, PoolingLayerTest,
    ::testing::Combine(
        resnet50_vgg16_maxpool_params,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(shapes),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    PoolingLayerTest::getTestCaseName);

}  // namespace resnet50_vgg16_maxpooling

namespace resnet50_avgpooling {
const std::vector<size_t> shape{1, 2048, 7, 7};
const std::vector<size_t> kernel{7, 7};
const std::vector<size_t> stride{1, 1};
const std::vector<size_t> padBegin{0, 0};
const std::vector<size_t> padEnd{0, 0};

const auto resnet50_avgpool_params =
    ::testing::Combine(::testing::Values(PoolingTypes::MAX),
                       ::testing::Values(kernel), ::testing::Values(stride),
                       ::testing::Values(padBegin), ::testing::Values(padEnd),
                       ::testing::Values(ngraph::op::RoundingType::FLOOR),
                       ::testing::Values(ngraph::op::PadType::VALID),
                       ::testing::Values(true)  // exclude pad
    );

INSTANTIATE_TEST_CASE_P(
    costly_Avg_Phase1, PoolingLayerTest,
    ::testing::Combine(
        resnet50_avgpool_params,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(shape),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    PoolingLayerTest::getTestCaseName);
}  // namespace resnet50_avgpooling
