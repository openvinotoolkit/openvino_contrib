// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/convolution.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const InferenceEngine::Precision inPrc = InferenceEngine::Precision::FP32;
const InferenceEngine::Precision outPrc = InferenceEngine::Precision::FP32;

const InferenceEngine::Layout inLayout = InferenceEngine::Layout::NHWC;
const InferenceEngine::Layout outLayout = InferenceEngine::Layout::NHWC;


/* ============= 1D Convolution ============= */
const std::vector<std::vector<size_t>> kernels1D = {
    {3},
    {5},
};
const std::vector<std::vector<size_t>> strides1D = {
    {1},
    {2},
};
const std::vector<std::vector<ptrdiff_t>> padBegins1D = {
    {0},
    {1},
};
const std::vector<std::vector<ptrdiff_t>> padEnds1D = {
    {0},
    {1},
};
const std::vector<std::vector<size_t>> dilations1D = {
    {1},
    {2},
};
const std::vector<size_t> numOutChannels1D = {
    1,
    3,
    5,
};

const auto conv1DParams_ExplicitPadding = ::testing::Combine(
    ::testing::ValuesIn(kernels1D), ::testing::ValuesIn(strides1D),
    ::testing::ValuesIn(padBegins1D), ::testing::ValuesIn(padEnds1D),
    ::testing::ValuesIn(dilations1D), ::testing::ValuesIn(numOutChannels1D),
    ::testing::Values(ngraph::op::PadType::EXPLICIT));

const auto conv1DParams_AutoPadValid = ::testing::Combine(
    ::testing::ValuesIn(kernels1D), ::testing::ValuesIn(strides1D),
    ::testing::Values(std::vector<ptrdiff_t>({0})),
    ::testing::Values(std::vector<ptrdiff_t>({0})),
    ::testing::ValuesIn(dilations1D), ::testing::ValuesIn(numOutChannels1D),
    ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(smoke_Convolution1D_ExplicitPadding, ConvolutionLayerTest,
    ::testing::Combine(
        conv1DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution1D_AutoPadValid, ConvolutionLayerTest,
    ::testing::Combine(
        conv1DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution ============= */
const std::vector<std::vector<size_t >> kernels = {
    {2, 2},
    {3, 3},
    {3, 5},
};
const std::vector<std::vector<size_t >> strides = {
    {1, 1},
    {2, 2},
    {1, 3},
};
const std::vector<std::vector<ptrdiff_t>> padBegins = {
    {0, 0},
    {1, 1},
    {0, 3},
};
const std::vector<std::vector<ptrdiff_t>> padEnds = {
    {0, 0},
    {1, 1},
    {0, 3},
};
const std::vector<std::vector<size_t >> dilations = {
    {1, 1},
    {2, 2},
    {3, 1},
};

const std::vector<size_t> numOutChannels = {
    1,
    3,
    5,
};

const std::vector<ngraph::op::PadType> padTypes = {
        ngraph::op::PadType::EXPLICIT,
        ngraph::op::PadType::VALID
};

const auto conv2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);
const auto conv2DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::VALID)
);

INSTANTIATE_TEST_CASE_P(Convolution2D_ExplicitPadding, ConvolutionLayerTest,
                        ::testing::Combine(
                                conv2DParams_ExplicitPadding,
                        ::testing::ValuesIn(netPrecisions),
                        ::testing::Values(inPrc), ::testing::Values(outPrc),
                        ::testing::Values(inLayout), ::testing::Values(outLayout),
                        ::testing::Values(InferenceEngine::SizeVector({1, 3, 30, 30})),
                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Convolution2D_AutoPadValid, ConvolutionLayerTest,
                        ::testing::Combine(
                                conv2DParams_AutoPadValid,
                        ::testing::ValuesIn(netPrecisions),
                        ::testing::Values(inPrc), ::testing::Values(outPrc),
                        ::testing::Values(inLayout), ::testing::Values(outLayout),
                        ::testing::Values(InferenceEngine::SizeVector({1, 3, 30, 30})),
                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ConvolutionLayerTest::getTestCaseName);
}  // namespace
