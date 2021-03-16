// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/convolution_backprop_data.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        // InferenceEngine::Precision::FP16
};

const std::vector<size_t> numOutChannels = {1, 5};

/* ============= 2D ConvolutionBackpropData ============= */
const std::vector<std::vector<size_t >> inputShapes2D = {{1, 3, 10, 11}};
const std::vector<std::vector<size_t >> kernels2D = {{1, 1}, {3, 3}, {3, 5}};
const std::vector<std::vector<size_t >> strides2D = {{1, 1}, {1, 3}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {{0, 0}, {1, 1}};
const std::vector<std::vector<size_t >> dilations2D = {{1, 1}, {2, 2}};

const auto conv2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels2D),
        ::testing::ValuesIn(strides2D),
        ::testing::ValuesIn(padBegins2D),
        ::testing::ValuesIn(padEnds2D),
        ::testing::ValuesIn(dilations2D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);
const auto conv2DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels2D),
        ::testing::ValuesIn(strides2D),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(dilations2D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::VALID)
);

INSTANTIATE_TEST_CASE_P(ConvolutionBackpropData2D_ExplicitPadding, ConvolutionBackpropDataLayerTest,
                        ::testing::Combine(
                                conv2DParams_ExplicitPadding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapes2D),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ConvolutionBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(ConvolutionBackpropData2D_AutoPadValid, ConvolutionBackpropDataLayerTest,
                        ::testing::Combine(
                                conv2DParams_AutoPadValid,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapes2D),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ConvolutionBackpropDataLayerTest::getTestCaseName);
}  // namespace
