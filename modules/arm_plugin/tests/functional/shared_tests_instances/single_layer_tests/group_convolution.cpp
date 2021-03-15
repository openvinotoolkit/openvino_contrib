// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <vector>

#include "single_layer_tests/group_convolution.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

/* ============= 1D GroupConvolution ============= */
const std::vector<std::vector<size_t>> kernels1d = {
        {1},
        {3},
};
const std::vector<std::vector<size_t>> strides1d = {
        {1},
        {2},
};
const std::vector<std::vector<ptrdiff_t>> padBegins1d = {
        {0},
        {1},
};
const std::vector<std::vector<ptrdiff_t>> padEnds1d = {
        {0},
        {1},
};
const std::vector<std::vector<size_t>> dilations1d = {
        {1},
        {2},
};
const std::vector<size_t> numOutChannels1d = {
        8,
        16,
};
const std::vector<size_t> numGroups1d = {
        2,
        8,
};

const auto groupConv1DParams_ExplicitPadding = ::testing::Combine(
    ::testing::ValuesIn(kernels1d), ::testing::ValuesIn(strides1d),
    ::testing::ValuesIn(padBegins1d), ::testing::ValuesIn(padEnds1d),
    ::testing::ValuesIn(dilations1d), ::testing::ValuesIn(numOutChannels1d),
    ::testing::ValuesIn(numGroups1d),
    ::testing::Values(ngraph::op::PadType::EXPLICIT));

const auto groupConv1DParams_AutoPadValid = ::testing::Combine(
    ::testing::ValuesIn(kernels1d), ::testing::ValuesIn(strides1d),
    ::testing::Values(std::vector<ptrdiff_t>({0})),
    ::testing::Values(std::vector<ptrdiff_t>({0})),
    ::testing::ValuesIn(dilations1d), ::testing::ValuesIn(numOutChannels1d),
    ::testing::ValuesIn(numGroups1d),
    ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(
    smoke_GroupConvolution1D_ExplicitPadding, GroupConvolutionLayerTest,
    ::testing::Combine(
        groupConv1DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{2, 8, 30}),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    GroupConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    smoke_GroupConvolution1D_AutoPadValid, GroupConvolutionLayerTest,
    ::testing::Combine(
        groupConv1DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({2, 8, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    GroupConvolutionLayerTest::getTestCaseName);

/* ============= 2D GroupConvolution ============= */
const std::vector<std::vector<size_t >> kernels = {{1, 1}, {3, 3}, {8, 8}};
const std::vector<std::vector<size_t >> strides = {{1, 1}, {2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
const std::vector<std::vector<size_t >> dilations = {{1, 1}, {2, 2}};
const std::vector<size_t> numOutChannels = {8, 16};
const std::vector<size_t> numGroups = {2, 8};

const auto groupConv2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);
const auto groupConv2DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::Values(ngraph::op::PadType::VALID)
);

INSTANTIATE_TEST_CASE_P(GroupConvolution2D_ExplicitPadding, GroupConvolutionLayerTest,
                        ::testing::Combine(
                                groupConv2DParams_ExplicitPadding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({1, 8, 30, 30})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GroupConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(GroupConvolution2D_AutoPadValid, GroupConvolutionLayerTest,
                        ::testing::Combine(
                                groupConv2DParams_AutoPadValid,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({1, 8, 30, 30})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GroupConvolutionLayerTest::getTestCaseName);
} // namespace
