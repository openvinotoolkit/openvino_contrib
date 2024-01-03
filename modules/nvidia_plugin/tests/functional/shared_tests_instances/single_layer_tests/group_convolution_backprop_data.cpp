// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/group_convolution_backprop_data.hpp"

#include <ie_common.h>

#include <ie_precision.hpp>
#include <ngraph/node.hpp>
#include <vector>

#include "cuda_test_constants.hpp"
#include "finite_comparer.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<size_t> numOutChannels = {16, 32};
const std::vector<size_t> numGroups = {2, 8, 16};

/* ============= 2D GroupConvolution ============= */
const std::vector<std::vector<size_t>> inputShapes2D = {{1, 16, 10, 10}, {1, 32, 10, 10}};
const std::vector<std::vector<size_t>> kernels2D = {{1, 1}, {3, 3}};
const std::vector<std::vector<size_t>> strides2D = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padBeginsAsym2D = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEndsAsym2D = {{1, 1}};
const std::vector<std::vector<size_t>> dilations2D = {{1, 1}};

const auto groupConvBackpropData2DParams_ExplicitPadding =
    ::testing::Combine(::testing::ValuesIn(kernels2D),
                       ::testing::ValuesIn(strides2D),
                       ::testing::ValuesIn(padBegins2D),
                       ::testing::ValuesIn(padEnds2D),
                       ::testing::ValuesIn(dilations2D),
                       ::testing::ValuesIn(numOutChannels),
                       ::testing::ValuesIn(numGroups),
                       ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto groupConvBackpropData2DParams_AutoPadValid =
    ::testing::Combine(::testing::ValuesIn(kernels2D),
                       ::testing::ValuesIn(strides2D),
                       ::testing::ValuesIn(padBegins2D),
                       ::testing::ValuesIn(padEnds2D),
                       ::testing::ValuesIn(dilations2D),
                       ::testing::ValuesIn(numOutChannels),
                       ::testing::ValuesIn(numGroups),
                       ::testing::Values(ngraph::op::PadType::VALID));
const auto groupConvBackpropData2DParams_AsymPad = ::testing::Combine(::testing::ValuesIn(kernels2D),
                                                                      ::testing::ValuesIn(strides2D),
                                                                      ::testing::ValuesIn(padBeginsAsym2D),
                                                                      ::testing::ValuesIn(padEndsAsym2D),
                                                                      ::testing::ValuesIn(dilations2D),
                                                                      ::testing::ValuesIn(numOutChannels),
                                                                      ::testing::ValuesIn(numGroups),
                                                                      ::testing::Values(ngraph::op::PadType::EXPLICIT));

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBackprop2D_ExplicitPadding,
                        GroupConvBackpropDataLayerTest,
                        ::testing::Combine(groupConvBackpropData2DParams_ExplicitPadding,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(inputShapes2D),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
                        GroupConvBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBackprop2D_AutoPadding,
                        GroupConvBackpropDataLayerTest,
                        ::testing::Combine(groupConvBackpropData2DParams_AutoPadValid,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(inputShapes2D),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
                        GroupConvBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBackprop2D_AsymPadding,
                        GroupConvBackpropDataLayerTest,
                        ::testing::Combine(groupConvBackpropData2DParams_AsymPad,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(inputShapes2D),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
                        GroupConvBackpropDataLayerTest::getTestCaseName);

/* ============= 3D GroupConvolution ============= */
const std::vector<std::vector<size_t>> inputShapes3D = {{1, 16, 5, 5, 5}, {1, 32, 5, 5, 5}};
const std::vector<std::vector<size_t>> kernels3D = {{1, 1, 1}, {3, 3, 3}};
const std::vector<std::vector<size_t>> strides3D = {{1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins3D = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds3D = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> padBeginsAsym3D = {{1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padEndsAsym3D = {{0, 0, 0}};
const std::vector<std::vector<size_t>> dilations3D = {{1, 1, 1}};

const auto groupConvBackpropData3DParams_ExplicitPadding =
    ::testing::Combine(::testing::ValuesIn(kernels3D),
                       ::testing::ValuesIn(strides3D),
                       ::testing::ValuesIn(padBegins3D),
                       ::testing::ValuesIn(padEnds3D),
                       ::testing::ValuesIn(dilations3D),
                       ::testing::ValuesIn(numOutChannels),
                       ::testing::ValuesIn(numGroups),
                       ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto groupConvBackpropData3DParams_AutoPadValid =
    ::testing::Combine(::testing::ValuesIn(kernels3D),
                       ::testing::ValuesIn(strides3D),
                       ::testing::ValuesIn(padBegins3D),
                       ::testing::ValuesIn(padEnds3D),
                       ::testing::ValuesIn(dilations3D),
                       ::testing::ValuesIn(numOutChannels),
                       ::testing::ValuesIn(numGroups),
                       ::testing::Values(ngraph::op::PadType::VALID));
const auto groupConvBackpropData3DParams_AsymPad = ::testing::Combine(::testing::ValuesIn(kernels3D),
                                                                      ::testing::ValuesIn(strides3D),
                                                                      ::testing::ValuesIn(padBeginsAsym3D),
                                                                      ::testing::ValuesIn(padEndsAsym3D),
                                                                      ::testing::ValuesIn(dilations3D),
                                                                      ::testing::ValuesIn(numOutChannels),
                                                                      ::testing::ValuesIn(numGroups),
                                                                      ::testing::Values(ngraph::op::PadType::EXPLICIT));

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBackprop3D_ExpicitPadding,
                        GroupConvBackpropDataLayerTest,
                        ::testing::Combine(groupConvBackpropData3DParams_ExplicitPadding,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(inputShapes3D),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
                        GroupConvBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBackprop3D_AutoPadding,
                        GroupConvBackpropDataLayerTest,
                        ::testing::Combine(groupConvBackpropData3DParams_AutoPadValid,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(inputShapes3D),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
                        GroupConvBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBackprop3D_AsymPadding,
                        GroupConvBackpropDataLayerTest,
                        ::testing::Combine(groupConvBackpropData3DParams_AsymPad,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(inputShapes3D),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
                        GroupConvBackpropDataLayerTest::getTestCaseName);

}  // namespace
