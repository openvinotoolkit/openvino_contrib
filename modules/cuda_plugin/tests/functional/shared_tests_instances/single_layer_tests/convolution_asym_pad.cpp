// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cuda_test_constants.hpp>
#include <vector>

#include "shared_test_classes/single_layer/convolution.hpp"

using namespace LayerTestsDefinitions;

namespace CUDALayerTestsDefinitions {

class ConvolutionAsymPadCUDALayerTest : public ConvolutionLayerTest {};

TEST_P(ConvolutionAsymPadCUDALayerTest, Run) { Run(); }

namespace {
const std::vector<InferenceEngine::Precision> precisions = {InferenceEngine::Precision::FP32};
const std::vector<std::vector<size_t>> kernels = {{3, 3}};
const std::vector<std::vector<size_t>> strides = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 2}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}};
const std::vector<size_t> numOutChannels = {2};
const std::vector<ngraph::op::PadType> pad_types = {ngraph::op::PadType::EXPLICIT};
const auto inputShapes = std::vector<size_t>({1, 4, 10, 10});

const auto groupConv2DParams = ::testing::Combine(::testing::ValuesIn(kernels),
                                                  ::testing::ValuesIn(strides),
                                                  ::testing::ValuesIn(padBegins),
                                                  ::testing::ValuesIn(padEnds),
                                                  ::testing::ValuesIn(dilations),
                                                  ::testing::ValuesIn(numOutChannels),
                                                  ::testing::ValuesIn(pad_types));

INSTANTIATE_TEST_CASE_P(smoke_ConvolutionAsymPadCUDA2D_Run,
                        ConvolutionAsymPadCUDALayerTest,
                        ::testing::Combine(groupConv2DParams,
                                           ::testing::ValuesIn(precisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(inputShapes),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        ConvolutionAsymPadCUDALayerTest::getTestCaseName);
}  // namespace
}  // namespace CUDALayerTestsDefinitions
