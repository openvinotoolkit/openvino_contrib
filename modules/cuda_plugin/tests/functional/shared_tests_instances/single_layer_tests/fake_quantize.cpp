// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/fake_quantize.hpp"

#include <cuda_test_constants.hpp>
#include <vector>

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::vector<size_t>> inputShapes = {{1, 1},
                                                      {2, 6},
                                                      {1, 1, 1},
                                                      {2, 6, 13},
                                                      {1, 1, 1, 1},
                                                      {3, 10, 5, 6},
                                                      {2, 8, 5, 18},
                                                      {2, 16, 3, 18},
                                                      {3, 49, 5, 6},
                                                      {1, 1, 1, 1, 1},
                                                      {3, 10, 2, 5, 6},
                                                      {2, 8, 1, 5, 18},
                                                      {2, 16, 4, 3, 18},
                                                      {3, 49, 7, 5, 6}};
const std::vector<std::vector<size_t>> constShapes = {{1}};
const std::vector<size_t> levels = {16, 255, 256};

const std::pair<std::string, std::map<std::string, std::string>> config = {};
const std::vector<float> fqArgs = {};
const std::vector<float> inputParams = {};

const auto fqParams = ::testing::Combine(::testing::ValuesIn(levels),
                                         ::testing::ValuesIn(constShapes),
                                         ::testing::Values(fqArgs),
                                         ::testing::Values(inputParams));

INSTANTIATE_TEST_CASE_P(smoke_CUDAFakeQuantize,
                        FakeQuantizeLayerTest,
                        ::testing::Combine(fqParams,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(inputShapes),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                                           ::testing::Values(config)),
                        FakeQuantizeLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> constShapesBr = {{2, 8}};
const std::vector<std::vector<size_t>> inputShapesBr = {{2, 8}};
const auto fqParamsBr = ::testing::Combine(::testing::ValuesIn(levels),
                                           ::testing::ValuesIn(constShapesBr),
                                           ::testing::Values(fqArgs),
                                           ::testing::Values(inputParams));

INSTANTIATE_TEST_CASE_P(smoke_CUDAFakeQuantizeBr,
                        FakeQuantizeLayerTest,
                        ::testing::Combine(fqParamsBr,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(inputShapesBr),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                                           ::testing::Values(config)),
                        FakeQuantizeLayerTest::getTestCaseName);

}  // namespace
