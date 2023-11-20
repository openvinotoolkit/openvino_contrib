// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/minimum_maximum.hpp"

#include <cuda_test_constants.hpp>
#include <vector>

using namespace LayerTestsDefinitions;

namespace {
using ov::test::utils::MinMaxOpType;
using ov::test::utils::InputLayerType;

const std::vector<std::vector<std::vector<size_t>>> inShapes = {
    {{2}, {1}},
    {{1, 1, 1, 3}, {1}},
    {{1, 2, 4}, {1}},
    {{1, 4, 4}, {1}},
    {{1, 4, 4, 1}, {1}},
    {{256, 56}, {256, 56}},
    {{8, 1, 6, 1}, {7, 1, 5}},
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const std::vector<MinMaxOpType> opType = {
    MinMaxOpType::MINIMUM,
    MinMaxOpType::MAXIMUM,
};

const std::vector<InputLayerType> inputType = {
    InputLayerType::CONSTANT,
    InputLayerType::PARAMETER,
};

INSTANTIATE_TEST_CASE_P(smoke_MaxMin,
                        MaxMinLayerTest,
                        ::testing::Combine(::testing::ValuesIn(inShapes),
                                           ::testing::ValuesIn(opType),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(inputType),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
                        MaxMinLayerTest::getTestCaseName);

}  // namespace
