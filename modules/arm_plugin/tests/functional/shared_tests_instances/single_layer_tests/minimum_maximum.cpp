// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <vector>
#include "single_layer_tests/minimum_maximum.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<std::vector<std::vector<size_t>>> inShapes = {
        {{2}, {1}},
        {{256, 56}, {256, 56}},
        {{1, 2, 4}, {1}},
        {{1, 4, 4, 1}, {1}},
        {{10, 3, 2}, {10, 3, 2}},
        {{1, 4, 4}, {1, 1, 1, 1}},
        {{1, 1, 1, 3}, {1, 1, 1, 1}},
        {{1, 15, 1, 3}, {1, 1, 1, 1}},
        {{8, 1, 6, 5}, {8, 1, 6, 5}},
        {{8, 1, 6, 1}, {1, 7, 1, 5}},
        {{8, 1, 6, 1}, {7, 1, 5}},
        {{7, 1, 5}, {8, 1, 6, 1}},
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
};

const std::vector<ngraph::helpers::MinMaxOpType> opType = {
        ngraph::helpers::MinMaxOpType::MINIMUM,
        ngraph::helpers::MinMaxOpType::MAXIMUM,
};

const std::vector<ngraph::helpers::InputLayerType> inputType = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
};

INSTANTIATE_TEST_CASE_P(minmax, MaxMinLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes),
                                ::testing::ValuesIn(opType),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputType),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        MaxMinLayerTest::getTestCaseName);

}  // namespace