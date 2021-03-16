// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "single_layer_tests/gather.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
};

const std::vector<std::vector<size_t>> inputShapes = {
        std::vector<size_t>{10, 20, 30, 40},
};

const std::vector<int> axes = {
        0,
        1,
        2,
        3,
        -1,
        -2,
};


const std::vector<std::vector<int>> indices1 = {
        std::vector<int>{0},
        std::vector<int>{1},
        std::vector<int>{4},
        std::vector<int>{9},
};

const std::vector<std::vector<size_t>> indicesShapes1 = {
        std::vector<size_t>{1},
};

const auto params1 = testing::Combine(
        testing::ValuesIn(indices1),
        testing::ValuesIn(indicesShapes1),
        testing::ValuesIn(axes),
        testing::ValuesIn(inputShapes),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);
INSTANTIATE_TEST_CASE_P(Gather1, GatherLayerTest, params1, GatherLayerTest::getTestCaseName);


const std::vector<std::vector<int>> indices2 = {
        std::vector<int>{0, 1},
        std::vector<int>{1, 3},
        std::vector<int>{0, 9},
};

const std::vector<std::vector<size_t>> indicesShapes2 = {
        std::vector<size_t>{2},
};

const auto params2 = testing::Combine(
        testing::ValuesIn(indices2),
        testing::ValuesIn(indicesShapes2),
        testing::ValuesIn(axes),
        testing::ValuesIn(inputShapes),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);
INSTANTIATE_TEST_CASE_P(Gather2, GatherLayerTest, params2, GatherLayerTest::getTestCaseName);


const std::vector<std::vector<int>> indices4 = {
        std::vector<int>{0, 1, 2, 3},
        std::vector<int>{0, 3, 2, 1},
        std::vector<int>{9, 3, 6, 4},
};

const std::vector<std::vector<size_t>> indicesShapes4 = {
        std::vector<size_t>{4},
};

const auto params4 = testing::Combine(
        testing::ValuesIn(indices4),
        testing::ValuesIn(indicesShapes4),
        testing::ValuesIn(axes),
        testing::ValuesIn(inputShapes),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(Gather4, GatherLayerTest, params4, GatherLayerTest::getTestCaseName);
}  // namespace
