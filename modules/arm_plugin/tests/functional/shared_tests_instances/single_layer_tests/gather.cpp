// Copyright (C) 2020-2022 Intel Corporation
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
INSTANTIATE_TEST_CASE_P(smoke_Gather1, GatherLayerTest, params1, GatherLayerTest::getTestCaseName);


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
INSTANTIATE_TEST_CASE_P(smoke_Gather2, GatherLayerTest, params2, GatherLayerTest::getTestCaseName);


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

INSTANTIATE_TEST_CASE_P(smoke_Gather4, GatherLayerTest, params4, GatherLayerTest::getTestCaseName);

const auto params_ref = testing::Combine(
        testing::ValuesIn(indices4),
        testing::Values(std::vector<size_t>{2, 2}),
        testing::ValuesIn(axes),
        testing::ValuesIn(inputShapes),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(smoke_Gather_refernce, GatherLayerTest, params_ref, GatherLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> indicesShapes5 = {
        std::vector<size_t>{10, 4},
        std::vector<size_t>{10, 20, 5},
};

const std::vector< std::tuple<int, int> > axes_batches = {
        std::tuple<int, int>(0, 0),
        std::tuple<int, int>(1, 0),
        std::tuple<int, int>(2, 0),
        std::tuple<int, int>(3, 0),
        std::tuple<int, int>(-1, 0),
        std::tuple<int, int>(-2, 0),
        std::tuple<int, int>(1, 1),
        std::tuple<int, int>(-1, 1),
        std::tuple<int, int>(1, -2),
        std::tuple<int, int>(-1, -2),
};

const auto params_g7 = testing::Combine(
        testing::ValuesIn(inputShapes),
        testing::ValuesIn(indicesShapes5),
        testing::ValuesIn(axes_batches),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(smoke_V7Gather4, Gather7LayerTest, params_g7, Gather7LayerTest::getTestCaseName);

const auto gatherParamsVec1 = testing::Combine(
        testing::ValuesIn(std::vector<std::vector<size_t>>({{10, 30, 50, 1}})),
        testing::ValuesIn(std::vector<std::vector<size_t>>({{10, 16, 16}, {10, 7, 8}, {10, 5, 7}, {10, 5}})),
        testing::ValuesIn(std::vector<std::tuple<int, int>>{std::tuple<int, int>{2, 1}}),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(smoke_V8_Vec1, Gather8LayerTest, gatherParamsVec1, Gather8LayerTest::getTestCaseName);

const auto gatherParamsVec2 = testing::Combine(
        testing::ValuesIn(std::vector<std::vector<size_t>>({{5, 4}, {11, 4}, {23, 4}, {35, 4}, {51, 4}, {71, 4}})),
        testing::ValuesIn(std::vector<std::vector<size_t>>({{1}})),
        testing::ValuesIn(std::vector<std::tuple<int, int>>{std::tuple<int, int>{1, 0}}),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(smoke_V8_Vec2, Gather8LayerTest, gatherParamsVec2, Gather8LayerTest::getTestCaseName);

const auto gatherParamsVec3 = testing::Combine(
        testing::ValuesIn(std::vector<std::vector<size_t>>({{4, 4}})),
        testing::ValuesIn(std::vector<std::vector<size_t>>({{5}, {11}, {21}, {35}, {55}, {70}})),
        testing::ValuesIn(std::vector<std::tuple<int, int>>{std::tuple<int, int>{1, 0}}),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(smoke_V8_Vec3, Gather8LayerTest, gatherParamsVec3, Gather8LayerTest::getTestCaseName);

}  // namespace
