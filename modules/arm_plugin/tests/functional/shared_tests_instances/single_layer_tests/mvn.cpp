// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/mvn.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::vector<size_t>> inputShapes = {
    {1, 32, 17},
    {1, 37, 9},
    {1, 16, 5, 8},
    {2, 19, 5, 10},
    {5, 8, 3, 5},
    {1, 9, 1, 15, 9},
    {3, 4, 5, 10, 6}
};

const std::vector<bool> acrossChannels = {
    true,
    false
};

const std::vector<bool> normalizeVariance = {
    true,
    false
};

const std::vector<double> epsilon = {
    1e-7,
    1e-8,
    1e-9,
};

const auto MvnCases2D = ::testing::Combine(
    ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 3}, {3, 1}, {32, 17}}),
    ::testing::ValuesIn(netPrecisions),
    ::testing::Values(true),
    ::testing::ValuesIn(normalizeVariance),
    ::testing::ValuesIn(epsilon),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(TestsMVN2D, MvnLayerTest, MvnCases2D, MvnLayerTest::getTestCaseName);

const auto MvnCases = ::testing::Combine(
    ::testing::ValuesIn(inputShapes),
    ::testing::ValuesIn(netPrecisions),
    ::testing::ValuesIn(acrossChannels),
    ::testing::ValuesIn(normalizeVariance),
    ::testing::ValuesIn(epsilon),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(TestsMVN, MvnLayerTest, MvnCases, MvnLayerTest::getTestCaseName);
