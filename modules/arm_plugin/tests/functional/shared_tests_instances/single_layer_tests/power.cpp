// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/power.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

std::vector<std::vector<std::vector<size_t>>> inShapes = {
        {{1, 8}},
        {{2, 16}},
        {{128, 3}},
        {{1, 2, 6}},
        {{6, 25, 1, 2}},
        {{1, 2, 5, 6, 7}},
};

std::vector<std::vector<float>> power = {
        {0.0f},
        {0.5f},
        {1.0f},
        {1.1f},
        {2.0f},
};


INSTANTIATE_TEST_CASE_P(Power, PowerLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::ValuesIn(power)),
                        PowerLayerTest::getTestCaseName);

// TODO: Add after fixing tests in OpenVINO
// INSTANTIATE_TEST_CASE_P(ElementwisePower, PowerLayerTest,
//                         ::testing::Combine(
//                                 ::testing::Values(std::vector<std::vector<size_t>>{{{2, 3, 1}}}),
//                                 ::testing::ValuesIn(netPrecisions),
//                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
//                                 ::testing::Values(std::vector<float>{2.0f, 1.5f, 0.5f, 1.1f, 0.8f, 2.2f})),
//                         PowerLayerTest::getTestCaseName);
}  // namespace