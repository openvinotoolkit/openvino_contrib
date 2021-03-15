// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <vector>

#include "single_layer_tests/transpose.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

// Empty order is not supported yet: CVS-32756
std::vector<std::vector<size_t>> inputShape2D = {{2, 10}, {10, 2}, {10, 10}};
std::vector<std::vector<size_t>> order2D      = {{0, 1}, {1, 0}, /*{}*/};

INSTANTIATE_TEST_CASE_P(Transpose2D, TransposeLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(order2D),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::ValuesIn(inputShape2D),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                TransposeLayerTest::getTestCaseName);

// TODO: fix Transpose for tensors with equal dimensions
std::vector<std::vector<size_t>> inputShape4D = {/*{2, 2, 2, 2},*/ {1, 10, 2, 3}, {2, 3, 4, 5}};
std::vector<std::vector<size_t>> order4D      = {
        // {}
        {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1},
        {1, 0, 2, 3}, {1, 0, 3, 2}, {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0},
        {2, 0, 1, 3}, {2, 0, 3, 1}, {2, 1, 0, 3}, {2, 1, 3, 0}, {2, 3, 0, 1}, {2, 3, 1, 0},
        {3, 0, 1, 2}, {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0}
};

INSTANTIATE_TEST_CASE_P(Transpose4D, TransposeLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(order4D),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::ValuesIn(inputShape4D),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                TransposeLayerTest::getTestCaseName);
}  // namespace
