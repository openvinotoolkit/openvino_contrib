// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/tile.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
};

const std::vector<std::vector<int64_t>> repeats = {
        {1, 2, 3},
        {2, 1, 1},
        {2, 3, 1},
        {2, 2, 2},
        {1, 2},
        {3},
        {1, 2, 3, 4},
};

const std::vector<std::vector<size_t>> inputShapes = {
        {2, 1},
        {1, 3, 3},
        {1, 2, 3, 4},
};

INSTANTIATE_TEST_CASE_P(Tile, TileLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(repeats),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::ValuesIn(inputShapes),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        TileLayerTest::getTestCaseName);
}  // namespace
