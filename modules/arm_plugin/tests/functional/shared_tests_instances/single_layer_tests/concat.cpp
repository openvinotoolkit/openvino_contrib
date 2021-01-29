// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <vector>

#include "single_layer_tests/concat.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

std::vector<size_t> axes4D = {0, 1, 2, 3};
std::vector<std::vector<std::vector<size_t>>> inShapes4D = {
        {{2, 2, 2, 2}, {2, 2, 2, 2}},
        {{3, 3, 3, 3}, {3, 3, 3, 3}, {3, 3, 3, 3}},
        {{4, 4, 4, 4}, {4, 4, 4, 4}, {4, 4, 4, 4}, {4, 4, 4, 4}},
        {{5, 5, 5, 5}, {5, 5, 5, 5}, {5, 5, 5, 5}, {5, 5, 5, 5}, {5, 5, 5, 5}}
};


std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

INSTANTIATE_TEST_CASE_P(Concat4DTest, ConcatLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(axes4D),
                                ::testing::ValuesIn(inShapes4D),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values("ARM")),
                        ConcatLayerTest::getTestCaseName);


std::vector<size_t> axes2D = {0, 1};
std::vector<std::vector<std::vector<size_t>>> inShapes2D = {
        {{1, 1}, {1, 1}},
        {{2, 2}, {2, 2}},
};

INSTANTIATE_TEST_CASE_P(Concat2DTest, ConcatLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(axes2D),
                                ::testing::ValuesIn(inShapes2D),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values("ARM")),
                        ConcatLayerTest::getTestCaseName);

}  // namespace
