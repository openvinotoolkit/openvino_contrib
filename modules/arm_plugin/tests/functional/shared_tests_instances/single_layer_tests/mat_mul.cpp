// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/mat_mul.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
};

const std::vector<std::vector<size_t>> shapesA2D = {
        {1, 4},
        {2, 4},
        {4, 4},
};

const std::vector<std::vector<size_t>> shapesB2D = {
        {4, 1},
        {4, 10},
};

std::map<std::string, std::string> additionalConfig = {};

INSTANTIATE_TEST_CASE_P(MatMul, MatMulTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(MatMulTest::combineShapes(shapesA2D, shapesB2D, false, false)),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::ValuesIn(secondaryInputTypes),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU),
                            ::testing::Values(additionalConfig)),
                        MatMulTest::getTestCaseName);

const std::vector<std::vector<size_t>> shapesA2DTranspose = {
        {4, 1},
        {4, 2},
        {4, 4},
};

INSTANTIATE_TEST_CASE_P(MatMul2DTransposeA, MatMulTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(MatMulTest::combineShapes(shapesA2DTranspose, shapesB2D, true, false)),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(secondaryInputTypes),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(additionalConfig)),
                        MatMulTest::getTestCaseName);

const std::vector<std::vector<size_t>> shapesA4D = {
        {1, 1, 2, 3},
        {1, 3, 5, 3},
        {3, 3, 3, 3},
};

const std::vector<std::vector<size_t>> shapesB4D = {
        {1, 1, 3, 4},
        {1, 3, 3, 4},
        {1, 3, 3, 1},
        {3, 3, 3, 3},
};

INSTANTIATE_TEST_CASE_P(MatMul4D, MatMulTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(MatMulTest::combineShapes(shapesA4D, shapesB4D, false, false)),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(secondaryInputTypes),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(additionalConfig)),
                        MatMulTest::getTestCaseName);

const std::vector<std::vector<size_t>> shapesB4DTranspose = {
        {1, 1, 4, 3},
        {1, 3, 4, 3},
        {1, 3, 1, 3},
        {3, 3, 3, 3},
};

INSTANTIATE_TEST_CASE_P(MatMul4DTransposeB, MatMulTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(MatMulTest::combineShapes(shapesA4D, shapesB4DTranspose, false, true)),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(secondaryInputTypes),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(additionalConfig)),
                        MatMulTest::getTestCaseName);

const std::vector<std::vector<size_t>> shapesA4DTranspose = {
        {1, 1, 3, 2},
        {1, 3, 3, 5},
        {3, 3, 3, 3},
};

INSTANTIATE_TEST_CASE_P(MatMul4DTransposeA, MatMulTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(MatMulTest::combineShapes(shapesA4DTranspose, shapesB4D, true, false)),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(secondaryInputTypes),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(additionalConfig)),
                        MatMulTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(MatMul4DTransposeAB, MatMulTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(MatMulTest::combineShapes(shapesA4DTranspose, shapesB4DTranspose, true, true)),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(secondaryInputTypes),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(additionalConfig)),
                        MatMulTest::getTestCaseName);

}  // namespace
