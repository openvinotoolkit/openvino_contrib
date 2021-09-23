// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/comparison.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "cuda_test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace LayerTestsDefinitions::ComparisonParams;

namespace {

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> smokeShapes = {
    {{1}, {{1}, {17}, {1, 1}, {2, 18}, {1, 1, 2}, {2, 2, 3}, {1, 1, 2, 3}}},
    {{2, 200}, {{2, 200}}},
    {{3, 10, 5}, {{3, 10, 5}}},
    {{2, 1, 1, 3, 1}, {{2, 1, 1, 3, 1}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> shapes = {
    {{1}, {{256}}},
};

std::vector<InferenceEngine::Precision> inputsPrecisions = {
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32,
};

std::vector<ngraph::helpers::ComparisonTypes> comparisonOpTypes = {
    ngraph::helpers::ComparisonTypes::GREATER,
};

std::vector<ngraph::helpers::InputLayerType> secondInputTypes = {
    ngraph::helpers::InputLayerType::PARAMETER,
};

std::map<std::string, std::string> additional_config = {};

const auto smokeComparisonTestParams =
    ::testing::Combine(::testing::ValuesIn(CommonTestUtils::combineParams(smokeShapes)),
                       ::testing::ValuesIn(inputsPrecisions),
                       ::testing::ValuesIn(comparisonOpTypes),
                       ::testing::ValuesIn(secondInputTypes),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                       ::testing::Values(additional_config));

const auto comparisonTestParams = ::testing::Combine(::testing::ValuesIn(CommonTestUtils::combineParams(shapes)),
                                                     ::testing::ValuesIn(inputsPrecisions),
                                                     ::testing::ValuesIn(comparisonOpTypes),
                                                     ::testing::ValuesIn(secondInputTypes),
                                                     ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                                     ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                                     ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                                                     ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(smoke_ComparisonCompareWithRefs,
                        ComparisonLayerTest,
                        smokeComparisonTestParams,
                        ComparisonLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(ComparisonCompareWithRefs,
                        ComparisonLayerTest,
                        comparisonTestParams,
                        ComparisonLayerTest::getTestCaseName);

}  // namespace
