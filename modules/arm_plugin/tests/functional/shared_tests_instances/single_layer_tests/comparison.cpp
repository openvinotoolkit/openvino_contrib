// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/comparison.hpp"

using namespace LayerTestsDefinitions;

namespace {
std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> inputShapes = {
        {{1}, {{1}, {5}, {2, 3, 4}}},
        {{5}, {{5}, {2, 5}}},
        {{2, 200}, {{1}, {2, 200}}},
        {{1, 3, 20}, {{1, 3, 20}}},
        {{2, 17, 3, 4}, {{2, 17, 3, 4}}},
        {{2, 1, 1, 3, 1}, {{2, 1, 1, 3, 1}}},
};

std::vector<InferenceEngine::Precision> inputsPrecisions = {
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::FP32,
};

std::vector<ngraph::helpers::ComparisonTypes> comparisonOpTypes = {
        ngraph::helpers::ComparisonTypes::EQUAL,
        ngraph::helpers::ComparisonTypes::NOT_EQUAL,
        ngraph::helpers::ComparisonTypes::GREATER,
        ngraph::helpers::ComparisonTypes::GREATER_EQUAL,
        ngraph::helpers::ComparisonTypes::LESS,
        ngraph::helpers::ComparisonTypes::LESS_EQUAL,
};

std::vector<ngraph::helpers::InputLayerType> secondInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
};

const auto ComparisonTestParams = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(inputShapes)),
        ::testing::ValuesIn(inputsPrecisions),
        ::testing::ValuesIn(comparisonOpTypes),
        ::testing::ValuesIn(secondInputTypes),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(std::map<std::string, std::string>{}));

INSTANTIATE_TEST_CASE_P(Comparison, ComparisonLayerTest, ComparisonTestParams, ComparisonLayerTest::getTestCaseName);

}  // namespace
