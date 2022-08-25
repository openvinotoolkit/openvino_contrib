// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/comparison.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "cuda_test_constants.hpp"
#include "unsymmetrical_comparer.hpp"

using namespace LayerTestsDefinitions;
using namespace LayerTestsDefinitions::ComparisonParams;

namespace LayerTestsDefinitions {

class UnsymmetricalComparisonLayerTest : public UnsymmetricalComparer<ComparisonLayerTest> {};

TEST_P(UnsymmetricalComparisonLayerTest, CompareWithRefs) { Run(); }
}  // namespace LayerTestsDefinitions

namespace {

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> smokeShapes = {
    {{1}, {{1}, {17}, {1, 1}, {2, 18}, {1, 1, 2}, {2, 2, 3}, {1, 1, 2, 3}}},
    {{5}, {{1}, {1, 1}, {2, 5}, {1, 1, 1}, {2, 2, 5}}},
    {{2, 200}, {{1}, {200}, {1, 200}, {2, 200}, {2, 2, 200}}},
    {{1, 3, 20}, {{20}, {2, 1, 1}}},
    {{2, 17, 3, 4}, {{4}, {1, 3, 4}, {2, 1, 3, 4}}},
    {{2, 1, 1, 3, 1}, {{1}, {1, 3, 4}, {2, 1, 3, 4}, {1, 1, 1, 1, 1}}},
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
    ngraph::helpers::ComparisonTypes::LESS,
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
                       ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                       ::testing::Values(additional_config));

const auto comparisonTestParams = ::testing::Combine(::testing::ValuesIn(CommonTestUtils::combineParams(shapes)),
                                                     ::testing::ValuesIn(inputsPrecisions),
                                                     ::testing::ValuesIn(comparisonOpTypes),
                                                     ::testing::ValuesIn(secondInputTypes),
                                                     ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                                     ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                                     ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                                                     ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(smoke_ComparisonCompareWithRefs,
                        UnsymmetricalComparisonLayerTest,
                        smokeComparisonTestParams,
                        ComparisonLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(ComparisonCompareWithRefs,
                        UnsymmetricalComparisonLayerTest,
                        comparisonTestParams,
                        ComparisonLayerTest::getTestCaseName);

// ------------- Benchmark -------------
#include "benchmark.hpp"

namespace LayerTestsDefinitions {
namespace benchmark {
struct ComparisonBenchmarkTest : BenchmarkLayerTest<ComparisonLayerTest> {};

TEST_P(ComparisonBenchmarkTest, DISABLED_benchmark) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run("Comparison", std::chrono::milliseconds(2000), 100);
}

INSTANTIATE_TEST_CASE_P(smoke_ComparisonCompareWithRefs,
                        ComparisonBenchmarkTest,
                        smokeComparisonTestParams,
                        ComparisonLayerTest::getTestCaseName);

}  // namespace benchmark

}  // namespace LayerTestsDefinitions
}  // namespace
