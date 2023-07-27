// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cuda_test_constants.hpp>
#include <single_layer_tests/logical.hpp>
#include <vector>

using namespace LayerTestsDefinitions;
using namespace LayerTestsDefinitions::LogicalParams;

namespace {

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> inputShapesNot = {
    {{256}, {}},
    {{50, 200}, {}},
    {{1, 3, 20}, {}},
    {{2, 17, 3, 4}, {}},
    {{2, 3, 25, 4, 13}, {}},
};

std::vector<InferenceEngine::Precision> inputsPrecisions = {
    InferenceEngine::Precision::BOOL,
};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::BOOL,
};

std::map<std::string, std::string> additional_config = {};

const auto LogicalTestParamsNot =
    ::testing::Combine(::testing::ValuesIn(LogicalLayerTest::combineShapes(inputShapesNot)),
                       ::testing::Values(ngraph::helpers::LogicalTypes::LOGICAL_NOT),
                       ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::ValuesIn(inputsPrecisions),
                       ::testing::Values(InferenceEngine::Precision::BOOL),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                       ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(smoke_LogicalNot, LogicalLayerTest, LogicalTestParamsNot, LogicalLayerTest::getTestCaseName);

}  // namespace
