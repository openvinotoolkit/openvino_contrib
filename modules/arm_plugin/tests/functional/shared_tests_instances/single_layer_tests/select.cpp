// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/select.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

const std::vector<InferenceEngine::Precision> inputPrecision = {
    InferenceEngine::Precision::U8,
    InferenceEngine::Precision::I16,
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const std::vector<std::vector<std::vector<size_t>>> noneShapes = {
    {{1}, {1}, {1}},
    {{8}, {8}, {8}},
    {{4, 5}, {4, 5}, {4, 5}},
    {{3, 4, 5}, {3, 4, 5}, {3, 4, 5}},
    {{2, 3, 4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}},
    {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}
};

const auto noneCases = ::testing::Combine(
    ::testing::ValuesIn(noneShapes),
    ::testing::ValuesIn(inputPrecision),
    ::testing::Values(ngraph::op::AutoBroadcastSpec::NONE),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(Select_none, SelectLayerTest, noneCases, SelectLayerTest::getTestCaseName);
