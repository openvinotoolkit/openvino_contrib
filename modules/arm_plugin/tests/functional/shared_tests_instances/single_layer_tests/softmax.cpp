// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "single_layer_tests/softmax.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
};

const std::vector<std::pair<ngraph::PartialShape, std::vector<ngraph::Shape>>> inputShapes2D = {
        {{}, {{1, 100}}},
        {{}, {{2, 2}}},
        {{}, {{3, 1}}},
        {{}, {{3, 2}}},
};

const std::vector<size_t> axis2D = {0, 1};

const auto params2D = testing::Combine(
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::NC),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputShapes2D),
        testing::ValuesIn(axis2D),
        testing::Values(CommonTestUtils::DEVICE_CPU),
        testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_CASE_P(
        smoke_SoftMax2D,
        SoftMaxLayerTest,
        params2D,
        SoftMaxLayerTest::getTestCaseName
);

const std::vector<std::pair<ngraph::PartialShape, std::vector<ngraph::Shape>>> inputShapes4D = {
        {{}, {{1, 10, 1, 1}}},
        {{}, {{1, 3, 10, 10}}},
        {{}, {{2, 3, 4, 5}}},
};

const std::vector<size_t> axis4D = {0, 1, 2, 3};

const auto params4D = testing::Combine(
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::NCHW),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputShapes4D),
        testing::ValuesIn(axis4D),
        testing::Values(CommonTestUtils::DEVICE_CPU),
        testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_CASE_P(
        smoke_SoftMax4D,
        SoftMaxLayerTest,
        params4D,
        SoftMaxLayerTest::getTestCaseName
);

}  // namespace