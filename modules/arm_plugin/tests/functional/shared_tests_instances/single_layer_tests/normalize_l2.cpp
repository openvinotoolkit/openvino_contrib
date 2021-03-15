// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "single_layer_tests/normalize_l2.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::vector<int64_t>> axes2D = {
        {0},
        {1},
        {0, 1}
};

const std::vector<float> eps = {
        1e-7f,
        1e-6f,
        1e-5f,
        1e-4f,
        1
};

const std::vector<ngraph::op::EpsMode> epsModes = {
        ngraph::op::EpsMode::ADD,
        ngraph::op::EpsMode::MAX
};

const auto normL2params2D = testing::Combine(
        testing::ValuesIn(axes2D),
        testing::ValuesIn(eps),
        testing::ValuesIn(epsModes),
        testing::Values(std::vector<size_t>{10, 5}),
        testing::ValuesIn(netPrecisions),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(
        NormalizeL2_2D,
        NormalizeL2LayerTest,
        normL2params2D,
        NormalizeL2LayerTest::getTestCaseName
);

const std::vector<std::vector<int64_t>> axesDecompose4D = {
        {0},
        {1},
        {2},
        {3},
        {0, 1},
        {1, 2, 3},
        {0, 1, 2, 3},
};

const auto normL2DecomposeParams4D = testing::Combine(
        testing::ValuesIn(axesDecompose4D),
        testing::ValuesIn(eps),
        testing::Values(ngraph::op::EpsMode::ADD),
        testing::Values(std::vector<size_t>{2, 3, 10, 5}),
        testing::ValuesIn(netPrecisions),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(
        NormalizeL2Decompose_4D,
        NormalizeL2LayerTest,
        normL2DecomposeParams4D,
        NormalizeL2LayerTest::getTestCaseName
);
}  // namespace
