// Copyright (C) 2020 Intel Corporation
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
};

const std::vector<float> eps = {
        1e-7f,
        1e-6f,
        1e-5f,
        1e-4f
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
        testing::Values("ARM")
);

INSTANTIATE_TEST_CASE_P(
        NormalizeL2_2D,
        NormalizeL2LayerTest,
        normL2params2D,
        NormalizeL2LayerTest::getTestCaseName
);

const std::vector<std::vector<int64_t>> axes4D = {
        {1},
        {2},
        {3},
};

const auto normL2params4D = testing::Combine(
        testing::ValuesIn(axes4D),
        testing::ValuesIn(eps),
        testing::ValuesIn(epsModes),
        testing::Values(std::vector<size_t>{1, 3, 10, 5}),
        testing::ValuesIn(netPrecisions),
        testing::Values("ARM")
);

INSTANTIATE_TEST_CASE_P(
        NormalizeL2_4D,
        NormalizeL2LayerTest,
        normL2params4D,
        NormalizeL2LayerTest::getTestCaseName
);
}  // namespace
