// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/variadic_split_pad.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecision = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<InferenceEngine::SizeVector> shapes = {
        {1, 8, 3, 2},
        {3, 8, 8, 8},
};

const std::vector<std::vector<size_t>> connectedIndexes = {
        {0},
        {0, 2},
        {0, 1, 3},
        {0, 1, 1, 0},
        {0, 0, 0, 1},
};

const std::vector<std::vector<size_t>> numSplits = {
        {2, 2, 2, 2},
        {1, 2, 4, 1},
        {3, 2, 2, 1}
};

const std::vector<std::vector<int64_t>> padsBegin = {
        {0, 0, 0, 0},
        {0, 0, 1, 1},
};

const std::vector<std::vector<int64_t>> padsEnd = {
        {0, 0, 0, 0},
        {0, 0, 1, 1},
};

const std::vector<ngraph::helpers::PadMode> padMode = {
        ngraph::helpers::PadMode::CONSTANT,
        ngraph::helpers::PadMode::REFLECT,
};

INSTANTIATE_TEST_CASE_P(smoke_Check, VariadicSplitPad,
                        ::testing::Combine(
                                ::testing::ValuesIn(shapes),
                                ::testing::Values(1),
                                ::testing::ValuesIn(numSplits),
                                ::testing::ValuesIn(connectedIndexes),
                                ::testing::ValuesIn(padsBegin),
                                ::testing::ValuesIn(padsEnd),
                                ::testing::ValuesIn(padMode),
                                ::testing::ValuesIn(netPrecision),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        VariadicSplitPad::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Check_SymmetricPad, VariadicSplitPad,
                        ::testing::Combine(
                                ::testing::ValuesIn(shapes),
                                ::testing::Values(1),
                                ::testing::ValuesIn(numSplits),
                                ::testing::ValuesIn(connectedIndexes),
                                ::testing::Values(padsBegin[0]),
                                ::testing::Values(padsEnd[0]),
                                ::testing::Values(ngraph::helpers::PadMode::SYMMETRIC),
                                ::testing::ValuesIn(netPrecision),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        VariadicSplitPad::getTestCaseName);
}  // namespace
