// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <vector>
#include "single_layer_tests/lrn.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<double> alpha  = {1e-06};
const std::vector<double> beta   = {0.75, 1, 1.1};
const std::vector<double> bias   = {0.9, 1, 2};
const std::vector<size_t> size   = {3, 5};
const std::vector<std::vector<int64_t>> reduction_axes = {{1}, {2, 3}};

INSTANTIATE_TEST_CASE_P(LrnCheck, LrnLayerTest,
                        ::testing::Combine(::testing::ValuesIn(alpha),
                                           ::testing::ValuesIn(beta),
                                           ::testing::ValuesIn(bias),
                                           ::testing::ValuesIn(size),
                                           ::testing::ValuesIn(reduction_axes),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(std::vector<size_t>({10, 10, 10, 10})),
                                           ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        LrnLayerTest::getTestCaseName);
}  // namespace
