// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_test_constants.hpp>

#include <vector>

#include "single_layer_tests/transpose.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<size_t>> inputShapes = {
        std::vector<size_t>{256, 3, 100, 100},
        std::vector<size_t>{1, 2048, 1, 1},
};

const std::vector<std::vector<size_t>> inputOrder = {
        std::vector<size_t>{0, 3, 2, 1},
// Empty inputs are currently unsupported in CUDAPlugin.
//        std::vector<size_t>{},
};

const auto params = testing::Combine(
        testing::ValuesIn(inputOrder),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputShapes),
        testing::Values(CommonTestUtils::DEVICE_CUDA)
);

INSTANTIATE_TEST_CASE_P(
        smoke_Transpose,
        TransposeLayerTest,
        params,
        TransposeLayerTest::getTestCaseName
);

}  // namespace
