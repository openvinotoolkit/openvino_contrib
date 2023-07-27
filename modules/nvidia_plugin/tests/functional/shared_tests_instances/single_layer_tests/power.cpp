// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/power.hpp"

#include <vector>

#include "cuda_test_constants.hpp"
#include "memory_manager/tensor_types.hpp"

using namespace LayerTestsDefinitions;

namespace {

std::vector<std::vector<std::vector<size_t>>> inShapes = {{{1, 8}},
                                                          {{2, 16}},
                                                          {{3, 32}},
                                                          {{4, 64}},
                                                          {{5, 128}},
                                                          {{6, 256}},
                                                          {{7, 512}},
                                                          {{8, 1024}},
                                                          {{5}},
                                                          {{8}},
                                                          // yolov5-640x640-IR power operation shapes
                                                          {{1, 3, 80, 80, 2}},
                                                          {{1, 3, 40, 40, 2}},
                                                          {{1, 3, 20, 20, 2}}};

std::vector<std::vector<float>> Power = {
    {0.0f},
    {0.5f},
    {1.0f},
    {1.1f},
    {1.5f},
    {2.0f},
};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

INSTANTIATE_TEST_CASE_P(smoke_powerCuda,
                        PowerLayerTest,
                        ::testing::Combine(::testing::ValuesIn(inShapes),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                                           ::testing::ValuesIn(Power)),
                        PowerLayerTest::getTestCaseName);

}  // namespace
