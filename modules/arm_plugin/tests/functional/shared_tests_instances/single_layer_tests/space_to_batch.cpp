// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <vector>

#include "single_layer_tests/space_to_batch.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

spaceToBatchParamsTuple cases[] = {
        spaceToBatchParamsTuple({1, 1, 2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {1, 1, 2, 2},
                                InferenceEngine::Precision::FP32,
                                InferenceEngine::Precision::UNSPECIFIED,
                                InferenceEngine::Precision::UNSPECIFIED,
                                InferenceEngine::Layout::ANY,
                                InferenceEngine::Layout::ANY,
                                CommonTestUtils::DEVICE_CPU),
        spaceToBatchParamsTuple({1, 1, 2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {1, 3, 2, 2},
                                InferenceEngine::Precision::FP32,
                                InferenceEngine::Precision::UNSPECIFIED,
                                InferenceEngine::Precision::UNSPECIFIED,
                                InferenceEngine::Layout::ANY,
                                InferenceEngine::Layout::ANY,
                                CommonTestUtils::DEVICE_CPU),
        spaceToBatchParamsTuple({1, 1, 2, 2}, {0, 0, 1, 1}, {0, 0, 1, 1}, {1, 1, 4, 4},
                                InferenceEngine::Precision::FP32,
                                InferenceEngine::Precision::UNSPECIFIED,
                                InferenceEngine::Precision::UNSPECIFIED,
                                InferenceEngine::Layout::ANY,
                                InferenceEngine::Layout::ANY,
                                CommonTestUtils::DEVICE_CPU),
        spaceToBatchParamsTuple({1, 1, 2, 2}, {0, 0, 0, 1}, {0, 0, 0, 1}, {1, 1, 4, 4},
                                InferenceEngine::Precision::FP32,
                                InferenceEngine::Precision::UNSPECIFIED,
                                InferenceEngine::Precision::UNSPECIFIED,
                                InferenceEngine::Layout::ANY,
                                InferenceEngine::Layout::ANY,
                                CommonTestUtils::DEVICE_CPU),
        spaceToBatchParamsTuple({1, 1, 2, 2}, {0, 0, 2, 2}, {0, 0, 2, 2}, {3, 3, 6, 4},
                                InferenceEngine::Precision::FP32,
                                InferenceEngine::Precision::UNSPECIFIED,
                                InferenceEngine::Precision::UNSPECIFIED,
                                InferenceEngine::Layout::ANY,
                                InferenceEngine::Layout::ANY,
                                CommonTestUtils::DEVICE_CPU),
        spaceToBatchParamsTuple({1, 1, 2, 2}, {0, 0, 1, 0}, {0, 0, 1, 0}, {1, 1, 4, 4},
                                InferenceEngine::Precision::FP32,
                                InferenceEngine::Precision::UNSPECIFIED,
                                InferenceEngine::Precision::UNSPECIFIED,
                                InferenceEngine::Layout::ANY,
                                InferenceEngine::Layout::ANY,
                                CommonTestUtils::DEVICE_CPU),
};

INSTANTIATE_TEST_CASE_P(SpaceToBatch, SpaceToBatchLayerTest, ::testing::ValuesIn(cases), SpaceToBatchLayerTest::getTestCaseName);


}  // namespace