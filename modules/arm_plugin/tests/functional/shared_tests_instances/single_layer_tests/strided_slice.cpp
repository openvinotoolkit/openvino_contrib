// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <vector>

#include "single_layer_tests/strided_slice.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

std::vector<StridedSliceSpecificParams> ss_only_test_cases = {
        StridedSliceSpecificParams{ { 2, 2, 2, 2 }, { 0, 0, 0, 0 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                            {}, {},  {},  {},  {} },
        StridedSliceSpecificParams{ { 2, 2, 4, 3 }, { 0, 0, 0, 0 }, { 2, 2, 4, 3 }, { 1, 1, 2, 1},
                            {1, 0, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},  {0, 1, 0, 0} },
        StridedSliceSpecificParams{ { 2, 2, 2, 2 }, { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                            {0, 0, 0, 0}, {0, 0, 0, 0},  {0, 0, 0, 0},  {0, 0, 0, 0},  {0, 0, 0, 0} },
        StridedSliceSpecificParams{ {1, 3, 4, 2}, {0, 0, -2, 0}, {1, -2, 4, -1}, {1, 1, 1, 1},
                            {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},  {0, 0, 0, 0},  {0, 0, 0, 0} },
        StridedSliceSpecificParams{ {2, 2, 4, 5}, {0, 0, 0, 2}, { 1, 2, 3, 4}, {1, 1, 1, 1},
                            {0, 0, 0, 0}, {1, 1, 1, 1},  {0, 0, 0, 0},  {0, 0, 0, 0},  {0, 0, 0, 0} },
        StridedSliceSpecificParams{ { 1, 12, 100 }, { 0, -1, 0 }, { 0, 0, 0 }, { 1, 1, 1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 1, 0 },  { 0, 0, 0 } },
        StridedSliceSpecificParams{ { 1, 12, 100 }, { 0, 1, 0 }, { 0, -1, 0 }, { 1, 1, 1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        StridedSliceSpecificParams{ { 1, 12, 100 }, { 0, 7, 0 }, { 0, 9, 0 }, { -1, 1, -1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        StridedSliceSpecificParams{ { 1, 12, 100 }, { 0, 4, 0 }, { 0, 10, 0 }, { -1, 2, -1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        StridedSliceSpecificParams{ { 1, 12, 100 }, { 0, 11, 0 }, { 0, 0, 0 }, { -1, -2, -1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        StridedSliceSpecificParams{ { 10, 12 }, { -1, 1 }, { -9999, 0 }, { -1, 1 },
                            { 0, 1 }, { 0, 1 },  { 0, 0 },  { 0, 0 },  { 0, 0 } },
        StridedSliceSpecificParams{ { 2, 2, 2, 2 }, { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                            {0, 0, 0, 0}, {1, 1, 1, 1},  {},  {},  {} },
        StridedSliceSpecificParams{ { 1, 12, 100, 10 }, { 0, 0, 0 }, { 1, 12, 100 }, { 1, 1, 1 },
                            {  }, {  },  { },  {  },  {} },
        StridedSliceSpecificParams{ { 2, 2, 4, 2 }, { 1, 0, 0, 1 }, { 2, 2, 4, 2 }, { 1, 1, 2, 1 },
                            {0, 1, 1, 0}, {1, 1, 0, 0},  {},  {},  {} },
        StridedSliceSpecificParams{ { 1, 2 }, { 0, 0 }, { 1, 2 }, { 1, -1 },
                            {1, 1}, {1, 1},  {},  {},  {} },
        StridedSliceSpecificParams{ { 2, 2, 4, 2 }, { 1, 0, 0, 0 }, { 1, 2, 4, 2 }, { 1, 1, -2, -1 },
                            {0, 1, 1, 1}, {1, 1, 1, 1},  {},  {},  {} },
        StridedSliceSpecificParams{ { 1, 12, 100, 1, 1 }, { 0, -1, 0, 0 }, { 0, 0, 0, 0 }, { 1, 1, 1, 1 },
                            { 1, 0, 1, 0 }, { 1, 0, 1, 0 },  { },  { 0, 1, 0, 1 },  {} },
        StridedSliceSpecificParams{ { 1, 12, 100, 2, 4 }, { 0, -1, 0, 0, 1 }, {1, 0, 0, 0, 0 }, { 1, 1, 2, 1, 1 },
                            { 1, 0, 1, 0, 0 }, { 1, 0, 1, 0, 0 },  { },  { 0, 1, 0, 1, 1 },  {} },
        StridedSliceSpecificParams{ { 2, 3, 4, 5, 6 }, { 0, 1, 0, 0, 0 }, { 2, 3, 4, 5, 6 }, { 1, 1, 1, 1, 1 },
                            {1, 0, 1, 1, 1}, {1, 0, 1, 1, 1},  {},  {0, 1, 0, 0, 0},  {} },
        StridedSliceSpecificParams{ { 5, 5, 5, 5 }, { -1, 0, -1, 0 }, { -50, 0, -60, 0 }, { -1, 1, -1, 1 },
                                { 0, 0, 0, 0 }, { 0, 1, 0, 1 },  { 0, 0, 0, 0 },  { 0, 0, 0, 0 },  { 0, 0, 0, 0 } },
        StridedSliceSpecificParams{ { 1, 12, 100 }, { 0, -6, 0 }, { 0, -8, 0 }, { -1, -2, -1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        };

const std::vector<InferenceEngine::Precision> precisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I64,
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::U8,
};

INSTANTIATE_TEST_CASE_P(smoke_StridedSliceTest,
        StridedSliceLayerTest,
        ::testing::Combine(
            ::testing::ValuesIn(ss_only_test_cases),
            ::testing::ValuesIn(precisions),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::Values(std::map<std::string, std::string>())),
        StridedSliceLayerTest::getTestCaseName);
}  // namespace





