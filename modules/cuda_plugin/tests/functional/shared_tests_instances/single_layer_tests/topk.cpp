// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/topk.hpp"

#include <vector>

#include "cuda_test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const std::vector<int64_t> k = {
    1,
    5,
    10,
};

const std::vector<ngraph::opset4::TopK::Mode> modes = {
    ngraph::opset4::TopK::Mode::MIN,
    ngraph::opset4::TopK::Mode::MAX,
};

const std::vector<ngraph::opset4::TopK::SortType> sortTypes = {
    ngraph::opset4::TopK::SortType::SORT_INDICES,
    ngraph::opset4::TopK::SortType::SORT_VALUES,
};

const std::vector<int64_t> axes3D = {
    0,
    1,
    2,
};

const std::vector<std::vector<size_t>> shapes3D = {
    {10, 10, 10},
    {15, 10, 10},
    {25, 15, 10},
};

INSTANTIATE_TEST_CASE_P(smoke_TopK3D,
                        TopKLayerTest,
                        ::testing::Combine(::testing::ValuesIn(k),
                                           ::testing::ValuesIn(axes3D),
                                           ::testing::ValuesIn(modes),
                                           ::testing::ValuesIn(sortTypes),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(shapes3D),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        TopKLayerTest::getTestCaseName);

const std::vector<int64_t> axes4D = {
    0,
    1,
    2,
    3,
};

const std::vector<std::vector<size_t>> shapes4D = {
    {10, 10, 10, 10},
    {15, 10, 10, 10},
    {25, 15, 10, 10},
};

INSTANTIATE_TEST_CASE_P(TopK4D,
                        TopKLayerTest,
                        ::testing::Combine(::testing::ValuesIn(k),
                                           ::testing::ValuesIn(axes4D),
                                           ::testing::ValuesIn(modes),
                                           ::testing::ValuesIn(sortTypes),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(shapes4D),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        TopKLayerTest::getTestCaseName);

const std::vector<int64_t> axes5D = {
    0,
    1,
    2,
    3,
    4,
};

const std::vector<std::vector<size_t>> shapes5D = {
    {10, 10, 10, 10, 10},
    {15, 10, 10, 10, 10},
    {25, 15, 10, 10, 10},
};

INSTANTIATE_TEST_CASE_P(TopK5D,
                        TopKLayerTest,
                        ::testing::Combine(::testing::ValuesIn(k),
                                           ::testing::ValuesIn(axes5D),
                                           ::testing::ValuesIn(modes),
                                           ::testing::ValuesIn(sortTypes),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(shapes5D),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        TopKLayerTest::getTestCaseName);

}  // namespace
