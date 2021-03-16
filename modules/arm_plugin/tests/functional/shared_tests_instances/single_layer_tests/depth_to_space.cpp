// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <vector>
#include <ngraph/opsets/opset3.hpp>

#include "single_layer_tests/depth_to_space.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::opset3;

namespace {
const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
};

const std::vector<DepthToSpace::DepthToSpaceMode> modes = {
        DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
};

const std::vector<std::vector<size_t >> inputShapesBS2 = {
        {1, 4, 1, 1}, {1, 4, 2, 2}, {1, 4, 3, 3}, {2, 32, 3, 3}, {2, 16, 5, 4},
};

const auto DepthToSpaceBS2 = ::testing::Combine(
        ::testing::ValuesIn(inputShapesBS2),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(modes),
        ::testing::Values(2),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(DepthToSpaceBS2Test, DepthToSpaceLayerTest, DepthToSpaceBS2, DepthToSpaceLayerTest::getTestCaseName);

const std::vector<std::vector<size_t >> inputShapesBS3 = {
        {1, 9, 1, 1}, {1, 9, 2, 2}, {1, 9, 3, 3}, {2, 36, 3, 3}, {2, 27, 5, 4},
};

const auto DepthToSpaceBS3 = ::testing::Combine(
        ::testing::ValuesIn(inputShapesBS3),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(modes),
        ::testing::Values(3),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(DepthToSpaceBS3Test, DepthToSpaceLayerTest, DepthToSpaceBS3, DepthToSpaceLayerTest::getTestCaseName);

}  // namespace