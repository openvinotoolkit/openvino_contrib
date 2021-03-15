// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/reorg_yolo.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

const std::vector<ngraph::Shape> inShapes = {
    {1, 4, 4, 4},
    {1, 8, 4, 4},
    {1, 24, 34, 62},
    {2, 8, 4, 4},
    {3, 8, 4, 4},
};

const std::vector<size_t> strides = {
    1, 2,
};


const auto testCase_caffe_yolov2 = ::testing::Combine(
    ::testing::Values(ngraph::Shape{1, 64, 26, 26}),
    ::testing::ValuesIn(strides),
    ::testing::Values(InferenceEngine::Precision::FP32),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto testCase_smallest = ::testing::Combine(
    ::testing::ValuesIn(inShapes),
    ::testing::ValuesIn(strides),
    ::testing::Values(InferenceEngine::Precision::FP32),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto testCase_stride_3 = ::testing::Combine(
    ::testing::Values(ngraph::Shape{1, 9, 3, 3}),
    ::testing::Values(3),
    ::testing::Values(InferenceEngine::Precision::FP32),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);


INSTANTIATE_TEST_CASE_P(smoke_TestsReorgYolo_caffe_YoloV2, ReorgYoloLayerTest, testCase_caffe_yolov2, ReorgYoloLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_TestsReorgYolo_stride_2_smallest, ReorgYoloLayerTest, testCase_smallest, ReorgYoloLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_TestsReorgYolo_stride_3, ReorgYoloLayerTest, testCase_stride_3, ReorgYoloLayerTest::getTestCaseName);
