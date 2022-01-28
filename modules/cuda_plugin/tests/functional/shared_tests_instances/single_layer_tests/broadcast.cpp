// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/broadcast.hpp"

#include <cuda_test_constants.hpp>
#include <vector>

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32, InferenceEngine::Precision::I32};

// NUMPY MODE

INSTANTIATE_TEST_CASE_P(smoke_TestNumpyBroadcast1,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ngraph::Shape{2, 3, 6}),  // target shape
                                           ::testing::Values(ngraph::AxisSet{}),       // not used in numpy mode
                                           ::testing::Values(ngraph::op::BroadcastType::NUMPY),
                                           ::testing::Values(ngraph::Shape{3, 1}),  // input shape
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_TestNumpyBroadcast2,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ngraph::Shape{1, 4, 4}),  // target shape
                                           ::testing::Values(ngraph::AxisSet{}),       // not used in numpy mode
                                           ::testing::Values(ngraph::op::BroadcastType::NUMPY),
                                           ::testing::Values(ngraph::Shape{1, 4, 1}),  // input shape
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_TestNumpyBroadcast3,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ngraph::Shape{3, 1, 4}),  // target shape
                                           ::testing::Values(ngraph::AxisSet{}),       // not used in numpy mode
                                           ::testing::Values(ngraph::op::BroadcastType::NUMPY),
                                           ::testing::Values(ngraph::Shape{3, 1, 1}),  // input shape
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_TestNumpyBroadcast4,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ngraph::Shape{2, 3, 3, 3, 3, 3, 3, 3}),  // target shape
                                           ::testing::Values(ngraph::AxisSet{}),  // not used in numpy mode
                                           ::testing::Values(ngraph::op::BroadcastType::NUMPY),
                                           ::testing::Values(ngraph::Shape{1, 3, 1, 3, 1, 3, 1}),  // input shape
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        BroadcastLayerTest::getTestCaseName);

// BIDIRECTIONAL MODE

INSTANTIATE_TEST_CASE_P(smoke_TestBidirectionalBroadcast1,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ngraph::Shape{2, 1, 4}),  // target shape
                                           ::testing::Values(ngraph::AxisSet{}),       // not used in bidirectional mode
                                           ::testing::Values(ngraph::op::BroadcastType::BIDIRECTIONAL),
                                           ::testing::Values(ngraph::Shape{4, 1}),  // input shape
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_TestBidirectionalBroadcast2,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ngraph::Shape{1, 4, 4}),  // target shape
                                           ::testing::Values(ngraph::AxisSet{}),       // not used in bidirectional mode
                                           ::testing::Values(ngraph::op::BroadcastType::BIDIRECTIONAL),
                                           ::testing::Values(ngraph::Shape{1, 4, 1}),  // input shape
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_TestBidirectionalBroadcas3,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ngraph::Shape{1, 1, 2, 2}),  // target shape
                                           ::testing::Values(ngraph::AxisSet{}),  // not used in bidirectional mode
                                           ::testing::Values(ngraph::op::BroadcastType::BIDIRECTIONAL),
                                           ::testing::Values(ngraph::Shape{4, 1, 1}),  // input shape
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        BroadcastLayerTest::getTestCaseName);

// EXPLICIT MODE

INSTANTIATE_TEST_CASE_P(smoke_TestExplicitBroadcast1,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ngraph::Shape{2, 3, 1}),  // target shape
                                           ::testing::Values(ngraph::AxisSet{1, 2}),   // axes
                                           ::testing::Values(ngraph::op::BroadcastType::EXPLICIT),
                                           ::testing::Values(ngraph::Shape{3, 1}),  // input shape
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_TestExplicitBroadcast2,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ngraph::Shape{5, 3, 7}),  // target shape
                                           ::testing::Values(ngraph::AxisSet{0, 2}),   // axes
                                           ::testing::Values(ngraph::op::BroadcastType::EXPLICIT),
                                           ::testing::Values(ngraph::Shape{5, 7}),  // input shape
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_TestExplicitBroadcast3,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ngraph::Shape{4, 4, 3, 7, 6, 6}),  // target shape
                                           ::testing::Values(ngraph::AxisSet{1, 3, 5}),         // axes
                                           ::testing::Values(ngraph::op::BroadcastType::EXPLICIT),
                                           ::testing::Values(ngraph::Shape{4, 7, 6}),  // input shape
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        BroadcastLayerTest::getTestCaseName);

// YOLOv5 operators

const std::vector<InferenceEngine::Precision> precisionsYOLOv5 = {
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32,
};

INSTANTIATE_TEST_CASE_P(yolov5_BroadcastTest1,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ngraph::Shape{1, 3, 80, 80, 2}),
                                           ::testing::Values(ngraph::AxisSet{}),  // not used in bidirectional mode
                                           ::testing::Values(ngraph::op::BroadcastType::BIDIRECTIONAL),
                                           ::testing::Values(ngraph::Shape{1, 3, 80, 80, 2}),
                                           ::testing::ValuesIn(precisionsYOLOv5),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(yolov5_BroadcastTest2,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ngraph::Shape{1, 3, 40, 40, 2}),
                                           ::testing::Values(ngraph::AxisSet{}),  // not used in bidirectional mode
                                           ::testing::Values(ngraph::op::BroadcastType::BIDIRECTIONAL),
                                           ::testing::Values(ngraph::Shape{1, 3, 40, 40, 2}),
                                           ::testing::ValuesIn(precisionsYOLOv5),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(yolov5_BroadcastTest3,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ngraph::Shape{1, 3, 20, 20, 2}),
                                           ::testing::Values(ngraph::AxisSet{}),  // not used in bidirectional mode
                                           ::testing::Values(ngraph::op::BroadcastType::BIDIRECTIONAL),
                                           ::testing::Values(ngraph::Shape{1, 3, 20, 20, 2}),
                                           ::testing::ValuesIn(precisionsYOLOv5),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        BroadcastLayerTest::getTestCaseName);

}  // namespace
