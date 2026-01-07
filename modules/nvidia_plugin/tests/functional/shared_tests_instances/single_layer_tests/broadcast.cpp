// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/broadcast.hpp"

#include <cuda_test_constants.hpp>
#include <vector>

namespace {

using namespace ov::test;
using namespace ov::test::utils;

const std::vector<ov::element::Type> input_precisions = {
    ov::element::f16, ov::element::f32, ov::element::i32
};

// NUMPY MODE

INSTANTIATE_TEST_CASE_P(smoke_TestNumpyBroadcast1,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ov::Shape{2, 3, 6}),  // target shape
                                           ::testing::Values(ov::AxisSet{}),       // not used in numpy mode
                                           ::testing::Values(ov::op::BroadcastType::NUMPY),
                                           ::testing::Values(static_shapes_to_test_representation({ov::Shape{3, 1}})),  // input shape
                                           ::testing::ValuesIn(input_precisions),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_TestNumpyBroadcast2,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ov::Shape{1, 4, 4}),  // target shape
                                           ::testing::Values(ov::AxisSet{}),       // not used in numpy mode
                                           ::testing::Values(ov::op::BroadcastType::NUMPY),
                                           ::testing::Values(static_shapes_to_test_representation({ov::Shape{1, 4, 1}})),  // input shape
                                           ::testing::ValuesIn(input_precisions),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_TestNumpyBroadcast3,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ov::Shape{3, 1, 4}),  // target shape
                                           ::testing::Values(ov::AxisSet{}),       // not used in numpy mode
                                           ::testing::Values(ov::op::BroadcastType::NUMPY),
                                           ::testing::Values(static_shapes_to_test_representation({ov::Shape{3, 1, 1}})),  // input shape
                                           ::testing::ValuesIn(input_precisions),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_TestNumpyBroadcast4,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ov::Shape{2, 3, 3, 3, 3, 3, 3, 3}),  // target shape
                                           ::testing::Values(ov::AxisSet{}),  // not used in numpy mode
                                           ::testing::Values(ov::op::BroadcastType::NUMPY),
                                           ::testing::Values(static_shapes_to_test_representation({ov::Shape{1, 3, 1, 3, 1, 3, 1}})),  // input shape
                                           ::testing::ValuesIn(input_precisions),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        BroadcastLayerTest::getTestCaseName);

// BIDIRECTIONAL MODE

INSTANTIATE_TEST_CASE_P(smoke_TestBidirectionalBroadcast1,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ov::Shape{2, 1, 4}),  // target shape
                                           ::testing::Values(ov::AxisSet{}),       // not used in bidirectional mode
                                           ::testing::Values(ov::op::BroadcastType::BIDIRECTIONAL),
                                           ::testing::Values(static_shapes_to_test_representation({ov::Shape{4, 1}})),  // input shape
                                           ::testing::ValuesIn(input_precisions),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_TestBidirectionalBroadcast2,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ov::Shape{1, 4, 4}),  // target shape
                                           ::testing::Values(ov::AxisSet{}),       // not used in bidirectional mode
                                           ::testing::Values(ov::op::BroadcastType::BIDIRECTIONAL),
                                           ::testing::Values(static_shapes_to_test_representation({ov::Shape{1, 4, 1}})),  // input shape
                                           ::testing::ValuesIn(input_precisions),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_TestBidirectionalBroadcas3,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ov::Shape{1, 1, 2, 2}),  // target shape
                                           ::testing::Values(ov::AxisSet{}),          // not used in bidirectional mode
                                           ::testing::Values(ov::op::BroadcastType::BIDIRECTIONAL),
                                           ::testing::Values(static_shapes_to_test_representation({ov::Shape{4, 1, 1}})),  // input shape
                                           ::testing::ValuesIn(input_precisions),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        BroadcastLayerTest::getTestCaseName);

// EXPLICIT MODE

INSTANTIATE_TEST_CASE_P(smoke_TestExplicitBroadcast1,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ov::Shape{2, 3, 1}),  // target shape
                                           ::testing::Values(ov::AxisSet{1, 2}),   // axes
                                           ::testing::Values(ov::op::BroadcastType::EXPLICIT),
                                           ::testing::Values(static_shapes_to_test_representation({ov::Shape{3, 1}})),  // input shape
                                           ::testing::ValuesIn(input_precisions),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_TestExplicitBroadcast2,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ov::Shape{5, 3, 7}),  // target shape
                                           ::testing::Values(ov::AxisSet{0, 2}),   // axes
                                           ::testing::Values(ov::op::BroadcastType::EXPLICIT),
                                           ::testing::Values(static_shapes_to_test_representation({ov::Shape{5, 7}})),  // input shape
                                           ::testing::ValuesIn(input_precisions),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_TestExplicitBroadcast3,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ov::Shape{4, 4, 3, 7, 6, 6}),  // target shape
                                           ::testing::Values(ov::AxisSet{1, 3, 5}),         // axes
                                           ::testing::Values(ov::op::BroadcastType::EXPLICIT),
                                           ::testing::Values(static_shapes_to_test_representation({ov::Shape{4, 7, 6}})),  // input shape
                                           ::testing::ValuesIn(input_precisions),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        BroadcastLayerTest::getTestCaseName);

// YOLOv5 operators

const std::vector<ov::element::Type> precisions_YOLOv5 = {
    ov::element::f16, ov::element::f32
};

INSTANTIATE_TEST_CASE_P(yolov5_BroadcastTest1,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ov::Shape{1, 3, 80, 80, 2}),
                                           ::testing::Values(ov::AxisSet{}),  // not used in bidirectional mode
                                           ::testing::Values(ov::op::BroadcastType::BIDIRECTIONAL),
                                           ::testing::Values(static_shapes_to_test_representation({ov::Shape{1, 3, 80, 80, 2}})),
                                           ::testing::ValuesIn(precisions_YOLOv5),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(yolov5_BroadcastTest2,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ov::Shape{1, 3, 40, 40, 2}),
                                           ::testing::Values(ov::AxisSet{}),  // not used in bidirectional mode
                                           ::testing::Values(ov::op::BroadcastType::BIDIRECTIONAL),
                                           ::testing::Values(static_shapes_to_test_representation({ov::Shape{1, 3, 40, 40, 2}})),
                                           ::testing::ValuesIn(precisions_YOLOv5),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(yolov5_BroadcastTest3,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::Values(ov::Shape{1, 3, 20, 20, 2}),
                                           ::testing::Values(ov::AxisSet{}),  // not used in bidirectional mode
                                           ::testing::Values(ov::op::BroadcastType::BIDIRECTIONAL),
                                           ::testing::Values(static_shapes_to_test_representation({ov::Shape{1, 3, 20, 20, 2}})),
                                           ::testing::ValuesIn(precisions_YOLOv5),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        BroadcastLayerTest::getTestCaseName);

}  // namespace
