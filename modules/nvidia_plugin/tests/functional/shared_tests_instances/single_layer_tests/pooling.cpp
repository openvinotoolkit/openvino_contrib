// Copyright (C) 2019 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_test_constants.hpp>
#include <single_layer_tests/pooling.hpp>
#include <vector>

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::vector<size_t>> kernels = {{3, 3}, {3, 5}, {2, 2}};
const std::vector<std::vector<size_t>> strides = {{1, 1}, {1, 2}, {2, 2}};
// TODO: find why paddings >0 for the first spatial axis, and/or paddings
// >1 for the second spatial axis fail.
// Note: asymmetric paddings (begin != end) are not supported in cuDNN.
const std::vector<std::vector<size_t>> padBegins = {{0, 1} /*, {0, 0}*/};
const std::vector<std::vector<size_t>> padEnds = {{0, 1} /*, {0, 0}*/};
const std::vector<ov::op::RoundingType> roundingTypes = {ov::op::RoundingType::CEIL, ov::op::RoundingType::FLOOR};

////* ========== Max Poolling ========== */
/* +========== Explicit Pad Floor Rounding ========== */
const auto maxPool_ExplicitPad_FloorRounding_Params =
    ::testing::Combine(::testing::Values(PoolingTypes::MAX),
                       ::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::ValuesIn(padBegins),
                       ::testing::ValuesIn(padEnds),
                       ::testing::Values(ov::op::RoundingType::FLOOR),
                       ::testing::Values(ov::op::PadType::EXPLICIT),
                       ::testing::Values(false)  // placeholder value - exclude pad not applicable
                                                 // for max pooling
    );

INSTANTIATE_TEST_CASE_P(smoke_MaxPool_ExplicitPad_FloorRounding,
                        PoolingLayerTest,
                        ::testing::Combine(maxPool_ExplicitPad_FloorRounding_Params,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 50, 50})),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        PoolingLayerTest::getTestCaseName);

/* ========== Explicit Pad Ceil Rounding ========== */
const auto maxPool_ExplicitPad_CeilRounding_Params =
    ::testing::Combine(::testing::Values(PoolingTypes::MAX),
                       ::testing::ValuesIn(kernels),
                       // TODO: Non 1 strides fails in ngraph reference implementation with error
                       // "The end corner is out of bounds at axis 3" thrown in the test body.
                       ::testing::Values(std::vector<size_t>({1, 1})),
                       ::testing::ValuesIn(padBegins),
                       ::testing::ValuesIn(padEnds),
                       ::testing::Values(ov::op::RoundingType::CEIL),
                       ::testing::Values(ov::op::PadType::EXPLICIT),
                       ::testing::Values(false)  // placeholder value - exclude pad not applicable
                                                 // for max pooling
    );

INSTANTIATE_TEST_CASE_P(smoke_MaxPool_ExplicitPad_CeilRounding,
                        PoolingLayerTest,
                        ::testing::Combine(maxPool_ExplicitPad_CeilRounding_Params,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 50, 50})),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        PoolingLayerTest::getTestCaseName);

////* ========== Avg Pooling ========== */
/* +========== Explicit Pad Ceil Rounding ========== */
const auto avgPoolExplicitPadCeilRoundingParams =
    ::testing::Combine(::testing::Values(PoolingTypes::AVG),
                       ::testing::ValuesIn(kernels),
                       // TODO: Non 1 strides fails in ngraph reference implementation with error
                       // "The end corner is out of bounds at axis 3" thrown in the test body.
                       ::testing::Values(std::vector<size_t>({1, 1})),
                       ::testing::ValuesIn(padBegins),
                       ::testing::ValuesIn(padEnds),
                       ::testing::Values(ov::op::RoundingType::CEIL),
                       ::testing::Values(ov::op::PadType::EXPLICIT),
                       ::testing::Values(true, false));

INSTANTIATE_TEST_CASE_P(smoke_AvgPool_ExplicitPad_CeilRounding,
                        PoolingLayerTest,
                        ::testing::Combine(avgPoolExplicitPadCeilRoundingParams,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        PoolingLayerTest::getTestCaseName);

/* +========== Explicit Pad Floor Rounding ========== */
const auto avgPoolExplicitPadFloorRoundingParams = ::testing::Combine(::testing::Values(PoolingTypes::AVG),
                                                                      ::testing::ValuesIn(kernels),
                                                                      ::testing::ValuesIn(strides),
                                                                      ::testing::ValuesIn(padBegins),
                                                                      ::testing::ValuesIn(padEnds),
                                                                      ::testing::Values(ov::op::RoundingType::FLOOR),
                                                                      ::testing::Values(ov::op::PadType::EXPLICIT),
                                                                      ::testing::Values(true, false));

INSTANTIATE_TEST_CASE_P(smoke_AvgPool_ExplicitPad_FloorRounding,
                        PoolingLayerTest,
                        ::testing::Combine(avgPoolExplicitPadFloorRoundingParams,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        PoolingLayerTest::getTestCaseName);

////* ========== Avg and Max Pooling Cases ========== */
/*    ========== Valid Pad Rounding Not Applicable ========== */
const auto allPools_ValidPad_Params =
    ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),
                       ::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::Values(std::vector<size_t>({0, 0})),
                       ::testing::ValuesIn(padEnds),
                       ::testing::Values(ov::op::RoundingType::FLOOR),  // placeholder value - Rounding Type
                                                                        // not applicable for Valid pad type
                       // TODO: PadType::VALID seems not to ignore padBegins
                       ::testing::Values(ov::op::PadType::VALID),
                       ::testing::Values(false)  // placeholder value - exclude pad not applicable
                                                 // for max pooling
    );

INSTANTIATE_TEST_CASE_P(smoke_MAX_and_AVGPool_ValidPad,
                        PoolingLayerTest,
                        ::testing::Combine(allPools_ValidPad_Params,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 50, 50})),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        PoolingLayerTest::getTestCaseName);

// =============================================================================
// clang-format off
// {AUTOGENERATED_TESTS_BEGIN_TAG_MAXPOOL}

// Attrs:  {'auto_pad': 'explicit', 'kernel': '13,13', 'pads_begin': '6,6', 'pads_end': '6,6', 'rounding_type': 'floor', 'strides': '1,1'}
// In:     (1, 256, 20, 20)
// Out:    (1, 256, 20, 20)
// Operators: 'yolov5-640x640-IR:opid191' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_yolov5_640x640_IR_opid191,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{13, 13}), // kernel
            ::testing::Values(std::vector<size_t>{1, 1}), // strides
            ::testing::Values(std::vector<size_t>{6, 6}), // pad begin
            ::testing::Values(std::vector<size_t>{6, 6}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::EXPLICIT), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 256, 20, 20}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'kernel': '2,2', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'ceil', 'strides': '2,2'}
// In:     (1, 128, 112, 112)
// Out:    (1, 128, 56, 56)
// Operators: 'vgg16-IR:opid24' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_vgg16_IR_opid24,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{2, 2}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::CEIL), // rounding type
            ::testing::Values(ov::op::PadType::EXPLICIT), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 128, 112, 112}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'kernel': '2,2', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'ceil', 'strides': '2,2'}
// In:     (1, 256, 56, 56)
// Out:    (1, 256, 28, 28)
// Operators: 'vgg16-IR:opid40' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_vgg16_IR_opid40,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{2, 2}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::CEIL), // rounding type
            ::testing::Values(ov::op::PadType::EXPLICIT), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 256, 56, 56}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'kernel': '2,2', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'ceil', 'strides': '2,2'}
// In:     (1, 512, 14, 14)
// Out:    (1, 512, 7, 7)
// Operators: 'vgg16-IR:opid72' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_vgg16_IR_opid72,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{2, 2}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::CEIL), // rounding type
            ::testing::Values(ov::op::PadType::EXPLICIT), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 512, 14, 14}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'kernel': '2,2', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'ceil', 'strides': '2,2'}
// In:     (1, 512, 28, 28)
// Out:    (1, 512, 14, 14)
// Operators: 'vgg16-IR:opid56' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_vgg16_IR_opid56,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{2, 2}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::CEIL), // rounding type
            ::testing::Values(ov::op::PadType::EXPLICIT), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 512, 28, 28}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'kernel': '2,2', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'ceil', 'strides': '2,2'}
// In:     (1, 64, 224, 224)
// Out:    (1, 64, 112, 112)
// Operators: 'vgg16-IR:opid13' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_vgg16_IR_opid13,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{2, 2}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::CEIL), // rounding type
            ::testing::Values(ov::op::PadType::EXPLICIT), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 64, 224, 224}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'ceil', 'strides': '2,2'}
// In:     (1, 128, 56, 56)
// Out:    (1, 128, 28, 28)
// Operators: 'squeezenet1.1:opid41' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    DISABLED_autogen_MaxPool_squeezenet1_1_opid41,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::CEIL), // rounding type
            ::testing::Values(ov::op::PadType::EXPLICIT), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 128, 56, 56}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'ceil', 'strides': '2,2'}
// In:     (1, 256, 28, 28)
// Out:    (1, 256, 14, 14)
// Operators: 'squeezenet1.1:opid74' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    DISABLED_autogen_MaxPool_squeezenet1_1_opid74,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::CEIL), // rounding type
            ::testing::Values(ov::op::PadType::EXPLICIT), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 256, 28, 28}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'ceil', 'strides': '2,2'}
// In:     (1, 64, 113, 113)
// Out:    (1, 64, 56, 56)
// Operators: 'squeezenet1.1:opid8' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_squeezenet1_1_opid8,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::CEIL), // rounding type
            ::testing::Values(ov::op::PadType::EXPLICIT), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 64, 113, 113}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'kernel': '3,3', 'pads_begin': '1,1', 'pads_end': '1,1', 'rounding_type': 'floor', 'strides': '2,2'}
// In:     (1, 64, 112, 112)
// Out:    (1, 64, 56, 56)
// Operators: 'resnet-50-caffe2:opid10' [FP16, FP32], 'resnet-50-pytorch:opid10' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_resnet_50_caffe2_opid10,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{1, 1}), // pad begin
            ::testing::Values(std::vector<size_t>{1, 1}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::EXPLICIT), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 64, 112, 112}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'kernel': '5,5', 'pads_begin': '2,2', 'pads_end': '2,2', 'rounding_type': 'floor', 'strides': '1,1'}
// In:     (1, 256, 20, 20)
// Out:    (1, 256, 20, 20)
// Operators: 'yolov5-640x640-IR:opid189' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_yolov5_640x640_IR_opid189,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{5, 5}), // kernel
            ::testing::Values(std::vector<size_t>{1, 1}), // strides
            ::testing::Values(std::vector<size_t>{2, 2}), // pad begin
            ::testing::Values(std::vector<size_t>{2, 2}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::EXPLICIT), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 256, 20, 20}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'kernel': '9,9', 'pads_begin': '4,4', 'pads_end': '4,4', 'rounding_type': 'floor', 'strides': '1,1'}
// In:     (1, 256, 20, 20)
// Out:    (1, 256, 20, 20)
// Operators: 'yolov5-640x640-IR:opid190' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_yolov5_640x640_IR_opid190,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{9, 9}), // kernel
            ::testing::Values(std::vector<size_t>{1, 1}), // strides
            ::testing::Values(std::vector<size_t>{4, 4}), // pad begin
            ::testing::Values(std::vector<size_t>{4, 4}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::EXPLICIT), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 256, 20, 20}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'kernel': '13,13', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '1,1'}
// In:     (1, 512, 19, 19)
// Out:    (1, 512, 19, 19)
// Operators: 'yolo-v4-tf:opid419' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_yolo_v4_tf_opid419,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{13, 13}), // kernel
            ::testing::Values(std::vector<size_t>{1, 1}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::SAME_UPPER), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 512, 19, 19}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '1,1'}
// In:     (100, 1024, 4, 4)
// Out:    (100, 1024, 4, 4)
// Operators: 'mask_rcnn_inception_v2_coco:opid405' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid563' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_mask_rcnn_inception_v2_coco_opid405,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{1, 1}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::SAME_UPPER), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{100, 1024, 4, 4}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '2,2'}
// In:     (1, 192, 200, 342)
// Out:    (1, 192, 100, 171)
// Operators: 'mask_rcnn_inception_v2_coco:opid23' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_mask_rcnn_inception_v2_coco_opid23,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::SAME_UPPER), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 192, 200, 342}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '2,2'}
// In:     (1, 320, 100, 171)
// Out:    (1, 320, 50, 86)
// Operators: 'mask_rcnn_inception_v2_coco:opid123' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_mask_rcnn_inception_v2_coco_opid123,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::SAME_UPPER), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 320, 100, 171}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '2,2'}
// In:     (1, 64, 112, 112)
// Out:    (1, 64, 56, 56)
// Operators: 'resnet-50-tf:opid8' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_resnet_50_tf_opid8,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::SAME_UPPER), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 64, 112, 112}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '2,2'}
// In:     (1, 64, 400, 683)
// Out:    (1, 64, 200, 342)
// Operators: 'mask_rcnn_inception_v2_coco:opid12' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_mask_rcnn_inception_v2_coco_opid12,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::SAME_UPPER), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 64, 400, 683}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '2,2'}
// In:     (1, 88, 10, 10)
// Out:    (1, 88, 5, 5)
// Operators: 'efficientdet-d1-tf:opid1001' [FP16, FP32], 'efficientdet-d1-tf:opid1234' [FP16, FP32], 'efficientdet-d1-tf:opid656' [FP16, FP32], 'efficientdet-d1-tf:opid767' [FP16, FP32], 'efficientdet-d1-tf:opid884' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_efficientdet_d1_tf_opid1001,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::SAME_UPPER), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 88, 10, 10}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '2,2'}
// In:     (1, 88, 20, 20)
// Out:    (1, 88, 10, 10)
// Operators: 'efficientdet-d1-tf:opid1190' [FP16, FP32], 'efficientdet-d1-tf:opid653' [FP16, FP32], 'efficientdet-d1-tf:opid752' [FP16, FP32], 'efficientdet-d1-tf:opid869' [FP16, FP32], 'efficientdet-d1-tf:opid986' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_efficientdet_d1_tf_opid1190,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::SAME_UPPER), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 88, 20, 20}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '2,2'}
// In:     (1, 88, 40, 40)
// Out:    (1, 88, 20, 20)
// Operators: 'efficientdet-d1-tf:opid1143' [FP16, FP32], 'efficientdet-d1-tf:opid734' [FP16, FP32], 'efficientdet-d1-tf:opid851' [FP16, FP32], 'efficientdet-d1-tf:opid968' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_efficientdet_d1_tf_opid1143,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::SAME_UPPER), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 88, 40, 40}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '2,2'}
// In:     (1, 88, 80, 80)
// Out:    (1, 88, 40, 40)
// Operators: 'efficientdet-d1-tf:opid1096' [FP16, FP32], 'efficientdet-d1-tf:opid714' [FP16, FP32], 'efficientdet-d1-tf:opid833' [FP16, FP32], 'efficientdet-d1-tf:opid950' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_efficientdet_d1_tf_opid1096,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::SAME_UPPER), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 88, 80, 80}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '2,2'}
// In:     (100, 576, 7, 7)
// Out:    (100, 576, 4, 4)
// Operators: 'mask_rcnn_inception_v2_coco:opid336' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid494' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_mask_rcnn_inception_v2_coco_opid336,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::SAME_UPPER), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{100, 576, 7, 7}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'kernel': '5,5', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '1,1'}
// In:     (1, 512, 19, 19)
// Out:    (1, 512, 19, 19)
// Operators: 'yolo-v4-tf:opid421' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_yolo_v4_tf_opid421,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{5, 5}), // kernel
            ::testing::Values(std::vector<size_t>{1, 1}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::SAME_UPPER), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 512, 19, 19}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'kernel': '9,9', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '1,1'}
// In:     (1, 512, 19, 19)
// Out:    (1, 512, 19, 19)
// Operators: 'yolo-v4-tf:opid420' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_yolo_v4_tf_opid420,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{9, 9}), // kernel
            ::testing::Values(std::vector<size_t>{1, 1}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::SAME_UPPER), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 512, 19, 19}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'kernel': '2,2', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '2,2'}
// In:     (1, 128, 16, 16)
// Out:    (1, 128, 8, 8)
// Operators: '2d_unet-graph-transform-cuda:opid28' [FP32], '2d_unet-graph-transform:opid44' [FP32], '2d_unet:opid44' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_2d_unet_graph_transform_cuda_opid28,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{2, 2}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::VALID), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 128, 16, 16}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'kernel': '2,2', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '2,2'}
// In:     (1, 16, 128, 128)
// Out:    (1, 16, 64, 64)
// Operators: '2d_unet-graph-transform-cuda:opid7' [FP32], '2d_unet-graph-transform:opid11' [FP32], '2d_unet:opid11' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_2d_unet_graph_transform_cuda_opid7,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{2, 2}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::VALID), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 16, 128, 128}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'kernel': '2,2', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '2,2'}
// In:     (1, 32, 64, 64)
// Out:    (1, 32, 32, 32)
// Operators: '2d_unet-graph-transform-cuda:opid14' [FP32], '2d_unet-graph-transform:opid22' [FP32], '2d_unet:opid22' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_2d_unet_graph_transform_cuda_opid14,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{2, 2}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::VALID), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 32, 64, 64}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'kernel': '2,2', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '2,2'}
// In:     (1, 64, 32, 32)
// Out:    (1, 64, 16, 16)
// Operators: '2d_unet-graph-transform-cuda:opid21' [FP32], '2d_unet-graph-transform:opid33' [FP32], '2d_unet:opid33' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_2d_unet_graph_transform_cuda_opid21,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{2, 2}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::VALID), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 64, 32, 32}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'kernel': '2,2', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '2,2'}
// In:     (100, 576, 14, 14)
// Out:    (100, 576, 7, 7)
// Operators: 'mask_rcnn_inception_v2_coco:opid310' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid468' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_mask_rcnn_inception_v2_coco_opid310,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{2, 2}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::VALID), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{100, 576, 14, 14}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'kernel': '2,2,2', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'rounding_type': 'floor', 'strides': '2,2,2'}
// In:     (1, 128, 18, 18, 18)
// Out:    (1, 128, 9, 9, 9)
// Operators: '3d_unet-graph-transform-cuda:opid28' [FP32], '3d_unet-graph-transform:opid44' [FP32], '3d_unet:opid44' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_3d_unet_graph_transform_cuda_opid28,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{2, 2, 2}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::VALID), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 128, 18, 18, 18}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'kernel': '2,2,2', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'rounding_type': 'floor', 'strides': '2,2,2'}
// In:     (1, 16, 144, 144, 144)
// Out:    (1, 16, 72, 72, 72)
// Operators: '3d_unet-graph-transform-cuda:opid7' [FP32], '3d_unet-graph-transform:opid11' [FP32], '3d_unet:opid11' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_3d_unet_graph_transform_cuda_opid7,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{2, 2, 2}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::VALID), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 16, 144, 144, 144}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'kernel': '2,2,2', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'rounding_type': 'floor', 'strides': '2,2,2'}
// In:     (1, 32, 72, 72, 72)
// Out:    (1, 32, 36, 36, 36)
// Operators: '3d_unet-graph-transform-cuda:opid14' [FP32], '3d_unet-graph-transform:opid22' [FP32], '3d_unet:opid22' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_3d_unet_graph_transform_cuda_opid14,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{2, 2, 2}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::VALID), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 32, 72, 72, 72}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'kernel': '2,2,2', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'rounding_type': 'floor', 'strides': '2,2,2'}
// In:     (1, 64, 36, 36, 36)
// Out:    (1, 64, 18, 18, 18)
// Operators: '3d_unet-graph-transform-cuda:opid21' [FP32], '3d_unet-graph-transform:opid33' [FP32], '3d_unet:opid33' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_3d_unet_graph_transform_cuda_opid21,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{2, 2, 2}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::VALID), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 64, 36, 36, 36}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '2,2'}
// In:     (1, 1024, 17, 17)
// Out:    (1, 1024, 8, 8)
// Operators: 'googlenet-v4-tf:opid629' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_googlenet_v4_tf_opid629,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::VALID), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 1024, 17, 17}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '2,2'}
// In:     (1, 192, 71, 71)
// Out:    (1, 192, 35, 35)
// Operators: 'googlenet-v4-tf:opid63' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_googlenet_v4_tf_opid63,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::VALID), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 192, 71, 71}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '2,2'}
// In:     (1, 384, 35, 35)
// Out:    (1, 384, 17, 17)
// Operators: 'googlenet-v4-tf:opid233' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_googlenet_v4_tf_opid233,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::VALID), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 384, 35, 35}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '2,2'}
// In:     (1, 64, 147, 147)
// Out:    (1, 64, 73, 73)
// Operators: 'googlenet-v4-tf:opid20' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MaxPool_googlenet_v4_tf_opid20,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::MAX),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{2, 2}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::VALID), // pad type
            ::testing::Values(false) // 'exclude pad' is N/A to MaxPool
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 64, 147, 147}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);

// {AUTOGENERATED_TESTS_END_TAG_MAXPOOL}
// clang-format on
// =============================================================================

// =============================================================================
// clang-format off
// {AUTOGENERATED_TESTS_BEGIN_TAG_AVGPOOL}

// Attrs:  {'auto_pad': 'explicit', 'exclude-pad': 'true', 'kernel': '7,7', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '1,1'}
// In:     (1, 2048, 7, 7)
// Out:    (1, 2048, 1, 1)
// Operators: 'resnet-50-caffe2:opid283' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_AvgPool_resnet_50_caffe2_opid283,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::AVG),
            ::testing::Values(std::vector<size_t>{7, 7}), // kernel
            ::testing::Values(std::vector<size_t>{1, 1}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::EXPLICIT), // pad type
            ::testing::Values(true) // exclude pad
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 2048, 7, 7}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'exclude-pad': 'true', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '1,1'}
// In:     (1, 1024, 17, 17)
// Out:    (1, 1024, 17, 17)
// Operators: 'googlenet-v4-tf:opid280' [FP16, FP32], 'googlenet-v4-tf:opid332' [FP16, FP32], 'googlenet-v4-tf:opid384' [FP16, FP32], 'googlenet-v4-tf:opid436' [FP16, FP32], 'googlenet-v4-tf:opid488' [FP16, FP32], 'googlenet-v4-tf:opid540' [FP16, FP32], 'googlenet-v4-tf:opid592' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_AvgPool_googlenet_v4_tf_opid280,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::AVG),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{1, 1}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::SAME_UPPER), // pad type
            ::testing::Values(true) // exclude pad
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 1024, 17, 17}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'exclude-pad': 'true', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '1,1'}
// In:     (1, 1536, 8, 8)
// Out:    (1, 1536, 8, 8)
// Operators: 'googlenet-v4-tf:opid678' [FP16, FP32], 'googlenet-v4-tf:opid732' [FP16, FP32], 'googlenet-v4-tf:opid786' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_AvgPool_googlenet_v4_tf_opid678,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::AVG),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{1, 1}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::SAME_UPPER), // pad type
            ::testing::Values(true) // exclude pad
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 1536, 8, 8}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'exclude-pad': 'true', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '1,1'}
// In:     (1, 192, 100, 171)
// Out:    (1, 192, 100, 171)
// Operators: 'mask_rcnn_inception_v2_coco:opid54' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_AvgPool_mask_rcnn_inception_v2_coco_opid54,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::AVG),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{1, 1}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::SAME_UPPER), // pad type
            ::testing::Values(true) // exclude pad
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 192, 100, 171}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'exclude-pad': 'true', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '1,1'}
// In:     (1, 256, 100, 171)
// Out:    (1, 256, 100, 171)
// Operators: 'mask_rcnn_inception_v2_coco:opid91' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_AvgPool_mask_rcnn_inception_v2_coco_opid91,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::AVG),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{1, 1}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::SAME_UPPER), // pad type
            ::testing::Values(true) // exclude pad
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 256, 100, 171}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'exclude-pad': 'true', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '1,1'}
// In:     (1, 384, 35, 35)
// Out:    (1, 384, 35, 35)
// Operators: 'googlenet-v4-tf:opid132' [FP16, FP32], 'googlenet-v4-tf:opid169' [FP16, FP32], 'googlenet-v4-tf:opid206' [FP16, FP32], 'googlenet-v4-tf:opid95' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_AvgPool_googlenet_v4_tf_opid132,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::AVG),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{1, 1}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::SAME_UPPER), // pad type
            ::testing::Values(true) // exclude pad
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 384, 35, 35}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'exclude-pad': 'true', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '1,1'}
// In:     (1, 576, 50, 86)
// Out:    (1, 576, 50, 86)
// Operators: 'mask_rcnn_inception_v2_coco:opid155' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid192' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid229' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid266' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_AvgPool_mask_rcnn_inception_v2_coco_opid155,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::AVG),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{1, 1}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::SAME_UPPER), // pad type
            ::testing::Values(true) // exclude pad
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 576, 50, 86}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'exclude-pad': 'true', 'kernel': '3,3', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '1,1'}
// In:     (100, 1024, 4, 4)
// Out:    (100, 1024, 4, 4)
// Operators: 'mask_rcnn_inception_v2_coco:opid368' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid526' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_AvgPool_mask_rcnn_inception_v2_coco_opid368,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::AVG),
            ::testing::Values(std::vector<size_t>{3, 3}), // kernel
            ::testing::Values(std::vector<size_t>{1, 1}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::SAME_UPPER), // pad type
            ::testing::Values(true) // exclude pad
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{100, 1024, 4, 4}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'exclude-pad': 'true', 'kernel': '8,8', 'pads_begin': '0,0', 'pads_end': '0,0', 'rounding_type': 'floor', 'strides': '1,1'}
// In:     (1, 1536, 8, 8)
// Out:    (1, 1536, 1, 1)
// Operators: 'googlenet-v4-tf:opid793' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_AvgPool_googlenet_v4_tf_opid793,
    PoolingLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(PoolingTypes::AVG),
            ::testing::Values(std::vector<size_t>{8, 8}), // kernel
            ::testing::Values(std::vector<size_t>{1, 1}), // strides
            ::testing::Values(std::vector<size_t>{0, 0}), // pad begin
            ::testing::Values(std::vector<size_t>{0, 0}), // pad end
            ::testing::Values(ov::op::RoundingType::FLOOR), // rounding type
            ::testing::Values(ov::op::PadType::VALID), // pad type
            ::testing::Values(true) // exclude pad
        ),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 1536, 8, 8}), // input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    PoolingLayerTest::getTestCaseName);

// {AUTOGENERATED_TESTS_END_TAG_AVGPOOL}
// clang-format on
// =============================================================================

}  // namespace
