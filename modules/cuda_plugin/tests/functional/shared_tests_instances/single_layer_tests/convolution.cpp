// Copyright (C) 2019-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <cuda_test_constants.hpp>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/convolution.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32,
};

/* ============= 1D Convolution ============= */
const std::vector<std::vector<size_t>> kernels1D = {{3}, {5}};
const std::vector<std::vector<size_t>> strides1D = {{1}, {3}};
const std::vector<std::vector<size_t>> dilations1D = {{1}, {3}};
const std::vector<size_t> numOutChannels1D = {1, 5};

const auto conv1DParams_ExplicitPaddingSymmetric1 = ::testing::Combine(
    ::testing::ValuesIn(kernels1D), ::testing::ValuesIn(strides1D),
    ::testing::Values(std::vector<ptrdiff_t>({0})),      // pads_begin
    ::testing::Values(std::vector<ptrdiff_t>({0})),      // pads_end
    ::testing::ValuesIn(dilations1D), ::testing::ValuesIn(numOutChannels1D),
    ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv1DParams_ExplicitPaddingSymmetric2 = ::testing::Combine(
    ::testing::ValuesIn(kernels1D), ::testing::ValuesIn(strides1D),
    ::testing::Values(std::vector<ptrdiff_t>({3})),      // pads_begin
    ::testing::Values(std::vector<ptrdiff_t>({3})),      // pads_end
    ::testing::ValuesIn(dilations1D), ::testing::ValuesIn(numOutChannels1D),
    ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv1DParams_ExplicitPaddingAsymmetric1 = ::testing::Combine(
    ::testing::ValuesIn(kernels1D), ::testing::ValuesIn(strides1D),
    ::testing::Values(std::vector<ptrdiff_t>({0})),      // pads_begin
    ::testing::Values(std::vector<ptrdiff_t>({3})),      // pads_end
    ::testing::ValuesIn(dilations1D), ::testing::ValuesIn(numOutChannels1D),
    ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv1DParams_ExplicitPaddingAsymmetric2 = ::testing::Combine(
    ::testing::ValuesIn(kernels1D), ::testing::ValuesIn(strides1D),
    ::testing::Values(std::vector<ptrdiff_t>({3})),      // pads_begin
    ::testing::Values(std::vector<ptrdiff_t>({0})),      // pads_end
    ::testing::ValuesIn(dilations1D), ::testing::ValuesIn(numOutChannels1D),
    ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv1DParams_AutoPadValid = ::testing::Combine(
    ::testing::ValuesIn(kernels1D), ::testing::ValuesIn(strides1D),
    ::testing::Values(std::vector<ptrdiff_t>({0})),
    ::testing::Values(std::vector<ptrdiff_t>({0})),
    ::testing::ValuesIn(dilations1D), ::testing::ValuesIn(numOutChannels1D),
    ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(
    smoke_Convolution1D_ExplicitPaddingSymmetric1, ConvolutionLayerTest,
    ::testing::Combine(
        conv1DParams_ExplicitPaddingSymmetric1, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(
    smoke_Convolution1D_ExplicitPaddingSymmetric2, ConvolutionLayerTest,
    ::testing::Combine(
        conv1DParams_ExplicitPaddingSymmetric2, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(
    DISABLED_smoke_Convolution1D_ExplicitPaddingAsymmetric1, ConvolutionLayerTest,
    ::testing::Combine(
        conv1DParams_ExplicitPaddingAsymmetric1, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(
    DISABLED_smoke_Convolution1D_ExplicitPaddingAsymmetric2, ConvolutionLayerTest,
    ::testing::Combine(
        conv1DParams_ExplicitPaddingAsymmetric2, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    smoke_Convolution1D_AutoPadValid, ConvolutionLayerTest,
    ::testing::Combine(
        conv1DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution ============= */
const std::vector<std::vector<size_t>> kernels = {{3, 3}, {3, 5}};
const std::vector<std::vector<size_t>> strides = {{1, 1}, {1, 3}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}, {3, 1}};
const std::vector<size_t> numOutChannels = {1, 5};

const auto conv2DParams_ExplicitPaddingSymmetric1 = ::testing::Combine(
    ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_begin
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_end
    ::testing::ValuesIn(dilations), ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv2DParams_ExplicitPaddingSymmetric2 = ::testing::Combine(
    ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
    ::testing::Values(std::vector<ptrdiff_t>({0, 3})),      // pads_begin
    ::testing::Values(std::vector<ptrdiff_t>({0, 3})),      // pads_end
    ::testing::ValuesIn(dilations), ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv2DParams_ExplicitPaddingAsymmetric1 = ::testing::Combine(
    ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_begin
    ::testing::Values(std::vector<ptrdiff_t>({0, 3})),      // pads_end
    ::testing::ValuesIn(dilations), ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv2DParams_ExplicitPaddingAsymmetric2 = ::testing::Combine(
    ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
    ::testing::Values(std::vector<ptrdiff_t>({0, 3})),      // pads_begin
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_end
    ::testing::ValuesIn(dilations), ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::EXPLICIT));

const auto conv2DParams_AutoPadValid = ::testing::Combine(
    ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::ValuesIn(dilations), ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(
    smoke_Convolution2D_ExplicitPaddingSymmetric1, ConvolutionLayerTest,
    ::testing::Combine(
        conv2DParams_ExplicitPaddingSymmetric1, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    smoke_Convolution2D_ExplicitPaddingSymmetric2, ConvolutionLayerTest,
    ::testing::Combine(
        conv2DParams_ExplicitPaddingSymmetric2, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    DISABLED_smoke_Convolution2D_ExplicitPaddingAsymmetric1, ConvolutionLayerTest,
    ::testing::Combine(
        conv2DParams_ExplicitPaddingAsymmetric1, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    DISABLED_smoke_Convolution2D_ExplicitPaddingAsymmetric2, ConvolutionLayerTest,
    ::testing::Combine(
        conv2DParams_ExplicitPaddingAsymmetric2, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    smoke_Convolution2D_AutoPadValid, ConvolutionLayerTest,
    ::testing::Combine(
        conv2DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

/* ============= 3D Convolution ============= */
const std::vector<std::vector<size_t>> kernels3d = {{3, 3, 3}, {3, 5, 3}};
const std::vector<std::vector<size_t>> strides3d = {{1, 1, 1}, {1, 2, 1}};
const std::vector<std::vector<size_t>> dilations3d = {{1, 1, 1}, {1, 2, 1}};
const std::vector<size_t> numOutChannels3D = {1, 5};

const auto conv3DParams_ExplicitPaddingSymmetric1 = ::testing::Combine(
    ::testing::ValuesIn(kernels3d), ::testing::ValuesIn(strides3d),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),      // pads_begin
    ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),      // pads_end
    ::testing::ValuesIn(dilations3d), ::testing::ValuesIn(numOutChannels3D),
    ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv3DParams_ExplicitPaddingSymmetric2 = ::testing::Combine(
    ::testing::ValuesIn(kernels3d), ::testing::ValuesIn(strides3d),
    ::testing::Values(std::vector<ptrdiff_t>({0, 2, 0})),      // pads_begin
    ::testing::Values(std::vector<ptrdiff_t>({0, 2, 0})),      // pads_end
    ::testing::ValuesIn(dilations3d), ::testing::ValuesIn(numOutChannels3D),
    ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv3DParams_ExplicitPaddingAsymmetric1 = ::testing::Combine(
    ::testing::ValuesIn(kernels3d), ::testing::ValuesIn(strides3d),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),      // pads_begin
    ::testing::Values(std::vector<ptrdiff_t>({0, 2, 0})),      // pads_end
    ::testing::ValuesIn(dilations3d), ::testing::ValuesIn(numOutChannels3D),
    ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv3DParams_ExplicitPaddingAsymmetric2 = ::testing::Combine(
    ::testing::ValuesIn(kernels3d), ::testing::ValuesIn(strides3d),
    ::testing::Values(std::vector<ptrdiff_t>({0, 2, 0})),      // pads_begin
    ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),      // pads_end
    ::testing::ValuesIn(dilations3d), ::testing::ValuesIn(numOutChannels3D),
    ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv3DParams_AutoPadValid = ::testing::Combine(
    ::testing::ValuesIn(kernels3d), ::testing::ValuesIn(strides3d),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
    ::testing::ValuesIn(dilations3d), ::testing::ValuesIn(numOutChannels3D),
    ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(
    smoke_Convolution3D_ExplicitPaddingSymmetric1, ConvolutionLayerTest,
    ::testing::Combine(
        conv3DParams_ExplicitPaddingSymmetric1, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    smoke_Convolution3D_ExplicitPaddingSymmetric2, ConvolutionLayerTest,
    ::testing::Combine(
        conv3DParams_ExplicitPaddingSymmetric2, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    DISABLED_smoke_Convolution3D_ExplicitPaddingAsymmetric1, ConvolutionLayerTest,
    ::testing::Combine(
        conv3DParams_ExplicitPaddingAsymmetric1, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    DISABLED_smoke_Convolution3D_ExplicitPaddingAsymmetric2, ConvolutionLayerTest,
    ::testing::Combine(
        conv3DParams_ExplicitPaddingAsymmetric2, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    smoke_Convolution3D_AutoPadValid, ConvolutionLayerTest,
    ::testing::Combine(
        conv3DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

/* ============= resnet50/vgg16 Convolutions ============= */

const auto resnet50_vgg16_precission = InferenceEngine::Precision::FP32;

// attrs: {'auto_pad': 'explicit', 'strides': '2,2', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 256, 28, 28), (256, 256, 3, 3); out: (1, 256, 14, 14)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group1_1, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})),         // kernel
            ::testing::Values(std::vector<size_t>({2, 2})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(256),                                 // Num out channels
            ::testing::Values(ngraph::op::PadType::EXPLICIT)),      // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 28, 28})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '2,2', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 128, 56, 56), (128, 128, 3, 3); out: (1, 128, 28, 28)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group1_2, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})),         // kernel
            ::testing::Values(std::vector<size_t>({2, 2})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(128),                                 // Num out channels
            ::testing::Values(ngraph::op::PadType::EXPLICIT)),      // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 56, 56})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '2,2', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 512, 14, 14), (512, 512, 3, 3); out: (1, 512, 7, 7)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group1_3, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})),         // kernel
            ::testing::Values(std::vector<size_t>({2, 2})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(512),                                 // Num out channels
            ::testing::Values(ngraph::op::PadType::EXPLICIT)),      // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 14, 14})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '2,2', 'dilations': '1,1', 'pads_begin': '3,3', 'pads_end': '3,3'},
// in: (1, 3, 224, 224), (64, 3, 7, 7); out: (1, 64, 112, 112)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group2_1, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({7, 7})),         // kernel
            ::testing::Values(std::vector<size_t>({2, 2})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({3, 3})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({3, 3})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(64),                                  // Num out channels
            ::testing::Values(ngraph::op::PadType::EXPLICIT)),      // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 3, 224, 224})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'valid', 'strides': '2,2', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 256, 56, 56), (512, 256, 1, 1); out: (1, 512, 28, 28)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group3_1, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})),         // kernel
            ::testing::Values(std::vector<size_t>({2, 2})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(512),                                 // Num out channels
            ::testing::Values(ngraph::op::PadType::VALID)),         // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 56, 56})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'valid', 'strides': '2,2', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 1024, 14, 14), (2048, 1024, 1, 1); out: (1, 2048, 7, 7)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group3_2, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})),         // kernel
            ::testing::Values(std::vector<size_t>({2, 2})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(2048),                                // Num out channels
            ::testing::Values(ngraph::op::PadType::VALID)),         // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 14, 14})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'valid', 'strides': '2,2', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 512, 28, 28), (1024, 512, 1, 1); out: (1, 1024, 14, 14)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group3_3, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})),         // kernel
            ::testing::Values(std::vector<size_t>({2, 2})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(1024),                                // Num out channels
            ::testing::Values(ngraph::op::PadType::VALID)),         // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 28, 28})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 256, 14, 14), (1024, 256, 1, 1); out: (1, 1024, 14, 14)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_1, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(1024),                                // Num out channels
            ::testing::Values(ngraph::op::PadType::SAME_UPPER)),    // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 14, 14})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 64, 56, 56), (64, 64, 1, 1); out: (1, 64, 56, 56)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_2, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(64),                                  // Num out channels
            ::testing::Values(ngraph::op::PadType::SAME_UPPER)),    // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 56, 56})),    // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 128, 28, 28), (512, 128, 1, 1); out: (1, 512, 28, 28)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_3, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(512),                                 // Num out channels
            ::testing::Values(ngraph::op::PadType::SAME_UPPER)),    // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 28, 28})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 256, 14, 14), (256, 256, 3, 3); out: (1, 256, 14, 14)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_4, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(256),                                 // Num out channels
            ::testing::Values(ngraph::op::PadType::SAME_UPPER)),    // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 14, 14})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 64, 56, 56), (256, 64, 1, 1); out: (1, 256, 56, 56)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_5, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(256),                                 // Num out channels
            ::testing::Values(ngraph::op::PadType::SAME_UPPER)),    // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 56, 56})),    // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 64, 56, 56), (64, 64, 3, 3); out: (1, 64, 56, 56)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_6, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(64),                                  // Num out channels
            ::testing::Values(ngraph::op::PadType::SAME_UPPER)),    // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 56, 56})),    // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 256, 56, 56), (64, 256, 1, 1); out: (1, 64, 56, 56)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_7, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(64),                                  // Num out channels
            ::testing::Values(ngraph::op::PadType::SAME_UPPER)),    // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 56, 56})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 512, 28, 28), (128, 512, 1, 1); out: (1, 128, 28, 28)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_8, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(128),                                 // Num out channels
            ::testing::Values(ngraph::op::PadType::SAME_UPPER)),    // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 28, 28})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 2048, 7, 7), (512, 2048, 1, 1); out: (1, 512, 7, 7)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_9, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(512),                                 // Num out channels
            ::testing::Values(ngraph::op::PadType::SAME_UPPER)),    // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 2048, 7, 7})),    // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 1024, 14, 14), (512, 1024, 1, 1); out: (1, 512, 14, 14)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_10, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(512),                                 // Num out channels
            ::testing::Values(ngraph::op::PadType::SAME_UPPER)),    // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 14, 14})),  // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 512, 7, 7), (512, 512, 3, 3); out: (1, 512, 7, 7)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_11, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(512),                                 // Num out channels
            ::testing::Values(ngraph::op::PadType::SAME_UPPER)),    // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 7, 7})),     // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 256, 56, 56), (128, 256, 1, 1); out: (1, 128, 56, 56)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_12, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(128),                                 // Num out channels
            ::testing::Values(ngraph::op::PadType::SAME_UPPER)),    // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 56, 56})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 512, 28, 28), (256, 512, 1, 1); out: (1, 256, 28, 28)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_13, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(256),                                 // Num out channels
            ::testing::Values(ngraph::op::PadType::SAME_UPPER)),    // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 28, 28})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 512, 7, 7), (2048, 512, 1, 1); out: (1, 2048, 7, 7)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_14, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(2048),                                // Num out channels
            ::testing::Values(ngraph::op::PadType::SAME_UPPER)),    // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 7, 7})),     // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 128, 28, 28), (128, 128, 3, 3); out: (1, 128, 28, 28)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_15, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(128),                                 // Num out channels
            ::testing::Values(ngraph::op::PadType::SAME_UPPER)),    // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 28, 28})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 1024, 14, 14), (256, 1024, 1, 1); out: (1, 256, 14, 14)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_16, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(256),                                 // Num out channels
            ::testing::Values(ngraph::op::PadType::SAME_UPPER)),    // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 14, 14})),  // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 64, 224, 224), (64, 64, 3, 3); out: (1, 64, 224, 224)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group5_1, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(64),                                  // Num out channels
            ::testing::Values(ngraph::op::PadType::EXPLICIT)),      // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 224, 224})),  // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 3, 224, 224), (64, 3, 3, 3); out: (1, 64, 224, 224)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group5_2, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(64),                                  // Num out channels
            ::testing::Values(ngraph::op::PadType::EXPLICIT)),      // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 3, 224, 224})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 128, 56, 56), (256, 128, 3, 3); out: (1, 256, 56, 56)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group5_3, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(256),                                 // Num out channels
            ::testing::Values(ngraph::op::PadType::EXPLICIT)),      // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 56, 56})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 512, 28, 28), (512, 512, 3, 3); out: (1, 512, 28, 28)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group5_4, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(512),                                 // Num out channels
            ::testing::Values(ngraph::op::PadType::EXPLICIT)),      // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 28, 28})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 512, 14, 14), (512, 512, 3, 3); out: (1, 512, 14, 14)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group5_5, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(512),                                 // Num out channels
            ::testing::Values(ngraph::op::PadType::EXPLICIT)),      // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 14, 14})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 256, 28, 28), (512, 256, 3, 3); out: (1, 512, 28, 28)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group5_6, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(512),                                 // Num out channels
            ::testing::Values(ngraph::op::PadType::EXPLICIT)),      // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 28, 28})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 256, 56, 56), (256, 256, 3, 3); out: (1, 256, 56, 56)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group5_7, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(256),                                 // Num out channels
            ::testing::Values(ngraph::op::PadType::EXPLICIT)),      // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 56, 56})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 64, 112, 112), (128, 64, 3, 3); out: (1, 128, 112, 112)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group5_8, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(128),                                 // Num out channels
            ::testing::Values(ngraph::op::PadType::EXPLICIT)),      // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 112, 112})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 128, 112, 112), (128, 128, 3, 3); out: (1, 128, 112, 112)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group5_9, ConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})),         // kernel
            ::testing::Values(std::vector<size_t>({1, 1})),         // stride
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})),      // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})),         // dilations
            ::testing::Values(128),                                 // Num out channels
            ::testing::Values(ngraph::op::PadType::EXPLICIT)),      // Padding type
        ::testing::Values(resnet50_vgg16_precission),               // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),            // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),            // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 112, 112})),   // Input shapes
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    ConvolutionLayerTest::getTestCaseName);

}  // namespace
