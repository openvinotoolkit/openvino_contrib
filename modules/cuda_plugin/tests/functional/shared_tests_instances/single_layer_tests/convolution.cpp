// Copyright (C) 2019-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/convolution.hpp"

#include <cuda_test_constants.hpp>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "finite_comparer.hpp"

using namespace LayerTestsDefinitions;

namespace LayerTestsDefinitions {

class ConvolutionLayerThresholdTest : public FiniteComparer<ConvolutionLayerTest> {
protected:
    void SetUp() override {
        ConvolutionLayerTest::SetUp();

        auto params = this->GetParam();
        auto netPrecision = std::get<1>(params);
        if (netPrecision.getPrecVal() == InferenceEngine::Precision::FP16) {
            this->threshold = 500;
            this->infinity_value = std::numeric_limits<std::uint16_t>::max();
        }
    }
};

TEST_P(ConvolutionLayerThresholdTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto params = GetParam();
    inPrc = std::get<2>(params);
    outPrc = std::get<3>(params);

    Run();
}

}  // namespace LayerTestsDefinitions

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

const auto conv1DParams_ExplicitPaddingSymmetric1 =
    ::testing::Combine(::testing::ValuesIn(kernels1D),
                       ::testing::ValuesIn(strides1D),
                       ::testing::Values(std::vector<ptrdiff_t>({0})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0})),  // pads_end
                       ::testing::ValuesIn(dilations1D),
                       ::testing::ValuesIn(numOutChannels1D),
                       ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv1DParams_ExplicitPaddingSymmetric2 =
    ::testing::Combine(::testing::ValuesIn(kernels1D),
                       ::testing::ValuesIn(strides1D),
                       ::testing::Values(std::vector<ptrdiff_t>({3})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({3})),  // pads_end
                       ::testing::ValuesIn(dilations1D),
                       ::testing::ValuesIn(numOutChannels1D),
                       ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv1DParams_ExplicitPaddingAsymmetric1 =
    ::testing::Combine(::testing::ValuesIn(kernels1D),
                       ::testing::ValuesIn(strides1D),
                       ::testing::Values(std::vector<ptrdiff_t>({0})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({3})),  // pads_end
                       ::testing::ValuesIn(dilations1D),
                       ::testing::ValuesIn(numOutChannels1D),
                       ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv1DParams_ExplicitPaddingAsymmetric2 =
    ::testing::Combine(::testing::ValuesIn(kernels1D),
                       ::testing::ValuesIn(strides1D),
                       ::testing::Values(std::vector<ptrdiff_t>({3})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0})),  // pads_end
                       ::testing::ValuesIn(dilations1D),
                       ::testing::ValuesIn(numOutChannels1D),
                       ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv1DParams_AutoPadValid = ::testing::Combine(::testing::ValuesIn(kernels1D),
                                                          ::testing::ValuesIn(strides1D),
                                                          ::testing::Values(std::vector<ptrdiff_t>({0})),
                                                          ::testing::Values(std::vector<ptrdiff_t>({0})),
                                                          ::testing::ValuesIn(dilations1D),
                                                          ::testing::ValuesIn(numOutChannels1D),
                                                          ::testing::Values(ov::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(smoke_Convolution1D_ExplicitPaddingSymmetric1,
                        ConvolutionLayerThresholdTest,
                        ::testing::Combine(conv1DParams_ExplicitPaddingSymmetric1,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        ConvolutionLayerThresholdTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_Convolution1D_ExplicitPaddingSymmetric2,
                        ConvolutionLayerThresholdTest,
                        ::testing::Combine(conv1DParams_ExplicitPaddingSymmetric2,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        ConvolutionLayerThresholdTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_Convolution1D_ExplicitPaddingAsymmetric1,
                        ConvolutionLayerThresholdTest,
                        ::testing::Combine(conv1DParams_ExplicitPaddingAsymmetric1,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        ConvolutionLayerThresholdTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_Convolution1D_ExplicitPaddingAsymmetric2,
                        ConvolutionLayerThresholdTest,
                        ::testing::Combine(conv1DParams_ExplicitPaddingAsymmetric2,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        ConvolutionLayerThresholdTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution1D_AutoPadValid,
                        ConvolutionLayerThresholdTest,
                        ::testing::Combine(conv1DParams_AutoPadValid,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        ConvolutionLayerThresholdTest::getTestCaseName);

/* ============= 2D Convolution ============= */
const std::vector<std::vector<size_t>> kernels = {{3, 3}, {3, 5}};
const std::vector<std::vector<size_t>> strides = {{1, 1}, {1, 3}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}, {3, 1}};
const std::vector<size_t> numOutChannels = {1, 5};

const auto conv2DParams_ExplicitPaddingSymmetric1 =
    ::testing::Combine(::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pads_end
                       ::testing::ValuesIn(dilations),
                       ::testing::ValuesIn(numOutChannels),
                       ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv2DParams_ExplicitPaddingSymmetric2 =
    ::testing::Combine(::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 3})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0, 3})),  // pads_end
                       ::testing::ValuesIn(dilations),
                       ::testing::ValuesIn(numOutChannels),
                       ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv2DParams_ExplicitPaddingAsymmetric1 =
    ::testing::Combine(::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0, 3})),  // pads_end
                       ::testing::ValuesIn(dilations),
                       ::testing::ValuesIn(numOutChannels),
                       ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv2DParams_ExplicitPaddingAsymmetric2 =
    ::testing::Combine(::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 3})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pads_end
                       ::testing::ValuesIn(dilations),
                       ::testing::ValuesIn(numOutChannels),
                       ::testing::Values(ov::op::PadType::EXPLICIT));

const auto conv2DParams_AutoPadValid = ::testing::Combine(::testing::ValuesIn(kernels),
                                                          ::testing::ValuesIn(strides),
                                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                                                          ::testing::ValuesIn(dilations),
                                                          ::testing::ValuesIn(numOutChannels),
                                                          ::testing::Values(ov::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_ExplicitPaddingSymmetric1,
                        ConvolutionLayerThresholdTest,
                        ::testing::Combine(conv2DParams_ExplicitPaddingSymmetric1,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        ConvolutionLayerThresholdTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_ExplicitPaddingSymmetric2_FP32,
                        ConvolutionLayerThresholdTest,
                        ::testing::Combine(conv2DParams_ExplicitPaddingSymmetric2,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        ConvolutionLayerThresholdTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_ExplicitPaddingSymmetric2,
                        ConvolutionLayerThresholdTest,
                        ::testing::Combine(conv2DParams_ExplicitPaddingSymmetric2,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        ConvolutionLayerThresholdTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_ExplicitPaddingAsymmetric1,
                        ConvolutionLayerThresholdTest,
                        ::testing::Combine(conv2DParams_ExplicitPaddingAsymmetric1,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        ConvolutionLayerThresholdTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_ExplicitPaddingAsymmetric2,
                        ConvolutionLayerThresholdTest,
                        ::testing::Combine(conv2DParams_ExplicitPaddingAsymmetric2,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        ConvolutionLayerThresholdTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_AutoPadValid,
                        ConvolutionLayerThresholdTest,
                        ::testing::Combine(conv2DParams_AutoPadValid,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        ConvolutionLayerThresholdTest::getTestCaseName);

/* ============= 3D Convolution ============= */
const std::vector<std::vector<size_t>> kernels3d = {{3, 3, 3}, {3, 5, 3}};
const std::vector<std::vector<size_t>> strides3d = {{1, 1, 1}, {1, 2, 1}};
const std::vector<std::vector<size_t>> dilations3d = {{1, 1, 1}, {1, 2, 1}};
const std::vector<size_t> numOutChannels3D = {1, 5};

const auto conv3DParams_ExplicitPaddingSymmetric1 =
    ::testing::Combine(::testing::ValuesIn(kernels3d),
                       ::testing::ValuesIn(strides3d),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),  // pads_end
                       ::testing::ValuesIn(dilations3d),
                       ::testing::ValuesIn(numOutChannels3D),
                       ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv3DParams_ExplicitPaddingSymmetric2 =
    ::testing::Combine(::testing::ValuesIn(kernels3d),
                       ::testing::ValuesIn(strides3d),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 2, 0})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0, 2, 0})),  // pads_end
                       ::testing::ValuesIn(dilations3d),
                       ::testing::ValuesIn(numOutChannels3D),
                       ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv3DParams_ExplicitPaddingAsymmetric1 =
    ::testing::Combine(::testing::ValuesIn(kernels3d),
                       ::testing::ValuesIn(strides3d),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0, 2, 0})),  // pads_end
                       ::testing::ValuesIn(dilations3d),
                       ::testing::ValuesIn(numOutChannels3D),
                       ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv3DParams_ExplicitPaddingAsymmetric2 =
    ::testing::Combine(::testing::ValuesIn(kernels3d),
                       ::testing::ValuesIn(strides3d),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 2, 0})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),  // pads_end
                       ::testing::ValuesIn(dilations3d),
                       ::testing::ValuesIn(numOutChannels3D),
                       ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv3DParams_AutoPadValid = ::testing::Combine(::testing::ValuesIn(kernels3d),
                                                          ::testing::ValuesIn(strides3d),
                                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
                                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
                                                          ::testing::ValuesIn(dilations3d),
                                                          ::testing::ValuesIn(numOutChannels3D),
                                                          ::testing::Values(ov::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(smoke_Convolution3D_ExplicitPaddingSymmetric1,
                        ConvolutionLayerThresholdTest,
                        ::testing::Combine(conv3DParams_ExplicitPaddingSymmetric1,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        ConvolutionLayerThresholdTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution3D_ExplicitPaddingSymmetric2,
                        ConvolutionLayerThresholdTest,
                        ::testing::Combine(conv3DParams_ExplicitPaddingSymmetric2,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        ConvolutionLayerThresholdTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution3D_ExplicitPaddingAsymmetric1,
                        ConvolutionLayerThresholdTest,
                        ::testing::Combine(conv3DParams_ExplicitPaddingAsymmetric1,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        ConvolutionLayerThresholdTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution3D_ExplicitPaddingAsymmetric2,
                        ConvolutionLayerThresholdTest,
                        ::testing::Combine(conv3DParams_ExplicitPaddingAsymmetric2,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        ConvolutionLayerThresholdTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution3D_AutoPadValid,
                        ConvolutionLayerThresholdTest,
                        ::testing::Combine(conv3DParams_AutoPadValid,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        ConvolutionLayerThresholdTest::getTestCaseName);

// =============================================================================
// clang-format off
// {AUTOGENERATED_TESTS_BEGIN_TAG}

// Attrs:  {'auto_pad': 'explicit', 'dilations': '1', 'pads_begin': '15', 'pads_end': '15', 'strides': '1'}
// In:     (1, 2, 1000), (32, 2, 31)
// Out:    (1, 32, 1000)
// Operators: 'Tacotron2-decoder_iter:opid107' [FP32], 'Tacotron2-graph-transform-cuda-decoder_iter:opid53' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_Tacotron2_decoder_iter_opid107,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({31})), // kernel
            ::testing::Values(std::vector<size_t>({1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({15})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({15})), // pads_end
            ::testing::Values(std::vector<size_t>({1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 2, 1000})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1', 'pads_begin': '2', 'pads_end': '2', 'strides': '1'}
// In:     (1, 512, 1000), (512, 512, 5)
// Out:    (1, 512, 1000)
// Operators: 'Tacotron2-encoder:opid12' [FP32], 'Tacotron2-encoder:opid17' [FP32], 'Tacotron2-encoder:opid7' [FP32], 'Tacotron2-graph-transform-cuda-encoder:opid13' [FP32], 'Tacotron2-graph-transform-cuda-encoder:opid18' [FP32], 'Tacotron2-graph-transform-cuda-encoder:opid8' [FP32], 'Tacotron2-graph-transform-cuda-postnet:opid12' [FP32], 'Tacotron2-graph-transform-cuda-postnet:opid17' [FP32], 'Tacotron2-graph-transform-cuda-postnet:opid7' [FP32], 'Tacotron2-postnet:opid12' [FP32], 'Tacotron2-postnet:opid17' [FP32], 'Tacotron2-postnet:opid7' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_Tacotron2_encoder_opid12,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({5})), // kernel
            ::testing::Values(std::vector<size_t>({1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({2})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({2})), // pads_end
            ::testing::Values(std::vector<size_t>({1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 1000})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1', 'pads_begin': '2', 'pads_end': '2', 'strides': '1'}
// In:     (1, 512, 1000), (80, 512, 5)
// Out:    (1, 80, 1000)
// Operators: 'Tacotron2-graph-transform-cuda-postnet:opid22' [FP32], 'Tacotron2-postnet:opid22' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_Tacotron2_graph_transform_cuda_postnet_opid22,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({5})), // kernel
            ::testing::Values(std::vector<size_t>({1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({2})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({2})), // pads_end
            ::testing::Values(std::vector<size_t>({1})), // dilations
            ::testing::Values(80), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 1000})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1', 'pads_begin': '2', 'pads_end': '2', 'strides': '1'}
// In:     (1, 80, 1000), (512, 80, 5)
// Out:    (1, 512, 1000)
// Operators: 'Tacotron2-graph-transform-cuda-postnet:opid2' [FP32], 'Tacotron2-postnet:opid2' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_Tacotron2_graph_transform_cuda_postnet_opid2,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({5})), // kernel
            ::testing::Values(std::vector<size_t>({1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({2})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({2})), // pads_end
            ::testing::Values(std::vector<size_t>({1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 80, 1000})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 14, 14), (256, 1024, 1, 1)
// Out:    (1, 256, 14, 14)
// Operators: 'resnet-50-caffe2:opid152' [FP16, FP32], 'resnet-50-caffe2:opid168' [FP16, FP32], 'resnet-50-caffe2:opid184' [FP16, FP32], 'resnet-50-caffe2:opid200' [FP16, FP32], 'resnet-50-caffe2:opid216' [FP16, FP32], 'resnet-50-pytorch:opid152' [FP32], 'resnet-50-pytorch:opid168' [FP32], 'resnet-50-pytorch:opid184' [FP32], 'resnet-50-pytorch:opid200' [FP32], 'resnet-50-pytorch:opid216' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid152,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 14, 14), (512, 1024, 1, 1)
// Out:    (1, 512, 14, 14)
// Operators: 'resnet-50-caffe2:opid232' [FP16, FP32], 'resnet-50-pytorch:opid232' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid232,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 20, 20), (512, 1024, 1, 1)
// Out:    (1, 512, 20, 20)
// Operators: 'yolov5-640x640-IR:opid194' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid194,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 114, 114), (64, 128, 3, 3)
// Out:    (1, 64, 112, 112)
// Operators: 'photo_style_transfer:opid184' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_photo_style_transfer_opid184,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 114, 114})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 28, 28), (32, 128, 1, 1)
// Out:    (1, 32, 28, 28)
// Operators: 'squeezenet1.1:opid43' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid43,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 28, 28), (512, 128, 1, 1)
// Out:    (1, 512, 28, 28)
// Operators: 'resnet-50-caffe2:opid110' [FP16, FP32], 'resnet-50-caffe2:opid126' [FP16, FP32], 'resnet-50-caffe2:opid74' [FP16, FP32], 'resnet-50-caffe2:opid94' [FP16, FP32], 'resnet-50-pytorch:opid110' [FP32], 'resnet-50-pytorch:opid126' [FP32], 'resnet-50-pytorch:opid74' [FP32], 'resnet-50-pytorch:opid94' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid110,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 40, 40), (128, 128, 1, 1)
// Out:    (1, 128, 40, 40)
// Operators: 'yolov5-640x640-IR:opid136' [FP32], 'yolov5-640x640-IR:opid147' [FP32], 'yolov5-640x640-IR:opid158' [FP32], 'yolov5-640x640-IR:opid251' [FP32], 'yolov5-640x640-IR:opid383' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid136,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 56, 56), (16, 128, 1, 1)
// Out:    (1, 16, 56, 56)
// Operators: 'squeezenet1.1:opid26' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid26,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 58, 58), (128, 128, 3, 3)
// Out:    (1, 128, 56, 56)
// Operators: 'photo_style_transfer:opid109' [FP16, FP32], 'photo_style_transfer:opid122' [FP16, FP32], 'photo_style_transfer:opid135' [FP16, FP32], 'photo_style_transfer:opid148' [FP16, FP32], 'photo_style_transfer:opid161' [FP16, FP32], 'photo_style_transfer:opid44' [FP16, FP32], 'photo_style_transfer:opid57' [FP16, FP32], 'photo_style_transfer:opid70' [FP16, FP32], 'photo_style_transfer:opid83' [FP16, FP32], 'photo_style_transfer:opid96' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_photo_style_transfer_opid109,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 58, 58})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 80, 80), (128, 128, 1, 1)
// Out:    (1, 128, 80, 80)
// Operators: 'yolov5-640x640-IR:opid121' [FP32], 'yolov5-640x640-IR:opid314' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid121,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 80, 80), (255, 128, 1, 1)
// Out:    (1, 255, 80, 80)
// Operators: 'yolov5-640x640-IR:opid319' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid319,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(255), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 80, 80), (64, 128, 1, 1)
// Out:    (1, 64, 80, 80)
// Operators: 'yolov5-640x640-IR:opid115' [FP32], 'yolov5-640x640-IR:opid77' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid115,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 16, 56, 56), (64, 16, 1, 1)
// Out:    (1, 64, 56, 56)
// Operators: 'squeezenet1.1:opid15' [FP16, FP32], 'squeezenet1.1:opid31' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid15,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 2048, 7, 7), (512, 2048, 1, 1)
// Out:    (1, 512, 7, 7)
// Operators: 'resnet-50-caffe2:opid252' [FP16, FP32], 'resnet-50-caffe2:opid268' [FP16, FP32], 'resnet-50-pytorch:opid252' [FP32], 'resnet-50-pytorch:opid268' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid252,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 2048, 7, 7})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 14, 14), (1024, 256, 1, 1)
// Out:    (1, 1024, 14, 14)
// Operators: 'resnet-50-caffe2:opid142' [FP16, FP32], 'resnet-50-caffe2:opid162' [FP16, FP32], 'resnet-50-caffe2:opid178' [FP16, FP32], 'resnet-50-caffe2:opid194' [FP16, FP32], 'resnet-50-caffe2:opid210' [FP16, FP32], 'resnet-50-caffe2:opid226' [FP16, FP32], 'resnet-50-pytorch:opid142' [FP32], 'resnet-50-pytorch:opid162' [FP32], 'resnet-50-pytorch:opid178' [FP32], 'resnet-50-pytorch:opid194' [FP32], 'resnet-50-pytorch:opid210' [FP32], 'resnet-50-pytorch:opid226' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid142,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1024), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 14, 14), (48, 256, 1, 1)
// Out:    (1, 48, 14, 14)
// Operators: 'squeezenet1.1:opid76' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid76,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(48), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 20, 20), (256, 256, 1, 1)
// Out:    (1, 256, 20, 20)
// Operators: 'yolov5-640x640-IR:opid204' [FP32], 'yolov5-640x640-IR:opid468' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid204,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 28, 28), (32, 256, 1, 1)
// Out:    (1, 32, 28, 28)
// Operators: 'squeezenet1.1:opid59' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid59,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 40, 40), (128, 256, 1, 1)
// Out:    (1, 128, 40, 40)
// Operators: 'yolov5-640x640-IR:opid131' [FP32], 'yolov5-640x640-IR:opid169' [FP32], 'yolov5-640x640-IR:opid272' [FP32], 'yolov5-640x640-IR:opid378' [FP32], 'yolov5-640x640-IR:opid393' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid131,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 40, 40), (255, 256, 1, 1)
// Out:    (1, 255, 40, 40)
// Operators: 'yolov5-640x640-IR:opid404' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid404,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(255), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 40, 40), (256, 256, 1, 1)
// Out:    (1, 256, 40, 40)
// Operators: 'yolov5-640x640-IR:opid175' [FP32], 'yolov5-640x640-IR:opid267' [FP32], 'yolov5-640x640-IR:opid399' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid175,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 56, 56), (128, 256, 1, 1)
// Out:    (1, 128, 56, 56)
// Operators: 'resnet-50-caffe2:opid64' [FP16, FP32], 'resnet-50-pytorch:opid64' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid64,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 56, 56), (64, 256, 1, 1)
// Out:    (1, 64, 56, 56)
// Operators: 'resnet-50-caffe2:opid32' [FP16, FP32], 'resnet-50-caffe2:opid48' [FP16, FP32], 'resnet-50-pytorch:opid32' [FP32], 'resnet-50-pytorch:opid48' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid32,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 80, 80), (64, 256, 1, 1)
// Out:    (1, 64, 80, 80)
// Operators: 'yolov5-640x640-IR:opid293' [FP32], 'yolov5-640x640-IR:opid308' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid293,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 3, 232, 232), (32, 3, 9, 9)
// Out:    (1, 32, 224, 224)
// Operators: 'photo_style_transfer:opid5' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_photo_style_transfer_opid5,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({9, 9})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 3, 232, 232})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 160, 160), (32, 32, 1, 1)
// Out:    (1, 32, 160, 160)
// Operators: 'yolov5-640x640-IR:opid50' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid50,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 160, 160})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 232, 232), (3, 32, 9, 9)
// Out:    (1, 3, 224, 224)
// Operators: 'photo_style_transfer:opid220' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_photo_style_transfer_opid220,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({9, 9})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(3), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 232, 232})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 28, 28), (128, 32, 1, 1)
// Out:    (1, 128, 28, 28)
// Operators: 'squeezenet1.1:opid48' [FP16, FP32], 'squeezenet1.1:opid64' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid48,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 384, 14, 14), (48, 384, 1, 1)
// Out:    (1, 48, 14, 14)
// Operators: 'squeezenet1.1:opid92' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid92,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(48), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 384, 14, 14), (64, 384, 1, 1)
// Out:    (1, 64, 14, 14)
// Operators: 'squeezenet1.1:opid108' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid108,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 48, 14, 14), (192, 48, 1, 1)
// Out:    (1, 192, 14, 14)
// Operators: 'squeezenet1.1:opid81' [FP16, FP32], 'squeezenet1.1:opid97' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid81,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 48, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 14, 14), (1000, 512, 1, 1)
// Out:    (1, 1000, 14, 14)
// Operators: 'squeezenet1.1:opid140' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid140,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1000), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 14, 14), (64, 512, 1, 1)
// Out:    (1, 64, 14, 14)
// Operators: 'squeezenet1.1:opid124' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid124,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 20, 20), (255, 512, 1, 1)
// Out:    (1, 255, 20, 20)
// Operators: 'yolov5-640x640-IR:opid489' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid489,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(255), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 20, 20), (256, 512, 1, 1)
// Out:    (1, 256, 20, 20)
// Operators: 'yolov5-640x640-IR:opid185' [FP32], 'yolov5-640x640-IR:opid199' [FP32], 'yolov5-640x640-IR:opid214' [FP32], 'yolov5-640x640-IR:opid225' [FP32], 'yolov5-640x640-IR:opid463' [FP32], 'yolov5-640x640-IR:opid478' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid185,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 20, 20), (512, 512, 1, 1)
// Out:    (1, 512, 20, 20)
// Operators: 'yolov5-640x640-IR:opid220' [FP32], 'yolov5-640x640-IR:opid484' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid220,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 28, 28), (128, 512, 1, 1)
// Out:    (1, 128, 28, 28)
// Operators: 'resnet-50-caffe2:opid100' [FP16, FP32], 'resnet-50-caffe2:opid116' [FP16, FP32], 'resnet-50-caffe2:opid84' [FP16, FP32], 'resnet-50-pytorch:opid100' [FP32], 'resnet-50-pytorch:opid116' [FP32], 'resnet-50-pytorch:opid84' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid100,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 28, 28), (256, 512, 1, 1)
// Out:    (1, 256, 28, 28)
// Operators: 'resnet-50-caffe2:opid132' [FP16, FP32], 'resnet-50-pytorch:opid132' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid132,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 40, 40), (128, 512, 1, 1)
// Out:    (1, 128, 40, 40)
// Operators: 'yolov5-640x640-IR:opid246' [FP32], 'yolov5-640x640-IR:opid261' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid246,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 7, 7), (2048, 512, 1, 1)
// Out:    (1, 2048, 7, 7)
// Operators: 'resnet-50-caffe2:opid242' [FP16, FP32], 'resnet-50-caffe2:opid262' [FP16, FP32], 'resnet-50-caffe2:opid278' [FP16, FP32], 'resnet-50-pytorch:opid242' [FP32], 'resnet-50-pytorch:opid262' [FP32], 'resnet-50-pytorch:opid278' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid242,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(2048), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 7, 7})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 14, 14), (256, 64, 1, 1)
// Out:    (1, 256, 14, 14)
// Operators: 'squeezenet1.1:opid113' [FP16, FP32], 'squeezenet1.1:opid129' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid113,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 160, 160), (32, 64, 1, 1)
// Out:    (1, 32, 160, 160)
// Operators: 'yolov5-640x640-IR:opid45' [FP32], 'yolov5-640x640-IR:opid61' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid45,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 160, 160})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 160, 160), (64, 64, 1, 1)
// Out:    (1, 64, 160, 160)
// Operators: 'yolov5-640x640-IR:opid67' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid67,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 160, 160})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 226, 226), (32, 64, 3, 3)
// Out:    (1, 32, 224, 224)
// Operators: 'photo_style_transfer:opid207' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_photo_style_transfer_opid207,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 226, 226})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 56, 56), (16, 64, 1, 1)
// Out:    (1, 16, 56, 56)
// Operators: 'squeezenet1.1:opid10' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid10,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 56, 56), (256, 64, 1, 1)
// Out:    (1, 256, 56, 56)
// Operators: 'resnet-50-caffe2:opid22' [FP16, FP32], 'resnet-50-caffe2:opid26' [FP16, FP32], 'resnet-50-caffe2:opid42' [FP16, FP32], 'resnet-50-caffe2:opid58' [FP16, FP32], 'resnet-50-pytorch:opid22' [FP32], 'resnet-50-pytorch:opid26' [FP32], 'resnet-50-pytorch:opid42' [FP32], 'resnet-50-pytorch:opid58' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid22,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 56, 56), (64, 64, 1, 1)
// Out:    (1, 64, 56, 56)
// Operators: 'resnet-50-caffe2:opid12' [FP16, FP32], 'resnet-50-pytorch:opid12' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid12,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 80, 80), (64, 64, 1, 1)
// Out:    (1, 64, 80, 80)
// Operators: 'yolov5-640x640-IR:opid104' [FP32], 'yolov5-640x640-IR:opid298' [FP32], 'yolov5-640x640-IR:opid82' [FP32], 'yolov5-640x640-IR:opid93' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid104,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 5, 1, 1), (5, 5, 1, 1)
// Out:    (100, 5, 1, 1)
// Operators: 'mask_rcnn_inception_v2_coco:opid306' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid306,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(5), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 5, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 1024, 14, 14), (2048, 1024, 1, 1)
// Out:    (1, 2048, 7, 7)
// Operators: 'resnet-50-caffe2:opid246' [FP16, FP32], 'resnet-50-pytorch:opid246' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid246,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(2048), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 256, 56, 56), (512, 256, 1, 1)
// Out:    (1, 512, 28, 28)
// Operators: 'resnet-50-caffe2:opid78' [FP16, FP32], 'resnet-50-pytorch:opid78' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid78,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 3, 227, 227), (64, 3, 3, 3)
// Out:    (1, 64, 113, 113)
// Operators: 'squeezenet1.1:opid4' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid4,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 3, 227, 227})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 32, 226, 226), (64, 32, 3, 3)
// Out:    (1, 64, 112, 112)
// Operators: 'photo_style_transfer:opid18' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_photo_style_transfer_opid18,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 226, 226})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 512, 28, 28), (1024, 512, 1, 1)
// Out:    (1, 1024, 14, 14)
// Operators: 'resnet-50-caffe2:opid146' [FP16, FP32], 'resnet-50-pytorch:opid146' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid146,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1024), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 64, 114, 114), (128, 64, 3, 3)
// Out:    (1, 128, 56, 56)
// Operators: 'photo_style_transfer:opid31' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_photo_style_transfer_opid31,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 114, 114})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 128, 104, 104), (256, 128, 3, 3)
// Out:    (1, 256, 52, 52)
// Operators: 'yolo-v3-tf:opid59' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid59,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 104, 104})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 128, 152, 152), (256, 128, 3, 3)
// Out:    (1, 256, 76, 76)
// Operators: 'yolo-v4-tf:opid92' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid92,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 152, 152})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 128, 76, 76), (256, 128, 3, 3)
// Out:    (1, 256, 38, 38)
// Operators: 'yolo-v4-tf:opid540' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid540,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 76, 76})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 256, 38, 38), (512, 256, 3, 3)
// Out:    (1, 512, 19, 19)
// Operators: 'yolo-v4-tf:opid588' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid588,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 256, 52, 52), (512, 256, 3, 3)
// Out:    (1, 512, 26, 26)
// Operators: 'yolo-v3-tf:opid169' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid169,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 52, 52})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 256, 76, 76), (512, 256, 3, 3)
// Out:    (1, 512, 38, 38)
// Operators: 'yolo-v4-tf:opid212' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid212,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 76, 76})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 32, 416, 416), (64, 32, 3, 3)
// Out:    (1, 64, 208, 208)
// Operators: 'yolo-v3-tf:opid8' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid8,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 416, 416})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 32, 608, 608), (64, 32, 3, 3)
// Out:    (1, 64, 304, 304)
// Operators: 'yolo-v4-tf:opid7' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid7,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 608, 608})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 512, 26, 26), (1024, 512, 3, 3)
// Out:    (1, 1024, 13, 13)
// Operators: 'yolo-v3-tf:opid279' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid279,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1024), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 26, 26})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 512, 38, 38), (1024, 512, 3, 3)
// Out:    (1, 1024, 19, 19)
// Operators: 'yolo-v4-tf:opid332' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid332,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1024), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 64, 208, 208), (128, 64, 3, 3)
// Out:    (1, 128, 104, 104)
// Operators: 'yolo-v3-tf:opid27' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid27,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 208, 208})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 64, 304, 304), (128, 64, 3, 3)
// Out:    (1, 128, 152, 152)
// Operators: 'yolo-v4-tf:opid44' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid44,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 304, 304})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 12, 320, 320), (32, 12, 3, 3)
// Out:    (1, 32, 320, 320)
// Operators: 'yolov5-640x640-IR:opid35' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid35,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 12, 320, 320})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 128, 112, 112), (128, 128, 3, 3)
// Out:    (1, 128, 112, 112)
// Operators: 'vgg16-IR:opid20' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_vgg16_IR_opid20,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 112, 112})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 128, 28, 28), (128, 128, 3, 3)
// Out:    (1, 128, 28, 28)
// Operators: 'resnet-50-caffe2:opid105' [FP16, FP32], 'resnet-50-caffe2:opid121' [FP16, FP32], 'resnet-50-caffe2:opid89' [FP16, FP32], 'resnet-50-pytorch:opid105' [FP32], 'resnet-50-pytorch:opid121' [FP32], 'resnet-50-pytorch:opid89' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid105,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 128, 40, 40), (128, 128, 3, 3)
// Out:    (1, 128, 40, 40)
// Operators: 'yolov5-640x640-IR:opid141' [FP32], 'yolov5-640x640-IR:opid152' [FP32], 'yolov5-640x640-IR:opid163' [FP32], 'yolov5-640x640-IR:opid256' [FP32], 'yolov5-640x640-IR:opid388' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid141,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 128, 56, 56), (256, 128, 3, 3)
// Out:    (1, 256, 56, 56)
// Operators: 'vgg16-IR:opid26' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_vgg16_IR_opid26,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 16, 56, 56), (64, 16, 3, 3)
// Out:    (1, 64, 56, 56)
// Operators: 'squeezenet1.1:opid20' [FP16, FP32], 'squeezenet1.1:opid36' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid20,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 256, 14, 14), (256, 256, 3, 3)
// Out:    (1, 256, 14, 14)
// Operators: 'resnet-50-caffe2:opid157' [FP16, FP32], 'resnet-50-caffe2:opid173' [FP16, FP32], 'resnet-50-caffe2:opid189' [FP16, FP32], 'resnet-50-caffe2:opid205' [FP16, FP32], 'resnet-50-caffe2:opid221' [FP16, FP32], 'resnet-50-pytorch:opid157' [FP32], 'resnet-50-pytorch:opid173' [FP32], 'resnet-50-pytorch:opid189' [FP32], 'resnet-50-pytorch:opid205' [FP32], 'resnet-50-pytorch:opid221' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid157,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 256, 20, 20), (256, 256, 3, 3)
// Out:    (1, 256, 20, 20)
// Operators: 'yolov5-640x640-IR:opid209' [FP32], 'yolov5-640x640-IR:opid473' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid209,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 256, 28, 28), (512, 256, 3, 3)
// Out:    (1, 512, 28, 28)
// Operators: 'vgg16-IR:opid42' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_vgg16_IR_opid42,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::Values(InferenceEngine::Precision::FP32), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 256, 56, 56), (256, 256, 3, 3)
// Out:    (1, 256, 56, 56)
// Operators: 'vgg16-IR:opid31' [FP16, FP32], 'vgg16-IR:opid36' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_vgg16_IR_opid31,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::Values(InferenceEngine::Precision::FP32), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 3, 224, 224), (64, 3, 3, 3)
// Out:    (1, 64, 224, 224)
// Operators: 'vgg16-IR:opid4' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_vgg16_IR_opid4,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 3, 224, 224})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 32, 160, 160), (32, 32, 3, 3)
// Out:    (1, 32, 160, 160)
// Operators: 'yolov5-640x640-IR:opid55' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid55,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 160, 160})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 32, 28, 28), (128, 32, 3, 3)
// Out:    (1, 128, 28, 28)
// Operators: 'squeezenet1.1:opid53' [FP16, FP32], 'squeezenet1.1:opid69' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid53,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 48, 14, 14), (192, 48, 3, 3)
// Out:    (1, 192, 14, 14)
// Operators: 'squeezenet1.1:opid102' [FP16, FP32], 'squeezenet1.1:opid86' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid102,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 48, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 512, 14, 14), (512, 512, 3, 3)
// Out:    (1, 512, 14, 14)
// Operators: 'vgg16-IR:opid58' [FP16, FP32], 'vgg16-IR:opid63' [FP16, FP32], 'vgg16-IR:opid68' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_vgg16_IR_opid58,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 512, 28, 28), (512, 512, 3, 3)
// Out:    (1, 512, 28, 28)
// Operators: 'vgg16-IR:opid47' [FP16, FP32], 'vgg16-IR:opid52' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_vgg16_IR_opid47,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 512, 7, 7), (512, 512, 3, 3)
// Out:    (1, 512, 7, 7)
// Operators: 'resnet-50-caffe2:opid257' [FP16, FP32], 'resnet-50-caffe2:opid273' [FP16, FP32], 'resnet-50-pytorch:opid257' [FP32], 'resnet-50-pytorch:opid273' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid257,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 7, 7})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 64, 112, 112), (128, 64, 3, 3)
// Out:    (1, 128, 112, 112)
// Operators: 'vgg16-IR:opid15' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_vgg16_IR_opid15,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 112, 112})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 64, 14, 14), (256, 64, 3, 3)
// Out:    (1, 256, 14, 14)
// Operators: 'squeezenet1.1:opid118' [FP16, FP32], 'squeezenet1.1:opid134' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid118,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 64, 224, 224), (64, 64, 3, 3)
// Out:    (1, 64, 224, 224)
// Operators: 'vgg16-IR:opid9' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_vgg16_IR_opid9,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 224, 224})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 64, 56, 56), (64, 64, 3, 3)
// Out:    (1, 64, 56, 56)
// Operators: 'resnet-50-caffe2:opid17' [FP16, FP32], 'resnet-50-caffe2:opid37' [FP16, FP32], 'resnet-50-caffe2:opid53' [FP16, FP32], 'resnet-50-pytorch:opid17' [FP32], 'resnet-50-pytorch:opid37' [FP32], 'resnet-50-pytorch:opid53' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid17,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 64, 80, 80), (64, 64, 3, 3)
// Out:    (1, 64, 80, 80)
// Operators: 'yolov5-640x640-IR:opid109' [FP32], 'yolov5-640x640-IR:opid303' [FP32], 'yolov5-640x640-IR:opid87' [FP32], 'yolov5-640x640-IR:opid98' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid109,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '2,2'}
// In:     (1, 128, 56, 56), (128, 128, 3, 3)
// Out:    (1, 128, 28, 28)
// Operators: 'resnet-50-caffe2:opid69' [FP16, FP32], 'resnet-50-pytorch:opid69' [FP32], 'resnet-50-tf:opid67' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid69,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '2,2'}
// In:     (1, 128, 80, 80), (128, 128, 3, 3)
// Out:    (1, 128, 40, 40)
// Operators: 'yolov5-640x640-IR:opid372' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid372,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '2,2'}
// In:     (1, 128, 80, 80), (256, 128, 3, 3)
// Out:    (1, 256, 40, 40)
// Operators: 'yolov5-640x640-IR:opid126' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid126,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '2,2'}
// In:     (1, 256, 28, 28), (256, 256, 3, 3)
// Out:    (1, 256, 14, 14)
// Operators: 'resnet-50-caffe2:opid137' [FP16, FP32], 'resnet-50-pytorch:opid137' [FP32], 'resnet-50-tf:opid135' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid137,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '2,2'}
// In:     (1, 256, 40, 40), (256, 256, 3, 3)
// Out:    (1, 256, 20, 20)
// Operators: 'yolov5-640x640-IR:opid457' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid457,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '2,2'}
// In:     (1, 256, 40, 40), (512, 256, 3, 3)
// Out:    (1, 512, 20, 20)
// Operators: 'yolov5-640x640-IR:opid180' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid180,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '2,2'}
// In:     (1, 32, 320, 320), (64, 32, 3, 3)
// Out:    (1, 64, 160, 160)
// Operators: 'yolov5-640x640-IR:opid40' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid40,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 320, 320})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '2,2'}
// In:     (1, 512, 14, 14), (512, 512, 3, 3)
// Out:    (1, 512, 7, 7)
// Operators: 'resnet-50-caffe2:opid237' [FP16, FP32], 'resnet-50-pytorch:opid237' [FP32], 'resnet-50-tf:opid235' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid237,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '2,2'}
// In:     (1, 64, 160, 160), (128, 64, 3, 3)
// Out:    (1, 128, 80, 80)
// Operators: 'yolov5-640x640-IR:opid72' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid72,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 160, 160})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '3,3', 'pads_end': '3,3', 'strides': '2,2'}
// In:     (1, 3, 224, 224), (64, 3, 7, 7)
// Out:    (1, 64, 112, 112)
// Operators: 'resnet-50-caffe2:opid6' [FP16, FP32], 'resnet-50-pytorch:opid6' [FP32], 'resnet-50-tf:opid4' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid6,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({7, 7})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({3, 3})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({3, 3})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 3, 224, 224})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1', 'pads_begin': '0', 'pads_end': '0', 'strides': '1'}
// In:     (64, 106, 64), (128, 106, 3)
// Out:    (64, 128, 64)
// Operators: 'LPCnet-lpcnet_enc:opid12' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_LPCnet_lpcnet_enc_opid12,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3})), // kernel
            ::testing::Values(std::vector<size_t>({1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0})), // pads_end
            ::testing::Values(std::vector<size_t>({1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({64, 106, 64})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1', 'pads_begin': '0', 'pads_end': '0', 'strides': '1'}
// In:     (64, 128, 64), (128, 128, 3)
// Out:    (64, 128, 64)
// Operators: 'LPCnet-lpcnet_enc:opid17' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_LPCnet_lpcnet_enc_opid17,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3})), // kernel
            ::testing::Values(std::vector<size_t>({1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0})), // pads_end
            ::testing::Values(std::vector<size_t>({1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({64, 128, 64})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1, 128, 128), (16, 1, 3, 3)
// Out:    (1, 16, 128, 128)
// Operators: '2d_unet-graph-transform:opid2' [FP32], '2d_unet:opid2' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid2,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1, 128, 128})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1, 224, 224), (32, 1, 5, 5)
// Out:    (1, 32, 224, 224)
// Operators: 'super_resolution:opid2' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_super_resolution_opid2,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({5, 5})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1, 224, 224})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 10, 1, 1), (240, 10, 1, 1)
// Out:    (1, 240, 1, 1)
// Operators: 'efficientdet-d1-tf:opid183' [FP16, FP32], 'efficientdet-d1-tf:opid211' [FP16, FP32], 'efficientdet-d1-tf:opid243' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid183,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(240), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 10, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 13, 13), (255, 1024, 1, 1)
// Out:    (1, 255, 13, 13)
// Operators: 'yolo-v3-tf:opid373' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid373,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(255), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 13, 13})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 13, 13), (512, 1024, 1, 1)
// Out:    (1, 512, 13, 13)
// Operators: 'yolo-v3-tf:opid285' [FP16, FP32], 'yolo-v3-tf:opid298' [FP16, FP32], 'yolo-v3-tf:opid311' [FP16, FP32], 'yolo-v3-tf:opid324' [FP16, FP32], 'yolo-v3-tf:opid337' [FP16, FP32], 'yolo-v3-tf:opid349' [FP16, FP32], 'yolo-v3-tf:opid361' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid285,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 13, 13})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 14, 14), (256, 1024, 1, 1)
// Out:    (1, 256, 14, 14)
// Operators: 'resnet-50-tf:opid150' [FP16, FP32], 'resnet-50-tf:opid166' [FP16, FP32], 'resnet-50-tf:opid182' [FP16, FP32], 'resnet-50-tf:opid198' [FP16, FP32], 'resnet-50-tf:opid214' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid150,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 14, 14), (512, 1024, 1, 1)
// Out:    (1, 512, 14, 14)
// Operators: 'resnet-50-tf:opid230' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid230,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 17, 17), (128, 1024, 1, 1)
// Out:    (1, 128, 17, 17)
// Operators: 'googlenet-v4-tf:opid282' [FP16, FP32], 'googlenet-v4-tf:opid334' [FP16, FP32], 'googlenet-v4-tf:opid386' [FP16, FP32], 'googlenet-v4-tf:opid438' [FP16, FP32], 'googlenet-v4-tf:opid490' [FP16, FP32], 'googlenet-v4-tf:opid542' [FP16, FP32], 'googlenet-v4-tf:opid594' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid282,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 17, 17), (192, 1024, 1, 1)
// Out:    (1, 192, 17, 17)
// Operators: 'googlenet-v4-tf:opid241' [FP16, FP32], 'googlenet-v4-tf:opid256' [FP16, FP32], 'googlenet-v4-tf:opid293' [FP16, FP32], 'googlenet-v4-tf:opid308' [FP16, FP32], 'googlenet-v4-tf:opid345' [FP16, FP32], 'googlenet-v4-tf:opid360' [FP16, FP32], 'googlenet-v4-tf:opid397' [FP16, FP32], 'googlenet-v4-tf:opid412' [FP16, FP32], 'googlenet-v4-tf:opid449' [FP16, FP32], 'googlenet-v4-tf:opid464' [FP16, FP32], 'googlenet-v4-tf:opid501' [FP16, FP32], 'googlenet-v4-tf:opid516' [FP16, FP32], 'googlenet-v4-tf:opid553' [FP16, FP32], 'googlenet-v4-tf:opid568' [FP16, FP32], 'googlenet-v4-tf:opid600' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid241,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 17, 17), (256, 1024, 1, 1)
// Out:    (1, 256, 17, 17)
// Operators: 'googlenet-v4-tf:opid610' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid610,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 17, 17), (384, 1024, 1, 1)
// Out:    (1, 384, 17, 17)
// Operators: 'googlenet-v4-tf:opid236' [FP16, FP32], 'googlenet-v4-tf:opid288' [FP16, FP32], 'googlenet-v4-tf:opid340' [FP16, FP32], 'googlenet-v4-tf:opid392' [FP16, FP32], 'googlenet-v4-tf:opid444' [FP16, FP32], 'googlenet-v4-tf:opid496' [FP16, FP32], 'googlenet-v4-tf:opid548' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid236,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(384), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 19, 19), (1024, 1024, 1, 1)
// Out:    (1, 1024, 19, 19)
// Operators: 'yolo-v4-tf:opid397' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid397,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1024), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 19, 19), (255, 1024, 1, 1)
// Out:    (1, 255, 19, 19)
// Operators: 'yolo-v4-tf:opid631' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid631,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(255), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 19, 19), (512, 1024, 1, 1)
// Out:    (1, 512, 19, 19)
// Operators: 'yolo-v4-tf:opid337' [FP16, FP32], 'yolo-v4-tf:opid391' [FP16, FP32], 'yolo-v4-tf:opid402' [FP16, FP32], 'yolo-v4-tf:opid414' [FP16, FP32], 'yolo-v4-tf:opid436' [FP16, FP32], 'yolo-v4-tf:opid595' [FP16, FP32], 'yolo-v4-tf:opid607' [FP16, FP32], 'yolo-v4-tf:opid619' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid337,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 112, 40, 40), (672, 112, 1, 1)
// Out:    (1, 672, 40, 40)
// Operators: 'efficientdet-d1-tf:opid364' [FP16, FP32], 'efficientdet-d1-tf:opid392' [FP16, FP32], 'efficientdet-d1-tf:opid420' [FP16, FP32], 'efficientdet-d1-tf:opid452' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid364,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(672), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 112, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 112, 40, 40), (88, 112, 1, 1)
// Out:    (1, 88, 40, 40)
// Operators: 'efficientdet-d1-tf:opid448' [FP16, FP32], 'efficientdet-d1-tf:opid708' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid448,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(88), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 112, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1152, 1, 1), (48, 1152, 1, 1)
// Out:    (1, 48, 1, 1)
// Operators: 'efficientdet-d1-tf:opid491' [FP16, FP32], 'efficientdet-d1-tf:opid519' [FP16, FP32], 'efficientdet-d1-tf:opid547' [FP16, FP32], 'efficientdet-d1-tf:opid575' [FP16, FP32], 'efficientdet-d1-tf:opid603' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid491,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(48), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1152, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1152, 20, 20), (192, 1152, 1, 1)
// Out:    (1, 192, 20, 20)
// Operators: 'efficientdet-d1-tf:opid502' [FP16, FP32], 'efficientdet-d1-tf:opid530' [FP16, FP32], 'efficientdet-d1-tf:opid558' [FP16, FP32], 'efficientdet-d1-tf:opid586' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid502,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1152, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1152, 20, 20), (320, 1152, 1, 1)
// Out:    (1, 320, 20, 20)
// Operators: 'efficientdet-d1-tf:opid614' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid614,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(320), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1152, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 1, 1), (24, 128, 3, 3)
// Out:    (1, 24, 1, 1)
// Operators: 'ssd_mobilenet_v2_coco:opid359' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid359,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(24), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 1, 1), (546, 128, 3, 3)
// Out:    (1, 546, 1, 1)
// Operators: 'ssd_mobilenet_v2_coco:opid410' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid410,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(546), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 104, 104), (64, 128, 1, 1)
// Out:    (1, 64, 104, 104)
// Operators: 'yolo-v3-tf:opid33' [FP16, FP32], 'yolo-v3-tf:opid46' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid33,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 104, 104})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 152, 152), (128, 128, 1, 1)
// Out:    (1, 128, 152, 152)
// Operators: 'yolo-v4-tf:opid87' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid87,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 152, 152})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 152, 152), (64, 128, 1, 1)
// Out:    (1, 64, 152, 152)
// Operators: 'yolo-v4-tf:opid49' [FP16, FP32], 'yolo-v4-tf:opid81' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid49,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 152, 152})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 16, 16), (128, 128, 3, 3)
// Out:    (1, 128, 16, 16)
// Operators: '2d_unet-graph-transform:opid40' [FP32], '2d_unet-graph-transform:opid67' [FP32], '2d_unet:opid102' [FP16, FP32], '2d_unet:opid40' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid40,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 16, 16})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 2, 2), (256, 128, 1, 1)
// Out:    (1, 256, 2, 2)
// Operators: 'ssd_mobilenet_v2_coco:opid331' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid331,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 2, 2})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 28, 28), (128, 128, 3, 3)
// Out:    (1, 128, 28, 28)
// Operators: 'resnet-50-tf:opid103' [FP16, FP32], 'resnet-50-tf:opid119' [FP16, FP32], 'resnet-50-tf:opid87' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid103,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 28, 28), (512, 128, 1, 1)
// Out:    (1, 512, 28, 28)
// Operators: 'resnet-50-tf:opid108' [FP16, FP32], 'resnet-50-tf:opid124' [FP16, FP32], 'resnet-50-tf:opid72' [FP16, FP32], 'resnet-50-tf:opid92' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid108,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 3, 3), (256, 128, 1, 1)
// Out:    (1, 256, 3, 3)
// Operators: 'ssd_mobilenet_v2_coco:opid308' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid308,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 3, 3})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 304, 304), (64, 128, 1, 1)
// Out:    (1, 64, 304, 304)
// Operators: 'yolo-v4-tf:opid39' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid39,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 304, 304})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 32, 32), (64, 128, 3, 3)
// Out:    (1, 64, 32, 32)
// Operators: '2d_unet-graph-transform:opid78' [FP32], '2d_unet:opid148' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid78,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 32, 32})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 50, 86), (128, 128, 3, 3)
// Out:    (1, 128, 50, 86)
// Operators: 'mask_rcnn_inception_v2_coco:opid151' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid188' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid151,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 50, 86), (160, 128, 3, 3)
// Out:    (1, 160, 50, 86)
// Operators: 'mask_rcnn_inception_v2_coco:opid210' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid220' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid210,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(160), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 50, 86), (192, 128, 3, 3)
// Out:    (1, 192, 50, 86)
// Operators: 'mask_rcnn_inception_v2_coco:opid247' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid247,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 52, 52), (256, 128, 3, 3)
// Out:    (1, 256, 52, 52)
// Operators: 'yolo-v3-tf:opid110' [FP16, FP32], 'yolo-v3-tf:opid123' [FP16, FP32], 'yolo-v3-tf:opid136' [FP16, FP32], 'yolo-v3-tf:opid149' [FP16, FP32], 'yolo-v3-tf:opid162' [FP16, FP32], 'yolo-v3-tf:opid461' [FP16, FP32], 'yolo-v3-tf:opid473' [FP16, FP32], 'yolo-v3-tf:opid485' [FP16, FP32], 'yolo-v3-tf:opid71' [FP16, FP32], 'yolo-v3-tf:opid84' [FP16, FP32], 'yolo-v3-tf:opid97' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid110,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 52, 52})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 76, 76), (128, 128, 1, 1)
// Out:    (1, 128, 76, 76)
// Operators: 'yolo-v4-tf:opid102' [FP16, FP32], 'yolo-v4-tf:opid113' [FP16, FP32], 'yolo-v4-tf:opid124' [FP16, FP32], 'yolo-v4-tf:opid135' [FP16, FP32], 'yolo-v4-tf:opid146' [FP16, FP32], 'yolo-v4-tf:opid157' [FP16, FP32], 'yolo-v4-tf:opid168' [FP16, FP32], 'yolo-v4-tf:opid179' [FP16, FP32], 'yolo-v4-tf:opid190' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid102,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 76, 76})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 76, 76), (128, 128, 3, 3)
// Out:    (1, 128, 76, 76)
// Operators: 'yolo-v4-tf:opid107' [FP16, FP32], 'yolo-v4-tf:opid118' [FP16, FP32], 'yolo-v4-tf:opid129' [FP16, FP32], 'yolo-v4-tf:opid140' [FP16, FP32], 'yolo-v4-tf:opid151' [FP16, FP32], 'yolo-v4-tf:opid162' [FP16, FP32], 'yolo-v4-tf:opid173' [FP16, FP32], 'yolo-v4-tf:opid184' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid107,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 76, 76})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 76, 76), (256, 128, 3, 3)
// Out:    (1, 256, 76, 76)
// Operators: 'yolo-v4-tf:opid516' [FP16, FP32], 'yolo-v4-tf:opid528' [FP16, FP32], 'yolo-v4-tf:opid636' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid516,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 76, 76})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 8, 8), (256, 128, 3, 3)
// Out:    (1, 256, 8, 8)
// Operators: '2d_unet-graph-transform:opid46' [FP32], '2d_unet:opid46' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid46,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 8, 8})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1280, 10, 10), (24, 1280, 3, 3)
// Out:    (1, 24, 10, 10)
// Operators: 'ssd_mobilenet_v2_coco:opid267' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid267,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(24), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1280, 10, 10})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1280, 10, 10), (256, 1280, 1, 1)
// Out:    (1, 256, 10, 10)
// Operators: 'ssd_mobilenet_v2_coco:opid275' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid275,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1280, 10, 10})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1280, 10, 10), (546, 1280, 3, 3)
// Out:    (1, 546, 10, 10)
// Operators: 'ssd_mobilenet_v2_coco:opid378' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid378,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(546), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1280, 10, 10})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 144, 1, 1), (6, 144, 1, 1)
// Out:    (1, 6, 1, 1)
// Operators: 'efficientdet-d1-tf:opid123' [FP16, FP32], 'efficientdet-d1-tf:opid151' [FP16, FP32], 'efficientdet-d1-tf:opid95' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid123,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(6), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 144, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 144, 160, 160), (24, 144, 1, 1)
// Out:    (1, 24, 160, 160)
// Operators: 'efficientdet-d1-tf:opid106' [FP16, FP32], 'efficientdet-d1-tf:opid134' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid106,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(24), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 144, 160, 160})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 144, 38, 38), (32, 144, 1, 1)
// Out:    (1, 32, 38, 38)
// Operators: 'ssd_mobilenet_v2_coco:opid59' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid59,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 144, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 144, 75, 75), (24, 144, 1, 1)
// Out:    (1, 24, 75, 75)
// Operators: 'ssd_mobilenet_v2_coco:opid44' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid44,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(24), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 144, 75, 75})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 144, 80, 80), (40, 144, 1, 1)
// Out:    (1, 40, 80, 80)
// Operators: 'efficientdet-d1-tf:opid162' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid162,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(40), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 144, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1536, 8, 8), (256, 1536, 1, 1)
// Out:    (1, 256, 8, 8)
// Operators: 'googlenet-v4-tf:opid632' [FP16, FP32], 'googlenet-v4-tf:opid680' [FP16, FP32], 'googlenet-v4-tf:opid686' [FP16, FP32], 'googlenet-v4-tf:opid734' [FP16, FP32], 'googlenet-v4-tf:opid740' [FP16, FP32], 'googlenet-v4-tf:opid788' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid632,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1536, 8, 8})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1536, 8, 8), (384, 1536, 1, 1)
// Out:    (1, 384, 8, 8)
// Operators: 'googlenet-v4-tf:opid637' [FP16, FP32], 'googlenet-v4-tf:opid653' [FP16, FP32], 'googlenet-v4-tf:opid691' [FP16, FP32], 'googlenet-v4-tf:opid707' [FP16, FP32], 'googlenet-v4-tf:opid745' [FP16, FP32], 'googlenet-v4-tf:opid761' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid637,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(384), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1536, 8, 8})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 16, 1, 1), (4, 16, 1, 1)
// Out:    (1, 4, 1, 1)
// Operators: 'efficientdet-d1-tf:opid40' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid40,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(4), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 16, 128, 128), (16, 16, 3, 3)
// Out:    (1, 16, 128, 128)
// Operators: '2d_unet-graph-transform:opid115' [FP32], '2d_unet-graph-transform:opid7' [FP32], '2d_unet:opid255' [FP16, FP32], '2d_unet:opid7' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid115,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 128, 128})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 16, 150, 150), (96, 16, 1, 1)
// Out:    (1, 96, 150, 150)
// Operators: 'ssd_mobilenet_v2_coco:opid20' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid20,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 150, 150})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 16, 320, 320), (16, 16, 1, 1)
// Out:    (1, 16, 320, 320)
// Operators: 'efficientdet-d1-tf:opid51' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid51,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 320, 320})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 16, 320, 320), (96, 16, 1, 1)
// Out:    (1, 96, 320, 320)
// Operators: 'efficientdet-d1-tf:opid56' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid56,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 320, 320})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 16, 64, 64), (32, 16, 3, 3)
// Out:    (1, 32, 64, 64)
// Operators: '2d_unet-graph-transform:opid13' [FP32], '2d_unet:opid13' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid13,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 64, 64})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 160, 10, 10), (960, 160, 1, 1)
// Out:    (1, 960, 10, 10)
// Operators: 'ssd_mobilenet_v2_coco:opid218' [FP16, FP32], 'ssd_mobilenet_v2_coco:opid233' [FP16, FP32], 'ssd_mobilenet_v2_coco:opid248' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid218,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(960), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 160, 10, 10})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 160, 50, 86), (160, 160, 3, 3)
// Out:    (1, 160, 50, 86)
// Operators: 'mask_rcnn_inception_v2_coco:opid225' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid225,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(160), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 160, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 160, 50, 86), (192, 160, 3, 3)
// Out:    (1, 192, 50, 86)
// Operators: 'mask_rcnn_inception_v2_coco:opid257' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid257,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 160, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 160, 73, 73), (64, 160, 1, 1)
// Out:    (1, 64, 73, 73)
// Operators: 'googlenet-v4-tf:opid28' [FP16, FP32], 'googlenet-v4-tf:opid38' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid28,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 160, 73, 73})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 192, 100, 171), (32, 192, 1, 1)
// Out:    (1, 32, 100, 171)
// Operators: 'mask_rcnn_inception_v2_coco:opid56' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid56,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 192, 100, 171})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 192, 100, 171), (64, 192, 1, 1)
// Out:    (1, 64, 100, 171)
// Operators: 'mask_rcnn_inception_v2_coco:opid25' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid30' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid40' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid25,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 192, 100, 171})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 192, 17, 17), (192, 192, 7, 1)
// Out:    (1, 192, 17, 17)
// Operators: 'googlenet-v4-tf:opid261' [FP16, FP32], 'googlenet-v4-tf:opid313' [FP16, FP32], 'googlenet-v4-tf:opid365' [FP16, FP32], 'googlenet-v4-tf:opid417' [FP16, FP32], 'googlenet-v4-tf:opid469' [FP16, FP32], 'googlenet-v4-tf:opid521' [FP16, FP32], 'googlenet-v4-tf:opid573' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid261,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({7, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 192, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 192, 17, 17), (224, 192, 1, 7)
// Out:    (1, 224, 17, 17)
// Operators: 'googlenet-v4-tf:opid246' [FP16, FP32], 'googlenet-v4-tf:opid266' [FP16, FP32], 'googlenet-v4-tf:opid298' [FP16, FP32], 'googlenet-v4-tf:opid318' [FP16, FP32], 'googlenet-v4-tf:opid350' [FP16, FP32], 'googlenet-v4-tf:opid370' [FP16, FP32], 'googlenet-v4-tf:opid402' [FP16, FP32], 'googlenet-v4-tf:opid422' [FP16, FP32], 'googlenet-v4-tf:opid454' [FP16, FP32], 'googlenet-v4-tf:opid474' [FP16, FP32], 'googlenet-v4-tf:opid506' [FP16, FP32], 'googlenet-v4-tf:opid526' [FP16, FP32], 'googlenet-v4-tf:opid558' [FP16, FP32], 'googlenet-v4-tf:opid578' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid246,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 7})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(224), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 192, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 192, 19, 19), (64, 192, 1, 1)
// Out:    (1, 64, 19, 19)
// Operators: 'ssd_mobilenet_v2_coco:opid103' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid103,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 192, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 192, 20, 20), (1152, 192, 1, 1)
// Out:    (1, 1152, 20, 20)
// Operators: 'efficientdet-d1-tf:opid479' [FP16, FP32], 'efficientdet-d1-tf:opid507' [FP16, FP32], 'efficientdet-d1-tf:opid535' [FP16, FP32], 'efficientdet-d1-tf:opid563' [FP16, FP32], 'efficientdet-d1-tf:opid591' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid479,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1152), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 192, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 192, 35, 35), (224, 192, 3, 3)
// Out:    (1, 224, 35, 35)
// Operators: 'googlenet-v4-tf:opid224' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid224,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(224), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 192, 35, 35})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 192, 38, 38), (32, 192, 1, 1)
// Out:    (1, 32, 38, 38)
// Operators: 'ssd_mobilenet_v2_coco:opid73' [FP16, FP32], 'ssd_mobilenet_v2_coco:opid88' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid73,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 192, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 192, 50, 86), (192, 192, 3, 3)
// Out:    (1, 192, 50, 86)
// Operators: 'mask_rcnn_inception_v2_coco:opid262' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid262,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 192, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1920, 1, 1), (80, 1920, 1, 1)
// Out:    (1, 80, 1, 1)
// Operators: 'efficientdet-d1-tf:opid630' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid630,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(80), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1920, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1920, 20, 20), (320, 1920, 1, 1)
// Out:    (1, 320, 20, 20)
// Operators: 'efficientdet-d1-tf:opid641' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid641,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(320), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1920, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 20, 1, 1), (480, 20, 1, 1)
// Out:    (1, 480, 1, 1)
// Operators: 'efficientdet-d1-tf:opid270' [FP16, FP32], 'efficientdet-d1-tf:opid298' [FP16, FP32], 'efficientdet-d1-tf:opid326' [FP16, FP32], 'efficientdet-d1-tf:opid354' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid270,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(480), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 20, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 2048, 19, 19), (512, 2048, 1, 1)
// Out:    (1, 512, 19, 19)
// Operators: 'yolo-v4-tf:opid424' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid424,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 2048, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 2048, 7, 7), (512, 2048, 1, 1)
// Out:    (1, 512, 7, 7)
// Operators: 'resnet-50-tf:opid250' [FP16, FP32], 'resnet-50-tf:opid266' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid250,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 2048, 7, 7})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 224, 17, 17), (224, 224, 7, 1)
// Out:    (1, 224, 17, 17)
// Operators: 'googlenet-v4-tf:opid271' [FP16, FP32], 'googlenet-v4-tf:opid323' [FP16, FP32], 'googlenet-v4-tf:opid375' [FP16, FP32], 'googlenet-v4-tf:opid427' [FP16, FP32], 'googlenet-v4-tf:opid479' [FP16, FP32], 'googlenet-v4-tf:opid531' [FP16, FP32], 'googlenet-v4-tf:opid583' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid271,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({7, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(224), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 224, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 224, 17, 17), (256, 224, 1, 7)
// Out:    (1, 256, 17, 17)
// Operators: 'googlenet-v4-tf:opid276' [FP16, FP32], 'googlenet-v4-tf:opid328' [FP16, FP32], 'googlenet-v4-tf:opid380' [FP16, FP32], 'googlenet-v4-tf:opid432' [FP16, FP32], 'googlenet-v4-tf:opid484' [FP16, FP32], 'googlenet-v4-tf:opid536' [FP16, FP32], 'googlenet-v4-tf:opid588' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid276,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 7})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 224, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 224, 17, 17), (256, 224, 7, 1)
// Out:    (1, 256, 17, 17)
// Operators: 'googlenet-v4-tf:opid251' [FP16, FP32], 'googlenet-v4-tf:opid303' [FP16, FP32], 'googlenet-v4-tf:opid355' [FP16, FP32], 'googlenet-v4-tf:opid407' [FP16, FP32], 'googlenet-v4-tf:opid459' [FP16, FP32], 'googlenet-v4-tf:opid511' [FP16, FP32], 'googlenet-v4-tf:opid563' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid251,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({7, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 224, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 24, 160, 160), (144, 24, 1, 1)
// Out:    (1, 144, 160, 160)
// Operators: 'efficientdet-d1-tf:opid111' [FP16, FP32], 'efficientdet-d1-tf:opid139' [FP16, FP32], 'efficientdet-d1-tf:opid83' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid111,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(144), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 24, 160, 160})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 24, 75, 75), (144, 24, 1, 1)
// Out:    (1, 144, 75, 75)
// Operators: 'ssd_mobilenet_v2_coco:opid34' [FP16, FP32], 'ssd_mobilenet_v2_coco:opid49' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid34,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(144), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 24, 75, 75})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 240, 1, 1), (10, 240, 1, 1)
// Out:    (1, 10, 1, 1)
// Operators: 'efficientdet-d1-tf:opid178' [FP16, FP32], 'efficientdet-d1-tf:opid206' [FP16, FP32], 'efficientdet-d1-tf:opid238' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid178,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(10), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 240, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 240, 40, 40), (80, 240, 1, 1)
// Out:    (1, 80, 40, 40)
// Operators: 'efficientdet-d1-tf:opid249' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid249,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(80), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 240, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 240, 80, 80), (40, 240, 1, 1)
// Out:    (1, 40, 80, 80)
// Operators: 'efficientdet-d1-tf:opid189' [FP16, FP32], 'efficientdet-d1-tf:opid217' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid189,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(40), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 240, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 100, 171), (64, 256, 1, 1)
// Out:    (1, 64, 100, 171)
// Operators: 'mask_rcnn_inception_v2_coco:opid62' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid67' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid77' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid93' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid62,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 100, 171})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 14, 14), (1024, 256, 1, 1)
// Out:    (1, 1024, 14, 14)
// Operators: 'resnet-50-tf:opid140' [FP16, FP32], 'resnet-50-tf:opid160' [FP16, FP32], 'resnet-50-tf:opid176' [FP16, FP32], 'resnet-50-tf:opid192' [FP16, FP32], 'resnet-50-tf:opid208' [FP16, FP32], 'resnet-50-tf:opid224' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid140,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1024), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 14, 14), (256, 256, 3, 3)
// Out:    (1, 256, 14, 14)
// Operators: 'resnet-50-tf:opid155' [FP16, FP32], 'resnet-50-tf:opid171' [FP16, FP32], 'resnet-50-tf:opid187' [FP16, FP32], 'resnet-50-tf:opid203' [FP16, FP32], 'resnet-50-tf:opid219' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid155,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 16, 16), (128, 256, 3, 3)
// Out:    (1, 128, 16, 16)
// Operators: '2d_unet-graph-transform:opid62' [FP32], '2d_unet:opid97' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid62,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 16, 16})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 17, 17), (256, 256, 1, 7)
// Out:    (1, 256, 17, 17)
// Operators: 'googlenet-v4-tf:opid615' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid615,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 7})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 17, 17), (320, 256, 7, 1)
// Out:    (1, 320, 17, 17)
// Operators: 'googlenet-v4-tf:opid620' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid620,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({7, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(320), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 2, 2), (24, 256, 3, 3)
// Out:    (1, 24, 2, 2)
// Operators: 'ssd_mobilenet_v2_coco:opid336' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid336,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(24), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 2, 2})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 2, 2), (546, 256, 3, 3)
// Out:    (1, 546, 2, 2)
// Operators: 'ssd_mobilenet_v2_coco:opid402' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid402,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(546), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 2, 2})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 2, 2), (64, 256, 1, 1)
// Out:    (1, 64, 2, 2)
// Operators: 'ssd_mobilenet_v2_coco:opid344' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid344,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 2, 2})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 26, 26), (128, 256, 1, 1)
// Out:    (1, 128, 26, 26)
// Operators: 'yolo-v3-tf:opid436' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid436,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 26, 26})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 26, 26), (512, 256, 3, 3)
// Out:    (1, 512, 26, 26)
// Operators: 'yolo-v3-tf:opid181' [FP16, FP32], 'yolo-v3-tf:opid194' [FP16, FP32], 'yolo-v3-tf:opid207' [FP16, FP32], 'yolo-v3-tf:opid220' [FP16, FP32], 'yolo-v3-tf:opid233' [FP16, FP32], 'yolo-v3-tf:opid246' [FP16, FP32], 'yolo-v3-tf:opid259' [FP16, FP32], 'yolo-v3-tf:opid272' [FP16, FP32], 'yolo-v3-tf:opid402' [FP16, FP32], 'yolo-v3-tf:opid414' [FP16, FP32], 'yolo-v3-tf:opid426' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid181,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 26, 26})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 3, 3), (128, 256, 1, 1)
// Out:    (1, 128, 3, 3)
// Operators: 'ssd_mobilenet_v2_coco:opid321' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid321,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 3, 3})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 3, 3), (24, 256, 3, 3)
// Out:    (1, 24, 3, 3)
// Operators: 'ssd_mobilenet_v2_coco:opid313' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid313,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(24), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 3, 3})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 3, 3), (546, 256, 3, 3)
// Out:    (1, 546, 3, 3)
// Operators: 'ssd_mobilenet_v2_coco:opid394' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid394,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(546), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 3, 3})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 38, 38), (128, 256, 1, 1)
// Out:    (1, 128, 38, 38)
// Operators: 'yolo-v4-tf:opid491' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid491,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 38, 38), (256, 256, 1, 1)
// Out:    (1, 256, 38, 38)
// Operators: 'yolo-v4-tf:opid222' [FP16, FP32], 'yolo-v4-tf:opid233' [FP16, FP32], 'yolo-v4-tf:opid244' [FP16, FP32], 'yolo-v4-tf:opid255' [FP16, FP32], 'yolo-v4-tf:opid266' [FP16, FP32], 'yolo-v4-tf:opid277' [FP16, FP32], 'yolo-v4-tf:opid288' [FP16, FP32], 'yolo-v4-tf:opid299' [FP16, FP32], 'yolo-v4-tf:opid310' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid222,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 38, 38), (256, 256, 3, 3)
// Out:    (1, 256, 38, 38)
// Operators: 'yolo-v4-tf:opid227' [FP16, FP32], 'yolo-v4-tf:opid238' [FP16, FP32], 'yolo-v4-tf:opid249' [FP16, FP32], 'yolo-v4-tf:opid260' [FP16, FP32], 'yolo-v4-tf:opid271' [FP16, FP32], 'yolo-v4-tf:opid282' [FP16, FP32], 'yolo-v4-tf:opid293' [FP16, FP32], 'yolo-v4-tf:opid304' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid227,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::Values(InferenceEngine::Precision::FP32), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 38, 38), (512, 256, 3, 3)
// Out:    (1, 512, 38, 38)
// Operators: 'yolo-v4-tf:opid467' [FP16, FP32], 'yolo-v4-tf:opid479' [FP16, FP32], 'yolo-v4-tf:opid553' [FP16, FP32], 'yolo-v4-tf:opid565' [FP16, FP32], 'yolo-v4-tf:opid577' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid467,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 5, 5), (512, 256, 1, 1)
// Out:    (1, 512, 5, 5)
// Operators: 'ssd_mobilenet_v2_coco:opid285' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid285,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 5, 5})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 52, 52), (128, 256, 1, 1)
// Out:    (1, 128, 52, 52)
// Operators: 'yolo-v3-tf:opid104' [FP16, FP32], 'yolo-v3-tf:opid117' [FP16, FP32], 'yolo-v3-tf:opid130' [FP16, FP32], 'yolo-v3-tf:opid143' [FP16, FP32], 'yolo-v3-tf:opid156' [FP16, FP32], 'yolo-v3-tf:opid467' [FP16, FP32], 'yolo-v3-tf:opid479' [FP16, FP32], 'yolo-v3-tf:opid65' [FP16, FP32], 'yolo-v3-tf:opid78' [FP16, FP32], 'yolo-v3-tf:opid91' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid104,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 52, 52})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 52, 52), (255, 256, 1, 1)
// Out:    (1, 255, 52, 52)
// Operators: 'yolo-v3-tf:opid491' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid491,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(255), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 52, 52})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 56, 56), (128, 256, 1, 1)
// Out:    (1, 128, 56, 56)
// Operators: 'resnet-50-tf:opid62' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid62,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 56, 56), (64, 256, 1, 1)
// Out:    (1, 64, 56, 56)
// Operators: 'resnet-50-tf:opid30' [FP16, FP32], 'resnet-50-tf:opid46' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid30,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 76, 76), (128, 256, 1, 1)
// Out:    (1, 128, 76, 76)
// Operators: 'yolo-v4-tf:opid195' [FP16, FP32], 'yolo-v4-tf:opid206' [FP16, FP32], 'yolo-v4-tf:opid510' [FP16, FP32], 'yolo-v4-tf:opid522' [FP16, FP32], 'yolo-v4-tf:opid534' [FP16, FP32], 'yolo-v4-tf:opid97' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid195,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 76, 76})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 76, 76), (255, 256, 1, 1)
// Out:    (1, 255, 76, 76)
// Operators: 'yolo-v4-tf:opid642' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid642,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(255), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 76, 76})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 76, 76), (256, 256, 1, 1)
// Out:    (1, 256, 76, 76)
// Operators: 'yolo-v4-tf:opid201' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid201,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 76, 76})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 8, 8), (256, 256, 3, 3)
// Out:    (1, 256, 8, 8)
// Operators: '2d_unet-graph-transform:opid51' [FP32], '2d_unet:opid51' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid51,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 8, 8})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 28, 1, 1), (672, 28, 1, 1)
// Out:    (1, 672, 1, 1)
// Operators: 'efficientdet-d1-tf:opid381' [FP16, FP32], 'efficientdet-d1-tf:opid409' [FP16, FP32], 'efficientdet-d1-tf:opid437' [FP16, FP32], 'efficientdet-d1-tf:opid469' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid381,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(672), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 28, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 3, 416, 416), (32, 3, 3, 3)
// Out:    (1, 32, 416, 416)
// Operators: 'yolo-v3-tf:opid2' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid2,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 3, 416, 416})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 3, 608, 608), (32, 3, 3, 3)
// Out:    (1, 32, 608, 608)
// Operators: 'yolo-v4-tf:opid2' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid2,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 3, 608, 608})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 1, 1), (8, 32, 1, 1)
// Out:    (1, 8, 1, 1)
// Operators: 'efficientdet-d1-tf:opid18' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid18,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(8), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 128, 128), (16, 32, 3, 3)
// Out:    (1, 16, 128, 128)
// Operators: '2d_unet-graph-transform:opid110' [FP32], '2d_unet:opid250' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid110,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 128, 128})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 147, 147), (64, 32, 3, 3)
// Out:    (1, 64, 147, 147)
// Operators: 'googlenet-v4-tf:opid16' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid16,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 147, 147})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 150, 150), (16, 32, 1, 1)
// Out:    (1, 16, 150, 150)
// Operators: 'ssd_mobilenet_v2_coco:opid16' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid16,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 150, 150})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 208, 208), (64, 32, 3, 3)
// Out:    (1, 64, 208, 208)
// Operators: 'yolo-v3-tf:opid20' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid20,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 208, 208})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 224, 224), (5, 32, 1, 1)
// Out:    (1, 5, 224, 224)
// Operators: 'super_resolution:opid8' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_super_resolution_opid8,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(5), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 224, 224})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 224, 224), (9, 32, 1, 1)
// Out:    (1, 9, 224, 224)
// Operators: 'super_resolution:opid26' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_super_resolution_opid26,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(9), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 224, 224})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 304, 304), (64, 32, 3, 3)
// Out:    (1, 64, 304, 304)
// Operators: 'yolo-v4-tf:opid22' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid22,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 304, 304})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 32, 32), (64, 32, 3, 3)
// Out:    (1, 64, 32, 32)
// Operators: '2d_unet-graph-transform:opid24' [FP32], '2d_unet:opid24' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid24,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 32, 32})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 320, 320), (16, 32, 1, 1)
// Out:    (1, 16, 320, 320)
// Operators: 'efficientdet-d1-tf:opid29' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid29,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 320, 320})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 38, 38), (192, 32, 1, 1)
// Out:    (1, 192, 38, 38)
// Operators: 'ssd_mobilenet_v2_coco:opid63' [FP16, FP32], 'ssd_mobilenet_v2_coco:opid78' [FP16, FP32], 'ssd_mobilenet_v2_coco:opid93' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid63,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 64, 64), (32, 32, 3, 3)
// Out:    (1, 32, 64, 64)
// Operators: '2d_unet-graph-transform:opid18' [FP32], '2d_unet-graph-transform:opid99' [FP32], '2d_unet:opid18' [FP16, FP32], '2d_unet:opid204' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid18,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 64, 64})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 320, 10, 10), (1280, 320, 1, 1)
// Out:    (1, 1280, 10, 10)
// Operators: 'ssd_mobilenet_v2_coco:opid262' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid262,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1280), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 320, 10, 10})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 320, 100, 171), (128, 320, 1, 1)
// Out:    (1, 128, 100, 171)
// Operators: 'mask_rcnn_inception_v2_coco:opid99' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid99,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 320, 100, 171})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 320, 100, 171), (64, 320, 1, 1)
// Out:    (1, 64, 100, 171)
// Operators: 'mask_rcnn_inception_v2_coco:opid109' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid109,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 320, 100, 171})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 320, 20, 20), (1920, 320, 1, 1)
// Out:    (1, 1920, 20, 20)
// Operators: 'efficientdet-d1-tf:opid618' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid618,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1920), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 320, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 320, 20, 20), (88, 320, 1, 1)
// Out:    (1, 88, 20, 20)
// Operators: 'efficientdet-d1-tf:opid646' [FP16, FP32], 'efficientdet-d1-tf:opid650' [FP16, FP32], 'efficientdet-d1-tf:opid728' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid646,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(88), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 320, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 384, 19, 19), (64, 384, 1, 1)
// Out:    (1, 64, 19, 19)
// Operators: 'ssd_mobilenet_v2_coco:opid117' [FP16, FP32], 'ssd_mobilenet_v2_coco:opid132' [FP16, FP32], 'ssd_mobilenet_v2_coco:opid147' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid117,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 384, 19, 19), (96, 384, 1, 1)
// Out:    (1, 96, 19, 19)
// Operators: 'ssd_mobilenet_v2_coco:opid162' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid162,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 384, 35, 35), (192, 384, 1, 1)
// Out:    (1, 192, 35, 35)
// Operators: 'googlenet-v4-tf:opid219' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid219,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 35, 35})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 384, 35, 35), (64, 384, 1, 1)
// Out:    (1, 64, 35, 35)
// Operators: 'googlenet-v4-tf:opid108' [FP16, FP32], 'googlenet-v4-tf:opid118' [FP16, FP32], 'googlenet-v4-tf:opid145' [FP16, FP32], 'googlenet-v4-tf:opid155' [FP16, FP32], 'googlenet-v4-tf:opid182' [FP16, FP32], 'googlenet-v4-tf:opid192' [FP16, FP32], 'googlenet-v4-tf:opid71' [FP16, FP32], 'googlenet-v4-tf:opid81' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid108,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 35, 35})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 384, 35, 35), (96, 384, 1, 1)
// Out:    (1, 96, 35, 35)
// Operators: 'googlenet-v4-tf:opid103' [FP16, FP32], 'googlenet-v4-tf:opid134' [FP16, FP32], 'googlenet-v4-tf:opid140' [FP16, FP32], 'googlenet-v4-tf:opid171' [FP16, FP32], 'googlenet-v4-tf:opid177' [FP16, FP32], 'googlenet-v4-tf:opid208' [FP16, FP32], 'googlenet-v4-tf:opid66' [FP16, FP32], 'googlenet-v4-tf:opid97' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid103,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 35, 35})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 384, 52, 52), (128, 384, 1, 1)
// Out:    (1, 128, 52, 52)
// Operators: 'yolo-v3-tf:opid455' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid455,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 52, 52})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 384, 8, 8), (256, 384, 1, 3)
// Out:    (1, 256, 8, 8)
// Operators: 'googlenet-v4-tf:opid642' [FP16, FP32], 'googlenet-v4-tf:opid696' [FP16, FP32], 'googlenet-v4-tf:opid750' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid642,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 8, 8})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 384, 8, 8), (256, 384, 3, 1)
// Out:    (1, 256, 8, 8)
// Operators: 'googlenet-v4-tf:opid647' [FP16, FP32], 'googlenet-v4-tf:opid701' [FP16, FP32], 'googlenet-v4-tf:opid755' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid647,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 8, 8})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 384, 8, 8), (448, 384, 3, 1)
// Out:    (1, 448, 8, 8)
// Operators: 'googlenet-v4-tf:opid658' [FP16, FP32], 'googlenet-v4-tf:opid712' [FP16, FP32], 'googlenet-v4-tf:opid766' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid658,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(448), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 8, 8})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 4, 1, 1), (16, 4, 1, 1)
// Out:    (1, 16, 1, 1)
// Operators: 'efficientdet-d1-tf:opid45' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid45,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 4, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 4, 1, 1), (96, 4, 1, 1)
// Out:    (1, 96, 1, 1)
// Operators: 'efficientdet-d1-tf:opid73' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid73,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 4, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 40, 80, 80), (240, 40, 1, 1)
// Out:    (1, 240, 80, 80)
// Operators: 'efficientdet-d1-tf:opid166' [FP16, FP32], 'efficientdet-d1-tf:opid194' [FP16, FP32], 'efficientdet-d1-tf:opid226' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid166,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(240), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 40, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 40, 80, 80), (88, 40, 1, 1)
// Out:    (1, 88, 80, 80)
// Operators: 'efficientdet-d1-tf:opid222' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid222,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(88), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 40, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 448, 8, 8), (512, 448, 1, 3)
// Out:    (1, 512, 8, 8)
// Operators: 'googlenet-v4-tf:opid663' [FP16, FP32], 'googlenet-v4-tf:opid717' [FP16, FP32], 'googlenet-v4-tf:opid771' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid663,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 448, 8, 8})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 48, 1, 1), (1152, 48, 1, 1)
// Out:    (1, 1152, 1, 1)
// Operators: 'efficientdet-d1-tf:opid496' [FP16, FP32], 'efficientdet-d1-tf:opid524' [FP16, FP32], 'efficientdet-d1-tf:opid552' [FP16, FP32], 'efficientdet-d1-tf:opid580' [FP16, FP32], 'efficientdet-d1-tf:opid608' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid496,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1152), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 48, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 480, 1, 1), (20, 480, 1, 1)
// Out:    (1, 20, 1, 1)
// Operators: 'efficientdet-d1-tf:opid265' [FP16, FP32], 'efficientdet-d1-tf:opid293' [FP16, FP32], 'efficientdet-d1-tf:opid321' [FP16, FP32], 'efficientdet-d1-tf:opid349' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid265,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(20), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 480, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 480, 40, 40), (112, 480, 1, 1)
// Out:    (1, 112, 40, 40)
// Operators: 'efficientdet-d1-tf:opid360' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid360,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(112), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 480, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 480, 40, 40), (80, 480, 1, 1)
// Out:    (1, 80, 40, 40)
// Operators: 'efficientdet-d1-tf:opid276' [FP16, FP32], 'efficientdet-d1-tf:opid304' [FP16, FP32], 'efficientdet-d1-tf:opid332' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid276,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(80), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 480, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 5, 224, 224), (32, 5, 1, 1)
// Out:    (1, 32, 224, 224)
// Operators: 'super_resolution:opid20' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_super_resolution_opid20,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 5, 224, 224})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 5, 224, 224), (5, 5, 3, 3)
// Out:    (1, 5, 224, 224)
// Operators: 'super_resolution:opid14' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_super_resolution_opid14,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(5), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 5, 224, 224})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 13, 13), (1024, 512, 3, 3)
// Out:    (1, 1024, 13, 13)
// Operators: 'yolo-v3-tf:opid291' [FP16, FP32], 'yolo-v3-tf:opid304' [FP16, FP32], 'yolo-v3-tf:opid317' [FP16, FP32], 'yolo-v3-tf:opid330' [FP16, FP32], 'yolo-v3-tf:opid343' [FP16, FP32], 'yolo-v3-tf:opid355' [FP16, FP32], 'yolo-v3-tf:opid367' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid291,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1024), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 13, 13})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 13, 13), (256, 512, 1, 1)
// Out:    (1, 256, 13, 13)
// Operators: 'yolo-v3-tf:opid377' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid377,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 13, 13})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 19, 19), (1024, 512, 3, 3)
// Out:    (1, 1024, 19, 19)
// Operators: 'yolo-v4-tf:opid408' [FP16, FP32], 'yolo-v4-tf:opid430' [FP16, FP32], 'yolo-v4-tf:opid601' [FP16, FP32], 'yolo-v4-tf:opid613' [FP16, FP32], 'yolo-v4-tf:opid625' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid408,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1024), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 19, 19), (256, 512, 1, 1)
// Out:    (1, 256, 19, 19)
// Operators: 'yolo-v4-tf:opid442' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid442,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 19, 19), (512, 512, 1, 1)
// Out:    (1, 512, 19, 19)
// Operators: 'yolo-v4-tf:opid342' [FP16, FP32], 'yolo-v4-tf:opid353' [FP16, FP32], 'yolo-v4-tf:opid364' [FP16, FP32], 'yolo-v4-tf:opid375' [FP16, FP32], 'yolo-v4-tf:opid386' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid342,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 19, 19), (512, 512, 3, 3)
// Out:    (1, 512, 19, 19)
// Operators: 'yolo-v4-tf:opid347' [FP16, FP32], 'yolo-v4-tf:opid358' [FP16, FP32], 'yolo-v4-tf:opid369' [FP16, FP32], 'yolo-v4-tf:opid380' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid347,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 26, 26), (255, 512, 1, 1)
// Out:    (1, 255, 26, 26)
// Operators: 'yolo-v3-tf:opid432' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid432,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(255), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 26, 26})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 26, 26), (256, 512, 1, 1)
// Out:    (1, 256, 26, 26)
// Operators: 'yolo-v3-tf:opid175' [FP16, FP32], 'yolo-v3-tf:opid188' [FP16, FP32], 'yolo-v3-tf:opid201' [FP16, FP32], 'yolo-v3-tf:opid214' [FP16, FP32], 'yolo-v3-tf:opid227' [FP16, FP32], 'yolo-v3-tf:opid240' [FP16, FP32], 'yolo-v3-tf:opid253' [FP16, FP32], 'yolo-v3-tf:opid266' [FP16, FP32], 'yolo-v3-tf:opid408' [FP16, FP32], 'yolo-v3-tf:opid420' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid175,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 26, 26})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 28, 28), (128, 512, 1, 1)
// Out:    (1, 128, 28, 28)
// Operators: 'resnet-50-tf:opid114' [FP16, FP32], 'resnet-50-tf:opid82' [FP16, FP32], 'resnet-50-tf:opid98' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid114,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 28, 28), (256, 512, 1, 1)
// Out:    (1, 256, 28, 28)
// Operators: 'resnet-50-tf:opid130' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid130,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 38, 38), (255, 512, 1, 1)
// Out:    (1, 255, 38, 38)
// Operators: 'yolo-v4-tf:opid583' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid583,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(255), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 38, 38), (256, 512, 1, 1)
// Out:    (1, 256, 38, 38)
// Operators: 'yolo-v4-tf:opid217' [FP16, FP32], 'yolo-v4-tf:opid315' [FP16, FP32], 'yolo-v4-tf:opid326' [FP16, FP32], 'yolo-v4-tf:opid461' [FP16, FP32], 'yolo-v4-tf:opid473' [FP16, FP32], 'yolo-v4-tf:opid485' [FP16, FP32], 'yolo-v4-tf:opid547' [FP16, FP32], 'yolo-v4-tf:opid559' [FP16, FP32], 'yolo-v4-tf:opid571' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid217,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 38, 38), (512, 512, 1, 1)
// Out:    (1, 512, 38, 38)
// Operators: 'yolo-v4-tf:opid321' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid321,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 5, 5), (128, 512, 1, 1)
// Out:    (1, 128, 5, 5)
// Operators: 'ssd_mobilenet_v2_coco:opid298' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid298,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 5, 5})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 5, 5), (24, 512, 3, 3)
// Out:    (1, 24, 5, 5)
// Operators: 'ssd_mobilenet_v2_coco:opid290' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid290,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(24), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 5, 5})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 5, 5), (546, 512, 3, 3)
// Out:    (1, 546, 5, 5)
// Operators: 'ssd_mobilenet_v2_coco:opid386' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid386,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(546), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 5, 5})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 50, 86), (24, 512, 1, 1)
// Out:    (1, 24, 50, 86)
// Operators: 'mask_rcnn_inception_v2_coco:opid279' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid279,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(24), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 50, 86), (48, 512, 1, 1)
// Out:    (1, 48, 50, 86)
// Operators: 'mask_rcnn_inception_v2_coco:opid296' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid296,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(48), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 7, 7), (2048, 512, 1, 1)
// Out:    (1, 2048, 7, 7)
// Operators: 'resnet-50-tf:opid240' [FP16, FP32], 'resnet-50-tf:opid260' [FP16, FP32], 'resnet-50-tf:opid276' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid240,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(2048), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 7, 7})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 7, 7), (512, 512, 3, 3)
// Out:    (1, 512, 7, 7)
// Operators: 'resnet-50-tf:opid255' [FP16, FP32], 'resnet-50-tf:opid271' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid255,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 7, 7})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 8, 8), (256, 512, 1, 3)
// Out:    (1, 256, 8, 8)
// Operators: 'googlenet-v4-tf:opid668' [FP16, FP32], 'googlenet-v4-tf:opid722' [FP16, FP32], 'googlenet-v4-tf:opid776' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid668,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 8, 8})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 8, 8), (256, 512, 3, 1)
// Out:    (1, 256, 8, 8)
// Operators: 'googlenet-v4-tf:opid673' [FP16, FP32], 'googlenet-v4-tf:opid727' [FP16, FP32], 'googlenet-v4-tf:opid781' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid673,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 8, 8})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 576, 10, 10), (160, 576, 1, 1)
// Out:    (1, 160, 10, 10)
// Operators: 'ssd_mobilenet_v2_coco:opid214' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid214,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(160), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 576, 10, 10})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 576, 19, 19), (12, 576, 3, 3)
// Out:    (1, 12, 19, 19)
// Operators: 'ssd_mobilenet_v2_coco:opid201' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid201,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(12), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 576, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 576, 19, 19), (273, 576, 3, 3)
// Out:    (1, 273, 19, 19)
// Operators: 'ssd_mobilenet_v2_coco:opid370' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid370,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(273), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 576, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 576, 19, 19), (96, 576, 1, 1)
// Out:    (1, 96, 19, 19)
// Operators: 'ssd_mobilenet_v2_coco:opid176' [FP16, FP32], 'ssd_mobilenet_v2_coco:opid191' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid176,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 576, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 576, 50, 86), (128, 576, 1, 1)
// Out:    (1, 128, 50, 86)
// Operators: 'mask_rcnn_inception_v2_coco:opid157' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid194' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid205' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid215' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid242' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid157,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 576, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 576, 50, 86), (160, 576, 1, 1)
// Out:    (1, 160, 50, 86)
// Operators: 'mask_rcnn_inception_v2_coco:opid200' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid252' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid200,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(160), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 576, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 576, 50, 86), (192, 576, 1, 1)
// Out:    (1, 192, 50, 86)
// Operators: 'mask_rcnn_inception_v2_coco:opid163' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid163,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 576, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 576, 50, 86), (224, 576, 1, 1)
// Out:    (1, 224, 50, 86)
// Operators: 'mask_rcnn_inception_v2_coco:opid126' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid126,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(224), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 576, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 576, 50, 86), (512, 576, 3, 3)
// Out:    (1, 512, 50, 86)
// Operators: 'mask_rcnn_inception_v2_coco:opid274' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid274,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 576, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 576, 50, 86), (64, 576, 1, 1)
// Out:    (1, 64, 50, 86)
// Operators: 'mask_rcnn_inception_v2_coco:opid131' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid131,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 576, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 576, 50, 86), (96, 576, 1, 1)
// Out:    (1, 96, 50, 86)
// Operators: 'mask_rcnn_inception_v2_coco:opid141' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid168' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid178' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid231' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid237' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid268' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid141,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 576, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 6, 1, 1), (144, 6, 1, 1)
// Out:    (1, 144, 1, 1)
// Operators: 'efficientdet-d1-tf:opid100' [FP16, FP32], 'efficientdet-d1-tf:opid128' [FP16, FP32], 'efficientdet-d1-tf:opid156' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid100,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(144), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 6, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 1, 1), (128, 64, 1, 1)
// Out:    (1, 128, 1, 1)
// Operators: 'ssd_mobilenet_v2_coco:opid354' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid354,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 100, 171), (64, 64, 3, 3)
// Out:    (1, 64, 100, 171)
// Operators: 'mask_rcnn_inception_v2_coco:opid35' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid35,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 100, 171})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 100, 171), (96, 64, 3, 3)
// Out:    (1, 96, 100, 171)
// Operators: 'mask_rcnn_inception_v2_coco:opid114' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid45' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid72' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid82' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid114,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 100, 171})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 104, 104), (128, 64, 3, 3)
// Out:    (1, 128, 104, 104)
// Operators: 'yolo-v3-tf:opid39' [FP16, FP32], 'yolo-v3-tf:opid52' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid39,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 104, 104})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 152, 152), (64, 64, 1, 1)
// Out:    (1, 64, 152, 152)
// Operators: 'yolo-v4-tf:opid54' [FP16, FP32], 'yolo-v4-tf:opid65' [FP16, FP32], 'yolo-v4-tf:opid76' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid54,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 152, 152})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 152, 152), (64, 64, 3, 3)
// Out:    (1, 64, 152, 152)
// Operators: 'yolo-v4-tf:opid59' [FP16, FP32], 'yolo-v4-tf:opid70' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid59,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 152, 152})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 16, 16), (128, 64, 3, 3)
// Out:    (1, 128, 16, 16)
// Operators: '2d_unet-graph-transform:opid35' [FP32], '2d_unet:opid35' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid35,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 16, 16})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 19, 19), (384, 64, 1, 1)
// Out:    (1, 384, 19, 19)
// Operators: 'ssd_mobilenet_v2_coco:opid107' [FP16, FP32], 'ssd_mobilenet_v2_coco:opid122' [FP16, FP32], 'ssd_mobilenet_v2_coco:opid137' [FP16, FP32], 'ssd_mobilenet_v2_coco:opid152' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid107,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(384), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 200, 342), (192, 64, 3, 3)
// Out:    (1, 192, 200, 342)
// Operators: 'mask_rcnn_inception_v2_coco:opid19' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid19,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 200, 342})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 200, 342), (64, 64, 1, 1)
// Out:    (1, 64, 200, 342)
// Operators: 'mask_rcnn_inception_v2_coco:opid14' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid14,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 200, 342})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 208, 208), (32, 64, 1, 1)
// Out:    (1, 32, 208, 208)
// Operators: 'yolo-v3-tf:opid14' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid14,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 208, 208})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 304, 304), (32, 64, 1, 1)
// Out:    (1, 32, 304, 304)
// Operators: 'yolo-v4-tf:opid17' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid17,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 304, 304})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 304, 304), (64, 64, 1, 1)
// Out:    (1, 64, 304, 304)
// Operators: 'yolo-v4-tf:opid12' [FP16, FP32], 'yolo-v4-tf:opid28' [FP16, FP32], 'yolo-v4-tf:opid33' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid12,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 304, 304})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 32, 32), (64, 64, 3, 3)
// Out:    (1, 64, 32, 32)
// Operators: '2d_unet-graph-transform:opid29' [FP32], '2d_unet-graph-transform:opid83' [FP32], '2d_unet:opid153' [FP16, FP32], '2d_unet:opid29' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid29,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 32, 32})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 35, 35), (96, 64, 3, 3)
// Out:    (1, 96, 35, 35)
// Operators: 'googlenet-v4-tf:opid113' [FP16, FP32], 'googlenet-v4-tf:opid123' [FP16, FP32], 'googlenet-v4-tf:opid150' [FP16, FP32], 'googlenet-v4-tf:opid160' [FP16, FP32], 'googlenet-v4-tf:opid187' [FP16, FP32], 'googlenet-v4-tf:opid197' [FP16, FP32], 'googlenet-v4-tf:opid76' [FP16, FP32], 'googlenet-v4-tf:opid86' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid113,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 35, 35})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 50, 86), (96, 64, 3, 3)
// Out:    (1, 96, 50, 86)
// Operators: 'mask_rcnn_inception_v2_coco:opid136' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid136,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 56, 56), (256, 64, 1, 1)
// Out:    (1, 256, 56, 56)
// Operators: 'resnet-50-tf:opid20' [FP16, FP32], 'resnet-50-tf:opid24' [FP16, FP32], 'resnet-50-tf:opid40' [FP16, FP32], 'resnet-50-tf:opid56' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid20,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 56, 56), (64, 64, 1, 1)
// Out:    (1, 64, 56, 56)
// Operators: 'resnet-50-tf:opid10' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid10,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 56, 56), (64, 64, 3, 3)
// Out:    (1, 64, 56, 56)
// Operators: 'resnet-50-tf:opid15' [FP16, FP32], 'resnet-50-tf:opid35' [FP16, FP32], 'resnet-50-tf:opid51' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid15,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 64, 64), (32, 64, 3, 3)
// Out:    (1, 32, 64, 64)
// Operators: '2d_unet-graph-transform:opid94' [FP32], '2d_unet:opid199' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid94,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 64, 64})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 73, 73), (64, 64, 1, 7)
// Out:    (1, 64, 73, 73)
// Operators: 'googlenet-v4-tf:opid43' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid43,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 7})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 73, 73})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 73, 73), (64, 64, 7, 1)
// Out:    (1, 64, 73, 73)
// Operators: 'googlenet-v4-tf:opid48' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid48,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({7, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 73, 73})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 672, 1, 1), (28, 672, 1, 1)
// Out:    (1, 28, 1, 1)
// Operators: 'efficientdet-d1-tf:opid376' [FP16, FP32], 'efficientdet-d1-tf:opid404' [FP16, FP32], 'efficientdet-d1-tf:opid432' [FP16, FP32], 'efficientdet-d1-tf:opid464' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid376,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(28), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 672, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 672, 20, 20), (192, 672, 1, 1)
// Out:    (1, 192, 20, 20)
// Operators: 'efficientdet-d1-tf:opid475' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid475,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 672, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 672, 40, 40), (112, 672, 1, 1)
// Out:    (1, 112, 40, 40)
// Operators: 'efficientdet-d1-tf:opid387' [FP16, FP32], 'efficientdet-d1-tf:opid415' [FP16, FP32], 'efficientdet-d1-tf:opid443' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid387,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(112), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 672, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 768, 26, 26), (256, 768, 1, 1)
// Out:    (1, 256, 26, 26)
// Operators: 'yolo-v3-tf:opid396' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid396,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 768, 26, 26})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 8, 1, 1), (32, 8, 1, 1)
// Out:    (1, 32, 1, 1)
// Operators: 'efficientdet-d1-tf:opid23' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid23,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 8, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 80, 1, 1), (1920, 80, 1, 1)
// Out:    (1, 1920, 1, 1)
// Operators: 'efficientdet-d1-tf:opid635' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid635,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1920), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 80, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 80, 40, 40), (480, 80, 1, 1)
// Out:    (1, 480, 40, 40)
// Operators: 'efficientdet-d1-tf:opid253' [FP16, FP32], 'efficientdet-d1-tf:opid281' [FP16, FP32], 'efficientdet-d1-tf:opid309' [FP16, FP32], 'efficientdet-d1-tf:opid337' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid253,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(480), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 80, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 96, 1, 1), (4, 96, 1, 1)
// Out:    (1, 4, 1, 1)
// Operators: 'efficientdet-d1-tf:opid68' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid68,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(4), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 96, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 96, 100, 171), (96, 96, 3, 3)
// Out:    (1, 96, 100, 171)
// Operators: 'mask_rcnn_inception_v2_coco:opid50' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid87' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid50,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 96, 100, 171})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 96, 160, 160), (24, 96, 1, 1)
// Out:    (1, 24, 160, 160)
// Operators: 'efficientdet-d1-tf:opid79' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid79,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(24), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 96, 160, 160})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 96, 19, 19), (576, 96, 1, 1)
// Out:    (1, 576, 19, 19)
// Operators: 'ssd_mobilenet_v2_coco:opid166' [FP16, FP32], 'ssd_mobilenet_v2_coco:opid181' [FP16, FP32], 'ssd_mobilenet_v2_coco:opid196' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid166,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(576), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 96, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 96, 35, 35), (96, 96, 3, 3)
// Out:    (1, 96, 35, 35)
// Operators: 'googlenet-v4-tf:opid128' [FP16, FP32], 'googlenet-v4-tf:opid165' [FP16, FP32], 'googlenet-v4-tf:opid202' [FP16, FP32], 'googlenet-v4-tf:opid91' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid128,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 96, 35, 35})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 96, 50, 86), (128, 96, 3, 3)
// Out:    (1, 128, 50, 86)
// Operators: 'mask_rcnn_inception_v2_coco:opid146' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid173' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid183' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid146,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 96, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 96, 75, 75), (24, 96, 1, 1)
// Out:    (1, 24, 75, 75)
// Operators: 'ssd_mobilenet_v2_coco:opid30' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid30,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(24), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 96, 75, 75})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 960, 10, 10), (160, 960, 1, 1)
// Out:    (1, 160, 10, 10)
// Operators: 'ssd_mobilenet_v2_coco:opid228' [FP16, FP32], 'ssd_mobilenet_v2_coco:opid243' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid228,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(160), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 960, 10, 10})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 960, 10, 10), (320, 960, 1, 1)
// Out:    (1, 320, 10, 10)
// Operators: 'ssd_mobilenet_v2_coco:opid258' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid258,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(320), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 960, 10, 10})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 1024, 15, 15), (256, 1024, 3, 3)
// Out:    (100, 256, 15, 15)
// Operators: 'mask_rcnn_inception_v2_coco:opid573' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid573,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 1024, 15, 15})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 1024, 4, 4), (128, 1024, 1, 1)
// Out:    (100, 128, 4, 4)
// Operators: 'mask_rcnn_inception_v2_coco:opid370' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid407' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid528' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid565' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid370,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 1024, 4, 4})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 1024, 4, 4), (160, 1024, 1, 1)
// Out:    (100, 160, 4, 4)
// Operators: 'mask_rcnn_inception_v2_coco:opid354' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid512' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid354,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(160), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 1024, 4, 4})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 1024, 4, 4), (192, 1024, 1, 1)
// Out:    (100, 192, 4, 4)
// Operators: 'mask_rcnn_inception_v2_coco:opid344' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid381' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid391' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid502' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid539' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid549' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid344,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 1024, 4, 4})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 1024, 4, 4), (352, 1024, 1, 1)
// Out:    (100, 352, 4, 4)
// Operators: 'mask_rcnn_inception_v2_coco:opid339' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid376' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid497' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid534' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid339,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(352), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 1024, 4, 4})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 160, 4, 4), (224, 160, 3, 3)
// Out:    (100, 224, 4, 4)
// Operators: 'mask_rcnn_inception_v2_coco:opid359' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid517' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid359,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(224), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 160, 4, 4})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 192, 4, 4), (224, 192, 3, 3)
// Out:    (100, 224, 4, 4)
// Operators: 'mask_rcnn_inception_v2_coco:opid396' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid554' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid396,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(224), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 192, 4, 4})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 192, 4, 4), (320, 192, 3, 3)
// Out:    (100, 320, 4, 4)
// Operators: 'mask_rcnn_inception_v2_coco:opid349' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid386' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid507' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid544' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid349,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(320), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 192, 4, 4})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 192, 7, 7), (256, 192, 3, 3)
// Out:    (100, 256, 7, 7)
// Operators: 'mask_rcnn_inception_v2_coco:opid327' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid485' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid327,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 192, 7, 7})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 224, 4, 4), (224, 224, 3, 3)
// Out:    (100, 224, 4, 4)
// Operators: 'mask_rcnn_inception_v2_coco:opid364' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid401' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid522' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid559' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid364,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(224), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 224, 4, 4})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 256, 15, 15), (90, 256, 3, 3)
// Out:    (100, 90, 15, 15)
// Operators: 'mask_rcnn_inception_v2_coco:opid578' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid578,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(90), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 256, 15, 15})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 576, 7, 7), (128, 576, 1, 1)
// Out:    (100, 128, 7, 7)
// Operators: 'mask_rcnn_inception_v2_coco:opid312' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid470' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid312,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 576, 7, 7})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 576, 7, 7), (192, 576, 1, 1)
// Out:    (100, 192, 7, 7)
// Operators: 'mask_rcnn_inception_v2_coco:opid322' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid480' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid322,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 576, 7, 7})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 128, 100, 171), (160, 128, 3, 3)
// Out:    (1, 160, 50, 86)
// Operators: 'mask_rcnn_inception_v2_coco:opid104' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid104,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(160), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 100, 171})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 3, 300, 300), (32, 3, 3, 3)
// Out:    (1, 32, 150, 150)
// Operators: 'ssd_mobilenet_v2_coco:opid6' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid6,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 3, 300, 300})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 3, 640, 640), (32, 3, 3, 3)
// Out:    (1, 32, 320, 320)
// Operators: 'efficientdet-d1-tf:opid6' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid6,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 3, 640, 640})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 96, 100, 171), (96, 96, 3, 3)
// Out:    (1, 96, 50, 86)
// Operators: 'mask_rcnn_inception_v2_coco:opid119' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid119,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 96, 100, 171})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (100, 128, 7, 7), (192, 128, 3, 3)
// Out:    (100, 192, 4, 4)
// Operators: 'mask_rcnn_inception_v2_coco:opid317' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid475' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid317,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 128, 7, 7})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (100, 256, 7, 7), (256, 256, 3, 3)
// Out:    (100, 256, 4, 4)
// Operators: 'mask_rcnn_inception_v2_coco:opid332' [FP16, FP32], 'mask_rcnn_inception_v2_coco:opid490' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid332,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 256, 7, 7})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 1, 144, 144, 144), (16, 1, 3, 3, 3)
// Out:    (1, 16, 144, 144, 144)
// Operators: '3d_unet-graph-transform:opid2' [FP32], '3d_unet:opid2' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid2,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1, 144, 144, 144})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 128, 18, 18, 18), (128, 128, 3, 3, 3)
// Out:    (1, 128, 18, 18, 18)
// Operators: '3d_unet-graph-transform:opid40' [FP32], '3d_unet-graph-transform:opid67' [FP32], '3d_unet:opid110' [FP16, FP32], '3d_unet:opid40' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid40,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 18, 18, 18})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 128, 36, 36, 36), (64, 128, 3, 3, 3)
// Out:    (1, 64, 36, 36, 36)
// Operators: '3d_unet-graph-transform:opid78' [FP32], '3d_unet:opid164' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid78,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 36, 36, 36})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 128, 9, 9, 9), (256, 128, 3, 3, 3)
// Out:    (1, 256, 9, 9, 9)
// Operators: '3d_unet-graph-transform:opid46' [FP32], '3d_unet:opid46' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid46,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 9, 9, 9})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 16, 144, 144, 144), (16, 16, 3, 3, 3)
// Out:    (1, 16, 144, 144, 144)
// Operators: '3d_unet-graph-transform:opid115' [FP32], '3d_unet-graph-transform:opid7' [FP32], '3d_unet:opid287' [FP16, FP32], '3d_unet:opid7' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid115,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 144, 144, 144})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 16, 72, 72, 72), (32, 16, 3, 3, 3)
// Out:    (1, 32, 72, 72, 72)
// Operators: '3d_unet-graph-transform:opid13' [FP32], '3d_unet:opid13' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid13,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 72, 72, 72})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 256, 18, 18, 18), (128, 256, 3, 3, 3)
// Out:    (1, 128, 18, 18, 18)
// Operators: '3d_unet-graph-transform:opid62' [FP32], '3d_unet:opid105' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid62,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 18, 18, 18})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 256, 9, 9, 9), (256, 256, 3, 3, 3)
// Out:    (1, 256, 9, 9, 9)
// Operators: '3d_unet-graph-transform:opid51' [FP32], '3d_unet:opid51' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid51,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 9, 9, 9})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 32, 144, 144, 144), (16, 32, 3, 3, 3)
// Out:    (1, 16, 144, 144, 144)
// Operators: '3d_unet-graph-transform:opid110' [FP32], '3d_unet:opid282' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid110,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 144, 144, 144})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 32, 36, 36, 36), (64, 32, 3, 3, 3)
// Out:    (1, 64, 36, 36, 36)
// Operators: '3d_unet-graph-transform:opid24' [FP32], '3d_unet:opid24' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid24,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 36, 36, 36})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 32, 72, 72, 72), (32, 32, 3, 3, 3)
// Out:    (1, 32, 72, 72, 72)
// Operators: '3d_unet-graph-transform:opid18' [FP32], '3d_unet-graph-transform:opid99' [FP32], '3d_unet:opid18' [FP16, FP32], '3d_unet:opid228' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid18,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 72, 72, 72})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 64, 18, 18, 18), (128, 64, 3, 3, 3)
// Out:    (1, 128, 18, 18, 18)
// Operators: '3d_unet-graph-transform:opid35' [FP32], '3d_unet:opid35' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid35,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 18, 18, 18})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 64, 36, 36, 36), (64, 64, 3, 3, 3)
// Out:    (1, 64, 36, 36, 36)
// Operators: '3d_unet-graph-transform:opid29' [FP32], '3d_unet-graph-transform:opid83' [FP32], '3d_unet:opid169' [FP16, FP32], '3d_unet:opid29' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid29,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 36, 36, 36})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 64, 72, 72, 72), (32, 64, 3, 3, 3)
// Out:    (1, 32, 72, 72, 72)
// Operators: '3d_unet-graph-transform:opid94' [FP32], '3d_unet:opid223' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid94,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 72, 72, 72})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 16, 128, 128), (1, 16, 1, 1)
// Out:    (1, 1, 128, 128)
// Operators: '2d_unet-graph-transform:opid120' [FP32], '2d_unet:opid260' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid120,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 128, 128})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 24, 400, 683), (64, 24, 1, 1)
// Out:    (1, 64, 400, 683)
// Operators: 'mask_rcnn_inception_v2_coco:opid8' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid8,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 24, 400, 683})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 149, 149), (32, 32, 3, 3)
// Out:    (1, 32, 147, 147)
// Operators: 'googlenet-v4-tf:opid11' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid11,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 149, 149})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 73, 73), (96, 64, 3, 3)
// Out:    (1, 96, 71, 71)
// Operators: 'googlenet-v4-tf:opid33' [FP16, FP32], 'googlenet-v4-tf:opid53' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid33,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 73, 73})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 10, 10), (36, 88, 1, 1)
// Out:    (1, 36, 10, 10)
// Operators: 'efficientdet-d1-tf:opid1225' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1225,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(36), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 10, 10})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 10, 10), (810, 88, 1, 1)
// Out:    (1, 810, 10, 10)
// Operators: 'efficientdet-d1-tf:opid1396' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1396,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(810), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 10, 10})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 10, 10), (88, 88, 1, 1)
// Out:    (1, 88, 10, 10)
// Operators: 'efficientdet-d1-tf:opid1021' [FP16, FP32], 'efficientdet-d1-tf:opid1198' [FP16, FP32], 'efficientdet-d1-tf:opid1204' [FP16, FP32], 'efficientdet-d1-tf:opid1211' [FP16, FP32], 'efficientdet-d1-tf:opid1218' [FP16, FP32], 'efficientdet-d1-tf:opid1375' [FP16, FP32], 'efficientdet-d1-tf:opid1382' [FP16, FP32], 'efficientdet-d1-tf:opid1389' [FP16, FP32], 'efficientdet-d1-tf:opid666' [FP16, FP32], 'efficientdet-d1-tf:opid760' [FP16, FP32], 'efficientdet-d1-tf:opid787' [FP16, FP32], 'efficientdet-d1-tf:opid877' [FP16, FP32], 'efficientdet-d1-tf:opid904' [FP16, FP32], 'efficientdet-d1-tf:opid994' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1021,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(88), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 10, 10})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 20, 20), (36, 88, 1, 1)
// Out:    (1, 36, 20, 20)
// Operators: 'efficientdet-d1-tf:opid1178' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1178,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(36), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 20, 20), (810, 88, 1, 1)
// Out:    (1, 810, 20, 20)
// Operators: 'efficientdet-d1-tf:opid1365' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1365,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(810), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 20, 20), (88, 88, 1, 1)
// Out:    (1, 88, 20, 20)
// Operators: 'efficientdet-d1-tf:opid1033' [FP16, FP32], 'efficientdet-d1-tf:opid1151' [FP16, FP32], 'efficientdet-d1-tf:opid1157' [FP16, FP32], 'efficientdet-d1-tf:opid1164' [FP16, FP32], 'efficientdet-d1-tf:opid1171' [FP16, FP32], 'efficientdet-d1-tf:opid1344' [FP16, FP32], 'efficientdet-d1-tf:opid1351' [FP16, FP32], 'efficientdet-d1-tf:opid1358' [FP16, FP32], 'efficientdet-d1-tf:opid678' [FP16, FP32], 'efficientdet-d1-tf:opid742' [FP16, FP32], 'efficientdet-d1-tf:opid799' [FP16, FP32], 'efficientdet-d1-tf:opid859' [FP16, FP32], 'efficientdet-d1-tf:opid916' [FP16, FP32], 'efficientdet-d1-tf:opid976' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1033,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(88), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 40, 40), (36, 88, 1, 1)
// Out:    (1, 36, 40, 40)
// Operators: 'efficientdet-d1-tf:opid1131' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1131,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(36), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 40, 40), (810, 88, 1, 1)
// Out:    (1, 810, 40, 40)
// Operators: 'efficientdet-d1-tf:opid1334' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1334,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(810), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 40, 40), (88, 88, 1, 1)
// Out:    (1, 88, 40, 40)
// Operators: 'efficientdet-d1-tf:opid1045' [FP16, FP32], 'efficientdet-d1-tf:opid1104' [FP16, FP32], 'efficientdet-d1-tf:opid1110' [FP16, FP32], 'efficientdet-d1-tf:opid1117' [FP16, FP32], 'efficientdet-d1-tf:opid1124' [FP16, FP32], 'efficientdet-d1-tf:opid1313' [FP16, FP32], 'efficientdet-d1-tf:opid1320' [FP16, FP32], 'efficientdet-d1-tf:opid1327' [FP16, FP32], 'efficientdet-d1-tf:opid690' [FP16, FP32], 'efficientdet-d1-tf:opid722' [FP16, FP32], 'efficientdet-d1-tf:opid811' [FP16, FP32], 'efficientdet-d1-tf:opid841' [FP16, FP32], 'efficientdet-d1-tf:opid928' [FP16, FP32], 'efficientdet-d1-tf:opid958' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1045,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(88), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 5, 5), (36, 88, 1, 1)
// Out:    (1, 36, 5, 5)
// Operators: 'efficientdet-d1-tf:opid1269' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1269,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(36), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 5, 5})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 5, 5), (810, 88, 1, 1)
// Out:    (1, 810, 5, 5)
// Operators: 'efficientdet-d1-tf:opid1427' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1427,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(810), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 5, 5})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 5, 5), (88, 88, 1, 1)
// Out:    (1, 88, 5, 5)
// Operators: 'efficientdet-d1-tf:opid1009' [FP16, FP32], 'efficientdet-d1-tf:opid1242' [FP16, FP32], 'efficientdet-d1-tf:opid1248' [FP16, FP32], 'efficientdet-d1-tf:opid1255' [FP16, FP32], 'efficientdet-d1-tf:opid1262' [FP16, FP32], 'efficientdet-d1-tf:opid1406' [FP16, FP32], 'efficientdet-d1-tf:opid1413' [FP16, FP32], 'efficientdet-d1-tf:opid1420' [FP16, FP32], 'efficientdet-d1-tf:opid775' [FP16, FP32], 'efficientdet-d1-tf:opid892' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1009,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(88), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 5, 5})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 80, 80), (36, 88, 1, 1)
// Out:    (1, 36, 80, 80)
// Operators: 'efficientdet-d1-tf:opid1084' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1084,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(36), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 80, 80), (810, 88, 1, 1)
// Out:    (1, 810, 80, 80)
// Operators: 'efficientdet-d1-tf:opid1303' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1303,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(810), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 80, 80), (88, 88, 1, 1)
// Out:    (1, 88, 80, 80)
// Operators: 'efficientdet-d1-tf:opid1057' [FP16, FP32], 'efficientdet-d1-tf:opid1063' [FP16, FP32], 'efficientdet-d1-tf:opid1070' [FP16, FP32], 'efficientdet-d1-tf:opid1077' [FP16, FP32], 'efficientdet-d1-tf:opid1282' [FP16, FP32], 'efficientdet-d1-tf:opid1289' [FP16, FP32], 'efficientdet-d1-tf:opid1296' [FP16, FP32], 'efficientdet-d1-tf:opid702' [FP16, FP32], 'efficientdet-d1-tf:opid823' [FP16, FP32], 'efficientdet-d1-tf:opid940' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1057,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(88), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 1024, 14, 14), (2048, 1024, 1, 1)
// Out:    (1, 2048, 7, 7)
// Operators: 'resnet-50-tf:opid244' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid244,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(2048), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 192, 17, 17), (192, 192, 3, 3)
// Out:    (1, 192, 8, 8)
// Operators: 'googlenet-v4-tf:opid605' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid605,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 192, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 192, 71, 71), (192, 192, 3, 3)
// Out:    (1, 192, 35, 35)
// Operators: 'googlenet-v4-tf:opid59' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid59,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 192, 71, 71})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 224, 35, 35), (256, 224, 3, 3)
// Out:    (1, 256, 17, 17)
// Operators: 'googlenet-v4-tf:opid229' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid229,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 224, 35, 35})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 256, 56, 56), (512, 256, 1, 1)
// Out:    (1, 512, 28, 28)
// Operators: 'resnet-50-tf:opid76' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid76,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 3, 299, 299), (32, 3, 3, 3)
// Out:    (1, 32, 149, 149)
// Operators: 'googlenet-v4-tf:opid6' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid6,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 3, 299, 299})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 320, 17, 17), (320, 320, 3, 3)
// Out:    (1, 320, 8, 8)
// Operators: 'googlenet-v4-tf:opid625' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid625,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(320), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 320, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 384, 35, 35), (384, 384, 3, 3)
// Out:    (1, 384, 17, 17)
// Operators: 'googlenet-v4-tf:opid214' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid214,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(384), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 35, 35})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 512, 28, 28), (1024, 512, 1, 1)
// Out:    (1, 1024, 14, 14)
// Operators: 'resnet-50-tf:opid144' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid144,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1024), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 64, 147, 147), (96, 64, 3, 3)
// Out:    (1, 96, 73, 73)
// Operators: 'googlenet-v4-tf:opid22' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid22,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 147, 147})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 16, 144, 144, 144), (1, 16, 1, 1, 1)
// Out:    (1, 1, 144, 144, 144)
// Operators: '3d_unet-graph-transform:opid120' [FP32], '3d_unet:opid292' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid120,
    ConvolutionLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(1), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 144, 144, 144})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
    ConvolutionLayerThresholdTest::getTestCaseName);

// {AUTOGENERATED_TESTS_END_TAG}
// clang-format on
// =============================================================================

}  // namespace
