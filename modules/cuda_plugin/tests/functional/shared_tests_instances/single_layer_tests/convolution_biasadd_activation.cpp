// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_biasadd_activation.hpp"

#include <gtest/gtest-param-test.h>
#include <ie_common.h>

#include <cstddef>
#include <cstdint>
#include <cuda_test_constants.hpp>
#include <functional_test_utils/skip_tests_config.hpp>
#include <ie_precision.hpp>
#include <limits>
#include <ngraph/op/util/attr_types.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <tuple>
#include <vector>

#include "finite_comparer.hpp"

namespace LayerTestsDefinitions {

class ConvolutionBiasAddActivationLayerFiniteComparerTest
    : public FiniteComparer<ConvolutionBiasAddActivationLayerTest> {
protected:
    void SetUp() override {
        ConvolutionBiasAddActivationLayerTest::SetUp();

        auto params = this->GetParam();
        auto netPrecision = std::get<1>(std::get<0>(params));
        if (netPrecision.getPrecVal() == InferenceEngine::Precision::FP16) {
            this->threshold = 600;
            this->infinity_value = std::numeric_limits<std::uint16_t>::max();
        }
    }
};

class ConvolutionBiasAddAddActivationLayerFiniteComparerTest
    : public FiniteComparer<ConvolutionBiasAddAddActivationLayerTest> {
protected:
    void SetUp() override {
        ConvolutionBiasAddAddActivationLayerTest::SetUp();

        auto params = this->GetParam();
        auto netPrecision = std::get<1>(std::get<0>(params));
        if (netPrecision.getPrecVal() == InferenceEngine::Precision::FP16) {
            this->threshold = 500;
            this->infinity_value = std::numeric_limits<std::uint16_t>::max();
        }
    }
};

TEST_P(ConvolutionBiasAddActivationLayerFiniteComparerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto params = GetParam();
    inPrc = std::get<2>(std::get<0>(params));
    outPrc = std::get<3>(std::get<0>(params));
    Run();
}

TEST_P(ConvolutionBiasAddAddActivationLayerFiniteComparerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto params = GetParam();
    inPrc = std::get<2>(std::get<0>(params));
    outPrc = std::get<3>(std::get<0>(params));
    Run();
}

namespace {

// NOTE: Default precision where difference between reference and actual
//       are bigger than defined threshold
const auto defaultPrecision = InferenceEngine::Precision::FP32;

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32,
};

const std::vector<ngraph::helpers::ActivationTypes> netActivations = {
    ngraph::helpers::ActivationTypes::None,
    ngraph::helpers::ActivationTypes::Relu,
};

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
                       ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv2DParams_ExplicitPaddingSymmetric2 =
    ::testing::Combine(::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 3})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0, 3})),  // pads_end
                       ::testing::ValuesIn(dilations),
                       ::testing::ValuesIn(numOutChannels),
                       ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv2DParams_ExplicitPaddingAsymmetric1 =
    ::testing::Combine(::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0, 3})),  // pads_end
                       ::testing::ValuesIn(dilations),
                       ::testing::ValuesIn(numOutChannels),
                       ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv2DParams_ExplicitPaddingAsymmetric2 =
    ::testing::Combine(::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 3})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pads_end
                       ::testing::ValuesIn(dilations),
                       ::testing::ValuesIn(numOutChannels),
                       ::testing::Values(ngraph::op::PadType::EXPLICIT));

const auto conv2DParams_AutoPadValid = ::testing::Combine(::testing::ValuesIn(kernels),
                                                          ::testing::ValuesIn(strides),
                                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                                                          ::testing::ValuesIn(dilations),
                                                          ::testing::ValuesIn(numOutChannels),
                                                          ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(
    smoke_Convolution2DBiasAddActivation_ExplicitPaddingSymmetric1,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(::testing::Combine(conv2DParams_ExplicitPaddingSymmetric1,
                                          ::testing::ValuesIn(netPrecisions),
                                          ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          ::testing::Values(InferenceEngine::Layout::ANY),
                                          ::testing::Values(InferenceEngine::Layout::ANY),
                                          ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                                          ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                       ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    smoke_Convolution2DBiasAddActivation_ExplicitPaddingSymmetric2_FP32,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(::testing::Combine(conv2DParams_ExplicitPaddingSymmetric2,
                                          ::testing::ValuesIn(netPrecisions),
                                          ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          ::testing::Values(InferenceEngine::Layout::ANY),
                                          ::testing::Values(InferenceEngine::Layout::ANY),
                                          ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                                          ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                       ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    DISABLED_smoke_Convolution2DBiasAddActivation_ExplicitPaddingSymmetric2_FP16,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(::testing::Combine(conv2DParams_ExplicitPaddingSymmetric2,
                                          ::testing::Values(InferenceEngine::Precision::FP16),
                                          ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          ::testing::Values(InferenceEngine::Layout::ANY),
                                          ::testing::Values(InferenceEngine::Layout::ANY),
                                          ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                                          ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                       ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    DISABLED_smoke_Convolution2DBiasAddActivation_ExplicitPaddingAsymmetric1,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(::testing::Combine(conv2DParams_ExplicitPaddingAsymmetric1,
                                          ::testing::ValuesIn(netPrecisions),
                                          ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          ::testing::Values(InferenceEngine::Layout::ANY),
                                          ::testing::Values(InferenceEngine::Layout::ANY),
                                          ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                                          ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                       ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    DISABLED_smoke_Convolution2DBiasAddActivation_ExplicitPaddingAsymmetric2,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(::testing::Combine(conv2DParams_ExplicitPaddingAsymmetric2,
                                          ::testing::ValuesIn(netPrecisions),
                                          ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          ::testing::Values(InferenceEngine::Layout::ANY),
                                          ::testing::Values(InferenceEngine::Layout::ANY),
                                          ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                                          ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                       ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    smoke_Convolution2DBiasAddActivation_AutoPadValid,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(::testing::Combine(conv2DParams_AutoPadValid,
                                          ::testing::ValuesIn(netPrecisions),
                                          ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          ::testing::Values(InferenceEngine::Layout::ANY),
                                          ::testing::Values(InferenceEngine::Layout::ANY),
                                          ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                                          ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                       ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

/* ============= resnet50/vgg16 Convolutions ============= */

// attrs: {'auto_pad': 'explicit', 'strides': '2,2', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 256, 28, 28), (256, 256, 3, 3); out: (1, 256, 14, 14)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group1_1,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({3, 3})),     // kernel
                                              ::testing::Values(std::vector<size_t>({2, 2})),     // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // dilations
                                              ::testing::Values(256),                             // Num out channels
                                              ::testing::Values(ngraph::op::PadType::EXPLICIT)),  // Padding type
                           ::testing::Values(defaultPrecision),                                   // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Output layout
                           ::testing::Values(std::vector<size_t>({1, 256, 28, 28})),              // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '2,2', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 128, 56, 56), (128, 128, 3, 3); out: (1, 128, 28, 28)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group1_2,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({3, 3})),     // kernel
                                              ::testing::Values(std::vector<size_t>({2, 2})),     // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // dilations
                                              ::testing::Values(128),                             // Num out channels
                                              ::testing::Values(ngraph::op::PadType::EXPLICIT)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                    // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Output layout
                           ::testing::Values(std::vector<size_t>({1, 128, 56, 56})),              // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '2,2', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 512, 14, 14), (512, 512, 3, 3); out: (1, 512, 7, 7)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group1_3,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({3, 3})),     // kernel
                                              ::testing::Values(std::vector<size_t>({2, 2})),     // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // dilations
                                              ::testing::Values(512),                             // Num out channels
                                              ::testing::Values(ngraph::op::PadType::EXPLICIT)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                    // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Output layout
                           ::testing::Values(std::vector<size_t>({1, 512, 14, 14})),              // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '2,2', 'dilations': '1,1', 'pads_begin': '3,3', 'pads_end': '3,3'},
// in: (1, 3, 224, 224), (64, 3, 7, 7); out: (1, 64, 112, 112)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group2_1,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({7, 7})),     // kernel
                                              ::testing::Values(std::vector<size_t>({2, 2})),     // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({3, 3})),  // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({3, 3})),  // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // dilations
                                              ::testing::Values(64),                              // Num out channels
                                              ::testing::Values(ngraph::op::PadType::EXPLICIT)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                    // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Output layout
                           ::testing::Values(std::vector<size_t>({1, 3, 224, 224})),              // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'valid', 'strides': '2,2', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 256, 56, 56), (512, 256, 1, 1); out: (1, 512, 28, 28)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group3_1,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({1, 1})),     // kernel
                                              ::testing::Values(std::vector<size_t>({2, 2})),     // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // dilations
                                              ::testing::Values(512),                             // Num out channels
                                              ::testing::Values(ngraph::op::PadType::VALID)),     // Padding type
                           ::testing::ValuesIn(netPrecisions),                                    // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Output layout
                           ::testing::Values(std::vector<size_t>({1, 256, 56, 56})),              // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'valid', 'strides': '2,2', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 1024, 14, 14), (2048, 1024, 1, 1); out: (1, 2048, 7, 7)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group3_2,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({1, 1})),     // kernel
                                              ::testing::Values(std::vector<size_t>({2, 2})),     // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // dilations
                                              ::testing::Values(2048),                            // Num out channels
                                              ::testing::Values(ngraph::op::PadType::VALID)),     // Padding type
                           ::testing::ValuesIn(netPrecisions),                                    // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Output layout
                           ::testing::Values(std::vector<size_t>({1, 1024, 14, 14})),             // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'valid', 'strides': '2,2', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 512, 28, 28), (1024, 512, 1, 1); out: (1, 1024, 14, 14)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group3_3,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({1, 1})),     // kernel
                                              ::testing::Values(std::vector<size_t>({2, 2})),     // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // dilations
                                              ::testing::Values(1024),                            // Num out channels
                                              ::testing::Values(ngraph::op::PadType::VALID)),     // Padding type
                           ::testing::ValuesIn(netPrecisions),                                    // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Output layout
                           ::testing::Values(std::vector<size_t>({1, 512, 28, 28})),              // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 256, 14, 14), (1024, 256, 1, 1); out: (1, 1024, 14, 14)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_1,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({1, 1})),       // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // dilations
                                              ::testing::Values(1024),                              // Num out channels
                                              ::testing::Values(ngraph::op::PadType::SAME_UPPER)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                      // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Output layout
                           ::testing::Values(std::vector<size_t>({1, 256, 14, 14})),                // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 64, 56, 56), (64, 64, 1, 1); out: (1, 64, 56, 56)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_2,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({1, 1})),       // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // dilations
                                              ::testing::Values(64),                                // Num out channels
                                              ::testing::Values(ngraph::op::PadType::SAME_UPPER)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                      // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Output layout
                           ::testing::Values(std::vector<size_t>({1, 64, 56, 56})),                 // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 128, 28, 28), (512, 128, 1, 1); out: (1, 512, 28, 28)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_3,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({1, 1})),       // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // dilations
                                              ::testing::Values(512),                               // Num out channels
                                              ::testing::Values(ngraph::op::PadType::SAME_UPPER)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                      // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Output layout
                           ::testing::Values(std::vector<size_t>({1, 128, 28, 28})),                // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 256, 14, 14), (256, 256, 3, 3); out: (1, 256, 14, 14)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_4,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({3, 3})),       // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // dilations
                                              ::testing::Values(256),                               // Num out channels
                                              ::testing::Values(ngraph::op::PadType::SAME_UPPER)),  // Padding type
                           ::testing::Values(defaultPrecision),                                     // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Output layout
                           ::testing::Values(std::vector<size_t>({1, 256, 14, 14})),                // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 64, 56, 56), (256, 64, 1, 1); out: (1, 256, 56, 56)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_5,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({1, 1})),       // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // dilations
                                              ::testing::Values(256),                               // Num out channels
                                              ::testing::Values(ngraph::op::PadType::SAME_UPPER)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                      // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Output layout
                           ::testing::Values(std::vector<size_t>({1, 64, 56, 56})),                 // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 64, 56, 56), (64, 64, 3, 3); out: (1, 64, 56, 56)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_6,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({3, 3})),       // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // dilations
                                              ::testing::Values(64),                                // Num out channels
                                              ::testing::Values(ngraph::op::PadType::SAME_UPPER)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                      // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Output layout
                           ::testing::Values(std::vector<size_t>({1, 64, 56, 56})),                 // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 256, 56, 56), (64, 256, 1, 1); out: (1, 64, 56, 56)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_7,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({1, 1})),       // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // dilations
                                              ::testing::Values(64),                                // Num out channels
                                              ::testing::Values(ngraph::op::PadType::SAME_UPPER)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                      // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Output layout
                           ::testing::Values(std::vector<size_t>({1, 256, 56, 56})),                // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 512, 28, 28), (128, 512, 1, 1); out: (1, 128, 28, 28)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_8,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({1, 1})),       // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // dilations
                                              ::testing::Values(128),                               // Num out channels
                                              ::testing::Values(ngraph::op::PadType::SAME_UPPER)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                      // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Output layout
                           ::testing::Values(std::vector<size_t>({1, 512, 28, 28})),                // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 2048, 7, 7), (512, 2048, 1, 1); out: (1, 512, 7, 7)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_9,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({1, 1})),       // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // dilations
                                              ::testing::Values(512),                               // Num out channels
                                              ::testing::Values(ngraph::op::PadType::SAME_UPPER)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                      // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Output layout
                           ::testing::Values(std::vector<size_t>({1, 2048, 7, 7})),                 // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 1024, 14, 14), (512, 1024, 1, 1); out: (1, 512, 14, 14)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_10,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({1, 1})),       // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // dilations
                                              ::testing::Values(512),                               // Num out channels
                                              ::testing::Values(ngraph::op::PadType::SAME_UPPER)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                      // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Output layout
                           ::testing::Values(std::vector<size_t>({1, 1024, 14, 14})),               // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 512, 7, 7), (512, 512, 3, 3); out: (1, 512, 7, 7)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_11,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({3, 3})),       // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // dilations
                                              ::testing::Values(512),                               // Num out channels
                                              ::testing::Values(ngraph::op::PadType::SAME_UPPER)),  // Padding type
                           ::testing::Values(defaultPrecision),                                     // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Output layout
                           ::testing::Values(std::vector<size_t>({1, 512, 7, 7})),                  // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 256, 56, 56), (128, 256, 1, 1); out: (1, 128, 56, 56)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_12,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({1, 1})),       // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // dilations
                                              ::testing::Values(128),                               // Num out channels
                                              ::testing::Values(ngraph::op::PadType::SAME_UPPER)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                      // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Output layout
                           ::testing::Values(std::vector<size_t>({1, 256, 56, 56})),                // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 512, 28, 28), (256, 512, 1, 1); out: (1, 256, 28, 28)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_13,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({1, 1})),       // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // dilations
                                              ::testing::Values(256),                               // Num out channels
                                              ::testing::Values(ngraph::op::PadType::SAME_UPPER)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                      // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Output layout
                           ::testing::Values(std::vector<size_t>({1, 512, 28, 28})),                // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 512, 7, 7), (2048, 512, 1, 1); out: (1, 2048, 7, 7)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_14,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({1, 1})),       // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // dilations
                                              ::testing::Values(2048),                              // Num out channels
                                              ::testing::Values(ngraph::op::PadType::SAME_UPPER)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                      // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Output layout
                           ::testing::Values(std::vector<size_t>({1, 512, 7, 7})),                  // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 128, 28, 28), (128, 128, 3, 3); out: (1, 128, 28, 28)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_15,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({3, 3})),       // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // dilations
                                              ::testing::Values(128),                               // Num out channels
                                              ::testing::Values(ngraph::op::PadType::SAME_UPPER)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                      // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Output layout
                           ::testing::Values(std::vector<size_t>({1, 128, 28, 28})),                // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'same_upper', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0'},
// in: (1, 1024, 14, 14), (256, 1024, 1, 1); out: (1, 256, 14, 14)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group4_16,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({1, 1})),       // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0})),    // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),       // dilations
                                              ::testing::Values(256),                               // Num out channels
                                              ::testing::Values(ngraph::op::PadType::SAME_UPPER)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                      // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),              // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                         // Output layout
                           ::testing::Values(std::vector<size_t>({1, 1024, 14, 14})),               // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 64, 224, 224), (64, 64, 3, 3); out: (1, 64, 224, 224)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group5_1,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({3, 3})),     // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // dilations
                                              ::testing::Values(64),                              // Num out channels
                                              ::testing::Values(ngraph::op::PadType::EXPLICIT)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                    // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Output layout
                           ::testing::Values(std::vector<size_t>({1, 64, 224, 224})),             // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 3, 224, 224), (64, 3, 3, 3); out: (1, 64, 224, 224)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group5_2,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({3, 3})),     // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // dilations
                                              ::testing::Values(64),                              // Num out channels
                                              ::testing::Values(ngraph::op::PadType::EXPLICIT)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                    // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Output layout
                           ::testing::Values(std::vector<size_t>({1, 3, 224, 224})),              // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 128, 56, 56), (256, 128, 3, 3); out: (1, 256, 56, 56)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group5_3,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({3, 3})),     // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // dilations
                                              ::testing::Values(256),                             // Num out channels
                                              ::testing::Values(ngraph::op::PadType::EXPLICIT)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                    // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Output layout
                           ::testing::Values(std::vector<size_t>({1, 128, 56, 56})),              // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 512, 28, 28), (512, 512, 3, 3); out: (1, 512, 28, 28)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group5_4,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({3, 3})),     // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // dilations
                                              ::testing::Values(512),                             // Num out channels
                                              ::testing::Values(ngraph::op::PadType::EXPLICIT)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                    // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Output layout
                           ::testing::Values(std::vector<size_t>({1, 512, 28, 28})),              // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 512, 14, 14), (512, 512, 3, 3); out: (1, 512, 14, 14)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group5_5,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({3, 3})),     // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // dilations
                                              ::testing::Values(512),                             // Num out channels
                                              ::testing::Values(ngraph::op::PadType::EXPLICIT)),  // Padding type
                           ::testing::Values(defaultPrecision),                                   // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Output layout
                           ::testing::Values(std::vector<size_t>({1, 512, 14, 14})),              // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 256, 28, 28), (512, 256, 3, 3); out: (1, 512, 28, 28)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group5_6,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({3, 3})),     // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // dilations
                                              ::testing::Values(512),                             // Num out channels
                                              ::testing::Values(ngraph::op::PadType::EXPLICIT)),  // Padding type
                           ::testing::Values(defaultPrecision),                                   // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Output layout
                           ::testing::Values(std::vector<size_t>({1, 256, 28, 28})),              // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 256, 56, 56), (256, 256, 3, 3); out: (1, 256, 56, 56)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group5_7,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({3, 3})),     // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // dilations
                                              ::testing::Values(256),                             // Num out channels
                                              ::testing::Values(ngraph::op::PadType::EXPLICIT)),  // Padding type
                           ::testing::Values(defaultPrecision),                                   // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Output layout
                           ::testing::Values(std::vector<size_t>({1, 256, 56, 56})),              // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 64, 112, 112), (128, 64, 3, 3); out: (1, 128, 112, 112)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group5_8,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({3, 3})),     // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // dilations
                                              ::testing::Values(128),                             // Num out channels
                                              ::testing::Values(ngraph::op::PadType::EXPLICIT)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                    // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Output layout
                           ::testing::Values(std::vector<size_t>({1, 64, 112, 112})),             // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 128, 112, 112), (128, 128, 3, 3); out: (1, 128, 112, 112)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group5_9,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({3, 3})),     // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // dilations
                                              ::testing::Values(128),                             // Num out channels
                                              ::testing::Values(ngraph::op::PadType::EXPLICIT)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                    // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Output layout
                           ::testing::Values(std::vector<size_t>({1, 128, 112, 112})),            // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// attrs: {'auto_pad': 'explicit', 'strides': '1,1', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1'},
// in: (1, 128, 112, 112), (128, 128, 3, 3); out: (1, 128, 112, 112)
INSTANTIATE_TEST_CASE_P(
    resnet50_vgg16_group5_9,
    ConvolutionBiasAddAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({3, 3})),     // kernel
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // stride
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_begin
                                              ::testing::Values(std::vector<ptrdiff_t>({1, 1})),  // pads_end
                                              ::testing::Values(std::vector<size_t>({1, 1})),     // dilations
                                              ::testing::Values(128),                             // Num out channels
                                              ::testing::Values(ngraph::op::PadType::EXPLICIT)),  // Padding type
                           ::testing::ValuesIn(netPrecisions),                                    // Net precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Input precision
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            // Output precision
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Input layout
                           ::testing::Values(InferenceEngine::Layout::ANY),                       // Output layout
                           ::testing::Values(std::vector<size_t>({1, 128, 112, 112})),            // Input shapes
                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(netActivations)),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

//
// Test generation for CUDA Fused Convolutions requires transformed models
// These tests cover only 2d_unet and 3d_unet
//
// WARNING: Currently the fusing of 3D Convolution is disabled in
// openvino_cuda_plugin/modules/cuda_plugin/src/transformer/fuse_conv_biasadd_activation.cpp
// so the the following tests for 3d_unet run on graphs without FusedConvolution nodes
//
// =============================================================================
// clang-format off
// {AUTOGENERATED_TESTS_BEGIN_TAG}

// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 1, 128, 128), (16, 1, 3, 3), (1, 16, 1, 1)
// Out:    (1, 16, 128, 128)
// Operators: '2d_unet-graph-transform-cuda:opid3' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_2d_unet_graph_transform_cuda_opid3,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1})), // dilations
                ::testing::Values(16), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 1, 128, 128})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 128, 16, 16), (128, 128, 3, 3), (1, 128, 1, 1)
// Out:    (1, 128, 16, 16)
// Operators: '2d_unet-graph-transform-cuda:opid27' [FP32], '2d_unet-graph-transform-cuda:opid45' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_2d_unet_graph_transform_cuda_opid27,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1})), // dilations
                ::testing::Values(128), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 128, 16, 16})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 128, 32, 32), (64, 128, 3, 3), (1, 64, 1, 1)
// Out:    (1, 64, 32, 32)
// Operators: '2d_unet-graph-transform-cuda:opid53' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_2d_unet_graph_transform_cuda_opid53,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1})), // dilations
                ::testing::Values(64), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 128, 32, 32})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 128, 8, 8), (256, 128, 3, 3), (1, 256, 1, 1)
// Out:    (1, 256, 8, 8)
// Operators: '2d_unet-graph-transform-cuda:opid31' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_2d_unet_graph_transform_cuda_opid31,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1})), // dilations
                ::testing::Values(256), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 128, 8, 8})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 16, 128, 128), (16, 16, 3, 3), (1, 16, 1, 1)
// Out:    (1, 16, 128, 128)
// Operators: '2d_unet-graph-transform-cuda:opid6' [FP32], '2d_unet-graph-transform-cuda:opid78' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_2d_unet_graph_transform_cuda_opid6,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1})), // dilations
                ::testing::Values(16), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 16, 128, 128})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 16, 64, 64), (32, 16, 3, 3), (1, 32, 1, 1)
// Out:    (1, 32, 64, 64)
// Operators: '2d_unet-graph-transform-cuda:opid10' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_2d_unet_graph_transform_cuda_opid10,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1})), // dilations
                ::testing::Values(32), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 16, 64, 64})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 256, 16, 16), (128, 256, 3, 3), (1, 128, 1, 1)
// Out:    (1, 128, 16, 16)
// Operators: '2d_unet-graph-transform-cuda:opid42' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_2d_unet_graph_transform_cuda_opid42,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1})), // dilations
                ::testing::Values(128), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 256, 16, 16})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 256, 8, 8), (256, 256, 3, 3), (1, 256, 1, 1)
// Out:    (1, 256, 8, 8)
// Operators: '2d_unet-graph-transform-cuda:opid34' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_2d_unet_graph_transform_cuda_opid34,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1})), // dilations
                ::testing::Values(256), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 256, 8, 8})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 32, 128, 128), (16, 32, 3, 3), (1, 16, 1, 1)
// Out:    (1, 16, 128, 128)
// Operators: '2d_unet-graph-transform-cuda:opid75' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_2d_unet_graph_transform_cuda_opid75,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1})), // dilations
                ::testing::Values(16), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 32, 128, 128})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 32, 32, 32), (64, 32, 3, 3), (1, 64, 1, 1)
// Out:    (1, 64, 32, 32)
// Operators: '2d_unet-graph-transform-cuda:opid17' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_2d_unet_graph_transform_cuda_opid17,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1})), // dilations
                ::testing::Values(64), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 32, 32, 32})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 32, 64, 64), (32, 32, 3, 3), (1, 32, 1, 1)
// Out:    (1, 32, 64, 64)
// Operators: '2d_unet-graph-transform-cuda:opid13' [FP32], '2d_unet-graph-transform-cuda:opid67' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_2d_unet_graph_transform_cuda_opid13,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1})), // dilations
                ::testing::Values(32), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 32, 64, 64})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 64, 16, 16), (128, 64, 3, 3), (1, 128, 1, 1)
// Out:    (1, 128, 16, 16)
// Operators: '2d_unet-graph-transform-cuda:opid24' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_2d_unet_graph_transform_cuda_opid24,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1})), // dilations
                ::testing::Values(128), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 64, 16, 16})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 64, 32, 32), (64, 64, 3, 3), (1, 64, 1, 1)
// Out:    (1, 64, 32, 32)
// Operators: '2d_unet-graph-transform-cuda:opid20' [FP32], '2d_unet-graph-transform-cuda:opid56' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_2d_unet_graph_transform_cuda_opid20,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1})), // dilations
                ::testing::Values(64), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 64, 32, 32})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 64, 64, 64), (32, 64, 3, 3), (1, 32, 1, 1)
// Out:    (1, 32, 64, 64)
// Operators: '2d_unet-graph-transform-cuda:opid64' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_2d_unet_graph_transform_cuda_opid64,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1})), // dilations
                ::testing::Values(32), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 64, 64, 64})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '1,1,1', 'pads_end': '1,1,1', 'strides': '1,1,1'}
// In:     (1, 1, 144, 144, 144), (16, 1, 3, 3, 3), (1, 16, 1, 1, 1)
// Out:    (1, 16, 144, 144, 144)
// Operators: '3d_unet-graph-transform-cuda:opid3' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_3d_unet_graph_transform_cuda_opid3,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
                ::testing::Values(16), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 1, 144, 144, 144})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '1,1,1', 'pads_end': '1,1,1', 'strides': '1,1,1'}
// In:     (1, 128, 18, 18, 18), (128, 128, 3, 3, 3), (1, 128, 1, 1, 1)
// Out:    (1, 128, 18, 18, 18)
// Operators: '3d_unet-graph-transform-cuda:opid27' [FP32], '3d_unet-graph-transform-cuda:opid45' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_3d_unet_graph_transform_cuda_opid27,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
                ::testing::Values(128), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 128, 18, 18, 18})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '1,1,1', 'pads_end': '1,1,1', 'strides': '1,1,1'}
// In:     (1, 128, 36, 36, 36), (64, 128, 3, 3, 3), (1, 64, 1, 1, 1)
// Out:    (1, 64, 36, 36, 36)
// Operators: '3d_unet-graph-transform-cuda:opid53' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_3d_unet_graph_transform_cuda_opid53,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
                ::testing::Values(64), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 128, 36, 36, 36})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '1,1,1', 'pads_end': '1,1,1', 'strides': '1,1,1'}
// In:     (1, 128, 9, 9, 9), (256, 128, 3, 3, 3), (1, 256, 1, 1, 1)
// Out:    (1, 256, 9, 9, 9)
// Operators: '3d_unet-graph-transform-cuda:opid31' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_3d_unet_graph_transform_cuda_opid31,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
                ::testing::Values(256), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 128, 9, 9, 9})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '1,1,1', 'pads_end': '1,1,1', 'strides': '1,1,1'}
// In:     (1, 16, 144, 144, 144), (16, 16, 3, 3, 3), (1, 16, 1, 1, 1)
// Out:    (1, 16, 144, 144, 144)
// Operators: '3d_unet-graph-transform-cuda:opid6' [FP32], '3d_unet-graph-transform-cuda:opid78' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_3d_unet_graph_transform_cuda_opid6,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
                ::testing::Values(16), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 16, 144, 144, 144})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '1,1,1', 'pads_end': '1,1,1', 'strides': '1,1,1'}
// In:     (1, 16, 72, 72, 72), (32, 16, 3, 3, 3), (1, 32, 1, 1, 1)
// Out:    (1, 32, 72, 72, 72)
// Operators: '3d_unet-graph-transform-cuda:opid10' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_3d_unet_graph_transform_cuda_opid10,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
                ::testing::Values(32), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 16, 72, 72, 72})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '1,1,1', 'pads_end': '1,1,1', 'strides': '1,1,1'}
// In:     (1, 256, 18, 18, 18), (128, 256, 3, 3, 3), (1, 128, 1, 1, 1)
// Out:    (1, 128, 18, 18, 18)
// Operators: '3d_unet-graph-transform-cuda:opid42' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_3d_unet_graph_transform_cuda_opid42,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
                ::testing::Values(128), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 256, 18, 18, 18})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '1,1,1', 'pads_end': '1,1,1', 'strides': '1,1,1'}
// In:     (1, 256, 9, 9, 9), (256, 256, 3, 3, 3), (1, 256, 1, 1, 1)
// Out:    (1, 256, 9, 9, 9)
// Operators: '3d_unet-graph-transform-cuda:opid34' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_3d_unet_graph_transform_cuda_opid34,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
                ::testing::Values(256), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 256, 9, 9, 9})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '1,1,1', 'pads_end': '1,1,1', 'strides': '1,1,1'}
// In:     (1, 32, 144, 144, 144), (16, 32, 3, 3, 3), (1, 16, 1, 1, 1)
// Out:    (1, 16, 144, 144, 144)
// Operators: '3d_unet-graph-transform-cuda:opid75' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_3d_unet_graph_transform_cuda_opid75,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
                ::testing::Values(16), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 32, 144, 144, 144})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '1,1,1', 'pads_end': '1,1,1', 'strides': '1,1,1'}
// In:     (1, 32, 36, 36, 36), (64, 32, 3, 3, 3), (1, 64, 1, 1, 1)
// Out:    (1, 64, 36, 36, 36)
// Operators: '3d_unet-graph-transform-cuda:opid17' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_3d_unet_graph_transform_cuda_opid17,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
                ::testing::Values(64), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 32, 36, 36, 36})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '1,1,1', 'pads_end': '1,1,1', 'strides': '1,1,1'}
// In:     (1, 32, 72, 72, 72), (32, 32, 3, 3, 3), (1, 32, 1, 1, 1)
// Out:    (1, 32, 72, 72, 72)
// Operators: '3d_unet-graph-transform-cuda:opid13' [FP32], '3d_unet-graph-transform-cuda:opid67' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_3d_unet_graph_transform_cuda_opid13,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
                ::testing::Values(32), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 32, 72, 72, 72})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '1,1,1', 'pads_end': '1,1,1', 'strides': '1,1,1'}
// In:     (1, 64, 18, 18, 18), (128, 64, 3, 3, 3), (1, 128, 1, 1, 1)
// Out:    (1, 128, 18, 18, 18)
// Operators: '3d_unet-graph-transform-cuda:opid24' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_3d_unet_graph_transform_cuda_opid24,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
                ::testing::Values(128), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 64, 18, 18, 18})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '1,1,1', 'pads_end': '1,1,1', 'strides': '1,1,1'}
// In:     (1, 64, 36, 36, 36), (64, 64, 3, 3, 3), (1, 64, 1, 1, 1)
// Out:    (1, 64, 36, 36, 36)
// Operators: '3d_unet-graph-transform-cuda:opid20' [FP32], '3d_unet-graph-transform-cuda:opid56' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_3d_unet_graph_transform_cuda_opid20,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
                ::testing::Values(64), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 64, 36, 36, 36})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '1,1,1', 'pads_end': '1,1,1', 'strides': '1,1,1'}
// In:     (1, 64, 72, 72, 72), (32, 64, 3, 3, 3), (1, 32, 1, 1, 1)
// Out:    (1, 32, 72, 72, 72)
// Operators: '3d_unet-graph-transform-cuda:opid64' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_3d_unet_graph_transform_cuda_opid64,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({1, 1, 1})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
                ::testing::Values(32), // Num out channels
                ::testing::Values(ngraph::op::PadType::SAME_UPPER)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 64, 72, 72, 72})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 16, 128, 128), (1, 16, 1, 1), (1, 1, 1, 1)
// Out:    (1, 1, 128, 128)
// Operators: '2d_unet-graph-transform-cuda:opid81' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_2d_unet_graph_transform_cuda_opid81,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({1, 1})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1})), // dilations
                ::testing::Values(1), // Num out channels
                ::testing::Values(ngraph::op::PadType::VALID)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 16, 128, 128})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 16, 144, 144, 144), (1, 16, 1, 1, 1), (1, 1, 1, 1, 1)
// Out:    (1, 1, 144, 144, 144)
// Operators: '3d_unet-graph-transform-cuda:opid81' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_FusedConvolution_3d_unet_graph_transform_cuda_opid81,
    ConvolutionBiasAddActivationLayerFiniteComparerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Combine(
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // kernel
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
                ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
                ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
                ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
                ::testing::Values(1), // Num out channels
                ::testing::Values(ngraph::op::PadType::VALID)), // Padding type
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16}), // Net precisions
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
            ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
            ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
            ::testing::Values(std::vector<size_t>({1, 16, 144, 144, 144})), // Input shape
            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ::testing::ValuesIn(std::vector<ngraph::helpers::ActivationTypes>{ngraph::helpers::ActivationTypes::None,ngraph::helpers::ActivationTypes::Relu})),
    ConvolutionBiasAddActivationLayerFiniteComparerTest::getTestCaseName);

// {AUTOGENERATED_TESTS_END_TAG}
// clang-format on
// =============================================================================

}  // namespace
}  // namespace LayerTestsDefinitions
