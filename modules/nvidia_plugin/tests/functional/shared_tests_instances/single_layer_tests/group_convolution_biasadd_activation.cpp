// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest-param-test.h>
#include <ie_common.h>

#include <cstddef>
#include <cuda_test_constants.hpp>
#include <functional_test_utils/skip_tests_config.hpp>
#include <ie_precision.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <openvino/op/util/attr_types.hpp>
#include <tuple>
#include <vector>

#include "convolution_biasadd_activation.hpp"

namespace LayerTestsDefinitions {

TEST_P(GroupConvolutionBiasAddActivationLayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto params = GetParam();
    inPrc = std::get<2>(std::get<0>(params));
    outPrc = std::get<3>(std::get<0>(params));
    Run();
}

TEST_P(GroupConvolutionBiasAddAddActivationLayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto params = GetParam();
    inPrc = std::get<2>(std::get<0>(params));
    outPrc = std::get<3>(std::get<0>(params));
    Run();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<ngraph::helpers::ActivationTypes> netActivations = {
    ngraph::helpers::ActivationTypes::None,
    // TODO: enable when ReLU fusing transformation is enabled for GroupConvolution
    // ngraph::helpers::ActivationTypes::Relu
};

/* ============= 1D Convolution ============= */
const std::vector<std::vector<size_t>> kernels1D = {{3}, {5}};
const std::vector<std::vector<size_t>> strides1D = {{1}, {3}};
const std::vector<std::vector<size_t>> dilations1D = {{1}, {3}};
const std::vector<size_t> numOutChannels1D = {4, 8};
const std::vector<size_t> numGroups1D = {2, 4};

const auto smoke_1D_AutoPadValid_Params =
    ::testing::Combine(::testing::Combine(::testing::Combine(::testing::ValuesIn(kernels1D),
                                                             ::testing::ValuesIn(strides1D),
                                                             ::testing::Values(std::vector<ptrdiff_t>({0})),
                                                             ::testing::Values(std::vector<ptrdiff_t>({0})),
                                                             ::testing::ValuesIn(dilations1D),
                                                             ::testing::ValuesIn(numOutChannels1D),
                                                             ::testing::ValuesIn(numGroups1D),
                                                             ::testing::Values(ov::op::PadType::VALID)),
                                          ::testing::ValuesIn(netPrecisions),
                                          ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          ::testing::Values(InferenceEngine::Layout::ANY),
                                          ::testing::Values(InferenceEngine::Layout::ANY),
                                          ::testing::Values(std::vector<size_t>({5, 8, 30})),
                                          ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
                       ::testing::ValuesIn(netActivations));

const auto smoke_1D_ExplicitPaddingAsymmetric_Params = ::testing::Combine(
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(kernels1D),
                                          ::testing::ValuesIn(strides1D),
                                          ::testing::Values(std::vector<ptrdiff_t>({0})),  // pads_begin
                                          ::testing::Values(std::vector<ptrdiff_t>({3})),  // pads_end
                                          ::testing::ValuesIn(dilations1D),
                                          ::testing::ValuesIn(numOutChannels1D),
                                          ::testing::ValuesIn(numGroups1D),
                                          ::testing::Values(ov::op::PadType::EXPLICIT)),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(std::vector<size_t>({2, 16, 15})),
                       ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
    ::testing::ValuesIn(netActivations));

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBiasAdd_1D_AutoPadValid_Params,
                        GroupConvolutionBiasAddActivationLayerTest,
                        smoke_1D_AutoPadValid_Params,
                        GroupConvolutionBiasAddActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBiasAddAdd_1D_AutoPadValid_Params,
                        GroupConvolutionBiasAddAddActivationLayerTest,
                        smoke_1D_AutoPadValid_Params,
                        GroupConvolutionBiasAddAddActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBiasAdd_1D_ExplicitPaddingAsymmetric_Params,
                        GroupConvolutionBiasAddActivationLayerTest,
                        smoke_1D_ExplicitPaddingAsymmetric_Params,
                        GroupConvolutionBiasAddActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBiasAddAdd_1D_ExplicitPaddingAsymmetric_Params,
                        GroupConvolutionBiasAddAddActivationLayerTest,
                        smoke_1D_ExplicitPaddingAsymmetric_Params,
                        GroupConvolutionBiasAddAddActivationLayerTest::getTestCaseName);

/* ============= 2D Convolution ============= */
const std::vector<std::vector<size_t>> kernels2D = {{3, 3}, {3, 5}};
const std::vector<std::vector<size_t>> strides2D = {{1, 1}, {1, 3}};
const std::vector<std::vector<size_t>> dilations2D = {{1, 1}, {3, 1}};
const std::vector<size_t> numOutChannels2D = {8, 32};
const std::vector<size_t> numGroups2D = {2, 8};

const auto smoke_2D_ExplicitPaddingSymmetric_Params = ::testing::Combine(
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(kernels2D),
                                          ::testing::ValuesIn(strides2D),
                                          ::testing::Values(std::vector<ptrdiff_t>({0, 3})),  // pads_begin
                                          ::testing::Values(std::vector<ptrdiff_t>({0, 3})),  // pads_end
                                          ::testing::ValuesIn(dilations2D),
                                          ::testing::ValuesIn(numOutChannels2D),
                                          ::testing::ValuesIn(numGroups2D),
                                          ::testing::Values(ov::op::PadType::EXPLICIT)),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(std::vector<size_t>({2, 16, 12, 6})),
                       ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
    ::testing::ValuesIn(netActivations));

const auto smoke_2D_ExplicitPaddingSymmetric_Params2 = ::testing::Combine(
    ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({3, 3})),    // kernels
                                          ::testing::Values(std::vector<size_t>({2, 2})),    // strides
                                          ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
                                          ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
                                          ::testing::Values(std::vector<size_t>({1, 1})),    // dilations
                                          ::testing::Values(96),                             // out channels
                                          ::testing::Values(2),                              // groups
                                          ::testing::Values(ov::op::PadType::EXPLICIT)),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(std::vector<size_t>({1, 96, 112, 112})),
                       ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
    ::testing::ValuesIn(netActivations));

const auto smoke_2D_ExplicitPaddingSymmetric_Params3 = ::testing::Combine(
    ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({1, 1})),    // kernels
                                          ::testing::Values(std::vector<size_t>({1, 1})),    // strides
                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
                                          ::testing::Values(std::vector<size_t>({1, 1})),    // dilations
                                          ::testing::Values(160),                            // out channels
                                          ::testing::Values(2),                              // groups
                                          ::testing::Values(ov::op::PadType::EXPLICIT)),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(std::vector<size_t>({1, 480, 14, 14})),
                       ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
    ::testing::ValuesIn(netActivations));

const auto smoke_2D_ExplicitPaddingSymmetric_Params4 = ::testing::Combine(
    ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>({1, 1})),    // kernels
                                          ::testing::Values(std::vector<size_t>({1, 1})),    // strides
                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
                                          ::testing::Values(std::vector<size_t>({1, 1})),    // dilations
                                          ::testing::Values(40),                             // out channels
                                          ::testing::Values(2),                              // groups
                                          ::testing::Values(ov::op::PadType::EXPLICIT)),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(std::vector<size_t>({1, 192, 56, 56})),
                       ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
    ::testing::ValuesIn(netActivations));

const auto smoke_2D_ExplicitPaddingAsymmetric_Params = ::testing::Combine(
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(kernels2D),
                                          ::testing::ValuesIn(strides2D),
                                          ::testing::Values(std::vector<ptrdiff_t>({0, 3})),  // pads_begin
                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pads_end
                                          ::testing::ValuesIn(dilations2D),
                                          ::testing::ValuesIn(numOutChannels2D),
                                          ::testing::ValuesIn(numGroups2D),
                                          ::testing::Values(ov::op::PadType::EXPLICIT)),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(std::vector<size_t>({3, 8, 21, 11})),
                       ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
    ::testing::ValuesIn(netActivations));

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBiasAdd_2D_ExplicitPaddingSymmetric,
                        GroupConvolutionBiasAddActivationLayerTest,
                        smoke_2D_ExplicitPaddingSymmetric_Params,
                        GroupConvolutionBiasAddActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBiasAddAdd_2D_ExplicitPaddingSymmetric,
                        GroupConvolutionBiasAddAddActivationLayerTest,
                        smoke_2D_ExplicitPaddingSymmetric_Params,
                        GroupConvolutionBiasAddAddActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBiasAdd_2D_ExplicitPaddingSymmetric2,
                        GroupConvolutionBiasAddActivationLayerTest,
                        smoke_2D_ExplicitPaddingSymmetric_Params2,
                        GroupConvolutionBiasAddActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBiasAddAdd_2D_ExplicitPaddingSymmetric2,
                        GroupConvolutionBiasAddAddActivationLayerTest,
                        smoke_2D_ExplicitPaddingSymmetric_Params2,
                        GroupConvolutionBiasAddAddActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBiasAdd_2D_ExplicitPaddingSymmetric3,
                        GroupConvolutionBiasAddActivationLayerTest,
                        smoke_2D_ExplicitPaddingSymmetric_Params3,
                        GroupConvolutionBiasAddActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBiasAddAdd_2D_ExplicitPaddingSymmetric3,
                        GroupConvolutionBiasAddAddActivationLayerTest,
                        smoke_2D_ExplicitPaddingSymmetric_Params3,
                        GroupConvolutionBiasAddAddActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBiasAdd_2D_ExplicitPaddingSymmetric4,
                        GroupConvolutionBiasAddActivationLayerTest,
                        smoke_2D_ExplicitPaddingSymmetric_Params4,
                        GroupConvolutionBiasAddActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBiasAddAdd_2D_ExplicitPaddingSymmetric4,
                        GroupConvolutionBiasAddAddActivationLayerTest,
                        smoke_2D_ExplicitPaddingSymmetric_Params4,
                        GroupConvolutionBiasAddAddActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBiasAdd_smoke_2D_ExplicitPaddingAsymmetric,
                        GroupConvolutionBiasAddActivationLayerTest,
                        smoke_2D_ExplicitPaddingAsymmetric_Params,
                        GroupConvolutionBiasAddActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBiasAddAdd_smoke_2D_ExplicitPaddingAsymmetric,
                        GroupConvolutionBiasAddAddActivationLayerTest,
                        smoke_2D_ExplicitPaddingAsymmetric_Params,
                        GroupConvolutionBiasAddAddActivationLayerTest::getTestCaseName);

// WARNING: Currently the fusing of 3D Convolution is disabled in
// openvino_nvidia_gpu_plugin/modules/nvidia_plugin/src/transformer/fuse_conv_biasadd_activation.cpp
// so the the following smoke tests on graphs without FusedConvolution nodes
//
/* ============= 3D Convolution ============= */
const std::vector<std::vector<size_t>> kernels3D = {{3, 3, 3}, {3, 5, 3}};
const std::vector<std::vector<size_t>> strides3D = {{1, 1, 1}, {1, 2, 1}};
const std::vector<std::vector<size_t>> dilations3D = {{1, 1, 1}, {1, 2, 1}};
const std::vector<size_t> numOutChannels3D = {6, 12};
const std::vector<size_t> numGroups3D = {2, 3};

const auto smoke_3D_ExplicitPaddingSymmetric_Params = ::testing::Combine(
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(kernels3D),
                                          ::testing::ValuesIn(strides3D),
                                          ::testing::Values(std::vector<ptrdiff_t>({0, 2, 0})),  // pads_begin
                                          ::testing::Values(std::vector<ptrdiff_t>({0, 2, 0})),  // pads_end
                                          ::testing::ValuesIn(dilations3D),
                                          ::testing::ValuesIn(numOutChannels3D),
                                          ::testing::ValuesIn(numGroups3D),
                                          ::testing::Values(ov::op::PadType::EXPLICIT)),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(std::vector<size_t>({1, 6, 9, 12, 10})),
                       ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
    ::testing::ValuesIn(netActivations));

const auto smoke_3D_ExplicitPaddingAsymmetric_Params = ::testing::Combine(
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(kernels3D),
                                          ::testing::ValuesIn(strides3D),
                                          ::testing::Values(std::vector<ptrdiff_t>({0, 2, 0})),  // pads_begin
                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),  // pads_end
                                          ::testing::ValuesIn(dilations3D),
                                          ::testing::ValuesIn(numOutChannels3D),
                                          ::testing::ValuesIn(numGroups3D),
                                          ::testing::Values(ov::op::PadType::EXPLICIT)),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(std::vector<size_t>({2, 6, 12, 15, 20})),
                       ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
    ::testing::ValuesIn(netActivations));

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBiasAdd_3D_ExplicitPaddingSymmetric,
                        GroupConvolutionBiasAddActivationLayerTest,
                        smoke_3D_ExplicitPaddingSymmetric_Params,
                        GroupConvolutionBiasAddActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBiasAddAdd_3D_ExplicitPaddingSymmetric,
                        GroupConvolutionBiasAddAddActivationLayerTest,
                        smoke_3D_ExplicitPaddingSymmetric_Params,
                        GroupConvolutionBiasAddAddActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBiasAdd_3D_ExplicitPaddingAsymmetric,
                        GroupConvolutionBiasAddActivationLayerTest,
                        smoke_3D_ExplicitPaddingAsymmetric_Params,
                        GroupConvolutionBiasAddActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolutionBiasAddAdd_3D_ExplicitPaddingAsymmetric,
                        GroupConvolutionBiasAddAddActivationLayerTest,
                        smoke_3D_ExplicitPaddingAsymmetric_Params,
                        GroupConvolutionBiasAddAddActivationLayerTest::getTestCaseName);
}  // namespace LayerTestsDefinitions
