// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <vector>
#include <cuda_test_constants.hpp>

#include "single_layer_tests/eltwise.hpp"

using namespace LayerTestsDefinitions;

namespace {

// Common parameters
const std::vector<ngraph::helpers::InputLayerType> input_layer_types = {
    ngraph::helpers::InputLayerType::CONSTANT,
    ngraph::helpers::InputLayerType::PARAMETER
};

const std::vector<CommonTestUtils::OpType> smoke_op_types = {CommonTestUtils::OpType::SCALAR,
                                                             CommonTestUtils::OpType::VECTOR};

const std::map<std::string, std::string> additional_config = {};

//
// 'Add' smoke tests
//
const std::vector<std::vector<std::vector<size_t>>> smoke_shapes_add = {
    {{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}},
    {{2, 3, 4, 5}, {2, 3, 4, 1}},
    {{2, 3, 4, 5}, {2, 1, 1, 5}},
    {{3, 1, 6, 1}, {1, 1, 1, 1}},
    {{10, 10}, {10, 10}},
    {{10, 10}, {1, 10}},
    {{10, 10}, {1}},
};
const std::vector<InferenceEngine::Precision> input_precisions_add = {InferenceEngine::Precision::FP16,
                                                                      InferenceEngine::Precision::FP32};

INSTANTIATE_TEST_CASE_P(smoke_Add,
                        EltwiseLayerTest,
                        ::testing::Combine(::testing::ValuesIn(smoke_shapes_add),
                                           ::testing::Values(ngraph::helpers::EltwiseTypes::ADD),
                                           ::testing::ValuesIn(input_layer_types),
                                           ::testing::ValuesIn(smoke_op_types),
                                           ::testing::ValuesIn(input_precisions_add),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                                           ::testing::Values(additional_config)),
                        EltwiseLayerTest::getTestCaseName);

//
// 'Multiply' smoke tests
//
const std::vector<std::vector<std::vector<size_t>>> smoke_shapes_multiply = {{{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}},
                                                                             {{2, 3, 4, 5}, {2, 3, 4, 1}},
                                                                             {{2, 3, 4, 5}, {2, 1, 1, 5}},
                                                                             {{3, 1, 6, 1}, {1, 1, 1, 1}},
                                                                             {{10, 10}, {10, 10}},
                                                                             {{10, 10}, {1, 10}},
                                                                             {{10, 10}, {1}},
                                                                             {{8, 1, 6, 1}, {7, 1, 5}}};
const std::vector<InferenceEngine::Precision> input_precisions_multiply = {
    InferenceEngine::Precision::I32, InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32};

INSTANTIATE_TEST_CASE_P(smoke_Multiply,
                        EltwiseLayerTest,
                        ::testing::Combine(::testing::ValuesIn(smoke_shapes_multiply),
                                           ::testing::Values(ngraph::helpers::EltwiseTypes::MULTIPLY),
                                           ::testing::ValuesIn(input_layer_types),
                                           ::testing::ValuesIn(smoke_op_types),
                                           ::testing::ValuesIn(input_precisions_multiply),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                                           ::testing::Values(additional_config)),
                        EltwiseLayerTest::getTestCaseName);

//
// Resnet-50 and VGG-16
//
const std::vector<std::vector<std::vector<size_t>>> renset50_vgg16_add_shapes = {{{1, 1000}, {1, 1000}},
                                                                                 {{1, 1001}, {1, 1001}},
                                                                                 {{1, 1024, 14, 14}, {1, 1024, 1, 1}},
                                                                                 {{1, 1024, 14, 14}, {1, 1024, 14, 14}},
                                                                                 {{1, 128, 112, 112}, {1, 128, 1, 1}},
                                                                                 {{1, 128, 28, 28}, {1, 128, 1, 1}},
                                                                                 {{1, 128, 56, 56}, {1, 128, 1, 1}},
                                                                                 {{1, 2048, 7, 7}, {1, 2048, 1, 1}},
                                                                                 {{1, 2048, 7, 7}, {1, 2048, 7, 7}},
                                                                                 {{1, 256, 14, 14}, {1, 256, 1, 1}},
                                                                                 {{1, 256, 28, 28}, {1, 256, 1, 1}},
                                                                                 {{1, 256, 56, 56}, {1, 256, 1, 1}},
                                                                                 {{1, 256, 56, 56}, {1, 256, 56, 56}},
                                                                                 {{1, 3, 224, 224}, {1, 3, 1, 1}},
                                                                                 {{1, 4096}, {1, 4096}},
                                                                                 {{1, 512, 14, 14}, {1, 512, 1, 1}},
                                                                                 {{1, 512, 28, 28}, {1, 512, 1, 1}},
                                                                                 {{1, 512, 28, 28}, {1, 512, 28, 28}},
                                                                                 {{1, 512, 7, 7}, {1, 512, 1, 1}},
                                                                                 {{1, 64, 112, 112}, {1, 64, 1, 1}},
                                                                                 {{1, 64, 224, 224}, {1, 64, 1, 1}},
                                                                                 {{1, 64, 56, 56}, {1, 64, 1, 1}}};

const std::vector<InferenceEngine::Precision> renset50_vgg16_input_precisions = {InferenceEngine::Precision::FP16,
                                                                                 InferenceEngine::Precision::FP32};

const std::vector<CommonTestUtils::OpType> renset50_vgg16_op_types = {CommonTestUtils::OpType::VECTOR};

INSTANTIATE_TEST_CASE_P(renset50_vgg16_Add,
                        EltwiseLayerTest,
                        ::testing::Combine(::testing::ValuesIn(renset50_vgg16_add_shapes),
                                           ::testing::Values(ngraph::helpers::EltwiseTypes::ADD),
                                           ::testing::ValuesIn(input_layer_types),
                                           ::testing::ValuesIn(renset50_vgg16_op_types),
                                           ::testing::ValuesIn(renset50_vgg16_input_precisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                                           ::testing::Values(additional_config)),
                        EltwiseLayerTest::getTestCaseName);

//
// Tacotron2
//
const std::vector<std::vector<std::vector<size_t>>> tacotron2_Multiply_shapes = {
    {{1}, {1}},
    {{2}, {1}},
    {{1}, {2}},
    {{10}, {1}},
    {{1}, {10}},
};

const std::vector<InferenceEngine::Precision> tacotron2_Multiply_input_precisions = {
    InferenceEngine::Precision::I32,
};

INSTANTIATE_TEST_CASE_P(tacotron2_Multiply,
                        EltwiseLayerTest,
                        ::testing::Combine(::testing::ValuesIn(tacotron2_Multiply_shapes),
                                           ::testing::Values(ngraph::helpers::EltwiseTypes::MULTIPLY),
                                           ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                                           ::testing::Values(CommonTestUtils::OpType::VECTOR),
                                           ::testing::ValuesIn(tacotron2_Multiply_input_precisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                                           ::testing::Values(additional_config)),
                        EltwiseLayerTest::getTestCaseName);

}  // namespace
