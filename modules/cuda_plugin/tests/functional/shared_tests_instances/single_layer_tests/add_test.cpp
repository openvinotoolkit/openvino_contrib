// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <vector>
#include <cuda_test_constants.hpp>

#include "single_layer_tests/eltwise.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<std::vector<std::vector<size_t>>> smoke_shapes = {
    { { 1, 2, 3, 4, 5 }, { 1, 2, 3, 4, 5 } },
    { { 2, 3, 4, 5 }, { 2, 3, 4, 1 } },
    { { 2, 3, 4, 5 }, { 2, 1, 1, 5 } },
    { { 3, 1, 6, 1 }, { 1, 1, 1, 1 } },
    { { 10, 10}, { 10, 10 } },
    { { 10, 10}, { 1, 10 } },
    { { 10, 10}, { 1 } }
};

const std::vector<ngraph::helpers::InputLayerType> smoke_input_layer_types = {
    ngraph::helpers::InputLayerType::CONSTANT,
    ngraph::helpers::InputLayerType::PARAMETER
};

const std::vector<CommonTestUtils::OpType> smoke_op_types = {
     CommonTestUtils::OpType::SCALAR,
     CommonTestUtils::OpType::VECTOR
};

const std::vector<InferenceEngine::Precision> smoke_inputPrecisions = {
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32
};

const std::map<std::string, std::string> smoke_additional_config = {};

INSTANTIATE_TEST_CASE_P(smoke_Add, EltwiseLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(smoke_shapes),
                            ::testing::Values(ngraph::helpers::EltwiseTypes::ADD),
                            ::testing::ValuesIn(smoke_input_layer_types),
                            ::testing::ValuesIn(smoke_op_types),
                            ::testing::ValuesIn(smoke_inputPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                            ::testing::Values(smoke_additional_config)),
                        EltwiseLayerTest::getTestCaseName);


const std::vector<std::vector<std::vector<size_t>>> renset50_vgg16_shapes = {
    { { 1, 1000 }, { 1, 1000 } },
    { { 1, 1001 }, { 1, 1001 } },
    { { 1, 1024, 14, 14 }, { 1, 1024, 1, 1 } },
    { { 1, 1024, 14, 14 }, { 1, 1024, 14, 14 } },
    { { 1, 128, 112, 112 }, { 1, 128, 1, 1 } },
    { { 1, 128, 28, 28 }, { 1, 128, 1, 1 } },
    { { 1, 128, 56, 56 }, { 1, 128, 1, 1 } },
    { { 1, 2048, 7, 7 }, { 1, 2048, 1, 1 } },
    { { 1, 2048, 7, 7 }, { 1, 2048, 7, 7 } },
    { { 1, 256, 14, 14 }, { 1, 256, 1, 1 } },
    { { 1, 256, 28, 28 }, { 1, 256, 1, 1 } },
    { { 1, 256, 56, 56 }, { 1, 256, 1, 1 } },
    { { 1, 256, 56, 56 }, { 1, 256, 56, 56 } },
    { { 1, 3, 224, 224 }, { 1, 3, 1, 1 } },
    { { 1, 4096 }, { 1, 4096 } },
    { { 1, 512, 14, 14 }, { 1, 512, 1, 1 } },
    { { 1, 512, 28, 28 }, { 1, 512, 1, 1 } },
    { { 1, 512, 28, 28 }, { 1, 512, 28, 28 } },
    { { 1, 512, 7, 7 }, { 1, 512, 1, 1 } },
    { { 1, 64, 112, 112 }, { 1, 64, 1, 1 } },
    { { 1, 64, 224, 224 }, { 1, 64, 1, 1 } },
    { { 1, 64, 56, 56 }, { 1, 64, 1, 1 } }
};

const std::vector<ngraph::helpers::InputLayerType> renset50_vgg16_input_layer_types = {
    ngraph::helpers::InputLayerType::CONSTANT,
    ngraph::helpers::InputLayerType::PARAMETER
};

const std::vector<CommonTestUtils::OpType> renset50_vgg16_op_types = {
     CommonTestUtils::OpType::VECTOR
};

const std::vector<InferenceEngine::Precision> renset50_vgg16_inputPrecisions = {
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32
};

const std::map<std::string, std::string> renset50_vgg16_additional_config = {};

INSTANTIATE_TEST_CASE_P(Add, EltwiseLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(renset50_vgg16_shapes),
                            ::testing::Values(ngraph::helpers::EltwiseTypes::ADD),
                            ::testing::ValuesIn(renset50_vgg16_input_layer_types),
                            ::testing::ValuesIn(renset50_vgg16_op_types),
                            ::testing::ValuesIn(renset50_vgg16_inputPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                            ::testing::Values(renset50_vgg16_additional_config)),
                        EltwiseLayerTest::getTestCaseName);

} // namespace
