// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/conversion.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<std::vector<size_t>> inShape = {{1, 2, 3, 4}};

const std::vector<InferenceEngine::Precision> targetPrecisionsU8 = {
        InferenceEngine::Precision::U16,
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::I32,
};

INSTANTIATE_TEST_CASE_P(smoke_ConvertU8, ConversionLayerTest,
                        ::testing::Combine(
                                ::testing::Values(ngraph::helpers::ConversionTypes::CONVERT),
                                ::testing::Values(inShape),
                                ::testing::Values(InferenceEngine::Precision::U8),
                                ::testing::ValuesIn(targetPrecisionsU8),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ConversionLayerTest::getTestCaseName);

const std::vector<InferenceEngine::Precision> targetPrecisionsU16 = {
        InferenceEngine::Precision::U8,
        // "Incorrect precision! openvino/inference-engine/tests/ie_test_utils/functional_test_utils/precision_utils.hpp:46"
        // InferenceEngine::Precision::U32,
};

INSTANTIATE_TEST_CASE_P(smoke_ConvertU16, ConversionLayerTest,
                        ::testing::Combine(
                                ::testing::Values(ngraph::helpers::ConversionTypes::CONVERT),
                                ::testing::Values(inShape),
                                ::testing::Values(InferenceEngine::Precision::U16),
                                ::testing::ValuesIn(targetPrecisionsU16),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ConversionLayerTest::getTestCaseName);

const std::vector<InferenceEngine::Precision> targetPrecisionsI16 = {
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::I32,
};

INSTANTIATE_TEST_CASE_P(smoke_ConvertI16, ConversionLayerTest,
                        ::testing::Combine(
                                ::testing::Values(ngraph::helpers::ConversionTypes::CONVERT),
                                ::testing::Values(inShape),
                                ::testing::Values(InferenceEngine::Precision::I16),
                                ::testing::ValuesIn(targetPrecisionsI16),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ConversionLayerTest::getTestCaseName);

const std::vector<InferenceEngine::Precision> precisions = {
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::FP32,
};

INSTANTIATE_TEST_CASE_P(smoke_ConvertAll, ConversionLayerTest,
                        ::testing::Combine(
                                ::testing::Values(ngraph::helpers::ConversionTypes::CONVERT),
                                ::testing::Values(inShape),
                                ::testing::ValuesIn(precisions),
                                ::testing::ValuesIn(precisions),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ConversionLayerTest::getTestCaseName);
}  // namespace
