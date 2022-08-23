// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_test_constants.hpp>
#include <vector>

#include "ie_precision.hpp"
#include "single_layer_tests/conversion.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine;

namespace CUDALayerTestsDefinitions {

class ConversionCUDALayerTest : public ConversionLayerTest {};

TEST_P(ConversionCUDALayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ConversionParamsTuple params = GetParam();
    inPrc = std::get<2>(params);
    outPrc = std::get<3>(params);

    Run();
}

namespace {
const std::vector<ngraph::helpers::ConversionTypes> conversionOpTypes = {
    ngraph::helpers::ConversionTypes::CONVERT,
};

const std::vector<std::vector<size_t>> inShape = {{1, 2, 3, 4}};

// List of precisions natively supported by CUDA.
// CUDA device supports only U8, FP16 and FP32 output precision
const std::vector<Precision> out_precisions = {
    Precision::U8,
    Precision::FP16,
    Precision::FP32,
};

// Supported formats are: BOOL, FP32, FP16, I16 and U8
const std::vector<Precision> in_precisions = {
    Precision::BOOL,
    Precision::U8,
    Precision::I16,
    // TODO: Uncomment when we find way to omit conversion from FP16 -> FP32 in tests
    //        Precision::FP16,
    Precision::FP32,
};

INSTANTIATE_TEST_SUITE_P(smoke_ConversionLayerTest_From_F32,
                         ConversionCUDALayerTest,
                         ::testing::Combine(::testing::ValuesIn(conversionOpTypes),
                                            ::testing::Values(inShape),
                                            ::testing::Values(Precision::FP32),
                                            ::testing::ValuesIn(out_precisions),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConversionLayerTest_To_F32,
                         ConversionCUDALayerTest,
                         ::testing::Combine(::testing::ValuesIn(conversionOpTypes),
                                            ::testing::Values(inShape),
                                            ::testing::ValuesIn(in_precisions),
                                            ::testing::Values(Precision::FP32),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                         ConversionLayerTest::getTestCaseName);

/* TODO Uncomment when BF16 support is implemented
INSTANTIATE_TEST_CASE_P(smoke_ConvertLayerTest_From_BF16, ConversionCUDALayerTest,
                        ::testing::Combine(
                                ::testing::Values(inShape),
                                ::testing::Values(Precision::BF16),
                                ::testing::ValuesIn(precisions),
                                ::testing::Values(Layout::ANY),
                                ::testing::Values(Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        ConvertLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_ConvertLayerTest_To_BF16, ConversionCUDALayerTest,
                        ::testing::Combine(
                                ::testing::Values(inShape),
                                ::testing::ValuesIn(precisions),
                                ::testing::Values(Precision::BF16),
                                ::testing::Values(Layout::ANY),
                                ::testing::Values(Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        ConvertLayerTest::getTestCaseName);
*/
}  // namespace
}  // namespace CUDALayerTestsDefinitions
