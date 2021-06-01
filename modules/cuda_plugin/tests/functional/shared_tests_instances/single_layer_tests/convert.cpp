// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/convert.hpp>
#include <cuda_test_constants.hpp>

using namespace LayerTestsDefinitions;
using namespace InferenceEngine;

namespace CUDALayerTestsDefinitions  {

class ConvertCUDALayerTest : public ConvertLayerTest {};

TEST_P(ConvertCUDALayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ConvertParamsTuple params = GetParam();
    inPrc = std::get<1>(params);
    outPrc = std::get<2>(params);

    Run();
}

namespace {
const std::vector<std::vector<size_t>> inShape = {{1, 2, 3, 4}};

// List of precisions natively supported by CUDA.
// CUDA device supports only U8, FP16 and FP32 output precision
const std::vector<Precision> out_precisions = {
        Precision::U8,
//        Precision::FP16, // FIXME Uncomment when FP16 blobs are supported
        Precision::FP32,
};

// Supported formats are: FP32, FP16, I16 and U8
const std::vector<Precision> in_precisions = {
        Precision::U8,
        Precision::I16,
//        Precision::FP16, // FIXME Uncomment when FP16 blobs are supported
        Precision::FP32,
};

INSTANTIATE_TEST_CASE_P(smoke_ConvertLayerTest_From_F32, ConvertCUDALayerTest,
                        ::testing::Combine(
                                ::testing::Values(inShape),
                                ::testing::Values(Precision::FP32),
                                ::testing::ValuesIn(out_precisions),
                                ::testing::Values(Layout::ANY),
                                ::testing::Values(Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        ConvertLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ConvertLayerTest_To_F32, ConvertCUDALayerTest,
                        ::testing::Combine(
                                ::testing::Values(inShape),
                                ::testing::ValuesIn(in_precisions),
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(Layout::ANY),
                                ::testing::Values(Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        ConvertLayerTest::getTestCaseName);

/* TODO Uncomment when BF16 support is implemented
INSTANTIATE_TEST_CASE_P(smoke_ConvertLayerTest_From_BF16, ConvertCUDALayerTest,
                        ::testing::Combine(
                                ::testing::Values(inShape),
                                ::testing::Values(Precision::BF16),
                                ::testing::ValuesIn(precisions),
                                ::testing::Values(Layout::ANY),
                                ::testing::Values(Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        ConvertLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ConvertLayerTest_To_BF16, ConvertCUDALayerTest,
                        ::testing::Combine(
                                ::testing::Values(inShape),
                                ::testing::ValuesIn(precisions),
                                ::testing::Values(Precision::BF16),
                                ::testing::Values(Layout::ANY),
                                ::testing::Values(Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        ConvertLayerTest::getTestCaseName);
*/
} // namespace
} // namespace CUDALayerTestsDefinitions
