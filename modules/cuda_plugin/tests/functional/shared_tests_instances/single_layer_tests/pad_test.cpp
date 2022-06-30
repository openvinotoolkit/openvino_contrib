// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/pad.hpp"

#include <cuda_test_constants.hpp>
#include <vector>

#include "benchmark.hpp"

using namespace LayerTestsDefinitions;

namespace {

/*
 * These tests instantiate more then 1000 test instances, therefore
 * some precisions are exectuted with small shapes.
 *
 * */
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::Precision> netPrecisionsForSmallShaped = {
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I16,
    InferenceEngine::Precision::U8,
    // TODO Now Openvino doesn't support the types listed below but CUDA Pad operation does.
    // Uncomment lines below when Openvino is ready for these types.
    // InferenceEngine::Precision::U16,
    // InferenceEngine::Precision::I8,
};

const std::vector<float> argPadValue = {0.f, 1.f, -1.f, 2.5f};

const std::vector<ngraph::helpers::PadMode> padMode = {
    ngraph::helpers::PadMode::EDGE, ngraph::helpers::PadMode::REFLECT, ngraph::helpers::PadMode::SYMMETRIC};

const std::vector<std::vector<int64_t>> padsBegin1D = {{2}};
const std::vector<std::vector<int64_t>> padsEnd1D = {{2}};
const std::vector<std::vector<size_t>> inputs1d = {{7}, {511}, {512}};

const auto pad1DConstparams = testing::Combine(testing::ValuesIn(padsBegin1D),
                                               testing::ValuesIn(padsEnd1D),
                                               testing::ValuesIn(argPadValue),
                                               testing::Values(ngraph::helpers::PadMode::CONSTANT),
                                               testing::ValuesIn(netPrecisions),
                                               testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                               testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                               testing::Values(InferenceEngine::Layout::ANY),
                                               testing::ValuesIn(inputs1d),
                                               testing::Values(CommonTestUtils::DEVICE_CUDA));

INSTANTIATE_TEST_CASE_P(smoke_Pad1DConst, PadLayerTest, pad1DConstparams, PadLayerTest::getTestCaseName);

const std::vector<std::vector<int64_t>> padsBegin2D = {{1, 1}, {2, 0}, {0, 3}};
const std::vector<std::vector<int64_t>> padsEnd2D = {{1, 1}, {0, 1}, {3, 2}};

const auto pad2DConstparams = testing::Combine(testing::ValuesIn(padsBegin2D),
                                               testing::ValuesIn(padsEnd2D),
                                               testing::ValuesIn(argPadValue),
                                               testing::Values(ngraph::helpers::PadMode::CONSTANT),
                                               testing::ValuesIn(netPrecisions),
                                               testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                               testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                               testing::Values(InferenceEngine::Layout::ANY),
                                               testing::Values(std::vector<size_t>{3, 512}),
                                               testing::Values(CommonTestUtils::DEVICE_CUDA));

INSTANTIATE_TEST_CASE_P(smoke_Pad2DConst, PadLayerTest, pad2DConstparams, PadLayerTest::getTestCaseName);

const std::vector<std::vector<int64_t>> padsBegin3D = {{2, 3, 1}, {0, 0, 1}, {0, 1, 1}, {2, 0, 1}};
const std::vector<std::vector<int64_t>> padsEnd3D = {{2, 3, 1}, {1, 0, 0}, {0, 0, 2}, {1, 3, 0}};

const auto pad3DConstparams = testing::Combine(testing::ValuesIn(padsBegin3D),
                                               testing::ValuesIn(padsEnd3D),
                                               testing::ValuesIn(argPadValue),
                                               testing::Values(ngraph::helpers::PadMode::CONSTANT),
                                               testing::ValuesIn(netPrecisions),
                                               testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                               testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                               testing::Values(InferenceEngine::Layout::ANY),
                                               testing::Values(std::vector<size_t>{3, 5, 11}),
                                               testing::Values(CommonTestUtils::DEVICE_CUDA));

INSTANTIATE_TEST_CASE_P(smoke_Pad3DConst, PadLayerTest, pad3DConstparams, PadLayerTest::getTestCaseName);

const std::vector<std::vector<int64_t>> padsBegin4D = {{0, 3, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 1}, {2, 0, 0, 0}};
const std::vector<std::vector<int64_t>> padsEnd4D = {{0, 3, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 2}, {1, 3, 0, 0}};

const auto pad4DConstparams = testing::Combine(testing::ValuesIn(padsBegin4D),
                                               testing::ValuesIn(padsEnd4D),
                                               testing::ValuesIn(argPadValue),
                                               testing::Values(ngraph::helpers::PadMode::CONSTANT),
                                               testing::ValuesIn(netPrecisions),
                                               testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                               testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                               testing::Values(InferenceEngine::Layout::ANY),
                                               testing::Values(std::vector<size_t>{3, 5, 10, 512}),
                                               testing::Values(CommonTestUtils::DEVICE_CUDA));

INSTANTIATE_TEST_CASE_P(smoke_Pad4DConst, PadLayerTest, pad4DConstparams, PadLayerTest::getTestCaseName);

const std::vector<std::vector<int64_t>> padsBegin5D = {{0, 1, 2, 3, 4}};
const std::vector<std::vector<int64_t>> padsEnd5D = {{4, 3, 2, 1, 0}};

const auto pad5DConstparams = testing::Combine(testing::ValuesIn(padsBegin5D),
                                               testing::ValuesIn(padsEnd5D),
                                               testing::ValuesIn(argPadValue),
                                               testing::Values(ngraph::helpers::PadMode::CONSTANT),
                                               testing::ValuesIn(netPrecisions),
                                               testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                               testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                               testing::Values(InferenceEngine::Layout::ANY),
                                               testing::Values(std::vector<size_t>{7, 3, 5, 10, 512}),
                                               testing::Values(CommonTestUtils::DEVICE_CUDA));

INSTANTIATE_TEST_CASE_P(smoke_Pad5DConst, PadLayerTest, pad5DConstparams, PadLayerTest::getTestCaseName);

const auto padPrecesionParams = testing::Combine(testing::ValuesIn(padsBegin2D),
                                                 testing::ValuesIn(padsEnd2D),
                                                 testing::ValuesIn(argPadValue),
                                                 testing::Values(ngraph::helpers::PadMode::CONSTANT),
                                                 testing::ValuesIn(netPrecisionsForSmallShaped),
                                                 testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                                 testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                                 testing::Values(InferenceEngine::Layout::ANY),
                                                 testing::Values(std::vector<size_t>{3, 128}),
                                                 testing::Values(CommonTestUtils::DEVICE_CUDA));

INSTANTIATE_TEST_CASE_P(smoke_PadPrecesionParams, PadLayerTest, padPrecesionParams, PadLayerTest::getTestCaseName);

struct NCHWFormatPadBenchmarkTest : BenchmarkLayerTest<PadLayerTest> {};

TEST_P(NCHWFormatPadBenchmarkTest, DISABLED_benchmark) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run("Pad", std::chrono::milliseconds(2000), 200);
}

const auto benchmarkShapes = std::vector<std::vector<size_t>>{
    {1, 96, 320, 320}, {1, 3, 640, 640}, {1, 672, 40, 40}, {1, 240, 80, 80}, {1, 144, 160, 160}};
const auto pad4DBenchmarkParams =
    testing::Combine(testing::ValuesIn(std::vector<std::vector<int64_t>>{{0, 0, 1, 1}, {0, 0, 3, 3}}),
                     testing::ValuesIn(std::vector<std::vector<int64_t>>{{0, 0, 0, 0}}),
                     testing::ValuesIn({0.f}),
                     testing::Values(ngraph::helpers::PadMode::CONSTANT),
                     testing::ValuesIn(netPrecisions),
                     testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                     testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                     testing::Values(InferenceEngine::Layout::ANY),
                     testing::ValuesIn(benchmarkShapes),
                     testing::Values(CommonTestUtils::DEVICE_CUDA));

INSTANTIATE_TEST_CASE_P(Benchmark4DPadOperation,
                        NCHWFormatPadBenchmarkTest,
                        pad4DBenchmarkParams,
                        PadLayerTest::getTestCaseName);

}  // namespace
