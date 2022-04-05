// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/interpolate.hpp"

#include <cuda_test_constants.hpp>
#include <vector>

#include "benchmark.hpp"
#include "common_test_utils/test_constants.hpp"
#include "unsymmetrical_comparer.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<size_t>> inShapes = {
    {1, 1, 23, 23},
};
const std::vector<std::vector<size_t>> targetShapes = {
    {1, 1, 46, 46},
};

const std::vector<std::vector<size_t>> downscaleInShapes = {
    {1, 1, 23, 23},
};
const std::vector<std::vector<size_t>> downscaleTargetShapes = {
    {1, 1, 46, 46},
};

const std::vector<std::vector<size_t>> efficientdetShapes = {
    {1, 88, 5, 5},
    {1, 88, 10, 10},
    {1, 88, 20, 20},
    {1, 88, 40, 40},
    {1, 88, 80, 80},
};

const std::vector<std::vector<size_t>> yolov5From20To40Shape = {
    {1, 256, 20, 20},
    {1, 256, 40, 40},
};

const std::vector<std::vector<size_t>> yolov5From40To80Shape = {
    {1, 128, 40, 40},
    {1, 128, 80, 80},
};

// TODO only nearest mode supported now
const std::vector<ngraph::op::v4::Interpolate::InterpolateMode> nearestMode = {
    ngraph::op::v4::Interpolate::InterpolateMode::nearest,
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModes = {
    ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric,
    ngraph::op::v4::Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn,
    // TODO CoordinateTransform mode not supported yet
    // ngraph::op::v4::Interpolate::CoordinateTransformMode::pytorch_half_pixel,
    // ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel,
    // ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners,
};

const std::vector<ngraph::op::v4::Interpolate::ShapeCalcMode> shapeCalculationMode = {
    ngraph::op::v4::Interpolate::ShapeCalcMode::sizes,
    ngraph::op::v4::Interpolate::ShapeCalcMode::scales,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> nearestModes = {
    ngraph::op::v4::Interpolate::NearestMode::simple, ngraph::op::v4::Interpolate::NearestMode::floor,
    // TODO nearest modes not supported yet
    // ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor,
    // ngraph::op::v4::Interpolate::NearestMode::ceil,
    // ngraph::op::v4::Interpolate::NearestMode::round_prefer_ceil,
};

const std::vector<std::vector<size_t>> pads = {
    // TODO only zero padding is supported now
    {0, 0, 0, 0},
};

const std::vector<bool> antialias = {
    // Not enabled in Inference Engine
    //        true,
    false,
};

const std::vector<double> cubeCoefs = {
    -0.75f,
};

const std::vector<std::vector<int64_t>> smokeTest4DAxes = {{0, 1, 2, 3}};
const std::vector<std::vector<float>> smokeTest4DScales = {{1.f, 1.f, 2.f, 2.f}};
const std::vector<std::vector<int64_t>> smokeTest2DAxes = {{2, 3}};
const std::vector<std::vector<float>> smokeTest2DScales = {{2.f, 2.f}};

std::map<std::string, std::string> additional_config = {};

const auto interpolate4DScaleParams = ::testing::Combine(::testing::ValuesIn(nearestMode),
                                                         ::testing::ValuesIn(shapeCalculationMode),
                                                         ::testing::ValuesIn(coordinateTransformModes),
                                                         ::testing::ValuesIn(nearestModes),
                                                         ::testing::ValuesIn(antialias),
                                                         ::testing::ValuesIn(pads),
                                                         ::testing::ValuesIn(pads),
                                                         ::testing::ValuesIn(cubeCoefs),
                                                         ::testing::ValuesIn(smokeTest4DAxes),
                                                         ::testing::ValuesIn(smokeTest4DScales));

const auto interpolate2DScaleParams = ::testing::Combine(::testing::ValuesIn(nearestMode),
                                                         ::testing::ValuesIn(shapeCalculationMode),
                                                         ::testing::ValuesIn(coordinateTransformModes),
                                                         ::testing::ValuesIn(nearestModes),
                                                         ::testing::ValuesIn(antialias),
                                                         ::testing::ValuesIn(pads),
                                                         ::testing::ValuesIn(pads),
                                                         ::testing::ValuesIn(cubeCoefs),
                                                         ::testing::ValuesIn(smokeTest2DAxes),
                                                         ::testing::ValuesIn(smokeTest2DScales));

class CUDNNInterpolateLayerTest : public UnsymmetricalComparer<InterpolateLayerTest> {};

TEST_P(CUDNNInterpolateLayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
};

const auto simpleCombine4DScaleParamTests =
    ::testing::Combine(interpolate4DScaleParams,
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::ValuesIn(inShapes),
                       ::testing::ValuesIn(targetShapes),
                       ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                       ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(smoke_Simple_Interpolate_4D_Scale_Param_Test,
                        CUDNNInterpolateLayerTest,
                        simpleCombine4DScaleParamTests,
                        InterpolateLayerTest::getTestCaseName);

const auto simpleCombine2DScaleParamTests =
    ::testing::Combine(interpolate2DScaleParams,
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::ValuesIn(inShapes),
                       ::testing::ValuesIn(targetShapes),
                       ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                       ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(smoke_Simple_Interpolate_2D_Scale_Param_Test,
                        CUDNNInterpolateLayerTest,
                        simpleCombine2DScaleParamTests,
                        InterpolateLayerTest::getTestCaseName);

const auto downscaleCombineTests = ::testing::Combine(interpolate4DScaleParams,
                                                      ::testing::ValuesIn(netPrecisions),
                                                      ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                                      ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                                      ::testing::Values(InferenceEngine::Layout::ANY),
                                                      ::testing::Values(InferenceEngine::Layout::ANY),
                                                      ::testing::ValuesIn(downscaleInShapes),
                                                      ::testing::ValuesIn(downscaleTargetShapes),
                                                      ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                                                      ::testing::Values(additional_config));
INSTANTIATE_TEST_CASE_P(smoke_Downscale_Interpolate_Nearest_Test,
                        CUDNNInterpolateLayerTest,
                        downscaleCombineTests,
                        InterpolateLayerTest::getTestCaseName);

const std::vector<std::vector<int64_t>> default4DAxes = {{0, 1, 2, 3}};
const std::vector<std::vector<float>> default4DScales = {{1.f, 1.f, 2.f, 2.f}};

const auto highResolutionCombineTest =
    ::testing::Combine(interpolate4DScaleParams,
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 88, 20, 20}}),
                       ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 88, 40, 40}}),
                       ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                       ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(smoke_High_Resolution_Interpolate_4D_Scale_Param_Test,
                        CUDNNInterpolateLayerTest,
                        highResolutionCombineTest,
                        InterpolateLayerTest::getTestCaseName);

const auto efficientdetCombinations = ::testing::Combine(interpolate4DScaleParams,
                                                         ::testing::ValuesIn(netPrecisions),
                                                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                                         ::testing::Values(InferenceEngine::Layout::ANY),
                                                         ::testing::Values(InferenceEngine::Layout::ANY),
                                                         ::testing::ValuesIn(efficientdetShapes),
                                                         ::testing::ValuesIn(efficientdetShapes),
                                                         ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                                                         ::testing::Values(additional_config));
INSTANTIATE_TEST_CASE_P(efficientdetInterpolateCombinationTests,
                        CUDNNInterpolateLayerTest,
                        efficientdetCombinations,
                        InterpolateLayerTest::getTestCaseName);

const auto yolov5InterpolateFrom20To40Shape =
    ::testing::Combine(interpolate4DScaleParams,
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::ValuesIn(yolov5From20To40Shape),
                       ::testing::ValuesIn(yolov5From20To40Shape),
                       ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                       ::testing::Values(additional_config));
INSTANTIATE_TEST_CASE_P(yolov5InterpolateFrom20To40ShapeTests,
                        CUDNNInterpolateLayerTest,
                        yolov5InterpolateFrom20To40Shape,
                        InterpolateLayerTest::getTestCaseName);

const auto yolov5InterpolateFrom40To80Shape =
    ::testing::Combine(interpolate4DScaleParams,
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::ValuesIn(yolov5From40To80Shape),
                       ::testing::ValuesIn(yolov5From40To80Shape),
                       ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                       ::testing::Values(additional_config));
INSTANTIATE_TEST_CASE_P(yolov5InterpolateFrom40To80ShapeTests,
                        CUDNNInterpolateLayerTest,
                        yolov5InterpolateFrom40To80Shape,
                        InterpolateLayerTest::getTestCaseName);

namespace benchmark {

struct CUDNNInterpolateLayerBenchmarkTest : BenchmarkLayerTest<CUDNNInterpolateLayerTest> {};

TEST_P(CUDNNInterpolateLayerBenchmarkTest, DISABLED_benchmark) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run("Interpolate", std::chrono::milliseconds(2000), 200);
}

const auto benchmarkParams =
    ::testing::Combine(interpolate4DScaleParams,
                       ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32}),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 88, 40, 40}}),
                       ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 88, 80, 80}}),
                       ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                       ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(CUDNNInterpolate_Benchmark,
                        CUDNNInterpolateLayerBenchmarkTest,
                        benchmarkParams,
                        InterpolateLayerTest::getTestCaseName);

}  // namespace benchmark

}  // namespace
