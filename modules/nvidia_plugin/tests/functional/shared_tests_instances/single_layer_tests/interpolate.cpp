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

using InterpolateMode = ngraph::op::v4::Interpolate::InterpolateMode;
using ShapeCalcMode = ngraph::op::v4::Interpolate::ShapeCalcMode;
using CoordinateTransformMode = ngraph::op::v4::Interpolate::CoordinateTransformMode;
using NearestMode = ngraph::op::v4::Interpolate::NearestMode;

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

const std::vector<CoordinateTransformMode> coordinateTransformModes = {CoordinateTransformMode::HALF_PIXEL,
                                                                       CoordinateTransformMode::PYTORCH_HALF_PIXEL,
                                                                       CoordinateTransformMode::ASYMMETRIC,
                                                                       CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN,
                                                                       CoordinateTransformMode::ALIGN_CORNERS};

const std::vector<ShapeCalcMode> shapeCalculationMode = {
    ShapeCalcMode::SIZES,
    ShapeCalcMode::SCALES,
};

const std::vector<NearestMode> nearestModes = {
    NearestMode::ROUND_PREFER_FLOOR,
    NearestMode::ROUND_PREFER_CEIL,
    NearestMode::FLOOR,
    NearestMode::CEIL,
    NearestMode::SIMPLE,
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

const std::vector<double> defaultCubeCoeff = {
    -0.75f,
};

const std::vector<std::vector<int64_t>> smokeTest4DAxes = {{0, 1, 2, 3}, {}};
const std::vector<std::vector<float>> smokeTest4DScales = {{1.f, 1.f, 2.f, 2.f}};
const std::vector<std::vector<int64_t>> smokeTest2DAxes = {{2, 3}};
const std::vector<std::vector<float>> smokeTest2DScales = {{2.f, 2.f}};

std::map<std::string, std::string> additional_config = {};

const auto interpolate4DScaleParams = ::testing::Combine(::testing::Values(InterpolateMode::NEAREST),
                                                         ::testing::ValuesIn(shapeCalculationMode),
                                                         ::testing::ValuesIn(coordinateTransformModes),
                                                         ::testing::ValuesIn(nearestModes),
                                                         ::testing::ValuesIn(antialias),
                                                         ::testing::ValuesIn(pads),
                                                         ::testing::ValuesIn(pads),
                                                         ::testing::ValuesIn(defaultCubeCoeff),
                                                         ::testing::ValuesIn(smokeTest4DAxes),
                                                         ::testing::ValuesIn(smokeTest4DScales));

const auto interpolate2DScaleParams = ::testing::Combine(::testing::Values(InterpolateMode::NEAREST),
                                                         ::testing::ValuesIn(shapeCalculationMode),
                                                         ::testing::ValuesIn(coordinateTransformModes),
                                                         ::testing::ValuesIn(nearestModes),
                                                         ::testing::ValuesIn(antialias),
                                                         ::testing::ValuesIn(pads),
                                                         ::testing::ValuesIn(pads),
                                                         ::testing::ValuesIn(defaultCubeCoeff),
                                                         ::testing::ValuesIn(smokeTest2DAxes),
                                                         ::testing::ValuesIn(smokeTest2DScales));

class CUDAInterpolateLayerTest : public UnsymmetricalComparer<InterpolateLayerTest> {
protected:
    void SetUp() override {
        UnsymmetricalComparer<InterpolateLayerTest>::SetUp();

        auto params = this->GetParam();
        auto netPrecision = std::get<1>(params);
        if (netPrecision.getPrecVal() == InferenceEngine::Precision::FP16) {
            this->threshold = 0.12;
        }
    }
};

TEST_P(CUDAInterpolateLayerTest, CompareWithRefs) {
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
                       ::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                       ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(smoke_Simple_Interpolate_Nearest_4D_Scale_Param_Test,
                        CUDAInterpolateLayerTest,
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
                       ::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                       ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(smoke_Simple_Interpolate_Nearest_2D_Scale_Param_Test,
                        CUDAInterpolateLayerTest,
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
                                                      ::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                                                      ::testing::Values(additional_config));
INSTANTIATE_TEST_CASE_P(smoke_Downscale_Interpolate_Nearest_Test,
                        CUDAInterpolateLayerTest,
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
                       ::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                       ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(smoke_High_Resolution_Interpolate_Nearest_4D_Scale_Param_Test,
                        CUDAInterpolateLayerTest,
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
                                                         ::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                                                         ::testing::Values(additional_config));
INSTANTIATE_TEST_CASE_P(efficientdetInterpolateCombinationTests,
                        CUDAInterpolateLayerTest,
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
                       ::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                       ::testing::Values(additional_config));
INSTANTIATE_TEST_CASE_P(yolov5InterpolateFrom20To40ShapeTests,
                        CUDAInterpolateLayerTest,
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
                       ::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                       ::testing::Values(additional_config));
INSTANTIATE_TEST_CASE_P(yolov5InterpolateFrom40To80ShapeTests,
                        CUDAInterpolateLayerTest,
                        yolov5InterpolateFrom40To80Shape,
                        InterpolateLayerTest::getTestCaseName);

const std::vector<InferenceEngine::Precision> linearNetPrecisions = {
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32,
};
const std::vector<ShapeCalcMode> linearShapeCalculationMode = {ShapeCalcMode::SIZES, ShapeCalcMode::SCALES};
const std::vector<CoordinateTransformMode> linearCoordinateTransformModes = {
    CoordinateTransformMode::HALF_PIXEL,
    CoordinateTransformMode::PYTORCH_HALF_PIXEL,
    CoordinateTransformMode::ASYMMETRIC,
    CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN,
    CoordinateTransformMode::ALIGN_CORNERS,
};
const std::vector<NearestMode> linearNearestModes = {NearestMode::SIMPLE};
const std::vector<bool> linearAntialias = {true, false};

const std::vector<std::vector<int64_t>> linearTest2DAxes = {{2, 3}};
const std::vector<std::vector<float>> linearTest2DScales = {{0.5f, 0.5f}, {1.5f, 1.5f}};
const std::vector<std::vector<size_t>> linearTest2DSizes = {{6, 10}, {14, 20}, {6, 20}};
const auto linear2DScaleParams = ::testing::Combine(::testing::Values(InterpolateMode::LINEAR),
                                                    ::testing::ValuesIn(linearShapeCalculationMode),
                                                    ::testing::ValuesIn(linearCoordinateTransformModes),
                                                    ::testing::ValuesIn(linearNearestModes),
                                                    ::testing::ValuesIn(linearAntialias),
                                                    ::testing::ValuesIn(pads),
                                                    ::testing::ValuesIn(pads),
                                                    ::testing::ValuesIn(defaultCubeCoeff),
                                                    ::testing::ValuesIn(linearTest2DAxes),
                                                    ::testing::ValuesIn(linearTest2DScales));
const std::vector<std::vector<size_t>> linearInput2DScaleShapes = {{1, 3, 10, 16}};
INSTANTIATE_TEST_CASE_P(smoke_InterpolateLinear_2D_Scale_Test,
                        CUDAInterpolateLayerTest,
                        ::testing::Combine(linear2DScaleParams,
                                           ::testing::ValuesIn(linearNetPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(linearInput2DScaleShapes),
                                           ::testing::ValuesIn(linearTest2DSizes),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                                           ::testing::Values(additional_config)),
                        InterpolateLayerTest::getTestCaseName);

const std::vector<std::vector<int64_t>> linearTest3DAxes = {{2, 3, 4}};
const std::vector<std::vector<float>> linearTest3DScales = {{0.5f, 0.4f, 0.6f}, {1.5f, 1.6f, 1.8f}};
const std::vector<std::vector<size_t>> linearTest3DSizes = {{6, 8, 10}, {10, 16, 18}, {10, 8, 18}};
const auto linear3DScaleParams = ::testing::Combine(::testing::Values(InterpolateMode::LINEAR),
                                                    ::testing::ValuesIn(linearShapeCalculationMode),
                                                    ::testing::ValuesIn(linearCoordinateTransformModes),
                                                    ::testing::ValuesIn(linearNearestModes),
                                                    ::testing::ValuesIn(linearAntialias),
                                                    ::testing::ValuesIn(pads),
                                                    ::testing::ValuesIn(pads),
                                                    ::testing::ValuesIn(defaultCubeCoeff),
                                                    ::testing::ValuesIn(linearTest3DAxes),
                                                    ::testing::ValuesIn(linearTest3DScales));
const std::vector<std::vector<size_t>> linearInput3DScaleShapes = {{1, 3, 8, 12, 14}};
INSTANTIATE_TEST_CASE_P(smoke_InterpolateLinear_3D_Scale_Test,
                        CUDAInterpolateLayerTest,
                        ::testing::Combine(linear3DScaleParams,
                                           ::testing::ValuesIn(linearNetPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(linearInput3DScaleShapes),
                                           ::testing::ValuesIn(linearTest3DSizes),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                                           ::testing::Values(additional_config)),
                        InterpolateLayerTest::getTestCaseName);

const std::vector<InferenceEngine::Precision> cubicNetPrecisions = {
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32,
};
const std::vector<double> cubeCoeffs = {-0.75f, -0.6f};
const std::vector<ShapeCalcMode> cubicShapeCalculationMode = {ShapeCalcMode::SIZES, ShapeCalcMode::SCALES};
const std::vector<CoordinateTransformMode> cubicCoordinateTransformModes = {
    CoordinateTransformMode::HALF_PIXEL,
    CoordinateTransformMode::PYTORCH_HALF_PIXEL,
    CoordinateTransformMode::ASYMMETRIC,
    CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN,
    CoordinateTransformMode::ALIGN_CORNERS,
};
const std::vector<NearestMode> cubicNearestModes = {
    NearestMode::SIMPLE  // Cubic interpolation algo doesn't use it.
};
const std::vector<bool> cubicAntialias = {false};  // Cubic interpolation algo doesn't use it.

const std::vector<std::vector<int64_t>> cubicTest2DAxes = {{2, 3}};
const std::vector<std::vector<float>> cubicTest2DScales = {{0.5f, 0.5f}, {1.5f, 1.5f}};
const std::vector<std::vector<size_t>> cubicTest2DSizes = {{6, 10}, {14, 20}, {6, 20}};
const auto cubic2DScaleParams = ::testing::Combine(::testing::Values(InterpolateMode::CUBIC),
                                                   ::testing::ValuesIn(cubicShapeCalculationMode),
                                                   ::testing::ValuesIn(cubicCoordinateTransformModes),
                                                   ::testing::ValuesIn(cubicNearestModes),
                                                   ::testing::ValuesIn(cubicAntialias),
                                                   ::testing::ValuesIn(pads),
                                                   ::testing::ValuesIn(pads),
                                                   ::testing::ValuesIn(cubeCoeffs),
                                                   ::testing::ValuesIn(cubicTest2DAxes),
                                                   ::testing::ValuesIn(cubicTest2DScales));
const std::vector<std::vector<size_t>> cubicInput2DScaleShapes = {{1, 3, 10, 16}};
INSTANTIATE_TEST_CASE_P(smoke_InterpolateCubic_2D_Scale_Test,
                        CUDAInterpolateLayerTest,
                        ::testing::Combine(cubic2DScaleParams,
                                           ::testing::ValuesIn(cubicNetPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(cubicInput2DScaleShapes),
                                           ::testing::ValuesIn(cubicTest2DSizes),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                                           ::testing::Values(additional_config)),
                        InterpolateLayerTest::getTestCaseName);

const std::vector<std::vector<int64_t>> cubicTest3DAxes = {{2, 3, 4}};
const std::vector<std::vector<float>> cubicTest3DScales = {{0.5f, 0.4f, 0.6f}, {1.5f, 1.6f, 1.8f}};
const std::vector<std::vector<size_t>> cubicTest3DSizes = {{6, 8, 10}, {10, 16, 18}, {10, 8, 18}};
const auto cubic3DScaleParams = ::testing::Combine(::testing::Values(InterpolateMode::CUBIC),
                                                   ::testing::ValuesIn(cubicShapeCalculationMode),
                                                   ::testing::ValuesIn(cubicCoordinateTransformModes),
                                                   ::testing::ValuesIn(cubicNearestModes),
                                                   ::testing::ValuesIn(cubicAntialias),
                                                   ::testing::ValuesIn(pads),
                                                   ::testing::ValuesIn(pads),
                                                   ::testing::ValuesIn(cubeCoeffs),
                                                   ::testing::ValuesIn(cubicTest3DAxes),
                                                   ::testing::ValuesIn(cubicTest3DScales));
const std::vector<std::vector<size_t>> cubicInput3DScaleShapes = {{1, 3, 8, 12, 14}};
INSTANTIATE_TEST_CASE_P(smoke_InterpolateCubic_3D_Scale_Test,
                        CUDAInterpolateLayerTest,
                        ::testing::Combine(cubic3DScaleParams,
                                           ::testing::ValuesIn(cubicNetPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(cubicInput3DScaleShapes),
                                           ::testing::ValuesIn(cubicTest3DSizes),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                                           ::testing::Values(additional_config)),
                        InterpolateLayerTest::getTestCaseName);

namespace benchmark {

struct CUDAInterpolateLayerBenchmarkTest : BenchmarkLayerTest<CUDAInterpolateLayerTest> {};

TEST_P(CUDAInterpolateLayerBenchmarkTest, DISABLED_benchmark) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run("Interpolate", std::chrono::milliseconds(2000), 200);
}

const auto nearestBenchmarkParams =
    ::testing::Combine(interpolate4DScaleParams,
                       ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32}),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 88, 40, 40}}),
                       ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 88, 80, 80}}),
                       ::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                       ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(CUDAInterpolate_Nearest_Benchmark,
                        CUDAInterpolateLayerBenchmarkTest,
                        nearestBenchmarkParams,
                        InterpolateLayerTest::getTestCaseName);

const std::vector<InterpolateMode> benchmarkInterpolateModes = {InterpolateMode::LINEAR, InterpolateMode::CUBIC};
const std::vector<InferenceEngine::Precision> benchmarkPrecisions = {
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32,
};
const std::vector<std::vector<float>> benchmarkScales = {{0.5f, 0.5f, 0.5f}, {1.5f, 1.5f, 1.5f}};
const auto benchmarkParams =
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(benchmarkInterpolateModes),
                                          ::testing::Values(ShapeCalcMode::SCALES),
                                          ::testing::Values(CoordinateTransformMode::HALF_PIXEL),
                                          ::testing::Values(NearestMode::SIMPLE),
                                          ::testing::Values(true),  // antialias
                                          ::testing::ValuesIn(pads),
                                          ::testing::ValuesIn(pads),
                                          ::testing::ValuesIn(defaultCubeCoeff),
                                          ::testing::Values(std::vector<int64_t>{2, 3, 4}),  // axes
                                          ::testing::ValuesIn(benchmarkScales)),
                       ::testing::ValuesIn(benchmarkPrecisions),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(std::vector<size_t>{1, 3, 50, 50, 50}),  // input data shape
                       ::testing::Values(std::vector<size_t>{0, 0, 0}),           // sizes, not used
                       ::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                       ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(CUDAInterpolate_Benchmark,
                        CUDAInterpolateLayerBenchmarkTest,
                        benchmarkParams,
                        InterpolateLayerTest::getTestCaseName);

}  // namespace benchmark

}  // namespace
