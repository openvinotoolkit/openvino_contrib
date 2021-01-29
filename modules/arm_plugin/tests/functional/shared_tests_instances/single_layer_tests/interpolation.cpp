// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/interpolate.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> prc = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
};

const std::vector<std::vector<size_t>> targetShapes = {
        {1, 1, 1, 1},
        {1, 1, 2, 2},
        {1, 1, 3, 3},
        {1, 1, 5, 5},
        {1, 1, 6, 6},

        {1, 1, 11, 11},
        {1, 1, 15, 15},
        {1, 1, 20, 20},
        {1, 1, 29, 29},

        {1, 1, 38, 38},
        {1, 1, 40, 40},
        {1, 1, 47, 47},
        {1, 1, 60, 60},
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModes = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::pytorch_half_pixel,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel,
};

const std::vector<bool> antialias = {
// Not enabled in Inference Engine
//        true,
        false,
};

const std::vector<ngraph::op::v4::Interpolate::ShapeCalcMode> shapeCalculationMode = {
        ngraph::op::v4::Interpolate::ShapeCalcMode::sizes,
        ngraph::op::v4::Interpolate::ShapeCalcMode::scales,
};

// --------------------Bilinear-----------------------------------

const  std::vector<ngraph::op::v4::Interpolate::InterpolateMode> linearModes = {
        ngraph::op::v4::Interpolate::InterpolateMode::linear,
        ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx,
};

const auto interpolateLinear = ::testing::Combine(
        ::testing::ValuesIn(linearModes),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::Values(ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor),
        ::testing::ValuesIn(antialias),
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}),
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}),
        ::testing::Values(static_cast<double>(-0.75)),
        ::testing::Values(std::vector<int64_t>{1}),
        ::testing::Values(std::vector<float>{1.33333f, 1.33333f}));

INSTANTIATE_TEST_CASE_P(Interpolate_Basic, InterpolateLayerTest, ::testing::Combine(
        interpolateLinear,
        ::testing::ValuesIn(prc),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 1, 30, 30}),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values("ARM")),
    InterpolateLayerTest::getTestCaseName);

// ------------------Common Nearest--------------------------------

const auto nearestFloorAsymCases = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::nearest),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::Values(ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric),
        ::testing::Values(ngraph::op::v4::Interpolate::NearestMode::floor),
        ::testing::ValuesIn(antialias),
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}),
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}),
        ::testing::Values(static_cast<double>(-0.75)),
        ::testing::Values(std::vector<int64_t>{1}),
        ::testing::Values(std::vector<float>{1.33333f, 1.33333f}));

INSTANTIATE_TEST_CASE_P(Interpolate_NearestFloorAsym, InterpolateLayerTest, ::testing::Combine(
        nearestFloorAsymCases,
        ::testing::ValuesIn(prc),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 1, 30, 30}),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values("ARM")),
    InterpolateLayerTest::getTestCaseName);

const auto nearestPrCeilAlignCases = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::nearest),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::Values(ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners),
        ::testing::Values(ngraph::op::v4::Interpolate::NearestMode::round_prefer_ceil),
        ::testing::ValuesIn(antialias),
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}),
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}),
        ::testing::Values(static_cast<double>(-0.75)),
        ::testing::Values(std::vector<int64_t>{1}),
        ::testing::Values(std::vector<float>{1.33333f, 1.33333f}));

INSTANTIATE_TEST_CASE_P(Interpolate_NearestPrCeilAlign, InterpolateLayerTest, ::testing::Combine(
        nearestPrCeilAlignCases,
        ::testing::ValuesIn(prc),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 1, 30, 30}),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values("ARM")),
    InterpolateLayerTest::getTestCaseName);

// ------------------Upsampling Nearest--------------------------------

const std::vector<std::vector<size_t>> upsampleShapes = {
        {1, 1, 31, 31},
        {1, 1, 38, 38},
        {1, 1, 40, 40},
        {1, 1, 47, 47},
        {1, 1, 60, 60},
        {1, 1, 90, 90},
};
const auto nearestSimpleUpCases = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::nearest),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::Values(ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric),
        ::testing::Values(ngraph::op::v4::Interpolate::NearestMode::simple),
        ::testing::ValuesIn(antialias),
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}),
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}),
        ::testing::Values(static_cast<double>(-0.75)),
        ::testing::Values(std::vector<int64_t>{1}),
        ::testing::Values(std::vector<float>{1.33333f, 1.33333f}));


INSTANTIATE_TEST_CASE_P(Interpolate_NearestSimpleUp, InterpolateLayerTest, ::testing::Combine(
        nearestSimpleUpCases,
        ::testing::ValuesIn(prc),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 1, 30, 30}),
        ::testing::ValuesIn(upsampleShapes),
        ::testing::Values("ARM")),
    InterpolateLayerTest::getTestCaseName);

// ------------------Upsampling Nearest int factor--------------------------------

const std::vector<std::vector<size_t>> upsampleIntShapes = {
        {1, 1, 60, 60},
        {1, 1, 90, 90},
        {1, 1, 120, 120},
};
const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> upPrCeilTransformModes = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::pytorch_half_pixel,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel,
};
const auto nearestUpPrCeilCases = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::nearest),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(upPrCeilTransformModes),
        ::testing::Values(ngraph::op::v4::Interpolate::NearestMode::round_prefer_ceil),
        ::testing::ValuesIn(antialias),
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}),
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}),
        ::testing::Values(static_cast<double>(-0.75)),
        ::testing::Values(std::vector<int64_t>{1}),
        ::testing::Values(std::vector<float>{1.33333f, 1.33333f}));

INSTANTIATE_TEST_CASE_P(Interpolate_NearestUpPrCeil, InterpolateLayerTest, ::testing::Combine(
        nearestUpPrCeilCases,
        ::testing::ValuesIn(prc),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 1, 30, 30}),
        ::testing::ValuesIn(upsampleIntShapes),
        ::testing::Values("ARM")),
    InterpolateLayerTest::getTestCaseName);

// ------------------Downsampling Nearest int factor--------------------------------

const std::vector<std::vector<size_t>> downsampleShapes = {
        {1, 1, 2, 2},
        {1, 1, 3, 3},
        {1, 1, 5, 5},
        {1, 1, 6, 6},
        {1, 1, 10, 10},
        {1, 1, 15, 15},
};
const auto nearestPrCeilCases = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::nearest),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::Values(ngraph::op::v4::Interpolate::NearestMode::round_prefer_ceil),
        ::testing::ValuesIn(antialias),
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}),
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}),
        ::testing::Values(static_cast<double>(-0.75)),
        ::testing::Values(std::vector<int64_t>{1}),
        ::testing::Values(std::vector<float>{1.33333f, 1.33333f}));

INSTANTIATE_TEST_CASE_P(Interpolate_NearestPrCeil, InterpolateLayerTest, ::testing::Combine(
        nearestPrCeilCases,
        ::testing::ValuesIn(prc),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 1, 30, 30}),
        ::testing::ValuesIn(downsampleShapes),
        ::testing::Values("ARM")),
    InterpolateLayerTest::getTestCaseName);

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> downSimpleTransformModes = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::pytorch_half_pixel,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel,
};
const auto nearestDownSimpleCases = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::nearest),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(downSimpleTransformModes),
        ::testing::Values(ngraph::op::v4::Interpolate::NearestMode::simple),
        ::testing::ValuesIn(antialias),
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}),
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}),
        ::testing::Values(static_cast<double>(-0.75)),
        ::testing::Values(std::vector<int64_t>{1}),
        ::testing::Values(std::vector<float>{1.33333f, 1.33333f}));

INSTANTIATE_TEST_CASE_P(Interpolate_NearestDownSimple, InterpolateLayerTest, ::testing::Combine(
        nearestDownSimpleCases,
        ::testing::ValuesIn(prc),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 1, 30, 30}),
        ::testing::ValuesIn(downsampleShapes),
        ::testing::Values("ARM")),
    InterpolateLayerTest::getTestCaseName);
} // namespace
