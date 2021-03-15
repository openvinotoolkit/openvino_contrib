// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/pad.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::vector<int64_t>> padsBegin2D = {{0, 0}, {1, 1}, {2, 0}, {0, 3}};
const std::vector<std::vector<int64_t>> padsEnd2D   = {{0, 0}, {1, 1}, {0, 1}, {3, 2}};
const std::vector<float> argPadValue = {0.f, 1.f, 2.f, -1.f};

const auto pad2DConstparams = testing::Combine(
        testing::ValuesIn(padsBegin2D),
        testing::ValuesIn(padsEnd2D),
        testing::ValuesIn(argPadValue),
        testing::Values(ngraph::helpers::PadMode::CONSTANT),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{13, 5}),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(
        Pad2DConst,
        PadLayerTest,
        pad2DConstparams,
        PadLayerTest::getTestCaseName
);

const auto pad2DParams = testing::Combine(
        testing::ValuesIn(padsBegin2D),
        testing::ValuesIn(padsEnd2D),
        testing::Values(0),
        testing::Values(ngraph::helpers::PadMode::REFLECT),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{13, 5}),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(
        Pad2D,
        PadLayerTest,
        pad2DParams,
        PadLayerTest::getTestCaseName
);

const std::vector<std::vector<int64_t>> padsBegin4D = {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 0, 1, 0}, {0, 3, 0, 1}};
const std::vector<std::vector<int64_t>> padsEnd4D   = {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 0, 0, 1}, {1, 3, 2, 0}};

const auto pad4DConstparams = testing::Combine(
        testing::ValuesIn(padsBegin4D),
        testing::ValuesIn(padsEnd4D),
        testing::ValuesIn(argPadValue),
        testing::Values(ngraph::helpers::PadMode::CONSTANT),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{3, 5, 10, 11}),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(
        Pad4DConst,
        PadLayerTest,
        pad4DConstparams,
        PadLayerTest::getTestCaseName
);

const auto pad4DReflectParams = testing::Combine(
        testing::ValuesIn(padsBegin4D),
        testing::ValuesIn(padsEnd4D),
        testing::Values(0),
        testing::Values(ngraph::helpers::PadMode::REFLECT),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{3, 5, 10, 11}),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(
        Pad4DReflect,
        PadLayerTest,
        pad4DReflectParams,
        PadLayerTest::getTestCaseName
);

const auto padSym4DParams = testing::Combine(
        testing::ValuesIn(padsBegin4D),
        testing::Values(std::vector<int64_t>{0, 0, 0, 0}),
        testing::Values(0),
        testing::Values(ngraph::helpers::PadMode::SYMMETRIC),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{3, 5, 10, 11}),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(
        Pad4DSymmetric,
        PadLayerTest,
        padSym4DParams,
        PadLayerTest::getTestCaseName
);

}  // namespace
