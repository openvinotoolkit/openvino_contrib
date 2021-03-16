// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/activation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::helpers;
namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
        {Sigmoid,     {}},
        {Tanh,        {}},
        {Relu,        {}},
        {Exp,         {}},
        {Abs,         {}},
        {Clamp,       {{-2.0f, 2.0f}}},
        {Negative,    {}},
        {Log,         {}},
        {Sin,         {}},
        {Floor,       {}},
        {Sqrt,        {}},
        {Elu,         {{0.1f}}},
        {HSwish,      {}},
        {SoftPlus,    {}},
        {Swish,       {{1.0f}, {0.5f}}},
        {Mish,        {}},
        {Sign,        {}},
        {Ceiling,     {}},
        {Gelu,        {}},
        // Reference
        {Asin,        {}},
        {Acos,        {}},
        {Atan,        {}},
        {Cos,         {}},
        {Cosh,        {}},
        {Atan,        {}},
        {Sinh,        {}},
        {Tan,         {}},
        {Erf,         {}},
        {HSigmoid,    {}},
        {HardSigmoid, {{1.0f}, {0.5f}}},
        {Selu,        {{1.6732f, 1.0507f}}},
        {RoundHalfToEven,       {}},
        {RoundHalfAwayFromZero, {}}
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationParamTypes = {
    {PReLu, {{-0.01f}}},
    {LeakyRelu, {{0.01f}}}
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
        {{1, 50}, {{}}},
        {{1, 128}, {{}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> preluBasic = {
        {{1, 50}, {{1}, {50}}},
        {{1, 128}, {{1}, {128}}},
};

const auto basicCases = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto basicPreluCases = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationParamTypes)),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(CommonTestUtils::combineParams(preluBasic)),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);


INSTANTIATE_TEST_CASE_P(Activation_Basic, ActivationLayerTest, basicCases, ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(Activation_Basic_Prelu, ActivationLayerTest, basicPreluCases, ActivationLayerTest::getTestCaseName);

// INSTANTIATE_TEST_CASE_P(Activation_Basic, ActivationParamLayerTest, basicPreluCases, ActivationLayerTest::getTestCaseName);

}  // namespace
