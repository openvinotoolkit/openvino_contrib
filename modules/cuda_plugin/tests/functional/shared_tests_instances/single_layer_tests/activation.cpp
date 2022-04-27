// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// TODO, this file is copied from cpu tests. Not checked tests are commented for now.

#include "shared_test_classes/single_layer/activation.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "cuda_test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::helpers;
namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

// TODO now for const parameter tests a slope node always is created with f32 precision
const std::vector<InferenceEngine::Precision> preluConstParamNetPrecisions = {InferenceEngine::Precision::FP32};

// TODO int parameter activation tests don't work for CUDA now
// const std::vector<InferenceEngine::Precision> intPrecisions = {
//        InferenceEngine::Precision::I32,
//};

// TODO commented tests don't work for CUDA now.
// The reason there are missing correspondent operations or transformation
const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
    {Sigmoid, {}},
    {Tanh, {}},
    {Relu, {}},
    //            {Exp,                   {}},
    //            {Log,                   {}},
    //            {Sign,                  {}},
    //            {Abs,                   {}},
    {Clamp, {{-2.0f, 2.0f}}},
    {Negative, {}},
    //            {Acos,                  {}},
    //            {Asin,                  {}},
    //            {Atan,                  {}},
    //            {Cos,                   {}},
    //            {Cosh,                  {}},
    {Floor, {}},
    //            {Sin,                   {}},
    //            {Sinh,                  {}},
    //            {Sqrt,                  {}},
    //            {Tan,                   {}},
    //            {Elu,                   {{0.1f}}},
    //            {Erf,                   {}},
    //            {HardSigmoid,           {{0.2f, 0.5f}}},
    //            {Selu,                  {{1.6732f, 1.0507f}}},
    //            {Ceiling,               {}},
    //            {Mish,                  {}},
    {HSwish, {}},
    //            {SoftPlus,              {}},
    {HSigmoid, {}},
    //            {RoundHalfToEven,       {}},
    //            {RoundHalfAwayFromZero, {}},
    //            {GeluErf,               {}},
    //            {GeluTanh,              {}}
};

// TODO tests below don't work for CUDA
// List of operations that should be tested also with integer precision
// const std::map<ActivationTypes, std::vector<std::vector<float>>> intActivationTypes = {
//        {Sqrt,                  {}},
//        {Tanh,                  {}},
//};

const std::map<ActivationTypes, std::vector<std::vector<float>>> preluActivationParamTypes = {
    {PReLu, {{}}},  // Slope will be filled with increasing values from -10 to match slope input shape
    {LeakyRelu, {{0.01f}}}};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
    {{1, 50}, {{}}},
    {{1, 128}, {{}}},
};

// TODO /*{2},*/is commented because it is not numpy broadcast,
// e.g. {2} shape can't be broadcasted to {3, 2, 5} according numpy rules
std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> preluBasic = {
    {{1, 50}, {{1}, {50}}},
    {{1, 128}, {{1}, {128}}},

    // Broadcast check
    {{3, 2}, {{1}, {2}, {3, 2}}},
    {{3, 2, 5}, {{1}, /*{2},*/ {5}, {2, 5}, {3, 1, 5}, {1, 2, 1}, {1, 1, 5}, {3, 1, 1}, {3, 2, 5}}},
    {{2, 1, 2}, {{2}, {2, 1, 1}}},
    {{3, 2, 5, 7}, {{1}, {7}, /*{2},*/ {5, 7}, {2, 5, 7}, {2, 1, 1}, {1, 2, 1, 1}, {3, 2, 1, 1}, {3, 2, 5, 7}}},
};

const auto basicCases = ::testing::Combine(::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA));

const auto basicPreluCases =
    ::testing::Combine(::testing::ValuesIn(CommonTestUtils::combineParams(preluActivationParamTypes)),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::ValuesIn(CommonTestUtils::combineParams(preluBasic)),
                       ::testing::Values(CommonTestUtils::DEVICE_CUDA));

const auto basicPReluConstParamCases =
    ::testing::Combine(::testing::ValuesIn(CommonTestUtils::combineParams(preluActivationParamTypes)),
                       ::testing::ValuesIn(preluConstParamNetPrecisions),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::ValuesIn(CommonTestUtils::combineParams(preluBasic)),
                       ::testing::Values(CommonTestUtils::DEVICE_CUDA));

// TODO int parameter activation tests don't work for CUDA now
// const auto basicIntegerOperations =
//    ::testing::Combine(::testing::ValuesIn(CommonTestUtils::combineParams(intActivationTypes)),
//                       ::testing::ValuesIn(intPrecisions),
//                       ::testing::ValuesIn(intPrecisions),
//                       ::testing::ValuesIn(intPrecisions),
//                       ::testing::Values(InferenceEngine::Layout::ANY),
//                       ::testing::Values(InferenceEngine::Layout::ANY),
//                       ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),
//                       ::testing::Values(CommonTestUtils::DEVICE_CUDA));

TEST_P(ActivationLayerTest, CompareWithRefs) { Run(); }

TEST_P(ActivationParamLayerTest, CompareWithRefs) { Run(); }

TEST_P(ActivationDynamicLayerTest, CompareWithRefs) { Run(); }

INSTANTIATE_TEST_CASE_P(smoke_Cuda_Activation_Basic,
                        ActivationLayerTest,
                        basicCases,
                        ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_Cuda_Activation_Basic,
                        ActivationDynamicLayerTest,
                        basicCases,
                        ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_Cuda_Activation_Prelu_Param,
                        ActivationParamLayerTest,
                        basicPreluCases,
                        ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_Cuda_Activation_PRelu_Const,
                        ActivationLayerTest,
                        basicPReluConstParamCases,
                        ActivationLayerTest::getTestCaseName);
// TODO Integer activation don't work for CUDA now
// INSTANTIATE_TEST_CASE_P(smoke_Cuda_Integer_Activation_Basic, ActivationLayerTest, basicIntegerOperations,
// ActivationLayerTest::getTestCaseName);

}  // namespace
