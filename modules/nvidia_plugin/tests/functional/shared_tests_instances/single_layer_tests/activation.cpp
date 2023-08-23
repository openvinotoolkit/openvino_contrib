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

const std::vector<InferenceEngine::Precision> intPrecisions = {
    InferenceEngine::Precision::I32,
};

// TODO commented tests don't work for CUDA now.
// The reason there are missing correspondent operations or transformation
const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
    {Sigmoid,               {}},
    {Tanh,                  {}},
    {Relu,                  {}},
    {Exp,                   {}},
    {Log,                   {}},
    //            {Sign,                  {}},
    {Abs,                   {}},
    {Clamp,    {{-2.0f, 2.0f}}},
    {Negative,              {}},
    //            {Acos,                  {}},
    //            {Asin,                  {}},
    //            {Atan,                  {}},
    {Cos,                   {}},
    {Cosh,                  {}},
    {Floor,                 {}},
    {Sin,                   {}},
    {Sinh,                  {}},
    {Sqrt,                  {}},
    //            {Tan,                   {}},
    //            {Elu,                   {{0.1f}}},
    //            {Erf,                   {}},
    //            {HardSigmoid,           {{0.2f, 0.5f}}},
    //            {Selu,                  {{1.6732f, 1.0507f}}},
    //            {Ceiling,               {}},
    {Mish,                  {}},
    {Swish,           {{0.5f}}},
    {HSwish,                {}},
    //            {SoftPlus,              {}},
    {HSigmoid,              {}},
    //            {RoundHalfToEven,       {}},
    //            {RoundHalfAwayFromZero, {}},
    {Gelu,                  {}},
    {GeluErf,               {}},
    {GeluTanh,              {}}
};

class CUDAActivationIntegerLayerTest : public ActivationLayerTest {
    void SetUp() override {
        ActivationLayerTest::SetUp();
        threshold = 1;
    }
};

// List of operations that should be tested also with integer precision
const std::map<ActivationTypes, std::vector<std::vector<float>>> intActivationTypes = {
        {Abs,                   {}},
        {Negative,              {}},
        {Cos,                   {}},
        {Cosh,                  {}},
        {Sinh,                  {}},
        {Sqrt,                  {}},
        {Log,                   {}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> preluActivationParamTypes = {
    {PReLu, {{}}},  // Slope will be filled with increasing values from -10 to match slope input shape
    {LeakyRelu, {{0.01f}}}};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
    {{1, 50}, {{}}},
    {{1, 128}, {{}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> preluBasic = {
    {{1, 50}, {{1}, {50}}},
    {{1, 128}, {{1}, {128}}},

    // Broadcast check
    {{3, 2}, {{1}, {2}, {3, 2}}},
    {{3, 2, 5}, {{1}, {2}, {5}, {2, 5}, {3, 1, 5}, {1, 2, 1}, {1, 1, 5}, {3, 1, 1}, {3, 2, 5}}},
    {{2, 1, 2}, {{2}, {2, 1, 1}}},
    {{3, 2, 5, 7}, {{1}, {7}, {2}, {5, 7}, {2, 5, 7}, {2, 1, 1}, {1, 2, 1, 1}, {3, 2, 1, 1}, {3, 2, 5, 7}}},
};

const auto basicCases = ::testing::Combine(::testing::ValuesIn(ov::test::utils::combineParams(activationTypes)),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(ov::test::utils::combineParams(basic)),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA));

const auto basicPreluCases =
    ::testing::Combine(::testing::ValuesIn(ov::test::utils::combineParams(preluActivationParamTypes)),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::ValuesIn(ov::test::utils::combineParams(preluBasic)),
                       ::testing::Values(ov::test::utils::DEVICE_NVIDIA));

const auto basicPReluConstParamCases =
    ::testing::Combine(::testing::ValuesIn(ov::test::utils::combineParams(preluActivationParamTypes)),
                       ::testing::ValuesIn(preluConstParamNetPrecisions),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::ValuesIn(ov::test::utils::combineParams(preluBasic)),
                       ::testing::Values(ov::test::utils::DEVICE_NVIDIA));

const auto basicIntegerOperations =
    ::testing::Combine(::testing::ValuesIn(ov::test::utils::combineParams(intActivationTypes)),
                       ::testing::ValuesIn(intPrecisions),
                       ::testing::ValuesIn(intPrecisions),
                       ::testing::ValuesIn(intPrecisions),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::ValuesIn(ov::test::utils::combineParams(basic)),
                       ::testing::Values(ov::test::utils::DEVICE_NVIDIA));

TEST_P(ActivationLayerTest, CompareWithRefs) { Run(); }

TEST_P(ActivationParamLayerTest, CompareWithRefs) { Run(); }

TEST_P(ActivationDynamicLayerTest, CompareWithRefs) { Run(); }

TEST_P(CUDAActivationIntegerLayerTest, CompareWithRefs) { Run(); }

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
INSTANTIATE_TEST_CASE_P(smoke_Cuda_Integer_Activation_Basic,
                        CUDAActivationIntegerLayerTest,
                        basicIntegerOperations,
                        CUDAActivationIntegerLayerTest::getTestCaseName);

}  // namespace
