// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda/cuda_config.hpp>
#include <cuda_test_constants.hpp>

#include "multi-device/multi_device_config.hpp"
#include "behavior/config.hpp"

using namespace BehaviorTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {{CUDA_CONFIG_KEY(THROUGHPUT_STREAMS), InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}},
    {{CUDA_CONFIG_KEY(THROUGHPUT_STREAMS), InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_NUMA}},
    {{CUDA_CONFIG_KEY(THROUGHPUT_STREAMS), "8"}},
};

const std::vector<std::map<std::string, std::string>> inconfigs = {
    {{CUDA_CONFIG_KEY(THROUGHPUT_STREAMS), CONFIG_VALUE(NO)}},
};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, IncorrectConfigTests,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                                ::testing::ValuesIn(inconfigs)),
                        IncorrectConfigTests::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, IncorrectConfigAPITests,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                                ::testing::ValuesIn(inconfigs)),
                        IncorrectConfigAPITests::getTestCaseName);


INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, CorrectConfigAPITests,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                                ::testing::ValuesIn(configs)),
                        CorrectConfigAPITests::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, CorrectConfigTests,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                                ::testing::ValuesIn(configs)),
                        CorrectConfigAPITests::getTestCaseName);

} // namespace