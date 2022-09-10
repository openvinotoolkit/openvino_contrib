// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <nvidia/cuda_config.hpp>
#include <cuda_test_constants.hpp>

#include "behavior/plugin/configuration_tests.hpp"

using namespace BehaviorTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {
    {{CONFIG_KEY(CPU_THROUGHPUT_STREAMS), InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}},
    {{CONFIG_KEY(CPU_THROUGHPUT_STREAMS), InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_NUMA}},
    {{CONFIG_KEY(CPU_THROUGHPUT_STREAMS), "8"}},
    {{CUDA_CONFIG_KEY(THROUGHPUT_STREAMS), InferenceEngine::CUDAConfigParams::CUDA_THROUGHPUT_AUTO}},
    {{CUDA_CONFIG_KEY(THROUGHPUT_STREAMS), "8"}},
};

const std::vector<std::map<std::string, std::string>> inconfigs = {
    {{CONFIG_KEY(CPU_THROUGHPUT_STREAMS), CONFIG_VALUE(NO)}},
    {{CUDA_CONFIG_KEY(THROUGHPUT_STREAMS), CONFIG_VALUE(NO)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         IncorrectConfigTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                                            ::testing::ValuesIn(inconfigs)),
                         IncorrectConfigTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         IncorrectConfigAPITests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                                            ::testing::ValuesIn(inconfigs)),
                         IncorrectConfigAPITests::getTestCaseName);
}  // namespace