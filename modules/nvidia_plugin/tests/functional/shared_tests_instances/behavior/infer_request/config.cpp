// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/config.hpp"

#include <cuda_test_constants.hpp>
#include <nvidia/nvidia_config.hpp>
#include <vector>

using namespace BehaviorTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests_IncorrectConfig,
                        InferRequestConfigTest,
                        ::testing::Combine(::testing::Values(0ul),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                                           ::testing::ValuesIn(configs)),
                        InferRequestConfigTest::getTestCaseName);

}  // namespace
