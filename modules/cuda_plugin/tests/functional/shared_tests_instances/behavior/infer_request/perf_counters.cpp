// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/perf_counters.hpp"

#include "base/behavior_test_utils.hpp"
#include "cuda_test_constants.hpp"
#include "multi-device/multi_device_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {

const std::vector<std::map<std::string, std::string>> configs = {{}};

const std::vector<std::map<std::string, std::string>> Multiconfigs = {
    {{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), CommonTestUtils::DEVICE_CUDA}}};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests,
                        InferRequestPerfCountersTest,
                        ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_CUDA),
                                           ::testing::ValuesIn(configs)),
                        InferRequestPerfCountersTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests,
                        InferRequestPerfCountersTest,
                        ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                           ::testing::ValuesIn(Multiconfigs)),
                        InferRequestPerfCountersTest::getTestCaseName);

}  // namespace
