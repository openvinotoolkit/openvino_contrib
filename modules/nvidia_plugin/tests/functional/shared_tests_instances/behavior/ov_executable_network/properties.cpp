// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/properties.hpp"

#include <cuda_test_constants.hpp>

#include "openvino/runtime/properties.hpp"
#include "nvidia/properties.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> inproperties = {
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

const std::vector<ov::AnyMap> hetero_inproperties = {
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

const std::vector<ov::AnyMap> multi_inproperties = {
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

const std::vector<ov::AnyMap> auto_inproperties = {
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

const std::vector<ov::AnyMap> auto_batch_inproperties = {
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
    {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(CommonTestUtils::DEVICE_NVIDIA) + "(4)"},
     {ov::auto_batch_timeout(-1)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                                            ::testing::ValuesIn(inproperties)),
                         OVCompiledModelPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                            ::testing::ValuesIn(hetero_inproperties)),
                         OVCompiledModelPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         OVCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multi_inproperties)),
                         OVCompiledModelPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         OVCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(auto_inproperties)),
                         OVCompiledModelPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         OVCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_inproperties)),
                         OVCompiledModelPropertiesIncorrectTests::getTestCaseName);

const std::vector<ov::AnyMap> default_properties = {
    {ov::num_streams(1)},
    {ov::inference_precision(ov::element::undefined)},
    {ov::hint::num_requests(0)},
    {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
    {ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE)},
    {ov::enable_profiling(false)},
    {ov::device::id("0")},
    {ov::nvidia_gpu::operation_benchmark(false)}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompiledModelPropertiesDefaultTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                                            ::testing::ValuesIn(default_properties)),
                         OVCompiledModelPropertiesDefaultTests::getTestCaseName);

const std::vector<ov::AnyMap> properties = {
    {ov::num_streams(8)},
    {ov::num_streams(ov::streams::AUTO)},
    {ov::inference_precision(ov::element::undefined)},
    {ov::inference_precision(ov::element::f32)},
    {ov::inference_precision(ov::element::f16)},
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
    {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
    {ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY)},
    {ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE)},
    {ov::enable_profiling(true)},
    {ov::enable_profiling(false)},
    {ov::device::id("0")},
    {ov::device::id("NVIDIA.0")},
};

const std::vector<ov::AnyMap> hetero_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_NVIDIA), ov::enable_profiling(true)},
    {ov::device::priorities(CommonTestUtils::DEVICE_NVIDIA), ov::device::id("0")},
};

const std::vector<ov::AnyMap> multi_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_NVIDIA), ov::enable_profiling(true)},
    {ov::device::priorities(CommonTestUtils::DEVICE_NVIDIA), ov::device::id("0")},
};

const std::vector<ov::AnyMap> auto_batch_properties = {
    {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(CommonTestUtils::DEVICE_NVIDIA) + "(4)"}},
    {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(CommonTestUtils::DEVICE_NVIDIA) + "(4)"},
     {CONFIG_KEY(AUTO_BATCH_TIMEOUT), "1"}},
    {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(CommonTestUtils::DEVICE_NVIDIA) + "(4)"},
     {ov::auto_batch_timeout(10)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                                            ::testing::ValuesIn(properties)),
                         OVCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                            ::testing::ValuesIn(hetero_properties)),
                         OVCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         OVCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multi_properties)),
                         OVCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         OVCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_properties)),
                         OVCompiledModelPropertiesTests::getTestCaseName);
}  // namespace