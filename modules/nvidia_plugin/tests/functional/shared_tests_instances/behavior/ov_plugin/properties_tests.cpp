// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/properties_tests.hpp"

#include <cuda_test_constants.hpp>

#include "openvino/runtime/properties.hpp"

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
};

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_BehaviorTests,
                         OVPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                                            ::testing::ValuesIn(inproperties)),
                         OVPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Hetero_BehaviorTests,
                         OVPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(hetero_inproperties)),
                         OVPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Multi_BehaviorTests,
                         OVPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multi_inproperties)),
                         OVPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Auto_BehaviorTests,
                         OVPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(auto_inproperties)),
                         OVPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_AutoBatch_BehaviorTests,
                         OVPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_inproperties)),
                         OVPropertiesIncorrectTests::getTestCaseName);

const std::vector<ov::AnyMap> default_properties = {
    {ov::num_streams(1)},
    {ov::hint::num_requests(0)},
    {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
    {ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE)},
    {ov::enable_profiling(false)},
    {ov::device::id(0)},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesDefaultTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                                            ::testing::ValuesIn(default_properties)),
                         OVPropertiesDefaultTests::getTestCaseName);

const std::vector<ov::AnyMap> properties = {
    {ov::num_streams(8)},
    {ov::num_streams(ov::streams::AUTO)},
    {ov::hint::inference_precision(ov::element::f32)},
    {ov::hint::inference_precision(ov::element::f16)},
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
    {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
    {ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY)},
    {ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE)},
    {ov::enable_profiling(true)},
    {ov::enable_profiling(false)},
    {ov::device::id(0)},
};

const std::vector<ov::AnyMap> hetero_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_NVIDIA), ov::enable_profiling(true)},
    {ov::device::priorities(ov::test::utils::DEVICE_NVIDIA), ov::device::id(0)},
};

const std::vector<ov::AnyMap> multi_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_NVIDIA), ov::enable_profiling(true)},
    {ov::device::priorities(ov::test::utils::DEVICE_NVIDIA), ov::device::id(0)},
};

const std::vector<ov::AnyMap> auto_batch_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_NVIDIA)},
    {{ov::device::priorities(ov::test::utils::DEVICE_NVIDIA)},
     {ov::auto_batch_timeout(1)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                                            ::testing::ValuesIn(properties)),
                         OVPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Hetero_BehaviorTests,
                         OVPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(hetero_properties)),
                         OVPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Multi_BehaviorTests,
                         OVPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multi_properties)),
                         OVPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_AutoBatch_BehaviorTests,
                         OVPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_properties)),
                         OVPropertiesTests::getTestCaseName);
}  // namespace
