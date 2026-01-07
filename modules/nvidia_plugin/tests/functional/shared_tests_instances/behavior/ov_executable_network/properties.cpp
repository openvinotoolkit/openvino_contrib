// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/properties.hpp"

#include <cuda_test_constants.hpp>

#include "nvidia/properties.hpp"
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
    {{ov::device::priorities(std::string(ov::test::utils::DEVICE_NVIDIA) + "(4)")},
     {ov::auto_batch_timeout(-1)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVClassCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                                            ::testing::ValuesIn(inproperties)),
                         OVClassCompiledModelPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVClassCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(hetero_inproperties)),
                         OVClassCompiledModelPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         OVClassCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multi_inproperties)),
                         OVClassCompiledModelPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         OVClassCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(auto_inproperties)),
                         OVClassCompiledModelPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         OVClassCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_inproperties)),
                         OVClassCompiledModelPropertiesIncorrectTests::getTestCaseName);

const std::vector<ov::AnyMap> default_properties = {{ov::num_streams(1)},
                                                    {ov::hint::num_requests(0)},
                                                    {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
                                                    {ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE)},
                                                    {ov::enable_profiling(false)},
                                                    {ov::device::id("0")},
                                                    {ov::nvidia_gpu::operation_benchmark(false)},
                                                    {ov::nvidia_gpu::use_cuda_graph(true)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompiledModelPropertiesDefaultSupportedTests,
                         ::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                         OVCompiledModelPropertiesDefaultSupportedTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVClassCompiledModelPropertiesDefaultTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                                            ::testing::ValuesIn(default_properties)),
                         OVClassCompiledModelPropertiesDefaultTests::getTestCaseName);

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
    {ov::device::id("0")},
    {ov::device::id("NVIDIA.0")},
};

const std::vector<ov::AnyMap> hetero_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_NVIDIA), ov::enable_profiling(true)},
    {ov::device::priorities(ov::test::utils::DEVICE_NVIDIA), ov::device::id("0")},
};

const std::vector<ov::AnyMap> multi_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_NVIDIA), ov::enable_profiling(true)},
    {ov::device::priorities(ov::test::utils::DEVICE_NVIDIA), ov::device::id("0")},
};

const std::vector<ov::AnyMap> auto_batch_properties = {
    {ov::device::priorities(std::string(ov::test::utils::DEVICE_NVIDIA) + "(4)")},
    {{ov::device::priorities(std::string(ov::test::utils::DEVICE_NVIDIA) + "(4)")},
     {ov::auto_batch_timeout(1)}},
    {{ov::device::priorities(std::string(ov::test::utils::DEVICE_NVIDIA) + "(4)")},
     {ov::auto_batch_timeout(10)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVClassCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                                            ::testing::ValuesIn(properties)),
                         OVClassCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVClassCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(hetero_properties)),
                         OVClassCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         OVClassCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multi_properties)),
                         OVClassCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         OVClassCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_properties)),
                         OVClassCompiledModelPropertiesTests::getTestCaseName);
}  // namespace