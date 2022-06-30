// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <string>
#include <vector>
#include <cuda_test_constants.hpp>

#include "behavior/executable_network/get_metric.hpp"

using namespace BehaviorTestsDefinitions;

namespace {
//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values(CommonTestUtils::DEVICE_CUDA, "MULTI:CUDA", "HETERO:CUDA"));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values(CommonTestUtils::DEVICE_CUDA, "MULTI:CUDA", "HETERO:CUDA"));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values(CommonTestUtils::DEVICE_CUDA, "MULTI:CUDA", "HETERO:CUDA"));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
        ::testing::Values(CommonTestUtils::DEVICE_CUDA, "MULTI:CUDA", "HETERO:CUDA"));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported, IEClassExecutableNetworkGetMetricTest,
        ::testing::Values(CommonTestUtils::DEVICE_CUDA, "MULTI:CUDA", "HETERO:CUDA"));
//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetConfigTest, IEClassExecutableNetworkGetConfigTest,
        ::testing::Values(CommonTestUtils::DEVICE_CUDA));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkSetConfigTest, IEClassExecutableNetworkSetConfigTest,
        ::testing::Values(CommonTestUtils::DEVICE_CUDA));

//
// Hetero Executable Network GetMetric
//

#ifdef ENABLE_INTEL_CPU

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values(CommonTestUtils::DEVICE_CUDA));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values(CommonTestUtils::DEVICE_CUDA));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values(CommonTestUtils::DEVICE_CUDA));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK,
        ::testing::Values(CommonTestUtils::DEVICE_CUDA));

#endif  // ENABLE_INTEL_CPU
} // namespace
