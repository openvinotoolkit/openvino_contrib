// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/get_metric.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov::test::behavior;

using namespace InferenceEngine::PluginConfigParams;

namespace {
//
// IE Class Common tests with <pluginName, deviceName params>
//



INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassImportExportTestP, CompiledModelImportExportTestP,
        ::testing::Values("HETERO:CPU"));

//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        smoke_CompiledModelGetMetricTest, CompiledModelGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("CPU", "MULTI:CPU", "HETERO:CPU", "AUTO:CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_CompiledModelGetMetricTest, CompiledModelGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("CPU", "MULTI:CPU", "HETERO:CPU", "AUTO:CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_CompiledModelGetMetricTest, CompiledModelGetMetricTest_NETWORK_NAME,
        ::testing::Values("CPU", "MULTI:CPU", "HETERO:CPU", "AUTO:CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_CompiledModelGetMetricTest, CompiledModelGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
        ::testing::Values("CPU", "MULTI:CPU", "HETERO:CPU", "AUTO:CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_CompiledModelGetMetricTest, CompiledModelGetMetricTest_ThrowsUnsupported,
        ::testing::Values("CPU", "MULTI:CPU", "HETERO:CPU", "AUTO:CPU"));

//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_SUITE_P(
        smoke_CompiledModelPropertyTest, CompiledModelPropertyTest,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_CompiledModelSetConfigTest, CompiledModelSetConfigTest,
        ::testing::Values("CPU"));

//
// Hetero Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(
       smoke_OVClassHeteroExecutableNetworkGetMetricTest, OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
       ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
       smoke_OVClassHeteroExecutableNetworkGetMetricTest, OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
       ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
       smoke_OVClassHeteroExecutableNetworkGetMetricTest, OVClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME,
       ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
       smoke_OVClassHeteroExecutableNetworkGetMetricTest, OVClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK,
       ::testing::Values("CPU"));

} // namespace

