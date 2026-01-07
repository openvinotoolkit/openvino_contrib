// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifcorer: Apache-2.0
//

#include "behavior/ov_plugin/properties_tests.hpp"
#include "behavior/ov_plugin/query_model.hpp"

#include <cuda_test_constants.hpp>
#include <string>
#include <utility>
#include <vector>

using namespace ov::test::behavior;

namespace {

//
// OV Class Common tests with <pluginName, device_name params>
//

INSTANTIATE_TEST_SUITE_P(smoke_OVClassNetworkTestP,
                         OVClassModelTestP,
                         ::testing::Values(ov::test::utils::DEVICE_NVIDIA));

//
// OV Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(smoke_OVClassGetMetricTest,
                         OVGetMetricPropsTest,
                         ::testing::Values(ov::test::utils::DEVICE_NVIDIA));

INSTANTIATE_TEST_SUITE_P(smoke_OVBasicPropertiesTestsP,
                         OVBasicPropertiesTestsP,
                         ::testing::Values(std::make_pair("openvino_nvidia_gpu_plugin",
                                                          ov::test::utils::DEVICE_NVIDIA)));

#ifdef PROXY_PLUGIN_ENABLED
INSTANTIATE_TEST_SUITE_P(smoke_OVClassGetAvailableDevices,
                         OVGetAvailableDevicesPropsTest,
                         ::testing::Values(ov::test::utils::DEVICE_GPU));
#else
INSTANTIATE_TEST_SUITE_P(smoke_OVClassGetAvailableDevices,
                         OVGetAvailableDevicesPropsTest,
                         ::testing::Values(ov::test::utils::DEVICE_NVIDIA));
#endif

//
// OV Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(smoke_OVClassGetConfigTest,
                         OVPropertiesDefaultSupportedTests,
                         ::testing::Values(ov::test::utils::DEVICE_NVIDIA));

TEST(OVClassBasicPropsTestP, smoke_CUDAGetSetConfigNoThrow) {
    ov::Core core = createCoreWithTemplate();

    auto device_name = ov::test::utils::DEVICE_NVIDIA;

    for (auto&& property : core.get_property(device_name, ov::supported_properties)) {
        if (ov::device::id == property) {
            std::cout << ov::device::id.name() << " : " << core.get_property(device_name, ov::device::id) << std::endl;
        } else if (ov::enable_profiling == property) {
            std::cout << ov::enable_profiling.name() << " : " << core.get_property(device_name, ov::enable_profiling)
                      << std::endl;
        } else if (ov::hint::performance_mode == property) {
            std::cout << "Default " << ov::hint::performance_mode.name() << " : "
                      << core.get_property(device_name, ov::hint::performance_mode) << std::endl;
            core.set_property(device_name, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
            ASSERT_EQ(ov::hint::PerformanceMode::THROUGHPUT,
                      core.get_property(device_name, ov::hint::performance_mode));
        }
    }
}

// OV Class Query network
INSTANTIATE_TEST_SUITE_P(smoke_OVClassQueryModelTest,
                         OVClassQueryModelTest,
                         ::testing::Values(ov::test::utils::DEVICE_NVIDIA));

}  // namespace
