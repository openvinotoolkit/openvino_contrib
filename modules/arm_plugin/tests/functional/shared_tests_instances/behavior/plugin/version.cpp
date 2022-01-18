// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multi-device/multi_device_config.hpp"
#include "hetero/hetero_plugin_config.hpp"
#include "behavior/plugin/version.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, VersionTest,
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
                            VersionTest::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, VersionTest,
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                            VersionTest::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Hetero_BehaviorTests, VersionTest,
                                    ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                            VersionTest::getTestCaseName);


}  // namespace
