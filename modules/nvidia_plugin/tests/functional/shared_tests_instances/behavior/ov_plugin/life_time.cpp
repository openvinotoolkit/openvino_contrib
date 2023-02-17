// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/life_time.hpp"

#include <cuda_test_constants.hpp>

using namespace ov::test::behavior;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVLifeTimeTest,
                         ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                         OVLifeTimeTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVLifeTimeTestOnImportedNetwork,
                         ::testing::Values(CommonTestUtils::DEVICE_NVIDIA, "HETERO:NVIDIA"),
                         OVLifeTimeTestOnImportedNetwork::getTestCaseName);

}  // namespace
