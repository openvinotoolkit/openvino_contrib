// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_test_constants.hpp>
#include "behavior/ov_infer_request/inference_chaining.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> configs = {
    {}
};

const std::vector<ov::AnyMap> HeteroConfigs = {
            {ov::device::priorities(CommonTestUtils::DEVICE_CUDA)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferenceChaining,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                                ::testing::ValuesIn(configs)),
                        OVInferenceChaining::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, OVInferenceChaining,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                ::testing::ValuesIn(HeteroConfigs)),
                        OVInferenceChaining::getTestCaseName);

}  // namespace
