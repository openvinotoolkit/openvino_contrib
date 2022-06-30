// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <cuda_test_constants.hpp>

#include "behavior/ov_infer_request/inference.hpp"

namespace {

using namespace ov::test::behavior;
using namespace ov;

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestInferenceTests,
                         ::testing::Combine(
                         ::testing::Values(tensor_roi::roi_nchw(), tensor_roi::roi_1d()),
                         ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                         OVInferRequestInferenceTests::getTestCaseName);

}  // namespace
