// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/multithreading.hpp"

#include <cuda_test_constants.hpp>
#include <vector>

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> configs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferRequestMultithreadingTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                                            ::testing::ValuesIn(configs)),
                         OVInferRequestMultithreadingTests::getTestCaseName);

}  // namespace
