// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/caching_tests.hpp"

#include <cuda_test_constants.hpp>

using namespace ov::test::behavior;

namespace {
static const std::vector<ov::element::Type> precisionsTemplate = {
    ov::element::f32,
};

static const std::vector<std::size_t> batchSizesTemplate = {1, 2};

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_CachingSupportCase_Template,
                         CompileModelCacheTestBase,
                         ::testing::Combine(::testing::ValuesIn(CompileModelCacheTestBase::getStandardFunctions()),
                                            ::testing::ValuesIn(precisionsTemplate),
                                            ::testing::ValuesIn(batchSizesTemplate),
                                            ::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                                            ::testing::Values(ov::AnyMap())),
                         CompileModelCacheTestBase::getTestCaseName);
}  // namespace
