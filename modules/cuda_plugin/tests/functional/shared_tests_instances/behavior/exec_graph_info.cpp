// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <cuda_test_constants.hpp>

// TODO remove when GTest ASSERT_NE(nullptr, ptr) macro will be fixed
#if defined(_WIN32)
#include "fix_win32_gtest_assert_ne_macro.hpp"
#endif
#include "behavior/exec_graph_info.hpp"

using namespace BehaviorTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {}
};

INSTANTIATE_TEST_CASE_P(DISABLED_smoke_BehaviorTests, ExecGraphTests,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                                ::testing::ValuesIn(configs)),
                        ExecGraphTests::getTestCaseName);

}  // namespace