// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <cuda_test_constants.hpp>

// TODO remove when GTest ASSERT_NE(nullptr, ptr) macro will be fixed
#if defined(_WIN32)
#include "fix_win32_gtest_assert_ne_macro.hpp"
#endif

#include "ie_precision.hpp"
#include "behavior/infer_request/callback.hpp"

using namespace BehaviorTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestCallbackTests,
                         ::testing::Combine(
                             ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                             ::testing::ValuesIn(configs)),
                         InferRequestCallbackTests::getTestCaseName);
}  // namespace
