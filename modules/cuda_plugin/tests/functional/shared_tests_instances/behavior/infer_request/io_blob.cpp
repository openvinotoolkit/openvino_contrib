// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_test_constants.hpp>

#include "multi-device/multi_device_config.hpp"
#include "behavior/infer_request/io_blob.hpp"

using namespace BehaviorTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestIOBBlobTest,
                         ::testing::Combine(
                         ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                         ::testing::ValuesIn(configs)),
                         InferRequestIOBBlobTest::getTestCaseName);

}  // namespace
