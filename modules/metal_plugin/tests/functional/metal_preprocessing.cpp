// clang-format off
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "metal_test_utils.hpp"
#include "test_constants.hpp"

#include "preprocessing/yuv_to_grey_tests.hpp"

using MetalPreprocessingYUV2GreyTest = ov::test::utils::MetalSkippedTests<ov::preprocess::PreprocessingYUV2GreyTest>;

INSTANTIATE_TEST_SUITE_P(smoke_Preprocessing,
                         MetalPreprocessingYUV2GreyTest,
                         testing::Values(ov::test::utils::DEVICE_METAL),
                         ov::preprocess::PreprocessingYUV2GreyTest::getTestCaseName);
