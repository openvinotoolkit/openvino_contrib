// clang-format off
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.hpp"
#include "integration/test_constants.hpp"

#include "preprocessing/yuv_to_grey_tests.hpp"

using GfxPreprocessingYUV2GreyTest = ov::test::utils::GfxSkippedTests<ov::preprocess::PreprocessingYUV2GreyTest>;

INSTANTIATE_TEST_SUITE_P(smoke_Preprocessing,
                         GfxPreprocessingYUV2GreyTest,
                         testing::Values(ov::test::utils::DEVICE_GFX),
                         ov::preprocess::PreprocessingYUV2GreyTest::getTestCaseName);
