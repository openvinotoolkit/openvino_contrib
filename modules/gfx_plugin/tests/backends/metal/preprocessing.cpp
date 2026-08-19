// clang-format off
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_tests_instances/test_utils.hpp"
#include "integration/test_constants.hpp"

#include "preprocessing/yuv_to_grey_tests.hpp"

namespace ov::preprocess {

INSTANTIATE_TEST_SUITE_P(smoke_Preprocessing,
                         PreprocessingYUV2GreyTest,
                         testing::Values(ov::test::utils::DEVICE_GFX),
                         PreprocessingYUV2GreyTest::getTestCaseName);

}  // namespace ov::preprocess
