// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/split.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "../../test_constants.hpp"
#include "../../metal_test_utils.hpp"

using ov::test::SplitLayerTest;

using MetalSplitLayerTest = ov::test::utils::MetalVsTemplateLayerTest<SplitLayerTest>;

TEST_P(MetalSplitLayerTest, CompareWithTemplate) {
    run_compare();
}

namespace {

INSTANTIATE_TEST_SUITE_P(
    Metal_smoke_NumSplitsCheck,
    MetalSplitLayerTest,
    ::testing::Combine(::testing::Values(1, 2, 3, 5, 6, 10, 30),
                       ::testing::Values(0, 1, 2, 3),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(ov::test::static_shapes_to_test_representation({{30, 30, 30, 30}})),
                       ::testing::Values(std::vector<size_t>({})),
                       ::testing::Values(ov::test::utils::DEVICE_METAL)),
    SplitLayerTest::getTestCaseName);

}  // namespace
