// Copyright (C) 2019-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/concat.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "cuda_test_constants.hpp"

namespace {

using namespace ov::test;
using namespace ov::test::utils;

std::vector<int> axes = {-3, -2, -1, 0, 1, 2, 3};
std::vector<std::vector<ov::Shape>> shapes_static = {
    {{10, 10, 10, 10}},
    {{10, 10, 10, 10}, {10, 10, 10, 10}},
    {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
    {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
    {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}}};

std::vector<ov::element::Type> model_precisions = {ov::element::f32,
                                                   ov::element::f16};

INSTANTIATE_TEST_CASE_P(smoke_NoReshape,
                        ConcatLayerTest,
                        ::testing::Combine(::testing::ValuesIn(axes),
                                           ::testing::ValuesIn(static_shapes_to_test_representation(shapes_static)),
                                           ::testing::ValuesIn(model_precisions),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        ConcatLayerTest::getTestCaseName);

}  // namespace
