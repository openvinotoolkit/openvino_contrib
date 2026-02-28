// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/shape_of.hpp"

#include "common_test_utils/test_constants.hpp"
#include "cuda_test_constants.hpp"

namespace {
using namespace ov::test;
using namespace ov::test::utils;

const std::vector<ov::element::Type> model_precisions = {
    ov::element::f32,
    ov::element::f16,
    ov::element::i32,
};

const std::vector<std::vector<ov::Shape>> input_shapes_static = {
    {{1}},
    {{1, 2}},
    {{1, 2, 3}},
    {{1, 2, 3, 4}},
    {{1, 2, 3, 4, 5}},
    {{10, 20, 30}},
};

INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf_i64,
                         ShapeOfLayerTest,
                         ::testing::Combine(::testing::ValuesIn(model_precisions),
                                            ::testing::Values(ov::element::i64),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                                            ::testing::Values(DEVICE_NVIDIA)),
                         ShapeOfLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf_i32,
                         ShapeOfLayerTest,
                         ::testing::Combine(::testing::ValuesIn(model_precisions),
                                            ::testing::Values(ov::element::i32),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                                            ::testing::Values(DEVICE_NVIDIA)),
                         ShapeOfLayerTest::getTestCaseName);
}  // namespace
