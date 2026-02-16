// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/eltwise.hpp"

#include <vector>

#include "cuda_test_constants.hpp"

namespace {

using namespace ov::test;
using namespace ov::test::utils;

std::vector<std::vector<ov::Shape>> bitwise_unary_shapes_static = {
    {{256}},
    {{1, 50}},
    {{2, 3, 4}},
    {{2, 17, 3, 4}},
};

std::vector<ov::element::Type> bitwise_integer_and_bool_types = {
    ov::element::i32,
    ov::element::boolean,
};

std::vector<std::vector<ov::Shape>> bitwise_input_shapes_static = {
    {{256}, {256}},
    {{256}, {1}},
    {{1, 50}, {1, 50}},
    {{1, 50}, {50}},
    {{2, 3, 4}, {2, 3, 4}},
    {{2, 3, 4}, {4}},
    {{2, 3, 4}, {1, 3, 4}},
};

std::vector<InputLayerType> secondary_input_types = {
    InputLayerType::CONSTANT,
    InputLayerType::PARAMETER,
};

// ---- BitwiseAnd ----
const auto bitwise_and_params = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bitwise_input_shapes_static)),
    ::testing::Values(EltwiseTypes::BITWISE_AND),
    ::testing::ValuesIn(secondary_input_types),
    ::testing::Values(OpType::VECTOR),
    ::testing::ValuesIn(bitwise_integer_and_bool_types),
    ::testing::Values(ov::element::Type_t::dynamic),
    ::testing::Values(ov::element::Type_t::dynamic),
    ::testing::Values(DEVICE_NVIDIA),
    ::testing::Values(ov::AnyMap()));

INSTANTIATE_TEST_SUITE_P(smoke_BitwiseAnd,
                         EltwiseLayerTest,
                         bitwise_and_params,
                         EltwiseLayerTest::getTestCaseName);

// ---- BitwiseNot ----
const auto bitwise_not_params = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bitwise_unary_shapes_static)),
    ::testing::Values(EltwiseTypes::BITWISE_NOT),
    ::testing::Values(InputLayerType::CONSTANT),
    ::testing::Values(OpType::VECTOR),
    ::testing::ValuesIn(bitwise_integer_and_bool_types),
    ::testing::Values(ov::element::Type_t::dynamic),
    ::testing::Values(ov::element::Type_t::dynamic),
    ::testing::Values(DEVICE_NVIDIA),
    ::testing::Values(ov::AnyMap()));

INSTANTIATE_TEST_SUITE_P(smoke_BitwiseNot,
                         EltwiseLayerTest,
                         bitwise_not_params,
                         EltwiseLayerTest::getTestCaseName);

}  // namespace
