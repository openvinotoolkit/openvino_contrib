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

// Note: InType and OutType must use concrete types (not ov::element::Type_t::dynamic)
// because "dynamic" in the test name matches the skip pattern in skip_tests_config.cpp:
//   std::regex(R"(.*(d|D)ynamic*.*)")
// which is intended for dynamic shapes but also catches InType=dynamic/OutType=dynamic.

#define BITWISE_BINARY_TESTS(OpEnum, Prefix, Type, TypeName)                                                     \
    INSTANTIATE_TEST_SUITE_P(                                                                                     \
        smoke_##Prefix##_##TypeName,                                                                              \
        EltwiseLayerTest,                                                                                         \
        ::testing::Combine(                                                                                       \
            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bitwise_input_shapes_static)),      \
            ::testing::Values(EltwiseTypes::OpEnum),                                                               \
            ::testing::ValuesIn(secondary_input_types),                                                            \
            ::testing::Values(OpType::VECTOR),                                                                     \
            ::testing::Values(Type),                                                                               \
            ::testing::Values(Type),                                                                               \
            ::testing::Values(Type),                                                                               \
            ::testing::Values(std::string(DEVICE_NVIDIA)),                                                         \
            ::testing::Values(ov::AnyMap())),                                                                      \
        EltwiseLayerTest::getTestCaseName)

#define BITWISE_UNARY_TESTS(OpEnum, Prefix, Type, TypeName)                                                      \
    INSTANTIATE_TEST_SUITE_P(                                                                                     \
        smoke_##Prefix##_##TypeName,                                                                              \
        EltwiseLayerTest,                                                                                         \
        ::testing::Combine(                                                                                       \
            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bitwise_unary_shapes_static)),      \
            ::testing::Values(EltwiseTypes::OpEnum),                                                               \
            ::testing::Values(InputLayerType::CONSTANT),                                                           \
            ::testing::Values(OpType::VECTOR),                                                                     \
            ::testing::Values(Type),                                                                               \
            ::testing::Values(Type),                                                                               \
            ::testing::Values(Type),                                                                               \
            ::testing::Values(std::string(DEVICE_NVIDIA)),                                                         \
            ::testing::Values(ov::AnyMap())),                                                                      \
        EltwiseLayerTest::getTestCaseName)

// ---- BitwiseAnd ----
BITWISE_BINARY_TESTS(BITWISE_AND, BitwiseAnd, ov::element::Type_t::i32, i32);
BITWISE_BINARY_TESTS(BITWISE_AND, BitwiseAnd, ov::element::Type_t::boolean, bool);

// ---- BitwiseOr ----
BITWISE_BINARY_TESTS(BITWISE_OR, BitwiseOr, ov::element::Type_t::i32, i32);
BITWISE_BINARY_TESTS(BITWISE_OR, BitwiseOr, ov::element::Type_t::boolean, bool);

// ---- BitwiseXor ----
BITWISE_BINARY_TESTS(BITWISE_XOR, BitwiseXor, ov::element::Type_t::i32, i32);
BITWISE_BINARY_TESTS(BITWISE_XOR, BitwiseXor, ov::element::Type_t::boolean, bool);

// ---- BitwiseNot ----
BITWISE_UNARY_TESTS(BITWISE_NOT, BitwiseNot, ov::element::Type_t::i32, i32);
BITWISE_UNARY_TESTS(BITWISE_NOT, BitwiseNot, ov::element::Type_t::boolean, bool);

// ---- BitwiseLeftShift ----
BITWISE_BINARY_TESTS(LEFT_SHIFT, BitwiseLeftShift, ov::element::Type_t::i32, i32);
BITWISE_BINARY_TESTS(LEFT_SHIFT, BitwiseLeftShift, ov::element::Type_t::i16, i16);
BITWISE_BINARY_TESTS(LEFT_SHIFT, BitwiseLeftShift, ov::element::Type_t::u32, u32);

// ---- BitwiseRightShift ----
BITWISE_BINARY_TESTS(RIGHT_SHIFT, BitwiseRightShift, ov::element::Type_t::i32, i32);
BITWISE_BINARY_TESTS(RIGHT_SHIFT, BitwiseRightShift, ov::element::Type_t::i16, i16);
BITWISE_BINARY_TESTS(RIGHT_SHIFT, BitwiseRightShift, ov::element::Type_t::u32, u32);

}  // namespace
