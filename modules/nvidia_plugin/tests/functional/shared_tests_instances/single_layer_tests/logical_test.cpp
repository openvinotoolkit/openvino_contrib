// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <vector>

#include "cuda_test_constants.hpp"
#include "single_op_tests/logical.hpp"
namespace {

using namespace ov::test;
using namespace ov::test::utils;
using ov::test::LogicalLayerTest;

std::map<ov::Shape, std::vector<ov::Shape>> input_shapes_not = {
    {{256}, {}},
    {{50, 200}, {}},
    {{1, 3, 20}, {}},
    {{2, 17, 3, 4}, {}},
    {{2, 3, 25, 4, 13}, {}},
};

std::vector<ov::element::Type> model_types = {
    ov::element::boolean,
};

std::map<std::string, std::string> additional_config = {};

std::vector<std::vector<ov::Shape>> combine_shapes(const std::map<ov::Shape, std::vector<ov::Shape>>& input_shapes_static) {
    std::vector<std::vector<ov::Shape>> result;
    for (const auto& input_shape : input_shapes_static) {
        for (auto& item : input_shape.second) {
            result.push_back({input_shape.first, item});
        }

        if (input_shape.second.empty()) {
            result.push_back({input_shape.first, {}});
        }
    }
    return result;
}

const auto LogicalTestParamsNot =
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(combine_shapes(input_shapes_not))),
                       ::testing::Values(LogicalTypes::LOGICAL_NOT),
                       ::testing::Values(InputLayerType::CONSTANT),
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(DEVICE_NVIDIA),
                       ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(smoke_LogicalNot,
                        LogicalLayerTest,
                        LogicalTestParamsNot,
                        LogicalLayerTest::getTestCaseName);

std::map<ov::Shape, std::vector<ov::Shape>> input_shapes_binary = {
    {{256}, {{256}}},
    {{256}, {{1}}},
    {{50, 200}, {{50, 200}}},
    {{50, 200}, {{200}}},
    {{1, 3, 20}, {{1, 3, 20}}},
    {{1, 3, 20}, {{20}}},
    {{2, 17, 3, 4}, {{4}}},
    {{2, 17, 3, 4}, {{1, 3, 4}}},
};

std::vector<InputLayerType> second_input_types_binary = {
    InputLayerType::CONSTANT,
    InputLayerType::PARAMETER,
};

const auto LogicalTestParamsAnd =
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(combine_shapes(input_shapes_binary))),
                       ::testing::Values(LogicalTypes::LOGICAL_AND),
                       ::testing::ValuesIn(second_input_types_binary),
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(DEVICE_NVIDIA),
                       ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(smoke_LogicalAnd,
                        LogicalLayerTest,
                        LogicalTestParamsAnd,
                        LogicalLayerTest::getTestCaseName);

}  // namespace
