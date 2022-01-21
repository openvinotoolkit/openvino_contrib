// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/eltwise.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ov::test::subgraph;

namespace {
std::vector<std::vector<ov::Shape>> inShapes = {
        {{2}},
        {{2, 200}},
        {{10, 200}},
        {{1, 10, 100}},
        {{4, 4, 16}},
        {{1, 1, 1, 3}},
        {{1, 2, 4}},
        {{1, 4, 4}},
        {{1, 4, 4, 1}},
};

std::vector<ov::test::ElementType> netPrecisions = {
        ov::element::f32,
        ov::element::f16,
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
};

std::vector<CommonTestUtils::OpType> opTypes = {
        CommonTestUtils::OpType::SCALAR,
        CommonTestUtils::OpType::VECTOR,
};

std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypes = {
        ngraph::helpers::EltwiseTypes::MULTIPLY,
        ngraph::helpers::EltwiseTypes::SUBTRACT,
        ngraph::helpers::EltwiseTypes::ADD,
        ngraph::helpers::EltwiseTypes::POWER,
        ngraph::helpers::EltwiseTypes::SQUARED_DIFF,
        ngraph::helpers::EltwiseTypes::DIVIDE,
        ngraph::helpers::EltwiseTypes::FLOOR_MOD,
        ngraph::helpers::EltwiseTypes::MOD,
};

ov::AnyMap additional_config = {};

const auto multiply_params = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)),
        ::testing::ValuesIn(eltwiseOpTypes),
        ::testing::ValuesIn(secondaryInputTypes),
        ::testing::ValuesIn(opTypes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(smoke_Eltwise, EltwiseLayerTest, multiply_params, EltwiseLayerTest::getTestCaseName);
}  // namespace