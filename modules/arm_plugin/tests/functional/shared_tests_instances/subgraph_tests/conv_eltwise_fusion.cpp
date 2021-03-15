// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/conv_eltwise_fusion.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
const std::vector<ngraph::element::Type> types{
    ngraph::element::f32,
    ngraph::element::f16
};
#define MUL(X) std::tuple<ngraph::NodeTypeInfo, int64_t>(ngraph::opset4::Multiply::type_info, X)
#define ADD(X) std::tuple<ngraph::NodeTypeInfo, int64_t>(ngraph::opset4::Add::type_info, X)
#define IN std::vector<std::tuple<ngraph::NodeTypeInfo, int64_t>>

    const std::vector<ngraph::Shape> const_shapes_2d{
        {},
        {20, 1, 1},
        {1, 1, 1},
        {1, 1, 1, 1},
    };

    INSTANTIATE_TEST_CASE_P(smoke_Convolution_2D, ConvEltwiseFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::Convolution::type_info),
                                    ::testing::ValuesIn(IN({MUL(4), ADD(5)})),
                                    ::testing::Values(ngraph::Shape{1, 3, 24, 24}),
                                    ::testing::Values(ngraph::Shape{20, 3, 3, 3}),
                                    ::testing::ValuesIn(const_shapes_2d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvEltwiseFusion::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Convolution_2D_4ops, ConvEltwiseFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::Convolution::type_info),
                                    ::testing::ValuesIn(IN({MUL(6), ADD(6)})),
                                    ::testing::Values(ngraph::Shape{1, 3, 24, 24}),
                                    ::testing::Values(ngraph::Shape{20, 3, 3, 3}),
                                    ::testing::Values(ngraph::Shape{1, 20, 22, 22}),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvEltwiseFusion::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_GroupConvolution_2D, ConvEltwiseFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::GroupConvolution::type_info),
                                    ::testing::ValuesIn(IN({MUL(4), ADD(5)})),
                                    ::testing::Values(ngraph::Shape{1, 12, 24, 24}),
                                    ::testing::Values(ngraph::Shape{4, 5, 3, 5, 5}),
                                    ::testing::ValuesIn(const_shapes_2d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvEltwiseFusion::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_DepthwiseConvolution_2D, ConvEltwiseFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::GroupConvolution::type_info),
                                    ::testing::ValuesIn(IN({MUL(4), ADD(5)})),
                                    ::testing::Values(ngraph::Shape{1, 20, 24, 24}),
                                    ::testing::Values(ngraph::Shape{20, 1, 1, 3, 3}),
                                    ::testing::ValuesIn(const_shapes_2d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvEltwiseFusion::getTestCaseName);

    const std::vector<ngraph::Shape> neg_const_shapes_2d{
        {1, 1, 1, 1, 1}, /* Broadcast output */
        {3},
        {3, 1},
        {3, 1, 1, 1},
        {1, 3},
        {1, 1, 3},

        {1, 3, 1}, // fused
    };

    INSTANTIATE_TEST_CASE_P(smoke_Convolution_2D_Negative, ConvEltwiseFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::Convolution::type_info),
                                    ::testing::ValuesIn(IN({MUL(6), ADD(6)})),
                                    ::testing::Values(ngraph::Shape{1, 3, 1, 1}),
                                    ::testing::Values(ngraph::Shape{3, 3, 1, 1}),
                                    ::testing::ValuesIn(neg_const_shapes_2d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvEltwiseFusion::getTestCaseName);

    const std::vector<ngraph::Shape> fused_neg_const_shapes_2d{
        {3, 1, 1}, // fused
        {1, 3, 1, 1}, // fused
    };

    INSTANTIATE_TEST_CASE_P(smoke_Convolution_2D_Negative_5ops, ConvEltwiseFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::Convolution::type_info),
                                    ::testing::ValuesIn(IN({ADD(5)})),
                                    ::testing::Values(ngraph::Shape{1, 3, 1, 1}),
                                    ::testing::Values(ngraph::Shape{3, 3, 1, 1}),
                                    ::testing::ValuesIn(fused_neg_const_shapes_2d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvEltwiseFusion::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_GroupConvolution_2D_Negative, ConvEltwiseFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::GroupConvolution::type_info),
                                    ::testing::ValuesIn(IN({MUL(6), ADD(6)})),
                                    ::testing::Values(ngraph::Shape{1, 12, 3, 3}),
                                    ::testing::Values(ngraph::Shape{4, 5, 3, 1, 1}),
                                    ::testing::ValuesIn(neg_const_shapes_2d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvEltwiseFusion::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_DepthwiseConvolution_2D_Negative, ConvEltwiseFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::GroupConvolution::type_info),
                                    ::testing::ValuesIn(IN({MUL(6), ADD(6)})),
                                    ::testing::Values(ngraph::Shape{1, 3, 3, 3}),
                                    ::testing::Values(ngraph::Shape{3, 1, 1, 1, 1}),
                                    ::testing::ValuesIn(neg_const_shapes_2d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvEltwiseFusion::getTestCaseName);
}  // namespace