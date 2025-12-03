// clang-format off
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "metal_test_utils.hpp"
#include "test_constants.hpp"

#include <cstddef>
#include <vector>

#include "single_op_tests/convolution.hpp"
#include "single_op_tests/eltwise.hpp"
#include "single_op_tests/reshape.hpp"
#include "single_op_tests/softmax.hpp"
#include "single_op_tests/split.hpp"

using ov::test::ConvolutionLayerTest;
using ov::test::EltwiseLayerTest;
using ov::test::ReshapeLayerTest;
using ov::test::subgraph::SoftMax8LayerTest;
using ov::test::SplitLayerTest;

// Wrap Convolution to keep the METAL skip guard while registering its own gtest
// suite (the base header only defines TEST_P for ConvolutionLayerTest).
class MetalConvolutionLayerTest : public ov::test::utils::MetalSkippedTests<ConvolutionLayerTest> {};
TEST_P(MetalConvolutionLayerTest, Inference) {
    run();
}

using MetalEltwiseLayerTest = ov::test::utils::MetalSkippedTests<EltwiseLayerTest>;
using MetalReshapeLayerTest = ov::test::utils::MetalSkippedTests<ReshapeLayerTest>;
using MetalSoftmaxLayerTest = ov::test::utils::MetalSkippedTests<SoftMax8LayerTest>;
using MetalSoftMax8LayerTest = ov::test::utils::MetalSkippedTests<SoftMax8LayerTest>;
using MetalSplitLayerTest = ov::test::utils::MetalSkippedTests<SplitLayerTest>;

namespace {

// Convolution (mirror TEMPLATE coverage; groups are implicit =1 in shared fixture)
const std::vector<ov::element::Type> model_types = {
    ov::element::f32,
    ov::element::f16,
};

const std::vector<std::vector<size_t>> kernels = {{3, 3}, {3, 5}};
const std::vector<std::vector<size_t>> strides = {{1, 1}, {1, 3}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0}, {0, 3}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0}, {0, 3}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}, {3, 1}};
const std::vector<size_t> numOutChannels = {1, 5};

const auto conv2DParams_ExplicitPadding = ::testing::Combine(::testing::ValuesIn(kernels),
                                                             ::testing::ValuesIn(strides),
                                                             ::testing::ValuesIn(padBegins),
                                                             ::testing::ValuesIn(padEnds),
                                                             ::testing::ValuesIn(dilations),
                                                             ::testing::ValuesIn(numOutChannels),
                                                             ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv2DParams_AutoPadValid = ::testing::Combine(::testing::ValuesIn(kernels),
                                                          ::testing::ValuesIn(strides),
                                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                                                          ::testing::ValuesIn(dilations),
                                                          ::testing::ValuesIn(numOutChannels),
                                                          ::testing::Values(ov::op::PadType::VALID));

INSTANTIATE_TEST_SUITE_P(
    Convolution2D_ExplicitPadding,
    MetalConvolutionLayerTest,
    ::testing::Combine(conv2DParams_ExplicitPadding,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation({{1, 3, 30, 30}})),
                       ::testing::Values(ov::test::utils::DEVICE_METAL)),
    ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    Convolution2D_AutoPadValid,
    MetalConvolutionLayerTest,
    ::testing::Combine(conv2DParams_AutoPadValid,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation({{1, 3, 30, 30}})),
                       ::testing::Values(ov::test::utils::DEVICE_METAL)),
    ConvolutionLayerTest::getTestCaseName);

// Reshape
const std::vector<ov::element::Type> reshape_types = {ov::element::f32};

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeCheckDynBatch,
                         MetalReshapeLayerTest,
                         ::testing::Combine(::testing::Values(true),
                                            ::testing::ValuesIn(reshape_types),
                                            ::testing::Values(std::vector<size_t>({30, 30, 30, 30})),
                                            ::testing::Values(std::vector<int64_t>({30, 30, 30, 30})),
                                            ::testing::Values(ov::test::utils::DEVICE_METAL)),
                         ReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeCheck,
                         MetalReshapeLayerTest,
                         ::testing::Combine(::testing::Values(true),
                                            ::testing::ValuesIn(reshape_types),
                                            ::testing::Values(std::vector<size_t>({10, 10, 10, 10})),
                                            ::testing::Values(std::vector<int64_t>({10, 0, 100})),
                                            ::testing::Values(ov::test::utils::DEVICE_METAL)),
                         ReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeCheckNegative,
                         MetalReshapeLayerTest,
                         ::testing::Combine(::testing::Values(true),
                                            ::testing::ValuesIn(reshape_types),
                                            ::testing::Values(std::vector<size_t>({10, 10, 10, 10})),
                                            ::testing::Values(std::vector<int64_t>({10, -1, 100})),
                                            ::testing::Values(ov::test::utils::DEVICE_METAL)),
                         ReshapeLayerTest::getTestCaseName);

// Split
INSTANTIATE_TEST_SUITE_P(
    smoke_NumSplitsCheck,
    MetalSplitLayerTest,
    ::testing::Combine(::testing::Values(1, 2, 3, 5, 6, 10, 30),
                       ::testing::Values(0, 1, 2, 3),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(ov::test::static_shapes_to_test_representation({{30, 30, 30, 30}})),
                       ::testing::Values(std::vector<size_t>({})),
                       ::testing::Values(ov::test::utils::DEVICE_METAL)),
    SplitLayerTest::getTestCaseName);

// Eltwise
std::vector<std::vector<ov::Shape>> inShapesStatic = {
    {{2}},
    {{2, 200}},
    {{10, 200}},
    {{1, 10, 100}},
    {{4, 4, 16}},
    {{1, 1, 1, 3}},
    {{2, 17, 5, 4}, {1, 17, 1, 1}},
    {{2, 17, 5, 1}, {1, 17, 1, 4}},
    {{1, 2, 4}},
    {{1, 4, 4}},
    {{1, 4, 4, 1}},
    {{1, 1, 1, 1, 1, 1, 3}},
    {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
};

std::vector<std::vector<ov::test::InputShape>> inShapesDynamic = {
    {{{ov::Dimension(1, 10), 200}, {{2, 200}, {1, 200}}}, {{ov::Dimension(1, 10), 200}, {{2, 200}, {5, 200}}}},
};

std::vector<ov::test::ElementType> netPrecisions = {
    ov::element::f32,
    ov::element::f16,
    ov::element::i32,
};

std::vector<ov::test::utils::InputLayerType> secondaryInputTypes = {
    ov::test::utils::InputLayerType::CONSTANT,
    ov::test::utils::InputLayerType::PARAMETER,
};

std::vector<ov::test::utils::InputLayerType> secondaryInputTypesDynamic = {
    ov::test::utils::InputLayerType::PARAMETER,
};

std::vector<ov::test::utils::OpType> opTypes = {
    ov::test::utils::OpType::SCALAR,
    ov::test::utils::OpType::VECTOR,
};

std::vector<ov::test::utils::OpType> opTypesDynamic = {
    ov::test::utils::OpType::VECTOR,
};

std::vector<ov::test::utils::EltwiseTypes> eltwiseOpTypes = {ov::test::utils::EltwiseTypes::ADD,
                                                             ov::test::utils::EltwiseTypes::MULTIPLY,
                                                             ov::test::utils::EltwiseTypes::SUBTRACT,
                                                             ov::test::utils::EltwiseTypes::DIVIDE,
                                                             ov::test::utils::EltwiseTypes::FLOOR_MOD,
                                                             ov::test::utils::EltwiseTypes::SQUARED_DIFF,
                                                             ov::test::utils::EltwiseTypes::POWER,
                                                             ov::test::utils::EltwiseTypes::MOD};

std::vector<ov::test::utils::EltwiseTypes> eltwiseOpTypesDynamic = {
    ov::test::utils::EltwiseTypes::ADD,
    ov::test::utils::EltwiseTypes::MULTIPLY,
    ov::test::utils::EltwiseTypes::SUBTRACT,
};

ov::test::Config additional_config = {};

const auto multiply_params =
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesStatic)),
                       ::testing::ValuesIn(eltwiseOpTypes),
                       ::testing::ValuesIn(secondaryInputTypes),
                       ::testing::ValuesIn(opTypes),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::test::utils::DEVICE_METAL),
                       ::testing::Values(additional_config));

const auto multiply_params_dynamic =
    ::testing::Combine(::testing::ValuesIn(inShapesDynamic),
                       ::testing::ValuesIn(eltwiseOpTypesDynamic),
                       ::testing::ValuesIn(secondaryInputTypesDynamic),
                       ::testing::ValuesIn(opTypesDynamic),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::test::utils::DEVICE_METAL),
                       ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_static,
                         MetalEltwiseLayerTest,
                         multiply_params,
                         EltwiseLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_dynamic,
                         MetalEltwiseLayerTest,
                         multiply_params_dynamic,
                         EltwiseLayerTest::getTestCaseName);

std::vector<std::vector<ov::Shape>> inShapesSingleThread = {
    {{1, 2, 3, 4}},
    {{2, 2, 2, 2}},
    {{2, 1, 2, 1, 2, 2}},
};

std::vector<ov::test::utils::EltwiseTypes> eltwiseOpTypesSingleThread = {
    ov::test::utils::EltwiseTypes::ADD,
    ov::test::utils::EltwiseTypes::POWER,
};

ov::AnyMap additional_config_single_thread = {ov::inference_num_threads(1)};

const auto single_thread_params =
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesSingleThread)),
                       ::testing::ValuesIn(eltwiseOpTypesSingleThread),
                       ::testing::ValuesIn(secondaryInputTypes),
                       ::testing::ValuesIn(opTypes),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::test::utils::DEVICE_METAL),
                       ::testing::Values(additional_config_single_thread));

INSTANTIATE_TEST_SUITE_P(smoke_SingleThread,
                         MetalEltwiseLayerTest,
                         single_thread_params,
                         EltwiseLayerTest::getTestCaseName);

// Softmax
const std::vector<ov::test::ElementType> netPrecisions2D = {
    ov::element::f32,
    ov::element::f16,
};

const std::vector<ov::Shape> inputStaticShape2D = {
    {1, 100},
    {100, 1},
    {10, 10},
};

const std::vector<ov::test::InputShape> inputDynamicShape2D = {
    {{ov::Dimension::dynamic(), 10}, {{1, 10}, {2, 10}, {10, 10}}},
    {{ov::Dimension(1, 10), 10}, {{1, 10}, {2, 10}, {10, 10}}},
    {{10, ov::Dimension::dynamic()}, {{10, 1}, {10, 5}, {10, 10}}},
    {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 10}, {2, 10}, {10, 10}}}};

const std::vector<int64_t> axis2D = {-2, -1, 0, 1};

const auto params2D_static =
    ::testing::Combine(::testing::ValuesIn(netPrecisions2D),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputStaticShape2D)),
                       ::testing::ValuesIn(axis2D),
                       ::testing::Values(ov::test::utils::DEVICE_METAL),
                       ::testing::Values(ov::AnyMap()));

const auto params2D_dynamic = ::testing::Combine(::testing::ValuesIn(netPrecisions2D),
                                                 ::testing::Values(ov::element::dynamic),
                                                 ::testing::Values(ov::element::dynamic),
                                                 ::testing::ValuesIn(inputDynamicShape2D),
                                                 ::testing::ValuesIn(axis2D),
                                                 ::testing::Values(ov::test::utils::DEVICE_METAL),
                                                 ::testing::Values(ov::AnyMap()));

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax2D_static,
                         MetalSoftmaxLayerTest,
                         params2D_static,
                         SoftMax8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax2D_dynamic,
                         MetalSoftmaxLayerTest,
                         params2D_dynamic,
                         SoftMax8LayerTest::getTestCaseName);

const std::vector<ov::Shape> inputStaticShape4D = {
    {1, 100, 1, 1},
    {50, 100, 4, 1},
    {2, 100, 10, 1},
};

const std::vector<ov::test::InputShape> inputDynamicShape4D = {
    {{ov::Dimension::dynamic(), 100, ov::Dimension(1, 10), 1}, {{1, 100, 1, 1}, {100, 100, 5, 1}}},
    {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
     {{1, 100, 1, 1}, {50, 100, 4, 1}, {2, 100, 10, 1}}},
};

const std::vector<ov::test::ElementType> netPrecisions4D = {
    ov::element::f32,
    ov::element::f16,
};

const std::vector<int64_t> axis4D = {0, 1, 2, 3, -1, -2, -3, -4};

const auto params4Dstatic =
    ::testing::Combine(::testing::ValuesIn(netPrecisions4D),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputStaticShape4D)),
                       ::testing::ValuesIn(axis4D),
                       ::testing::Values(ov::test::utils::DEVICE_METAL),
                       ::testing::Values(ov::AnyMap()));

const auto params4Ddynamic = ::testing::Combine(::testing::ValuesIn(netPrecisions4D),
                                                ::testing::Values(ov::element::dynamic),
                                                ::testing::Values(ov::element::dynamic),
                                                ::testing::ValuesIn(inputDynamicShape4D),
                                                ::testing::ValuesIn(axis4D),
                                                ::testing::Values(ov::test::utils::DEVICE_METAL),
                                                ::testing::Values(ov::AnyMap()));

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax4D_static,
                         MetalSoftmaxLayerTest,
                         params4Dstatic,
                         SoftMax8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax4D_dynamic,
                         MetalSoftmaxLayerTest,
                         params4Ddynamic,
                         SoftMax8LayerTest::getTestCaseName);

const std::vector<ov::Shape> inputStaticShape5D = {
    {1, 100, 1, 1, 1},
    {50, 100, 4, 1, 1},
    {2, 100, 10, 1, 1},
};

const std::vector<ov::test::InputShape> inputDynamicShape5D = {
    {{ov::Dimension::dynamic(), 100, ov::Dimension(1, 10), 1, 1}, {{1, 100, 1, 1, 1}, {100, 100, 5, 1, 1}}},
    {{ov::Dimension::dynamic(),
      ov::Dimension::dynamic(),
      ov::Dimension::dynamic(),
      ov::Dimension::dynamic(),
      ov::Dimension::dynamic()},
     {{1, 100, 1, 1, 1}, {50, 100, 4, 1, 1}, {2, 100, 10, 1, 1}}},
};

const std::vector<ov::test::ElementType> netPrecisions5D = {
    ov::element::f32,
};

const std::vector<int64_t> axis5D = {0, 1, 2, 3, 4, -1, -2, -3, -4, -5};

const auto params5Dstatic =
    ::testing::Combine(::testing::ValuesIn(netPrecisions5D),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputStaticShape5D)),
                       ::testing::ValuesIn(axis5D),
                       ::testing::Values(ov::test::utils::DEVICE_METAL),
                       ::testing::Values(ov::AnyMap()));

const auto params5Ddynamic = ::testing::Combine(::testing::ValuesIn(netPrecisions5D),
                                                ::testing::Values(ov::element::dynamic),
                                                ::testing::Values(ov::element::dynamic),
                                                ::testing::ValuesIn(inputDynamicShape5D),
                                                ::testing::ValuesIn(axis5D),
                                                ::testing::Values(ov::test::utils::DEVICE_METAL),
                                                ::testing::Values(ov::AnyMap()));

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax5D_static,
                         MetalSoftmaxLayerTest,
                         params5Dstatic,
                         SoftMax8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax5D_dynamic,
                         MetalSoftmaxLayerTest,
                         params5Ddynamic,
                         SoftMax8LayerTest::getTestCaseName);

TEST_P(MetalSoftmaxLayerTest, CompareWithRefs) {
    run();
}

TEST_P(MetalSoftmaxLayerTest, CompareQueryModel) {
    query_model();
}

}  // namespace
