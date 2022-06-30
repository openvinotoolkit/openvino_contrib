// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <cuda_test_constants.hpp>

#include "single_layer_tests/softmax.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ov::test::subgraph;

namespace {

/********************* SoftMax 2D tests **************************/

const std::vector<ov::test::ElementType> netPrecisions = {
    ov::element::f32,
};

const std::vector<InferenceEngine::Layout> inputLayouts2D = {
    InferenceEngine::Layout::NC,
};

const std::vector<ov::Shape> inputShapes2D = {
    {1, 100},
    {100, 1},
    {10, 10},
};

const std::vector<size_t> axis2D = {
    0, 1
};

const auto params2D = testing::Combine(testing::ValuesIn(netPrecisions),
                                       testing::Values(ov::element::undefined),
                                       testing::Values(ov::element::undefined),
                                       testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes2D)),
                                       testing::ValuesIn(axis2D),
                                       testing::Values(CommonTestUtils::DEVICE_CUDA),
                                       testing::Values(ov::AnyMap()));

INSTANTIATE_TEST_CASE_P(
        SoftMax2D,
        SoftMaxLayerTest,
        params2D,
        SoftMaxLayerTest::getTestCaseName
);

/********************* SoftMax 3D tests **************************/

const std::vector<ov::Shape> inputShapes3D = {
    {5, 5, 1},
    {5, 5, 5},
};

const std::vector<size_t> axis3D = {0, 1, 2};

const auto params3D = testing::Combine(testing::ValuesIn(netPrecisions),
                                       testing::Values(ov::element::undefined),
                                       testing::Values(ov::element::undefined),
                                       testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes3D)),
                                       testing::ValuesIn(axis3D),
                                       testing::Values(CommonTestUtils::DEVICE_CUDA),
                                       testing::Values(ov::AnyMap()));

INSTANTIATE_TEST_CASE_P(
        SoftMax3D,
        SoftMaxLayerTest,
        params3D,
        SoftMaxLayerTest::getTestCaseName
);

/********************* SoftMax 4D tests **************************/

const std::vector<ov::Shape> inputShapes4D = {
    {1, 100, 1, 1},
    {1, 3, 4, 3},
    {2, 3, 4, 5},
    {5, 5, 5, 5},
    {5, 5, 1, 1},
};

const std::vector<size_t> axis4D = {0, 1, 2, 3};

const auto params4D = testing::Combine(testing::ValuesIn(netPrecisions),
                                       testing::Values(ov::element::undefined),
                                       testing::Values(ov::element::undefined),
                                       testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes4D)),
                                       testing::ValuesIn(axis4D),
                                       testing::Values(CommonTestUtils::DEVICE_CUDA),
                                       testing::Values(ov::AnyMap()));

INSTANTIATE_TEST_CASE_P(
        SoftMax4D,
        SoftMaxLayerTest,
        params4D,
        SoftMaxLayerTest::getTestCaseName
);

/********************* SoftMax 5D tests **************************/

const std::vector<ov::Shape> inputShapes5D = {
    InferenceEngine::SizeVector{5, 5, 5, 5, 5},
    InferenceEngine::SizeVector{5, 5, 1, 1, 1},
};

const std::vector<size_t> axis5D = {0, 1, 2, 3, 4};

const auto params5D = testing::Combine(testing::ValuesIn(netPrecisions),
                                       testing::Values(ov::element::undefined),
                                       testing::Values(ov::element::undefined),
                                       testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes5D)),
                                       testing::ValuesIn(axis5D),
                                       testing::Values(CommonTestUtils::DEVICE_CUDA),
                                       testing::Values(ov::AnyMap()));

INSTANTIATE_TEST_CASE_P(
        SoftMax5D,
        SoftMaxLayerTest,
        params5D,
        SoftMaxLayerTest::getTestCaseName
);

/**************** SoftMax NN specific tests **********************/
// resnet5: shape (1, 1001), axis 1
// vgg: shape (1, 1000), axis 1

const std::vector<ov::Shape> resnet5Shapes = {
    {1, 1001},
};

const auto resnet5Params =
    testing::Combine(testing::ValuesIn(netPrecisions),
                     testing::Values(ov::element::undefined),
                     testing::Values(ov::element::undefined),
                     testing::ValuesIn(ov::test::static_shapes_to_test_representation(resnet5Shapes)),
                     testing::ValuesIn(axis2D),
                     testing::Values(CommonTestUtils::DEVICE_CUDA),
                     testing::Values(ov::AnyMap()));

INSTANTIATE_TEST_CASE_P(
        SoftMax2Dresnet5,
        SoftMaxLayerTest,
        resnet5Params,
        SoftMaxLayerTest::getTestCaseName
);

const std::vector<ov::Shape> vggShapes = {
    {1, 1000},
};

const auto vggParams = testing::Combine(testing::ValuesIn(netPrecisions),
                                        testing::Values(ov::element::undefined),
                                        testing::Values(ov::element::undefined),
                                        testing::ValuesIn(ov::test::static_shapes_to_test_representation(vggShapes)),
                                        testing::ValuesIn(axis2D),
                                        testing::Values(CommonTestUtils::DEVICE_CUDA),
                                        testing::Values(ov::AnyMap()));

INSTANTIATE_TEST_CASE_P(
        SoftMax2Dvgg,
        SoftMaxLayerTest,
        vggParams,
        SoftMaxLayerTest::getTestCaseName
);

}  // namespace
