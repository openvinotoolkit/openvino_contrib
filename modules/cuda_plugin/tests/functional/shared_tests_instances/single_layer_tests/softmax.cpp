// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <cuda_test_constants.hpp>

#include "single_layer_tests/softmax.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

/********************* SoftMax 2D tests **************************/

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::Layout> inputLayouts2D = {
    InferenceEngine::Layout::NC,
};

const std::vector<InferenceEngine::SizeVector> inputShapes2D = {
    InferenceEngine::SizeVector {1, 100},
    InferenceEngine::SizeVector {100, 1},
    InferenceEngine::SizeVector {10, 10},
};

const std::vector<size_t> axis2D = {
    0, 1
};

const auto params2D = testing::Combine(
    testing::ValuesIn(netPrecisions),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::ValuesIn(inputLayouts2D),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::ValuesIn(inputShapes2D),
    testing::ValuesIn(axis2D),
    testing::Values(CommonTestUtils::DEVICE_CUDA),
    testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_CASE_P(
        SoftMax2D,
        SoftMaxLayerTest,
        params2D,
        SoftMaxLayerTest::getTestCaseName
);

/********************* SoftMax 3D tests **************************/

const std::vector<InferenceEngine::SizeVector> inputShapes3D = {
    InferenceEngine::SizeVector{5, 5, 1},
    InferenceEngine::SizeVector{5, 5, 5},
};

const std::vector<size_t> axis3D = {0, 1, 2};

const auto params3D = testing::Combine(
    testing::ValuesIn(netPrecisions),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::ValuesIn(inputShapes3D),
    testing::ValuesIn(axis3D),
    testing::Values(CommonTestUtils::DEVICE_CUDA),
    testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_CASE_P(
        SoftMax3D,
        SoftMaxLayerTest,
        params3D,
        SoftMaxLayerTest::getTestCaseName
);

/********************* SoftMax 4D tests **************************/

const std::vector<InferenceEngine::SizeVector> inputShapes4D = {
    InferenceEngine::SizeVector {1, 100, 1, 1},
    InferenceEngine::SizeVector {1, 3, 4, 3},
    InferenceEngine::SizeVector {2, 3, 4, 5},
    InferenceEngine::SizeVector {5, 5, 5, 5},
    InferenceEngine::SizeVector {5, 5, 1, 1},
};

const std::vector<size_t> axis4D = {0, 1, 2, 3};

const auto params4D = testing::Combine(
    testing::ValuesIn(netPrecisions),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Layout::NCHW),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::ValuesIn(inputShapes4D),
    testing::ValuesIn(axis4D),
    testing::Values(CommonTestUtils::DEVICE_CUDA),
    testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_CASE_P(
        SoftMax4D,
        SoftMaxLayerTest,
        params4D,
        SoftMaxLayerTest::getTestCaseName
);

/********************* SoftMax 5D tests **************************/

const std::vector<InferenceEngine::SizeVector> inputShapes5D = {
    InferenceEngine::SizeVector{5, 5, 5, 5, 5},
    InferenceEngine::SizeVector{5, 5, 1, 1, 1},
};

const std::vector<size_t> axis5D = {0, 1, 2, 3, 4};


const auto params5D = testing::Combine(
    testing::ValuesIn(netPrecisions),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::ValuesIn(inputShapes5D),
    testing::ValuesIn(axis5D),
    testing::Values(CommonTestUtils::DEVICE_CUDA),
    testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_CASE_P(
        SoftMax5D,
        SoftMaxLayerTest,
        params5D,
        SoftMaxLayerTest::getTestCaseName
);

/**************** SoftMax NN specific tests **********************/
// resnet5: shape (1, 1001), axis 1
// vgg: shape (1, 1000), axis 1

const std::vector<InferenceEngine::SizeVector> resnet5Shapes = {
    InferenceEngine::SizeVector {1, 1001},
};

const auto resnet5Params = testing::Combine(
    testing::ValuesIn(netPrecisions),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::ValuesIn(inputLayouts2D),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::ValuesIn(resnet5Shapes),
    testing::ValuesIn(axis2D),
    testing::Values(CommonTestUtils::DEVICE_CUDA),
    testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_CASE_P(
        SoftMax2Dresnet5,
        SoftMaxLayerTest,
        resnet5Params,
        SoftMaxLayerTest::getTestCaseName
);

const std::vector<InferenceEngine::SizeVector> vggShapes = {
    InferenceEngine::SizeVector {1, 1000},
};

const auto vggParams = testing::Combine(
    testing::ValuesIn(netPrecisions),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::ValuesIn(inputLayouts2D),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::ValuesIn(vggShapes),
    testing::ValuesIn(axis2D),
    testing::Values(CommonTestUtils::DEVICE_CUDA),
    testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_CASE_P(
        SoftMax2Dvgg,
        SoftMaxLayerTest,
        vggParams,
        SoftMaxLayerTest::getTestCaseName
);

}  // namespace
