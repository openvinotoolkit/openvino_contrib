// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <cuda_test_constants.hpp>

#include "single_layer_tests/mat_mul.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
    // TODO: Uncomment when ExecNetwork will support FP16
//    InferenceEngine::Precision::FP16,
};

// General
const std::vector<ShapeRelatedParams> smokeShapeRelatedParams = {
    // ND x ND: [B, ..., X, Y] x [B, ..., Y, Z] => [B, ..., X, Z]
    { { {3, 2, 10, 10}, false }, { {3, 2, 10, 20}, false } },
    { { {3, 2, 10, 10}, true }, { {3, 2, 10, 20}, false } },
    { { {3, 2, 10, 20}, false }, { {3, 2, 10, 20}, true } },
    { { {3, 2, 20, 10}, true }, { {3, 2, 10, 20}, true } },

    // ND x ND: [B, ..., X, Y] x [B, ..., Y, Z] => [B, ..., X, Z]
    { { {2, 10, 10}, false }, { {2, 10, 20}, false } },
    { { {2, 10, 10}, true }, { {2, 10, 20}, false } },
    { { {2, 10, 20}, false }, { {2, 10, 20}, true } },
    { { {2, 20, 10}, true }, { {2, 10, 20}, true } },

    // ND x ND: [B, ..., X, Y] x [B, ..., Y, Z] => [B, ..., X, Z]
    { { {10, 10}, false }, { {10, 20}, false } },
    { { {10, 10}, true }, { {10, 20}, false } },
    { { {10, 20}, false }, { {10, 20}, true } },
    { { {20, 10}, true }, { {10, 20}, true } },

    // ND x 1D: [B, ..., X, Y] x [Y] -> [B, ..., X, Y] x [Y, 1] -> [B, ..., X, 1] => [B, ..., X]
    { { {2, 10, 10}, false }, { {10}, false } },
    { { {2, 10, 10}, true }, { {10}, false } },
    { { {2, 10, 20}, false }, { {20}, true } },
    { { {2, 20, 10}, true }, { {20}, true } },

    // ND x 1D: [B, ..., X, Y] x [Y] -> [B, ..., X, Y] x [Y, 1] -> [B, ..., X, 1] => [B, ..., X]
    { { {10, 10}, false }, { {10}, false } },
    { { {10, 10}, true }, { {10}, false } },
    { { {10, 20}, false }, { {20}, true } },
    { { {20, 10}, true }, { {20}, true } },

    // 1D x ND: [X] x [B, ..., X, Y] -> [1, X] x [B, ..., X, Y] -> [B, ..., 1, Y] => [B, ..., Y]
    { { {10}, false }, { {10, 20}, false } },
    { { {10}, true }, { {10, 20}, false } },
    { { {20}, false }, { {10, 20}, true } },
    { { {20}, true }, { {10, 20}, true } },

    // 1D x ND: [X] x [B, ..., X, Y] -> [1, X] x [B, ..., X, Y] -> [B, ..., 1, Y] => [B, ..., Y]
    { { {10}, false }, { {2, 10, 20}, false } },
    { { {10}, true }, { {2, 10, 20}, false } },
    { { {20}, false }, { {2, 10, 20}, true } },
    { { {20}, true }, { {2, 10, 20}, true } },

    // 1D x 1D: [X] x [X] -> [1, X] x [X, 1] -> [1, 1] => [] (scalar)
    { { {10}, false }, { {10}, false } },
    { { {10}, true }, { {10}, false } },
    { { {10}, false }, { {10}, true } },
    { { {10}, true }, { {10}, true } },
};

// NOTE: Resnet-50 shapes
const std::vector<ShapeRelatedParams> resnet50ShapeRelatedParams = {
    { { {1, 2048}, false }, { {2048, 1001}, false } },
    { { {1, 2048}, false }, { {1001, 2048}, true } },
};

// NOTE: VGG-16 shapes
const std::vector<ShapeRelatedParams> vgg16ShapeRelatedParams = {
    { { {1, 25088}, false }, { {4096, 25088}, true } },
    { { {1, 25088}, false }, { {25088, 4096}, false } },
    { { {1, 4096}, false }, { {4096, 4096}, true } },
    { { {1, 4096}, false }, { {4096, 4096}, false } },
    { { {1, 4096}, false }, { {1000, 4096}, true } },
    { { {1, 4096}, false }, { {4096, 1000}, false } },
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
    ngraph::helpers::InputLayerType::CONSTANT,
    ngraph::helpers::InputLayerType::PARAMETER,
};

std::map<std::string, std::string> additional_config = {};

INSTANTIATE_TEST_CASE_P(smoke_MatMul, MatMulTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(smokeShapeRelatedParams),
                            ::testing::ValuesIn(inputPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::ValuesIn(secondaryInputTypes),
                            ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                            ::testing::Values(additional_config)),
                        MatMulTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(MatMul_Resnet50, MatMulTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(resnet50ShapeRelatedParams),
                            ::testing::ValuesIn(inputPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::ValuesIn(secondaryInputTypes),
                            ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                            ::testing::Values(additional_config)),
                        MatMulTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(MatMul_VGG16, MatMulTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(vgg16ShapeRelatedParams),
                            ::testing::ValuesIn(inputPrecisions),
                            ::testing::Values(InferenceEngine::Precision::FP32),
                            ::testing::Values(InferenceEngine::Precision::FP32),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::ValuesIn(secondaryInputTypes),
                            ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                            ::testing::Values(additional_config)),
                        MatMulTest::getTestCaseName);

} // namespace
