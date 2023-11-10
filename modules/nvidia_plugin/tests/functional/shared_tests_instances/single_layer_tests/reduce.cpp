// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "cuda_test_constants.hpp"
#include "single_layer_tests/reduce_ops.hpp"

namespace {
using namespace LayerTestsDefinitions;
using ov::test::utils::ReductionTypel

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const std::vector<bool> keepDims = {true, false};

const std::vector<std::vector<size_t>> inputShapes = {
    std::vector<size_t>{10, 20, 30, 40},
    std::vector<size_t>{10, 20, 1, 40},
    std::vector<size_t>{3, 5, 7, 9},
};

const std::vector<std::vector<size_t>> inputShapesOneAxis = {
    std::vector<size_t>{10, 20, 30, 40},
    std::vector<size_t>{10, 20, 1, 40},
    std::vector<size_t>{3, 5, 7, 9},
    std::vector<size_t>{10},
};

const std::vector<std::vector<int>> axes = {{0},
                                            {1},
                                            {2},
                                            {3},
                                            {0, 1},
                                            {0, 2},
                                            {0, 3},
                                            {1, 2},
                                            {1, 3},
                                            {2, 3},
                                            {0, 1, 2},
                                            {0, 1, 3},
                                            {0, 2, 3},
                                            {1, 2, 3},
                                            {0, 1, 2, 3},
                                            {1, -1}};

std::vector<ov::test::utils::OpType> opTypes = {
    ov::test::utils::OpType::SCALAR,
    ov::test::utils::OpType::VECTOR,
};

const std::vector<ReductionType> reductionTypes = {
    ReductionType::Mean,
    ReductionType::Min,
    ReductionType::Max,
    ReductionType::Sum,
    ReductionType::Prod,
    ReductionType::L1,
    ReductionType::L2,
};

const std::vector<ReductionType> reductionLogicalTypes = {ReductionType::LogicalOr,
                                                                           ReductionType::LogicalAnd};

const auto paramsOneAxis = testing::Combine(testing::Values(std::vector<int>{0}),
                                            testing::ValuesIn(opTypes),
                                            testing::ValuesIn(keepDims),
                                            testing::ValuesIn(reductionTypes),
                                            testing::Values(netPrecisions[0]),
                                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            testing::Values(InferenceEngine::Layout::ANY),
                                            testing::ValuesIn(inputShapesOneAxis),
                                            testing::Values(ov::test::utils::DEVICE_NVIDIA));

const auto params_Precisions = testing::Combine(testing::Values(std::vector<int>{1, 3}),
                                                testing::Values(opTypes[1]),
                                                testing::ValuesIn(keepDims),
                                                testing::Values(ReductionType::Max,
                                                                ReductionType::Mean,
                                                                ReductionType::Min,
                                                                ReductionType::Sum,
                                                                ReductionType::Prod),
                                                testing::ValuesIn(netPrecisions),
                                                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                                testing::Values(InferenceEngine::Layout::ANY),
                                                testing::Values(std::vector<size_t>{2, 2, 2, 2}),
                                                testing::Values(ov::test::utils::DEVICE_NVIDIA));

const auto params_Precisions_ReduceL1 = testing::Combine(testing::Values(std::vector<int>{1, 3}),
                                                         testing::Values(opTypes[1]),
                                                         testing::ValuesIn(keepDims),
                                                         testing::Values(ReductionType::L1),
                                                         testing::ValuesIn(netPrecisions),
                                                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                                         testing::Values(InferenceEngine::Layout::ANY),
                                                         testing::Values(std::vector<size_t>{2, 2, 2, 2}),
                                                         testing::Values(ov::test::utils::DEVICE_NVIDIA));

const auto params_Precisions_ReduceL2 =
    testing::Combine(testing::Values(std::vector<int>{1, 3}),
                     testing::Values(opTypes[1]),
                     testing::ValuesIn(keepDims),
                     testing::Values(ReductionType::L2),
                     testing::Values(InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16),
                     testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                     testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                     testing::Values(InferenceEngine::Layout::ANY),
                     testing::Values(std::vector<size_t>{2, 2, 2, 2}),
                     testing::Values(ov::test::utils::DEVICE_NVIDIA));

const auto params_InputShapes = testing::Combine(testing::Values(std::vector<int>{0}),
                                                 testing::Values(opTypes[1]),
                                                 testing::ValuesIn(keepDims),
                                                 testing::Values(ReductionType::Sum),
                                                 testing::Values(netPrecisions[0]),
                                                 testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                                 testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                                 testing::Values(InferenceEngine::Layout::ANY),
                                                 testing::Values(std::vector<size_t>{3},
                                                                 std::vector<size_t>{3, 5},
                                                                 std::vector<size_t>{2, 4, 6},
                                                                 std::vector<size_t>{2, 4, 6, 8},
                                                                 std::vector<size_t>{2, 2, 2, 2, 2},
                                                                 std::vector<size_t>{2, 2, 2, 2, 2, 2}),
                                                 testing::Values(ov::test::utils::DEVICE_NVIDIA));

const auto params_Axes = testing::Combine(testing::ValuesIn(axes),
                                          testing::Values(opTypes[1]),
                                          testing::ValuesIn(keepDims),
                                          testing::Values(ReductionType::Sum),
                                          testing::Values(netPrecisions[0]),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Layout::ANY),
                                          testing::ValuesIn(inputShapes),
                                          testing::Values(ov::test::utils::DEVICE_NVIDIA));

const auto params_ReductionTypes = testing::Combine(testing::Values(std::vector<int>{0, 1, 3}),
                                                    testing::Values(opTypes[1]),
                                                    testing::ValuesIn(keepDims),
                                                    testing::ValuesIn(reductionTypes),
                                                    testing::Values(netPrecisions[0]),
                                                    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                                    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                                    testing::Values(InferenceEngine::Layout::ANY),
                                                    testing::Values(std::vector<size_t>{2, 9, 2, 9}),
                                                    testing::Values(ov::test::utils::DEVICE_NVIDIA));

const auto params_ReduceSum_accuracy = testing::Combine(testing::Values(std::vector<int>{0}),
                                                        testing::Values(opTypes[1]),
                                                        testing::Values(true),
                                                        testing::Values(ReductionType::Sum),
                                                        testing::Values(InferenceEngine::Precision::FP32),
                                                        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                                        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                                        testing::Values(InferenceEngine::Layout::ANY),
                                                        testing::Values(std::vector<size_t>{1000000}),
                                                        testing::Values(ov::test::utils::DEVICE_NVIDIA));

INSTANTIATE_TEST_SUITE_P(smoke_ReduceSum_Accuracy,
                         ReduceOpsLayerTest,
                         params_ReduceSum_accuracy,
                         ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReduceOneAxis, ReduceOpsLayerTest, paramsOneAxis, ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reduce_Precisions,
                         ReduceOpsLayerTest,
                         params_Precisions,
                         ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reduce_Precisions_L1,
                         ReduceOpsLayerTest,
                         params_Precisions_ReduceL1,
                         ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reduce_Precisions_L2,
                         ReduceOpsLayerTest,
                         params_Precisions_ReduceL2,
                         ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reduce_InputShapes,
                         ReduceOpsLayerTest,
                         params_InputShapes,
                         ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reduce_Axes, ReduceOpsLayerTest, params_Axes, ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reduce_ReductionTypes,
                         ReduceOpsLayerTest,
                         params_ReductionTypes,
                         ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reduce,
                         ReduceOpsLayerWithSpecificInputTest,
                         testing::Combine(testing::ValuesIn(decltype(axes){{0}, {1}}),
                                          testing::Values(opTypes[1]),
                                          testing::Values(true),
                                          testing::Values(ReductionType::Sum),
                                          testing::ValuesIn(netPrecisions),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Layout::ANY),
                                          testing::Values(std::vector<size_t>{2, 10}),
                                          testing::Values(ov::test::utils::DEVICE_NVIDIA)),
                         ReduceOpsLayerWithSpecificInputTest::getTestCaseName);

}  // namespace
