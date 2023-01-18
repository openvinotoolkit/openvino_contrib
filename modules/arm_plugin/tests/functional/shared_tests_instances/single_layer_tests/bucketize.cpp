// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/bucketize.hpp"

#include <vector>

using namespace LayerTestsDefinitions;

namespace {

    const std::vector<std::vector<size_t>> data_shapes = {
            {1, 20, 20},         // 3D
            {40, 22, 13, 9},     // 4D
            {6, 7, 3, 2, 8},     // 5D
            {6, 7, 3, 2, 8, 5},  // 6D
    };

    const std::vector<std::vector<size_t>> buckets_shapes = {
            {5},
            {20},
            {100},
    };

    const std::vector<bool> with_right_bound = {true, false};

    const std::vector<InferenceEngine::Precision> out_precision = {
            InferenceEngine::Precision::I32,
            InferenceEngine::Precision::I64,
    };

// We won't test FP32 and FP16 together as it won't make sense for now
// as ngraph reference implementation use FP32 for FP16 case

    INSTANTIATE_TEST_SUITE_P(smoke_Bucketize_input_fp16,
                             BucketizeLayerTest,
                             testing::Combine(testing::ValuesIn(data_shapes),
                                              testing::ValuesIn(buckets_shapes),
                                              testing::ValuesIn(with_right_bound),
                                              testing::Values(InferenceEngine::Precision::FP16),
                                              testing::Values(InferenceEngine::Precision::FP16,
                                                              InferenceEngine::Precision::I32,
                                                              InferenceEngine::Precision::I64),
                                              testing::ValuesIn(out_precision),
                                              testing::Values(CommonTestUtils::DEVICE_CPU)),
                             BucketizeLayerTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Bucketize_input_fp32,
                             BucketizeLayerTest,
                             testing::Combine(testing::ValuesIn(data_shapes),
                                              testing::ValuesIn(buckets_shapes),
                                              testing::ValuesIn(with_right_bound),
                                              testing::Values(InferenceEngine::Precision::FP32),
                                              testing::Values(InferenceEngine::Precision::FP32,
                                                              InferenceEngine::Precision::I32,
                                                              InferenceEngine::Precision::I64),
                                              testing::ValuesIn(out_precision),
                                              testing::Values(CommonTestUtils::DEVICE_CPU)),
                             BucketizeLayerTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Bucketize_input_i32,
                             BucketizeLayerTest,
                             testing::Combine(testing::ValuesIn(data_shapes),
                                              testing::ValuesIn(buckets_shapes),
                                              testing::ValuesIn(with_right_bound),
                                              testing::Values(InferenceEngine::Precision::I32),
                                              testing::Values(InferenceEngine::Precision::FP16,
                                                              InferenceEngine::Precision::FP32,
                                                              InferenceEngine::Precision::I32,
                                                              InferenceEngine::Precision::I64),
                                              testing::ValuesIn(out_precision),
                                              testing::Values(CommonTestUtils::DEVICE_CPU)),
                             BucketizeLayerTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Bucketize_input_i64,
                             BucketizeLayerTest,
                             testing::Combine(testing::ValuesIn(data_shapes),
                                              testing::ValuesIn(buckets_shapes),
                                              testing::ValuesIn(with_right_bound),
                                              testing::Values(InferenceEngine::Precision::I64),
                                              testing::Values(InferenceEngine::Precision::FP16,
                                                              InferenceEngine::Precision::FP32,
                                                              InferenceEngine::Precision::I32,
                                                              InferenceEngine::Precision::I64),
                                              testing::ValuesIn(out_precision),
                                              testing::Values(CommonTestUtils::DEVICE_CPU)),
                             BucketizeLayerTest::getTestCaseName);
}  // namespace
