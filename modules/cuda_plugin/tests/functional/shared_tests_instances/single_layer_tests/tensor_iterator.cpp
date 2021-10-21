// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/tensor_iterator.hpp"

#include <cuda/runtime.hpp>
#include <cuda_operation_base.hpp>
#include <cuda_operation_registry.hpp>
#include <cuda_test_constants.hpp>
#include <ngraph/op/util/attr_types.hpp>
#include <vector>

#include "benchmark.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
std::vector<bool> should_decompose = {false};
std::vector<size_t> smoke_seq_lengths_clip_non_zero{20};
std::vector<size_t> batch{1, 10};
std::vector<size_t> hidden_size{1, 10, 384, 512, 768};
std::vector<size_t> sequence_axis{0, 1};
std::vector<ngraph::helpers::TensorIteratorBody> body_type = {ngraph::helpers::TensorIteratorBody::LSTM};
std::vector<float> clip_zeros{0.f};
std::vector<ngraph::op::RecurrentSequenceDirection> direction = {
    ngraph::op::RecurrentSequenceDirection::FORWARD,
    ngraph::op::RecurrentSequenceDirection::REVERSE,
};
std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                         InferenceEngine::Precision::FP16};

INSTANTIATE_TEST_CASE_P(
    smoke_TensorIteratorNoClip,
    TensorIteratorTest,
    ::testing::Combine(::testing::ValuesIn(should_decompose),
                       ::testing::ValuesIn(smoke_seq_lengths_clip_non_zero),
                       ::testing::ValuesIn(batch),
                       ::testing::ValuesIn(hidden_size),
                       //::testing::ValuesIn(input_size), // hardcoded to 10 due to Combine supports up to 10 args
                       ::testing::ValuesIn(sequence_axis),
                       ::testing::ValuesIn(clip_zeros),
                       ::testing::ValuesIn(body_type),
                       ::testing::ValuesIn(direction),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    TensorIteratorTest::getTestCaseName);

std::vector<size_t> seq_lengths_clip_non_zero{1000};
INSTANTIATE_TEST_CASE_P(
    TensorIteratorNoClip,
    TensorIteratorTest,
    ::testing::Combine(::testing::ValuesIn(should_decompose),
                       ::testing::ValuesIn(seq_lengths_clip_non_zero),
                       ::testing::ValuesIn(batch),
                       ::testing::ValuesIn(hidden_size),
                       //::testing::ValuesIn(input_size), // hardcoded to 10 due to Combine supports up to 10 args
                       ::testing::ValuesIn(sequence_axis),
                       ::testing::ValuesIn(clip_zeros),
                       ::testing::ValuesIn(body_type),
                       ::testing::ValuesIn(direction),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    TensorIteratorTest::getTestCaseName);

// ------------- Benchmark -------------
namespace benchmark {

struct TensorIteratorBenchmarkTest : BenchmarkLayerTest<TensorIteratorTest> {};

TEST_P(TensorIteratorBenchmarkTest, DISABLED_benchmark) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run("TensorIterator", std::chrono::milliseconds(2000), 100);
}

std::vector<bool> should_decompose = {false};
std::vector<size_t> seq_lengths_clip_non_zero{20};
std::vector<size_t> batch{1, 10};
std::vector<size_t> hidden_size{1, 10};
std::vector<size_t> sequence_axis{0, 1};
std::vector<ngraph::helpers::TensorIteratorBody> body_type = {ngraph::helpers::TensorIteratorBody::LSTM};
std::vector<float> clip_zeros{0.f};
std::vector<ngraph::op::RecurrentSequenceDirection> direction = {
    ngraph::op::RecurrentSequenceDirection::FORWARD,
    ngraph::op::RecurrentSequenceDirection::REVERSE,
};
std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                         InferenceEngine::Precision::FP16};

INSTANTIATE_TEST_CASE_P(
    smoke_TensorIteratorNoClip,
    TensorIteratorBenchmarkTest,
    ::testing::Combine(::testing::ValuesIn(should_decompose),
                       ::testing::ValuesIn(seq_lengths_clip_non_zero),
                       ::testing::ValuesIn(batch),
                       ::testing::ValuesIn(hidden_size),
                       //::testing::ValuesIn(input_size), // hardcoded to 10 due to Combine supports up to 10 args
                       ::testing::ValuesIn(sequence_axis),
                       ::testing::ValuesIn(clip_zeros),
                       ::testing::ValuesIn(body_type),
                       ::testing::ValuesIn(direction),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    TensorIteratorTest::getTestCaseName);

}  // namespace benchmark

}  // namespace
