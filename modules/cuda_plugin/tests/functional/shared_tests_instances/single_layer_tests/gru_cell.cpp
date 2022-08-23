// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/gru_cell.hpp"

#include <cuda_test_constants.hpp>
#include <vector>

#include "benchmark.hpp"
#include "common_test_utils/test_constants.hpp"
#include "unsymmetrical_comparer.hpp"

namespace LayerTestsDefinitions {

class CUDNNGRUCellTest : public UnsymmetricalComparer<GRUCellTest> {
protected:
    void SetUp() override {
        GRUCellTest::SetUp();

        constexpr float up_to = 1.5f;
        constexpr float start_from = -1.5f;

        int seed = 1;
        const auto& ops = function->get_ordered_ops();
        for (const auto& op : ops) {
            if (std::dynamic_pointer_cast<ngraph::opset1::Constant>(op)) {
                const auto constant = ngraph::builder::makeConstant(
                    op->get_element_type(), op->get_shape(), std::vector<float>{}, true, up_to, start_from, seed++);
                function->replace_node(op, constant);
            }
        }
    }
};

// this class sets lesser precision because of test failures on some hardware, e.g. RTX2080
class FP16CUDNNGRUCellTest : public CUDNNGRUCellTest {
protected:
    void SetUp() override {
        CUDNNGRUCellTest::SetUp();
        threshold = 0.07f;
    }
};

TEST_P(CUDNNGRUCellTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
}

TEST_P(FP16CUDNNGRUCellTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
}

namespace {

const bool should_decompose = false;
const std::vector<std::string> activations{"sigmoid", "tanh"};

// only 0.0 clip is supported now
const std::vector<float> clips{0.0f};

// for now only true value is supported
// TODO: check OV answer regarding reference implementation
const std::vector<bool> linear_before_reset{true};

const std::vector<InferenceEngine::Precision> net_precisions = {InferenceEngine::Precision::FP32,
                                                                InferenceEngine::Precision::FP16};

// ------------- Smoke shapes -------------
const std::vector<size_t> smoke_batches_01{1, 2};
const std::vector<size_t> smoke_hidden_sizes_01{1, 4, 47};
const std::vector<size_t> smoke_input_sizes_01{1, 5, 16};

INSTANTIATE_TEST_CASE_P(smoke_GRUCellCommon_01,
                        CUDNNGRUCellTest,
                        ::testing::Combine(::testing::Values(should_decompose),
                                           ::testing::ValuesIn(smoke_batches_01),
                                           ::testing::ValuesIn(smoke_hidden_sizes_01),
                                           ::testing::ValuesIn(smoke_input_sizes_01),
                                           ::testing::Values(activations),
                                           ::testing::ValuesIn(clips),
                                           ::testing::ValuesIn(linear_before_reset),
                                           ::testing::ValuesIn(net_precisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        GRUCellTest::getTestCaseName);

const size_t smoke_batch_02 = 9;
const std::vector<size_t> smoke_hidden_sizes_02{1, 4, 47};
const std::vector<size_t> smoke_input_sizes_02{1, 5, 16};

INSTANTIATE_TEST_CASE_P(smoke_GRUCellCommon_02_FP32,
                        CUDNNGRUCellTest,
                        ::testing::Combine(::testing::Values(should_decompose),
                                           ::testing::Values(smoke_batch_02),
                                           ::testing::ValuesIn(smoke_hidden_sizes_02),
                                           ::testing::ValuesIn(smoke_input_sizes_02),
                                           ::testing::Values(activations),
                                           ::testing::ValuesIn(clips),
                                           ::testing::ValuesIn(linear_before_reset),
                                           ::testing::Values(InferenceEngine::Precision::FP32),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        GRUCellTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GRUCellCommon_02_FP16,
                        FP16CUDNNGRUCellTest,
                        ::testing::Combine(::testing::Values(should_decompose),
                                           ::testing::Values(smoke_batch_02),
                                           ::testing::ValuesIn(smoke_hidden_sizes_02),
                                           ::testing::ValuesIn(smoke_input_sizes_02),
                                           ::testing::Values(activations),
                                           ::testing::ValuesIn(clips),
                                           ::testing::ValuesIn(linear_before_reset),
                                           ::testing::Values(InferenceEngine::Precision::FP16),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        GRUCellTest::getTestCaseName);

// ------------- LPCNet shapes -------------
const size_t lpc_batch = 64;
const std::vector<size_t> lpc_hidden_sizes = {16, 384};
const size_t lpc_input_size = 512;

INSTANTIATE_TEST_CASE_P(GRUCell_LPCNet,
                        CUDNNGRUCellTest,
                        ::testing::Combine(::testing::Values(should_decompose),
                                           ::testing::Values(lpc_batch),
                                           ::testing::ValuesIn(lpc_hidden_sizes),
                                           ::testing::Values(lpc_input_size),
                                           ::testing::Values(activations),
                                           ::testing::ValuesIn(clips),
                                           ::testing::ValuesIn(linear_before_reset),
                                           ::testing::ValuesIn(net_precisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        GRUCellTest::getTestCaseName);

}  // namespace

// ------------- Benchmark -------------
const auto benchmark_params = ::testing::Combine(::testing::Values(should_decompose),
                                                 ::testing::Values(lpc_batch),
                                                 ::testing::Values(lpc_hidden_sizes[0]),
                                                 ::testing::Values(lpc_input_size),
                                                 ::testing::Values(activations),
                                                 ::testing::Values(clips[0]),
                                                 ::testing::Values(linear_before_reset[0]),
                                                 ::testing::Values(net_precisions[0]),
                                                 ::testing::Values(CommonTestUtils::DEVICE_CUDA));

namespace benchmark {

struct GruCellBenchmarkTest : BenchmarkLayerTest<CUDNNGRUCellTest> {};

TEST_P(GruCellBenchmarkTest, DISABLED_benchmark) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run("Comparison", std::chrono::milliseconds(2000), 100);
}

INSTANTIATE_TEST_CASE_P(GRUCell_Benchmark, GruCellBenchmarkTest, benchmark_params, GRUCellTest::getTestCaseName);

}  // namespace benchmark

}  // namespace LayerTestsDefinitions
