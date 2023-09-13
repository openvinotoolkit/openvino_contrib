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

constexpr int SEED_FIRST = 10;
constexpr float THRESHOLD_FP16 = 0.05f;

class CUDNNGRUCellTest : public UnsymmetricalComparer<GRUCellTest> {
protected:
    void SetUp() override {
        GRUCellTest::SetUp();

        const auto hiddenSize = std::get<2>(this->GetParam());

        // All the weights and biases are initialized from u(-sqrt(k), sqrt(k)), where k = 1 / hidden_size
        // https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
        const auto k_root = std::sqrt(1.0f / static_cast<float>(hiddenSize));

        const float up_to = k_root;
        const float start_from = -k_root;

        const auto& ops = function->get_ordered_ops();
        int seed = SEED_FIRST;
        for (const auto& op : ops) {
            if (std::dynamic_pointer_cast<ngraph::opset1::Constant>(op)) {
                ov::Tensor random_tensor(op->get_element_type(), op->get_shape());
                ov::test::utils::fill_tensor_random(random_tensor, up_to - start_from, start_from, 1, seed++);
                function->replace_node(op, std::make_shared<ov::op::v0::Constant>(random_tensor));
            }
        }

        const auto& netPrecision = std::get<InferenceEngine::Precision>(this->GetParam());
        if (netPrecision == InferenceEngine::Precision::FP16) {
            this->threshold = THRESHOLD_FP16;
        }
    }
};

TEST_P(CUDNNGRUCellTest, CompareWithRefs) {
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
const std::vector WRBLayerTypes = {ngraph::helpers::InputLayerType::CONSTANT};

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
                                           ::testing::ValuesIn(WRBLayerTypes),
                                           ::testing::ValuesIn(WRBLayerTypes),
                                           ::testing::ValuesIn(WRBLayerTypes),
                                           ::testing::ValuesIn(net_precisions),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
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
                                           ::testing::ValuesIn(WRBLayerTypes),
                                           ::testing::ValuesIn(WRBLayerTypes),
                                           ::testing::ValuesIn(WRBLayerTypes),
                                           ::testing::Values(InferenceEngine::Precision::FP32),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
                        GRUCellTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GRUCellCommon_02_FP16,
                        CUDNNGRUCellTest,
                        ::testing::Combine(::testing::Values(should_decompose),
                                           ::testing::Values(smoke_batch_02),
                                           ::testing::ValuesIn(smoke_hidden_sizes_02),
                                           ::testing::ValuesIn(smoke_input_sizes_02),
                                           ::testing::Values(activations),
                                           ::testing::ValuesIn(clips),
                                           ::testing::ValuesIn(linear_before_reset),
                                           ::testing::ValuesIn(WRBLayerTypes),
                                           ::testing::ValuesIn(WRBLayerTypes),
                                           ::testing::ValuesIn(WRBLayerTypes),
                                           ::testing::Values(InferenceEngine::Precision::FP16),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
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
                                           ::testing::ValuesIn(WRBLayerTypes),
                                           ::testing::ValuesIn(WRBLayerTypes),
                                           ::testing::ValuesIn(WRBLayerTypes),
                                           ::testing::ValuesIn(net_precisions),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
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
                                                 ::testing::ValuesIn(WRBLayerTypes),
                                                 ::testing::ValuesIn(WRBLayerTypes),
                                                 ::testing::ValuesIn(WRBLayerTypes),
                                                 ::testing::Values(net_precisions[0]),
                                                 ::testing::Values(ov::test::utils::DEVICE_NVIDIA));

namespace benchmark {

struct GruCellBenchmarkTest : BenchmarkLayerTest<CUDNNGRUCellTest> {};

TEST_P(GruCellBenchmarkTest, DISABLED_benchmark) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run("Comparison", std::chrono::milliseconds(2000), 100);
}

INSTANTIATE_TEST_CASE_P(GRUCell_Benchmark, GruCellBenchmarkTest, benchmark_params, GRUCellTest::getTestCaseName);

}  // namespace benchmark

}  // namespace LayerTestsDefinitions
