// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/gru_cell.hpp"

#include <cuda_test_constants.hpp>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "unsymmetrical_comparer.hpp"

using namespace LayerTestsDefinitions;

#include <iostream>

class CUDAGRUCellTest : public UnsymmetricalComparer<GRUCellTest> {
    // !!! TODO remove when GRU reference cell works as CUDA
    void Validate() override {}
    void SetUp() {
        GRUCellTest::SetUp();
        constexpr float up_to = 1.5f;
        constexpr float start_from = -1.5f;
        // std::random_device rd;

        int couter = 1;
        const auto& ops = function->get_ordered_ops();
        for (const auto& op : ops) {
            if (std::dynamic_pointer_cast<ngraph::opset1::Constant>(op)) {
                const auto constant = ngraph::builder::makeConstant(
                    op->get_element_type(), op->get_shape(), std::vector<float>{}, true, up_to, start_from, couter++);
                function->replace_node(op, constant);
            }
        }
    }
};

TEST_P(CUDAGRUCellTest, CompareWithRefs) { Run(); }

namespace {
std::vector<bool> should_decompose{false};
std::vector<size_t> batch{2, 64};
std::vector<size_t> hidden_size{4, 16, 384};
std::vector<size_t> input_size{2, 512};
std::vector<std::vector<std::string>> activations{{"sigmoid", "tanh"}};
std::vector<float> clip = {0.0f};  // only 0.0 clip is supported now
std::vector<bool> linear_before_reset = {false, true};
std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                         InferenceEngine::Precision::FP16};

INSTANTIATE_TEST_CASE_P(smoke_GRUCellCommon,
                        CUDAGRUCellTest,
                        ::testing::Combine(::testing::ValuesIn(should_decompose),
                                           ::testing::ValuesIn(batch),
                                           ::testing::ValuesIn(hidden_size),
                                           ::testing::ValuesIn(input_size),
                                           ::testing::ValuesIn(activations),
                                           ::testing::ValuesIn(clip),
                                           ::testing::ValuesIn(linear_before_reset),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
                        GRUCellTest::getTestCaseName);

}  // namespace

// ------------- Benchmark -------------
#include "benchmark.hpp"

const auto BenchmarkGRUCellTestParams = ::testing::Combine(
    ::testing::ValuesIn(std::vector<bool>{false}),                                    // should_decompose
    ::testing::ValuesIn(std::vector<size_t>{64}),                                     // batch
    ::testing::ValuesIn(std::vector<size_t>{16}),                                     // hidden_size
    ::testing::ValuesIn(std::vector<size_t>{512}),                                    // input size
    ::testing::ValuesIn(std::vector<std::vector<std::string>>{{"sigmoid", "tanh"}}),  // activiation
    ::testing::ValuesIn(std::vector<float>{0.0f}),                                    // clip
    ::testing::ValuesIn(std::vector<bool>{true}),                                     // linear before reset
    ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32}),  // precision
    ::testing::Values(CommonTestUtils::DEVICE_CUDA));                                                // device

namespace LayerTestsDefinitions {
namespace benchmark {
struct GruCellBenchmarkTest : BenchmarkLayerTest<CUDAGRUCellTest> {};

TEST_P(GruCellBenchmarkTest, DISABLED_benchmark) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run("Comparison", std::chrono::milliseconds(2000), 100);
}

INSTANTIATE_TEST_CASE_P(GruCellBenchmark,
                        GruCellBenchmarkTest,
                        BenchmarkGRUCellTestParams,
                        GRUCellTest::getTestCaseName);

}  // namespace benchmark

}  // namespace LayerTestsDefinitions
