// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "single_layer_tests/lstm_sequence.hpp"

#include <cuda_profiler.hpp>
#include <cuda_test_constants.hpp>
#include <functional>
#include <vector>

#include "cuda/device_pointers.hpp"
#include "cuda_operation_registry.hpp"
#include "cuda_thread_context.hpp"
#include "unsymmetrical_comparer.hpp"

namespace LayerTestsDefinitions {

class CUDALSTMSequenceTest : public UnsymmetricalComparer<LSTMSequenceTest> {
    void SetUp() override {
        LSTMSequenceTest::SetUp();
        threshold = 0.01;
        constexpr float up_to = -1.0f;
        constexpr float start_from = 1.0f;
        int counter = 1;
        const auto& ops = function->get_ordered_ops();
        for (const auto& op : ops) {
            if (std::dynamic_pointer_cast<ngraph::opset1::Constant>(op)) {
                if (op->get_element_type() == ov::element::Type_t::f32) {
                    const auto constant = ngraph::builder::makeConstant(op->get_element_type(),
                                                                        op->get_shape(),
                                                                        std::vector<float>{},
                                                                        true,
                                                                        up_to,
                                                                        start_from,
                                                                        counter++);
                    function->replace_node(op, constant);
                }
            }
        }
    }
};

TEST_P(CUDALSTMSequenceTest, CompareWithRefs) { Run(); }

namespace {

const auto testMode = ngraph::helpers::SequenceTestsMode::PURE_SEQ;
const std::vector<std::string> activations{"sigmoid", "tanh", "tanh"};
const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};
const std::vector<ov::op::RecurrentSequenceDirection> sequenceDirections = {
    ov::op::RecurrentSequenceDirection::FORWARD, ov::op::RecurrentSequenceDirection::BIDIRECTIONAL};
// Currently LSTMSequence cuDNN implementation doesn't support clipping
const float no_clip = 0.0f;
const std::vector<size_t> batches{1, 2, 3, 10};

// ------------- Smoke Tests -------------

const std::vector<size_t> smoke_max_seq_lengths{1, 2, 3, 10};

// Currently LSTMSequence cuDNN implementation doesn't support combination of input_size == 1 and hidden_size == 1
const std::vector<size_t> smoke_01_input_sizes{1, 2, 3, 20};
const std::vector<size_t> smoke_01_hidden_sizes{2, 3, 10};
const std::vector<size_t> smoke_02_input_sizes{2, 3, 20};
const std::vector<size_t> smoke_02_hidden_sizes{1};

INSTANTIATE_TEST_CASE_P(smoke_LSTMSequence_01,
                        CUDALSTMSequenceTest,
                        ::testing::Combine(::testing::Values(testMode),
                                           ::testing::ValuesIn(smoke_max_seq_lengths),
                                           ::testing::ValuesIn(batches),
                                           ::testing::ValuesIn(smoke_01_hidden_sizes),
                                           ::testing::ValuesIn(smoke_01_input_sizes),
                                           ::testing::Values(activations),
                                           ::testing::Values(no_clip),
                                           ::testing::ValuesIn(sequenceDirections),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        CUDALSTMSequenceTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_LSTMSequence_02,
                        CUDALSTMSequenceTest,
                        ::testing::Combine(::testing::Values(testMode),
                                           ::testing::ValuesIn(smoke_max_seq_lengths),
                                           ::testing::ValuesIn(batches),
                                           ::testing::ValuesIn(smoke_02_hidden_sizes),
                                           ::testing::ValuesIn(smoke_02_input_sizes),
                                           ::testing::Values(activations),
                                           ::testing::Values(no_clip),
                                           ::testing::ValuesIn(sequenceDirections),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        CUDALSTMSequenceTest::getTestCaseName);

// ------------- Tacotron2 Tests -------------

INSTANTIATE_TEST_CASE_P(LSTMSequence_Tacotron2_decoder_01,
                        CUDALSTMSequenceTest,
                        ::testing::Combine(::testing::Values(testMode),
                                           ::testing::Values(1),          // seq_lengths
                                           ::testing::ValuesIn(batches),  // batch
                                           ::testing::Values(1024),       // hidden size
                                           ::testing::Values(768),        // input size
                                           ::testing::Values(activations),
                                           ::testing::Values(no_clip),  // clip
                                           ::testing::Values(ov::op::RecurrentSequenceDirection::FORWARD),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        CUDALSTMSequenceTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(LSTMSequence_Tacotron2_decoder_02,
                        CUDALSTMSequenceTest,
                        ::testing::Combine(::testing::Values(testMode),
                                           ::testing::Values(1),          // seq_lengths
                                           ::testing::ValuesIn(batches),  // batch
                                           ::testing::Values(1024),       // hidden size
                                           ::testing::Values(1536),       // input size
                                           ::testing::Values(activations),
                                           ::testing::Values(no_clip),  // clip
                                           ::testing::Values(ov::op::RecurrentSequenceDirection::FORWARD),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        CUDALSTMSequenceTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(LSTMSequence_Tacotron2_encoder_01,
                        CUDALSTMSequenceTest,
                        ::testing::Combine(::testing::Values(testMode),
                                           ::testing::Values(1000),       // seq_lengths
                                           ::testing::ValuesIn(batches),  // batch
                                           ::testing::Values(256),        // hidden size
                                           ::testing::Values(512),        // input size
                                           ::testing::Values(activations),
                                           ::testing::Values(no_clip),  // clip
                                           ::testing::Values(ov::op::RecurrentSequenceDirection::BIDIRECTIONAL),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        CUDALSTMSequenceTest::getTestCaseName);

}  // namespace
}  // namespace LayerTestsDefinitions
