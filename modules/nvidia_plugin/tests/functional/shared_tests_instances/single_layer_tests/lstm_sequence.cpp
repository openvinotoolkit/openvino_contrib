// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "single_layer_tests/lstm_sequence.hpp"

#include <cuda_simple_execution_delegator.hpp>
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
                    ov::Tensor random_tensor(op->get_element_type(), op->get_shape());
                    ov::test::utils::fill_tensor_random(random_tensor, up_to - start_from, start_from, 1, counter++);
                    function->replace_node(op, std::make_shared<ov::op::v0::Constant>(random_tensor));
                }
            }
        }
    }
};

using LSTMSequenceOptimizedParams =
    typename std::tuple<ngraph::helpers::SequenceTestsMode,  // pure Sequence or TensorIterator
                        size_t,                              // seq_lengths
                        size_t,                              // batch
                        size_t,                              // hidden size
                        size_t,                              // input size
                        std::vector<std::string>,            // activations
                        float,                               // clip
                        std::string,                         // major batch
                        ngraph::helpers::InputLayerType,     // WRB input type (Constant or Parameter)
                        InferenceEngine::Precision,          // Network precision
                        std::string>;                        // Device name

class CUDALSTMSequenceOptimizedTest : public testing::WithParamInterface<LSTMSequenceOptimizedParams>,
                                      virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LSTMSequenceOptimizedParams>& obj) {
        ngraph::helpers::SequenceTestsMode mode;
        size_t seq_lengths;
        size_t batch;
        size_t hidden_size;
        size_t input_size;
        std::vector<std::string> activations;
        std::vector<float> activations_alpha;
        std::vector<float> activations_beta;
        float clip;
        std::string major_batch;
        ngraph::helpers::InputLayerType WRBType;
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::tie(mode,
                 seq_lengths,
                 batch,
                 hidden_size,
                 input_size,
                 activations,
                 clip,
                 major_batch,
                 WRBType,
                 netPrecision,
                 targetDevice) = obj.param;
        std::vector<std::vector<size_t>> input_shapes = {
            {{batch, input_size},
             {batch, hidden_size},
             {batch, hidden_size},
             {4 * hidden_size, input_size},
             {4 * hidden_size, hidden_size},
             {4 * hidden_size}},
        };
        std::ostringstream result;
        result << "mode=" << mode << "_";
        result << "seq_lengths=" << seq_lengths << "_";
        result << "batch=" << batch << "_";
        result << "hidden_size=" << hidden_size << "_";
        result << "input_size=" << input_size << "_";
        result << "IS=" << ov::test::utils::vec2str(input_shapes) << "_";
        result << "activations=" << ov::test::utils::vec2str(activations) << "_";
        result << "major_batch=" << major_batch << "_";
        result << "clip=" << clip << "_";
        result << "WRBType=" << WRBType << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        return result.str();
    }

protected:
    void GenerateInputs() override {
        for (const auto& input : executableNetwork.GetInputsInfo()) {
            const auto& info = input.second;
            auto blob = GenerateInput(*info);
            if (input.first == "seq_lengths") {
                blob = FuncTestUtils::createAndFillBlob(info->getTensorDesc(), m_max_seq_len, 0);
            }

            inputs.push_back(blob);
        }
    }
    void SetUp() override {
        using namespace ngraph::helpers;
        using namespace ngraph::builder;
        threshold = 0.01;
        constexpr float up_to = -1.0f;
        constexpr float start_from = 1.0f;
        int counter = 1;
        size_t seq_lengths;
        size_t batch;
        size_t hidden_size;
        size_t input_size;
        std::vector<std::string> activations;
        std::vector<float> activations_alpha;
        std::vector<float> activations_beta;
        float clip;
        ov::op::RecurrentSequenceDirection direction = ov::op::RecurrentSequenceDirection::BIDIRECTIONAL;
        std::string major_batch;
        ngraph::helpers::InputLayerType WRBType;
        InferenceEngine::Precision netPrecision;
        std::tie(m_mode,
                 seq_lengths,
                 batch,
                 hidden_size,
                 input_size,
                 activations,
                 clip,
                 major_batch,
                 WRBType,
                 netPrecision,
                 targetDevice) = this->GetParam();
        size_t num_directions = 2;
        m_max_seq_len = seq_lengths;
        auto x_shape = (major_batch == "BatchMajor") ? ov::Shape{batch, seq_lengths, input_size}
                                                     : ov::Shape{seq_lengths, batch, input_size};
        std::vector<ov::Shape> input_shapes = {
            {x_shape,
             {batch, num_directions, hidden_size},
             {batch, num_directions, hidden_size},
             {batch},
             {num_directions, 4 * hidden_size, input_size},
             {num_directions, 4 * hidden_size, hidden_size},
             {num_directions, 4 * hidden_size}},
        };

        const auto& W_shape = input_shapes[4];
        const auto& R_shape = input_shapes[5];
        const auto& B_shape = input_shapes[6];

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, input_shapes[0]),
                                   std::make_shared<ov::op::v0::Parameter>(ngPrc, input_shapes[1]),
                                   std::make_shared<ov::op::v0::Parameter>(ngPrc, input_shapes[2])};
        std::shared_ptr<ov::Node> x_node = params[0];
        if (major_batch == "SequenceMajor") {
            x_node = std::make_shared<ov::op::v1::Transpose>(
                params[0], ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, 0, 2}));
        }
        std::shared_ptr<ov::Node> seq_lengths_node;
        bool is_pure_sequence =
            (m_mode == SequenceTestsMode::PURE_SEQ || m_mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM ||
             m_mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST);
        EXPECT_EQ(is_pure_sequence, true);
        if (m_mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM) {
            auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, input_shapes[3]);
            seq_lengths_node = param;
            seq_lengths_node->set_friendly_name("seq_lengths");
            params.push_back(param);
        } else if (m_mode == ngraph::helpers::SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST) {
            seq_lengths_node = ngraph::builder::makeConstant<int64_t>(
                ov::element::i64, input_shapes[3], {}, true, static_cast<int64_t>(seq_lengths), 0.f);
        } else {
            std::vector<int64_t> lengths(input_shapes[3][0], seq_lengths);
            seq_lengths_node = ngraph::builder::makeConstant(ov::element::i64, input_shapes[3], lengths, false);
        }
        std::shared_ptr<ov::Node> W, R, B;
        if (WRBType == InputLayerType::PARAMETER) {
            const auto W_param = std::make_shared<ov::op::v0::Parameter>(ngPrc, W_shape);
            const auto R_param = std::make_shared<ov::op::v0::Parameter>(ngPrc, R_shape);
            const auto B_param = std::make_shared<ov::op::v0::Parameter>(ngPrc, B_shape);
            W = W_param;
            R = R_param;
            B = B_param;
            params.push_back(W_param);
            params.push_back(R_param);
            params.push_back(B_param);
        } else {
            W = ngraph::builder::makeConstant(ngPrc, W_shape, {}, true, up_to, start_from, counter++);
            R = ngraph::builder::makeConstant(ngPrc, R_shape, {}, true, up_to, start_from, counter++);
            B = ngraph::builder::makeConstant(ngPrc, B_shape, {}, true, up_to, start_from, counter++);
        }

        auto lstm_sequence = std::make_shared<ov::op::v5::LSTMSequence>(x_node,
                                                                        params[1],
                                                                        params[2],
                                                                        seq_lengths_node,
                                                                        W,
                                                                        R,
                                                                        B,
                                                                        hidden_size,
                                                                        direction,
                                                                        std::vector<float>{},
                                                                        std::vector<float>{},
                                                                        activations,
                                                                        clip);

        std::shared_ptr<ov::Node> transpose_y;
        std::shared_ptr<ov::Node> transpose_ho;
        std::shared_ptr<ov::Node> transpose_co;
        if (major_batch == "BatchMajor") {
            if (1 == seq_lengths) {
                auto shape = lstm_sequence->output(0).get_shape();
                transpose_y = std::make_shared<ov::op::v1::Reshape>(
                    lstm_sequence->output(0),
                    ov::op::v0::Constant::create(
                        ov::element::i32, ov::Shape{4}, {shape[0], shape[2], shape[1], shape[3]}),
                    true);
            } else {
                transpose_y = std::make_shared<ov::op::v1::Transpose>(
                    lstm_sequence->output(0),
                    ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 2, 1, 3}));
            }
        } else {
            if (1 == seq_lengths) {
                auto shape = lstm_sequence->output(0).get_shape();
                transpose_y = std::make_shared<ov::op::v1::Reshape>(
                    lstm_sequence->output(0),
                    ov::op::v0::Constant::create(
                        ov::element::i32, ov::Shape{4}, {shape[2], shape[0], shape[1], shape[3]}),
                    true);
            } else {
                transpose_y = std::make_shared<ov::op::v1::Transpose>(
                    lstm_sequence->output(0),
                    ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {2, 0, 1, 3}));
            }
        }
        transpose_y->set_friendly_name("output_0");
        if (1 == batch) {
            auto shape = lstm_sequence->output(1).get_shape();
            transpose_ho = std::make_shared<ov::op::v1::Reshape>(
                lstm_sequence->output(1),
                ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {shape[1], shape[0], shape[2]}),
                true);
            transpose_co = std::make_shared<ov::op::v1::Reshape>(
                lstm_sequence->output(2),
                ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {shape[1], shape[0], shape[2]}),
                true);
        } else {
            transpose_ho = std::make_shared<ov::op::v1::Transpose>(
                lstm_sequence->output(1), ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, 0, 2}));
            transpose_co = std::make_shared<ov::op::v1::Transpose>(
                lstm_sequence->output(2), ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, 0, 2}));
        }
        transpose_ho->set_friendly_name("output_1");
        transpose_co->set_friendly_name("output_2");

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(transpose_y),
                                 std::make_shared<ov::op::v0::Result>(transpose_ho),
                                 std::make_shared<ov::op::v0::Result>(transpose_co)};
        function = std::make_shared<ov::Model>(results, params, "lstm_sequence_optimized");
    }
    ngraph::helpers::SequenceTestsMode m_mode;
    int64_t m_max_seq_len = 0;
};

TEST_P(CUDALSTMSequenceTest, CompareWithRefs) { Run(); }

TEST_P(CUDALSTMSequenceOptimizedTest, CompareWithRefs) { Run(); }

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
                                           ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
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
                                           ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
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
                                           ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
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
                                           ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
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
                                           ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
                        CUDALSTMSequenceTest::getTestCaseName);

// ------------- LSTM Sequence optimized -------------

const std::vector<std::string> majors = {"BatchMajor", "SequenceMajor"};

INSTANTIATE_TEST_CASE_P(smoke_LSTMSequenceOptmized_01,
                        CUDALSTMSequenceOptimizedTest,
                        ::testing::Combine(::testing::Values(testMode),
                                           ::testing::ValuesIn(smoke_max_seq_lengths),
                                           ::testing::ValuesIn(batches),
                                           ::testing::ValuesIn(smoke_01_hidden_sizes),
                                           ::testing::ValuesIn(smoke_01_input_sizes),
                                           ::testing::Values(activations),
                                           ::testing::Values(no_clip),
                                           ::testing::ValuesIn(majors),
                                           ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
                        CUDALSTMSequenceOptimizedTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_LSTMSequenceOptmized_02,
                        CUDALSTMSequenceOptimizedTest,
                        ::testing::Combine(::testing::Values(testMode),
                                           ::testing::ValuesIn(smoke_max_seq_lengths),
                                           ::testing::ValuesIn(batches),
                                           ::testing::ValuesIn(smoke_02_hidden_sizes),
                                           ::testing::ValuesIn(smoke_02_input_sizes),
                                           ::testing::Values(activations),
                                           ::testing::Values(no_clip),
                                           ::testing::ValuesIn(majors),
                                           ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
                        CUDALSTMSequenceOptimizedTest::getTestCaseName);

}  // namespace
}  // namespace LayerTestsDefinitions
