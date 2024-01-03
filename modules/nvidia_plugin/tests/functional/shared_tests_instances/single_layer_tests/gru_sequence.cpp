// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/gru_sequence.hpp"

#include <cuda_test_constants.hpp>
#include <openvino/op/util/attr_types.hpp>
#include <vector>

#include "common_test_utils/node_builders/gru_cell.hpp"
#include "unsymmetrical_comparer.hpp"

using ov::test::utils::InputLayerType;
using ov::test::utils::SequenceTestsMode;

namespace LayerTestsDefinitions {

class CUDNNGRUSequenceTest : public UnsymmetricalComparer<GRUSequenceTest> {
public:
    void SetUp() override {
        GRUSequenceTest::SetUp();
        threshold = 0.01f;
        constexpr float up_to = 1.0f;
        constexpr float start_from = -1.0f;

        const auto& ops = function->get_ordered_ops();
        int seed = 1;
        for (const auto& op : ops) {
            if (std::dynamic_pointer_cast<ov::op::v0::Constant>(op)) {
                if (op->get_element_type() == ov::element::Type_t::f32) {
                    ov::Tensor random_tensor(op->get_element_type(), op->get_shape());
                    ov::test::utils::fill_tensor_random(random_tensor, up_to - start_from, start_from, 1, seed++);
                    function->replace_node(op, std::make_shared<ov::op::v0::Constant>(random_tensor));
                }
            }
        }
    }
};
TEST_P(CUDNNGRUSequenceTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
};

class LPCNetCUDNNGRUSequenceTest : public UnsymmetricalComparer<GRUSequenceTest> {
public:
    void SetUp() override {
        threshold = 0.05f;  // there is one place with 0.0254 difference when sequence length and shapes are big, e.g.
                            // 10 and shapes input 512, hidden size 384
        updatedGRUSequenceTest_SetUp();
        UpdateConstantNoded();
    }
    void UpdateConstantNoded() {
        constexpr float up_to = 1.0f;
        constexpr float start_from = -1.0f;

        const auto& ops = function->get_ordered_ops();
        int seed = 1;
        for (const auto& op : ops) {
            if (std::dynamic_pointer_cast<ov::op::v0::Constant>(op)) {
                if (op->get_element_type() == ov::element::Type_t::f32) {
                    ov::Tensor random_tensor(op->get_element_type(), op->get_shape());
                    ov::test::utils::fill_tensor_random(random_tensor, up_to - start_from, start_from, 1, seed++);
                    function->replace_node(op, std::make_shared<ov::op::v0::Constant>(random_tensor));
                }
            }
        }
    }

    void updatedGRUSequenceTest_SetUp() {
        size_t seq_lengths;
        size_t batch;
        size_t hidden_size;
        size_t input_size = 512;
        std::vector<std::string> activations;
        float clip;
        bool linear_before_reset;
        SequenceTestsMode mode;
        ov::op::RecurrentSequenceDirection direction;
        InputLayerType WRBType;
        InferenceEngine::Precision netPrecision;
        std::tie(mode,
                 seq_lengths,
                 batch,
                 hidden_size,
                 activations,
                 clip,
                 linear_before_reset,
                 direction,
                 WRBType,
                 netPrecision,
                 targetDevice) = this->GetParam();
        size_t num_directions = direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;
        std::vector<std::vector<size_t>> inputShapes = {
            {{batch, seq_lengths, input_size},
             {batch, num_directions, hidden_size},
             {batch},
             {num_directions, 3 * hidden_size, input_size},
             {num_directions, 3 * hidden_size, hidden_size},
             {num_directions, (linear_before_reset ? 4 : 3) * hidden_size}},
        };
        m_max_seq_len_ = seq_lengths;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector params;
        params.push_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[0])));
        params.push_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[1])));

        ASSERT_EQ(InputLayerType::CONSTANT, WRBType);
        std::vector<ov::Shape> WRB = {inputShapes[3], inputShapes[4], inputShapes[5], inputShapes[2]};
        auto gru_sequence = ov::test::utils::make_gru(ov::OutputVector{params[0], params[1]},
                                                      WRB,
                                                      hidden_size,
                                                      activations,
                                                      {},
                                                      {},
                                                      clip,
                                                      linear_before_reset,
                                                      true,
                                                      direction,
                                                      mode);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gru_sequence->output(0)),
                                 std::make_shared<ov::op::v0::Result>(gru_sequence->output(1))};
        function = std::make_shared<ngraph::Function>(results, params, "gru_sequence");
        bool ti_found = ngraph::helpers::is_tensor_iterator_exist(function);
        EXPECT_EQ(ti_found, false);
    }

    void GenerateInputs() override {
        inputs.clear();
        for (const auto& input : executableNetwork.GetInputsInfo()) {
            const auto& info = input.second;
            auto blob = LayerTestsCommon::GenerateInput(*info);
            if (input.first == "seq_lengths") {
                blob = FuncTestUtils::createAndFillBlob(info->getTensorDesc(), m_max_seq_len_, 0);
            }
            inputs.push_back(blob);
        }
    }

private:
    int64_t m_max_seq_len_ = 0;
};

TEST_P(LPCNetCUDNNGRUSequenceTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
};

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
SequenceTestsMode mode{SequenceTestsMode::PURE_SEQ};
// output values increase rapidly without clip, so use only seq_lengths = 2
std::vector<size_t> seq_lengths{1, 2, 5, 10};
std::vector<size_t> batch{1};
std::vector<size_t> hidden_size{1, 10};
std::vector<std::vector<std::string>> activations = {{"sigmoid", "tanh"}};
std::vector<bool> linear_before_reset = {true};  // false doesn't work properly in reference GRUCell implementation
std::vector<float> clip{0.f};
std::vector<ov::op::RecurrentSequenceDirection> direction = {ov::op::RecurrentSequenceDirection::FORWARD};
std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                         InferenceEngine::Precision::FP16};

INSTANTIATE_TEST_CASE_P(
    GRUSequenceCommon,
    CUDNNGRUSequenceTest,
    ::testing::Combine(::testing::Values(mode),
                       ::testing::ValuesIn(seq_lengths),
                       ::testing::ValuesIn(batch),
                       ::testing::ValuesIn(hidden_size),
                       // ::testing::ValuesIn(input_size), // hardcoded to 10 due to Combine supports up to 10 args
                       ::testing::ValuesIn(activations),
                       ::testing::ValuesIn(clip),
                       ::testing::ValuesIn(linear_before_reset),
                       ::testing::ValuesIn(direction),
                       ::testing::Values(InputLayerType::CONSTANT),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
    GRUSequenceTest::getTestCaseName);

// ------------- Smoke test -------------
const std::vector<size_t> smoke_seq_lengths{1, 2, 5};
const std::vector<size_t> smoke_batchs{1, 16};
const std::vector<size_t> smoke_hidden_sizes{1, 5, 10};
INSTANTIATE_TEST_CASE_P(
    smoke_GRUSequenceCommon,
    CUDNNGRUSequenceTest,
    ::testing::Combine(::testing::Values(mode),
                       ::testing::ValuesIn(smoke_seq_lengths),
                       ::testing::ValuesIn(smoke_batchs),
                       ::testing::ValuesIn(smoke_hidden_sizes),
                       // ::testing::ValuesIn(input_size), // hardcoded to 10 due to Combine supports up to 10 args
                       ::testing::ValuesIn(activations),
                       ::testing::ValuesIn(clip),
                       ::testing::ValuesIn(linear_before_reset),
                       ::testing::ValuesIn(direction),
                       ::testing::Values(InputLayerType::CONSTANT),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
    GRUSequenceTest::getTestCaseName);

// -------------  LPCNet shapes  -------------
const std::vector<size_t> lpcnet_seq_lengths{5, 10};
const std::vector<size_t> lpcnet_batchs{1};
const std::vector<size_t> lpcnet_hidden_sizes{16, 384};
INSTANTIATE_TEST_CASE_P(LPCNetCUDNNGRUSequenceShapeTest,
                        LPCNetCUDNNGRUSequenceTest,
                        ::testing::Combine(::testing::Values(mode),
                                           ::testing::ValuesIn(lpcnet_seq_lengths),
                                           ::testing::ValuesIn(lpcnet_batchs),
                                           ::testing::ValuesIn(lpcnet_hidden_sizes),
                                           // ::testing::ValuesIn(input_size), // 512 in this test
                                           ::testing::ValuesIn(activations),
                                           ::testing::ValuesIn(clip),
                                           ::testing::ValuesIn(linear_before_reset),
                                           ::testing::ValuesIn(direction),
                                           ::testing::Values(InputLayerType::CONSTANT),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
                        GRUSequenceTest::getTestCaseName);

}  // namespace
