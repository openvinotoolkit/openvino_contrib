// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformer/bidirectional_lstm_sequence_composition.hpp"

#include <gtest/gtest.h>

#include <tuple>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
#include "transformer/nodes/lstm_sequence_optimized.hpp"

using ov::nvidia_gpu::nodes::LSTMSequenceOptimized;
using MajorFormat = LSTMSequenceOptimized::MajorFormat;
using namespace ov;
using namespace std;

namespace testing {

namespace {
struct InputParameters {
    uint64_t batch_size = 1;
    uint64_t seq_length = 326;
    uint64_t input_size = 512;
    uint64_t hidden_size = 512;
    ov::op::RecurrentSequenceDirection direction = ov::op::RecurrentSequenceDirection::BIDIRECTIONAL;
    std::vector<float> activations_alpha = {};
    std::vector<float> activations_beta = {};
    std::vector<std::string> activations = {"sigmoid", "tanh", "tanh"};
    float clip = 0.f;

    uint64_t num_directions() const { return (direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL) ? 2 : 1; }
};
struct Inputs {
    std::shared_ptr<op::v0::Parameter> X;
    std::shared_ptr<op::v0::Parameter> H;
    std::shared_ptr<op::v0::Parameter> C;

    ParameterVector get_parameter_vector() { return ParameterVector{X, H, C}; }
};
struct Constants {
    std::shared_ptr<op::v0::Constant> S;
    std::shared_ptr<op::v0::Constant> W;
    std::shared_ptr<op::v0::Constant> R;
    std::shared_ptr<op::v0::Constant> B;
};
std::shared_ptr<ov::op::v5::LSTMSequence> create_lstm_sequence(const InputParameters& params,
                                                               const Inputs& inputs,
                                                               const Constants& constants) {
    return std::make_shared<ov::op::v5::LSTMSequence>(inputs.X,
                                                      inputs.H,
                                                      inputs.C,
                                                      constants.S,
                                                      constants.W,
                                                      constants.R,
                                                      constants.B,
                                                      params.hidden_size,
                                                      params.direction,
                                                      params.activations_alpha,
                                                      params.activations_beta,
                                                      params.activations,
                                                      params.clip);
}
std::shared_ptr<ov::op::v5::LSTMSequence> create_lstm_sequence(const InputParameters& params,
                                                               const std::shared_ptr<Node>& X,
                                                               const std::shared_ptr<Node>& H,
                                                               const std::shared_ptr<Node>& C,
                                                               const Constants& constants) {
    return std::make_shared<ov::op::v5::LSTMSequence>(X,
                                                      H,
                                                      C,
                                                      constants.S,
                                                      constants.W,
                                                      constants.R,
                                                      constants.B,
                                                      params.hidden_size,
                                                      params.direction,
                                                      params.activations_alpha,
                                                      params.activations_beta,
                                                      params.activations,
                                                      params.clip);
}
std::shared_ptr<LSTMSequenceOptimized> create_lstm_sequence_optimized(
    const InputParameters& params,
    const Inputs& inputs,
    const Constants& constants,
    const LSTMSequenceOptimized::MajorFormat major_format = MajorFormat::BatchMajor) {
    return std::make_shared<LSTMSequenceOptimized>(inputs.X,
                                                   inputs.H,
                                                   inputs.C,
                                                   constants.S,
                                                   constants.W,
                                                   constants.R,
                                                   constants.B,
                                                   params.hidden_size,
                                                   params.direction,
                                                   major_format,
                                                   params.activations_alpha,
                                                   params.activations_beta,
                                                   params.activations,
                                                   params.clip);
}
Inputs get_inputs(const InputParameters& params, bool transpose_x = false) {
    Inputs inputs;

    auto num_directions = params.num_directions();

    if (transpose_x) {
        inputs.X = std::make_shared<op::v0::Parameter>(element::f32,
                                                       Shape{params.seq_length, params.batch_size, params.input_size});
    } else {
        inputs.X = std::make_shared<op::v0::Parameter>(element::f32,
                                                       Shape{params.batch_size, params.seq_length, params.input_size});
    }
    inputs.H =
        std::make_shared<op::v0::Parameter>(element::f32, Shape{params.batch_size, num_directions, params.hidden_size});
    inputs.C =
        std::make_shared<op::v0::Parameter>(element::f32, Shape{params.batch_size, num_directions, params.hidden_size});

    return inputs;
}
Constants get_constants(const InputParameters& params, const uint64_t num_gates = 4) {
    Constants constants;

    auto num_directions = params.num_directions();

    const auto seq_len_val = std::vector<int>(params.batch_size, params.seq_length);
    const auto w_val = std::vector<float>(num_directions * num_gates * params.hidden_size * params.input_size, 0);
    const auto r_val = std::vector<float>(num_directions * num_gates * params.hidden_size * params.hidden_size, 0);
    const auto b_val = std::vector<float>(num_directions * num_gates * params.hidden_size, 0);

    constants.S = std::make_shared<op::v0::Constant>(element::i32, Shape{params.batch_size}, seq_len_val);
    constants.W = std::make_shared<op::v0::Constant>(
        element::f32, Shape{num_directions, num_gates * params.hidden_size, params.input_size}, w_val);
    constants.R = std::make_shared<op::v0::Constant>(
        element::f32, Shape{num_directions, num_gates * params.hidden_size, params.hidden_size}, r_val);
    constants.B =
        std::make_shared<op::v0::Constant>(element::f32, Shape{num_directions, num_gates * params.hidden_size}, b_val);

    return constants;
}

std::shared_ptr<ov::op::v1::Transpose> get_4d_batch_transpose(const ov::Output<ov::Node>& output) {
    return std::make_shared<ov::op::v1::Transpose>(output,
                                                   op::v0::Constant::create(element::i64, Shape{4}, {0, 2, 1, 3}));
}

std::shared_ptr<ov::op::v1::Transpose> get_4d_sequence_transpose(const ov::Output<ov::Node>& output) {
    return std::make_shared<ov::op::v1::Transpose>(output,
                                                   op::v0::Constant::create(element::i64, Shape{4}, {2, 0, 1, 3}));
}

std::shared_ptr<ov::op::v1::Transpose> get_3d_transpose(const ov::Output<ov::Node>& output) {
    return std::make_shared<ov::op::v1::Transpose>(output, op::v0::Constant::create(element::i64, Shape{3}, {1, 0, 2}));
}
}  // namespace

TEST(bidirectional_lstm_sequence_composition, bidirectional_lstm_no_ho_co) {
    shared_ptr<ov::Model> model, model_ref;
    InputParameters params;
    {
        auto inputs = get_inputs(params);
        auto constants = get_constants(params);
        auto lstm_sequence = create_lstm_sequence(params, inputs, constants);
        auto transpose = get_4d_sequence_transpose(lstm_sequence->output(0));
        model = std::make_shared<Model>(
            OutputVector{transpose->output(0), lstm_sequence->output(1), lstm_sequence->output(2)},
            inputs.get_parameter_vector());
        model_ref = model->clone();
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::BidirectionalSequenceComposition>();
    pass_manager.run_passes(model);
    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(bidirectional_lstm_sequence_composition, bidirectional_lstm_no_ho_co_batch) {
    shared_ptr<ov::Model> model, model_ref;
    InputParameters params;
    {
        auto inputs = get_inputs(params);
        auto constants = get_constants(params);
        auto lstm_sequence = create_lstm_sequence(params, inputs, constants);
        auto transpose0 = get_4d_sequence_transpose(lstm_sequence->output(0));
        auto reshape_const =
            op::v0::Constant::create<int64_t>(element::i64,
                                              Shape{3},
                                              {static_cast<int64_t>(params.seq_length),
                                               1,
                                               static_cast<int64_t>(params.num_directions() * params.hidden_size)});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(transpose0, reshape_const, false);
        auto transpose1 = get_3d_transpose(reshape);
        model = std::make_shared<Model>(OutputVector{transpose1->output(0)}, inputs.get_parameter_vector());
    }
    {
        auto inputs = get_inputs(params);
        auto constants = get_constants(params);
        auto lstm_sequence = create_lstm_sequence_optimized(params, inputs, constants, MajorFormat::BatchMajor);
        auto y_reshape_const =
            op::v0::Constant::create<int32_t>(element::i32,
                                              Shape{3},
                                              {static_cast<int32_t>(params.batch_size),
                                               static_cast<int32_t>(params.seq_length),
                                               static_cast<int32_t>(params.num_directions() * params.hidden_size)});
        auto y_reshape = std::make_shared<ov::op::v1::Reshape>(lstm_sequence->output(0), y_reshape_const, false);

        model_ref = std::make_shared<Model>(OutputVector{y_reshape->output(0)}, inputs.get_parameter_vector());
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::BidirectionalSequenceComposition>();
    pass_manager.run_passes(model);
    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(bidirectional_lstm_sequence_composition, bidirectional_lstm_batch_major_4d_out_h0_c0_fail) {
    shared_ptr<ov::Model> model, model_ref;
    InputParameters params;
    {
        auto inputs = get_inputs(params);
        auto constants = get_constants(params);
        auto lstm_sequence = create_lstm_sequence(params, inputs, constants);
        auto y_transpose = get_4d_batch_transpose(lstm_sequence->output(0));
        // Unexpected transpose
        auto hc_transpose_const = op::v0::Constant::create(element::i64, Shape{3}, {0, 2, 1});
        auto h_transpose = std::make_shared<ov::op::v1::Transpose>(lstm_sequence->output(1), hc_transpose_const);
        auto c_transpose = std::make_shared<ov::op::v1::Transpose>(lstm_sequence->output(2), hc_transpose_const);
        model =
            std::make_shared<Model>(OutputVector{y_transpose, h_transpose, c_transpose}, inputs.get_parameter_vector());
        model_ref = model->clone();
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::BidirectionalSequenceComposition>();
    pass_manager.run_passes(model);
    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(bidirectional_lstm_sequence_composition, bidirectional_lstm_batch_major_4d_out) {
    shared_ptr<ov::Model> model, model_ref;
    InputParameters params;
    {
        auto inputs = get_inputs(params);
        auto constants = get_constants(params);
        auto lstm_sequence = create_lstm_sequence(params, inputs, constants);
        auto y_transpose = get_4d_batch_transpose(lstm_sequence->output(0));
        auto h_transpose = get_3d_transpose(lstm_sequence->output(1));
        auto c_transpose = get_3d_transpose(lstm_sequence->output(2));
        model =
            std::make_shared<Model>(OutputVector{y_transpose, h_transpose, c_transpose}, inputs.get_parameter_vector());
    }
    {
        auto inputs = get_inputs(params);
        auto constants = get_constants(params);
        auto lstm_sequence = create_lstm_sequence_optimized(params, inputs, constants, MajorFormat::BatchMajor);
        model_ref = std::make_shared<Model>(
            OutputVector{lstm_sequence->output(0), lstm_sequence->output(1), lstm_sequence->output(2)},
            inputs.get_parameter_vector());
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::BidirectionalSequenceComposition>();
    pass_manager.run_passes(model);
    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(bidirectional_lstm_sequence_composition, bidirectional_lstm_batch_major_3d_out) {
    shared_ptr<ov::Model> model, model_ref;
    InputParameters params;
    {
        auto inputs = get_inputs(params);
        auto constants = get_constants(params);
        auto lstm_sequence = create_lstm_sequence(params, inputs, constants);
        auto y_transpose0 = get_4d_sequence_transpose(lstm_sequence->output(0));
        auto y_reshape_const =
            op::v0::Constant::create<int64_t>(element::i64,
                                              Shape{3},
                                              {static_cast<int64_t>(params.seq_length),
                                               static_cast<int64_t>(params.batch_size),
                                               static_cast<int64_t>(params.num_directions() * params.hidden_size)});
        auto y_reshape = std::make_shared<ov::op::v1::Reshape>(y_transpose0, y_reshape_const, false);
        auto y_transpose1 = get_3d_transpose(y_reshape);
        auto h_transpose = get_3d_transpose(lstm_sequence->output(1));
        auto c_transpose = get_3d_transpose(lstm_sequence->output(2));
        model = std::make_shared<Model>(OutputVector{y_transpose1, h_transpose, c_transpose},
                                        inputs.get_parameter_vector());
    }
    {
        auto inputs = get_inputs(params);
        auto constants = get_constants(params);
        auto lstm_sequence = create_lstm_sequence_optimized(params, inputs, constants, MajorFormat::BatchMajor);
        auto y_reshape_const =
            op::v0::Constant::create<int32_t>(element::i32,
                                              Shape{3},
                                              {static_cast<int32_t>(params.batch_size),
                                               static_cast<int32_t>(params.seq_length),
                                               static_cast<int32_t>(params.num_directions() * params.hidden_size)});
        auto y_reshape = std::make_shared<ov::op::v1::Reshape>(lstm_sequence->output(0), y_reshape_const, false);
        model_ref = std::make_shared<Model>(OutputVector{y_reshape, lstm_sequence->output(1), lstm_sequence->output(2)},
                                            inputs.get_parameter_vector());
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::BidirectionalSequenceComposition>();
    pass_manager.run_passes(model);
    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(bidirectional_lstm_sequence_composition, bidirectional_lstm_batch_major_3d_out_all_equal_fail) {
    shared_ptr<ov::Model> model, model_ref;
    InputParameters params;
    params.batch_size = 2;
    params.seq_length = 2;
    params.input_size = 2;
    params.hidden_size = 2;
    {
        auto inputs = get_inputs(params);
        auto constants = get_constants(params);
        auto lstm_sequence = create_lstm_sequence(params, inputs, constants);
        // Unexpected transpose
        auto y_transpose0_const = op::v0::Constant::create(element::i64, Shape{4}, {3, 0, 2, 1});
        auto y_transpose0 = std::make_shared<ov::op::v1::Transpose>(lstm_sequence->output(0), y_transpose0_const);
        auto y_reshape_const =
            op::v0::Constant::create<int64_t>(element::i64,
                                              Shape{3},
                                              {static_cast<int64_t>(params.seq_length),
                                               static_cast<int64_t>(params.batch_size),
                                               static_cast<int64_t>(params.num_directions() * params.hidden_size)});
        auto y_reshape = std::make_shared<ov::op::v1::Reshape>(y_transpose0, y_reshape_const, false);
        auto y_transpose1 = get_3d_transpose(y_reshape);
        auto h_transpose = get_3d_transpose(lstm_sequence->output(1));
        auto c_transpose = get_3d_transpose(lstm_sequence->output(2));
        model = std::make_shared<Model>(OutputVector{y_transpose1, h_transpose, c_transpose},
                                        inputs.get_parameter_vector());
        model_ref = model->clone();
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::BidirectionalSequenceComposition>();
    pass_manager.run_passes(model);
    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(bidirectional_lstm_sequence_composition, bidirectional_lstm_sequence_major_no_tranpose) {
    shared_ptr<ov::Model> model, model_ref;
    InputParameters params;
    {
        auto inputs = get_inputs(params);
        auto constants = get_constants(params);
        auto lstm_sequence = create_lstm_sequence(params, inputs, constants);
        auto y_transpose = get_4d_sequence_transpose(lstm_sequence->output(0));
        auto h_transpose = get_3d_transpose(lstm_sequence->output(1));
        auto c_transpose = get_3d_transpose(lstm_sequence->output(2));
        model =
            std::make_shared<Model>(OutputVector{y_transpose, h_transpose, c_transpose}, inputs.get_parameter_vector());
        model_ref = model->clone();
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::BidirectionalSequenceComposition>();
    pass_manager.run_passes(model);
    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(bidirectional_lstm_sequence_composition, bidirectional_lstm_sequence_major_batch1) {
    shared_ptr<ov::Model> model, model_ref;
    InputParameters params;
    {
        auto inputs = get_inputs(params, true);
        auto constants = get_constants(params);
        auto x_transpose = get_3d_transpose(inputs.X);
        auto lstm_sequence = create_lstm_sequence(params, x_transpose, inputs.H, inputs.C, constants);
        auto y_transpose = get_4d_sequence_transpose(lstm_sequence->output(0));
        auto h_transpose = get_3d_transpose(lstm_sequence->output(1));
        auto c_transpose = get_3d_transpose(lstm_sequence->output(2));
        model =
            std::make_shared<Model>(OutputVector{y_transpose, h_transpose, c_transpose}, inputs.get_parameter_vector());
    }
    {
        auto inputs = get_inputs(params, true);
        auto constants = get_constants(params);
        auto lstm_sequence = create_lstm_sequence_optimized(params, inputs, constants, MajorFormat::SequenceMajor);
        model_ref = std::make_shared<Model>(
            OutputVector{lstm_sequence->output(0), lstm_sequence->output(1), lstm_sequence->output(2)},
            inputs.get_parameter_vector());
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::BidirectionalSequenceComposition>();
    pass_manager.run_passes(model);
    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(bidirectional_lstm_sequence_composition, bidirectional_lstm_sequence_major_all_params_equal) {
    shared_ptr<ov::Model> model, model_ref;
    InputParameters params;
    params.batch_size = 2;
    params.seq_length = 2;
    params.input_size = 2;
    params.hidden_size = 2;
    {
        auto inputs = get_inputs(params, true);
        auto constants = get_constants(params);
        auto x_transpose = get_3d_transpose(inputs.X);
        auto lstm_sequence = create_lstm_sequence(params, x_transpose, inputs.H, inputs.C, constants);
        auto y_transpose = get_4d_sequence_transpose(lstm_sequence->output(0));
        auto h_transpose = get_3d_transpose(lstm_sequence->output(1));
        auto c_transpose = get_3d_transpose(lstm_sequence->output(2));
        model =
            std::make_shared<Model>(OutputVector{y_transpose, h_transpose, c_transpose}, inputs.get_parameter_vector());
    }
    {
        auto inputs = get_inputs(params, true);
        auto constants = get_constants(params);
        auto lstm_sequence = create_lstm_sequence_optimized(params, inputs, constants, MajorFormat::SequenceMajor);
        model_ref = std::make_shared<Model>(
            OutputVector{lstm_sequence->output(0), lstm_sequence->output(1), lstm_sequence->output(2)},
            inputs.get_parameter_vector());
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::BidirectionalSequenceComposition>();
    pass_manager.run_passes(model);
    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}
TEST(bidirectional_lstm_sequence_composition, bidirectional_lstm_sequence_major_all_params_equal_fail) {
    shared_ptr<ov::Model> model, model_ref;
    InputParameters params;
    params.batch_size = 2;
    params.seq_length = 2;
    params.input_size = 2;
    params.hidden_size = 2;
    {
        auto inputs = get_inputs(params, true);
        auto constants = get_constants(params);
        auto x_transpose = get_3d_transpose(inputs.X);
        auto lstm_sequence = create_lstm_sequence(params, x_transpose, inputs.H, inputs.C, constants);
        // Unexpected transpose
        auto y_transpose_const = op::v0::Constant::create(element::i64, Shape{4}, {3, 0, 2, 1});
        auto y_transpose = std::make_shared<ov::op::v1::Transpose>(lstm_sequence->output(0), y_transpose_const);
        auto h_transpose = get_3d_transpose(lstm_sequence->output(1));
        auto c_transpose = get_3d_transpose(lstm_sequence->output(2));
        model =
            std::make_shared<Model>(OutputVector{y_transpose, h_transpose, c_transpose}, inputs.get_parameter_vector());
        model_ref = model->clone();
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::BidirectionalSequenceComposition>();
    pass_manager.run_passes(model);
    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(bidirectional_lstm_sequence_composition, bidirectional_lstm_sequence_major_batch4) {
    shared_ptr<ov::Model> model, model_ref;
    InputParameters params;
    params.batch_size = 4;
    {
        auto inputs = get_inputs(params, true);
        auto constants = get_constants(params);
        auto x_transpose = get_3d_transpose(inputs.X);
        auto lstm_sequence = create_lstm_sequence(params, x_transpose, inputs.H, inputs.C, constants);
        auto y_transpose = get_4d_sequence_transpose(lstm_sequence->output(0));
        auto h_transpose = get_3d_transpose(lstm_sequence->output(1));
        auto c_transpose = get_3d_transpose(lstm_sequence->output(2));
        model =
            std::make_shared<Model>(OutputVector{y_transpose, h_transpose, c_transpose}, inputs.get_parameter_vector());
    }
    {
        auto inputs = get_inputs(params, true);
        auto constants = get_constants(params);
        auto lstm_sequence = create_lstm_sequence_optimized(params, inputs, constants, MajorFormat::SequenceMajor);
        model_ref = std::make_shared<Model>(
            OutputVector{lstm_sequence->output(0), lstm_sequence->output(1), lstm_sequence->output(2)},
            inputs.get_parameter_vector());
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::BidirectionalSequenceComposition>();
    pass_manager.run_passes(model);
    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(bidirectional_lstm_sequence_composition, bidirectional_lstm_batch_major_matmul) {
    shared_ptr<ov::Model> model, model_ref;
    InputParameters params;
    {
        auto inputs = get_inputs(params);
        auto constants = get_constants(params);
        auto lstm_sequence = create_lstm_sequence(params, inputs, constants);
        auto y_transpose = get_4d_sequence_transpose(lstm_sequence->output(0));
        auto y_reshape_const =
            op::v0::Constant::create<int32_t>(element::i32,
                                              Shape{3},
                                              {static_cast<int32_t>(params.batch_size),
                                               static_cast<int32_t>(params.seq_length),
                                               static_cast<int32_t>(params.num_directions() * params.hidden_size)});
        auto y_reshape = std::make_shared<ov::op::v1::Reshape>(y_transpose, y_reshape_const, false);
        auto h_transpose = get_3d_transpose(lstm_sequence->output(1));
        auto c_transpose = get_3d_transpose(lstm_sequence->output(2));
        auto matmul_const =
            op::v0::Constant::create(element::f32, {80, params.num_directions() * params.input_size}, {1});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(y_reshape, matmul_const, false, true);

        model = std::make_shared<Model>(OutputVector{matmul, h_transpose, c_transpose}, inputs.get_parameter_vector());
    }
    {
        auto inputs = get_inputs(params);
        auto constants = get_constants(params);
        auto lstm_sequence = create_lstm_sequence_optimized(params, inputs, constants, MajorFormat::BatchMajor);
        auto y_reshape_const =
            op::v0::Constant::create<int32_t>(element::i32,
                                              Shape{3},
                                              {static_cast<int32_t>(params.batch_size),
                                               static_cast<int32_t>(params.seq_length),
                                               static_cast<int32_t>(params.num_directions() * params.hidden_size)});
        auto y_reshape = std::make_shared<ov::op::v1::Reshape>(lstm_sequence->output(0), y_reshape_const, false);
        auto matmul_const =
            op::v0::Constant::create(element::f32, {80, params.num_directions() * params.input_size}, {1});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(y_reshape, matmul_const, false, true);
        model_ref =
            std::make_shared<Model>(OutputVector{matmul->output(0), lstm_sequence->output(1), lstm_sequence->output(2)},
                                    inputs.get_parameter_vector());
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::BidirectionalSequenceComposition>();
    pass_manager.run_passes(model);
    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(bidirectional_lstm_sequence_composition, LSTMSequence_compose_batch_major) {
    shared_ptr<ov::Model> model, model_ref;
    InputParameters params;
    {
        auto inputs = get_inputs(params);
        auto constants = get_constants(params);
        auto axis_0 = op::v0::Constant::create(element::i64, Shape{}, {0});
        auto axis_1 = op::v0::Constant::create(element::i64, Shape{}, {1});
        auto H_split = std::make_shared<op::v1::Split>(inputs.H, axis_1, 2);
        auto C_split = std::make_shared<op::v1::Split>(inputs.C, axis_1, 2);
        auto W_split = std::make_shared<op::v1::Split>(constants.W, axis_0, 2);
        auto R_split = std::make_shared<op::v1::Split>(constants.R, axis_0, 2);
        auto B_split = std::make_shared<op::v1::Split>(constants.B, axis_0, 2);

        auto lstm_seq_forward = std::make_shared<ov::op::v5::LSTMSequence>(inputs.X,
                                                                           H_split->output(0),
                                                                           C_split->output(0),
                                                                           constants.S,
                                                                           W_split->output(0),
                                                                           R_split->output(0),
                                                                           B_split->output(0),
                                                                           params.hidden_size,
                                                                           op::RecurrentSequenceDirection::FORWARD);
        auto lstm_seq_reverse = std::make_shared<ov::op::v5::LSTMSequence>(inputs.X,
                                                                           H_split->output(1),
                                                                           C_split->output(1),
                                                                           constants.S,
                                                                           W_split->output(1),
                                                                           R_split->output(1),
                                                                           B_split->output(1),
                                                                           params.hidden_size,
                                                                           op::RecurrentSequenceDirection::REVERSE);

        auto y_forward_transpose = get_4d_batch_transpose(lstm_seq_forward->output(0));
        auto y_reverse_transpose = get_4d_batch_transpose(lstm_seq_reverse->output(0));
        auto h_forward_transpose = get_3d_transpose(lstm_seq_forward->output(1));
        auto h_reverse_transpose = get_3d_transpose(lstm_seq_reverse->output(1));
        auto c_forward_transpose = get_3d_transpose(lstm_seq_forward->output(2));
        auto c_reverse_transpose = get_3d_transpose(lstm_seq_reverse->output(2));

        auto concat_0 = std::make_shared<op::v0::Concat>(OutputVector{y_forward_transpose, y_reverse_transpose}, 2);
        auto concat_1 = std::make_shared<op::v0::Concat>(OutputVector{h_forward_transpose, h_reverse_transpose}, 0);
        auto concat_2 = std::make_shared<op::v0::Concat>(OutputVector{c_forward_transpose, c_reverse_transpose}, 0);

        model = std::make_shared<Model>(OutputVector{concat_0, concat_1, concat_2}, inputs.get_parameter_vector());
    }

    {
        auto inputs = get_inputs(params);
        auto constants = get_constants(params);
        auto lstm_seq = create_lstm_sequence_optimized(params, inputs, constants, MajorFormat::BatchMajor);
        model_ref = std::make_shared<Model>(OutputVector{lstm_seq->output(0), lstm_seq->output(1), lstm_seq->output(2)},
                                            inputs.get_parameter_vector());
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::BidirectionalSequenceComposition>();
    pass_manager.run_passes(model);
    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(bidirectional_lstm_sequence_composition, LSTMSequence_compose_sequence_major) {
    shared_ptr<ov::Model> model, model_ref;
    InputParameters params;
    {
        auto inputs = get_inputs(params, true);
        auto x_transpose = get_3d_transpose(inputs.X);
        auto constants = get_constants(params, 4);
        auto axis_0 = op::v0::Constant::create(element::i64, Shape{}, {0});
        auto axis_1 = op::v0::Constant::create(element::i64, Shape{}, {1});
        auto H_split = std::make_shared<op::v1::Split>(inputs.H, axis_1, 2);
        auto C_split = std::make_shared<op::v1::Split>(inputs.C, axis_1, 2);
        auto W_split = std::make_shared<op::v1::Split>(constants.W, axis_0, 2);
        auto R_split = std::make_shared<op::v1::Split>(constants.R, axis_0, 2);
        auto B_split = std::make_shared<op::v1::Split>(constants.B, axis_0, 2);

        auto lstm_seq_forward = std::make_shared<ov::op::v5::LSTMSequence>(x_transpose,
                                                                           H_split->output(0),
                                                                           C_split->output(0),
                                                                           constants.S,
                                                                           W_split->output(0),
                                                                           R_split->output(0),
                                                                           B_split->output(0),
                                                                           params.hidden_size,
                                                                           op::RecurrentSequenceDirection::FORWARD);
        auto lstm_seq_reverse = std::make_shared<ov::op::v5::LSTMSequence>(x_transpose,
                                                                           H_split->output(1),
                                                                           C_split->output(1),
                                                                           constants.S,
                                                                           W_split->output(1),
                                                                           R_split->output(1),
                                                                           B_split->output(1),
                                                                           params.hidden_size,
                                                                           op::RecurrentSequenceDirection::REVERSE);

        auto y_forward_transpose = get_4d_sequence_transpose(lstm_seq_forward->output(0));
        auto y_reverse_transpose = get_4d_sequence_transpose(lstm_seq_reverse->output(0));
        auto h_forward_transpose = get_3d_transpose(lstm_seq_forward->output(1));
        auto h_reverse_transpose = get_3d_transpose(lstm_seq_reverse->output(1));
        auto c_forward_transpose = get_3d_transpose(lstm_seq_forward->output(2));
        auto c_reverse_transpose = get_3d_transpose(lstm_seq_reverse->output(2));

        auto concat_0 = std::make_shared<op::v0::Concat>(OutputVector{y_forward_transpose, y_reverse_transpose}, 2);
        auto concat_1 = std::make_shared<op::v0::Concat>(OutputVector{h_forward_transpose, h_reverse_transpose}, 0);
        auto concat_2 = std::make_shared<op::v0::Concat>(OutputVector{c_forward_transpose, c_reverse_transpose}, 0);

        model = std::make_shared<Model>(OutputVector{concat_0, concat_1, concat_2}, inputs.get_parameter_vector());
    }

    {
        auto inputs = get_inputs(params, true);
        auto constants = get_constants(params, 4);
        auto lstm_seq = create_lstm_sequence_optimized(params, inputs, constants, MajorFormat::SequenceMajor);
        model_ref = std::make_shared<Model>(OutputVector{lstm_seq->output(0), lstm_seq->output(1), lstm_seq->output(2)},
                                            inputs.get_parameter_vector());
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::BidirectionalSequenceComposition>();
    pass_manager.run_passes(model);
    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(bidirectional_lstm_sequence_composition, LSTMSequence_compose_no_tranpose) {
    shared_ptr<ov::Model> model, model_ref;
    InputParameters params;
    {
        auto inputs = get_inputs(params);
        auto constants = get_constants(params, 4);
        auto axis_0 = op::v0::Constant::create(element::i64, Shape{}, {0});
        auto axis_1 = op::v0::Constant::create(element::i64, Shape{}, {1});
        auto H_split = std::make_shared<op::v1::Split>(inputs.H, axis_1, 2);
        auto C_split = std::make_shared<op::v1::Split>(inputs.C, axis_1, 2);
        auto W_split = std::make_shared<op::v1::Split>(constants.W, axis_0, 2);
        auto R_split = std::make_shared<op::v1::Split>(constants.R, axis_0, 2);
        auto B_split = std::make_shared<op::v1::Split>(constants.B, axis_0, 2);

        auto lstm_seq_forward = std::make_shared<ov::op::v5::LSTMSequence>(inputs.X,
                                                                           H_split->output(0),
                                                                           C_split->output(0),
                                                                           constants.S,
                                                                           W_split->output(0),
                                                                           R_split->output(0),
                                                                           B_split->output(0),
                                                                           params.hidden_size,
                                                                           op::RecurrentSequenceDirection::FORWARD);
        auto lstm_seq_reverse = std::make_shared<ov::op::v5::LSTMSequence>(inputs.X,
                                                                           H_split->output(1),
                                                                           C_split->output(1),
                                                                           constants.S,
                                                                           W_split->output(1),
                                                                           R_split->output(1),
                                                                           B_split->output(1),
                                                                           params.hidden_size,
                                                                           op::RecurrentSequenceDirection::REVERSE);

        auto concat_0 =
            std::make_shared<op::v0::Concat>(OutputVector{lstm_seq_forward->output(0), lstm_seq_reverse->output(0)}, 1);
        auto concat_1 =
            std::make_shared<op::v0::Concat>(OutputVector{lstm_seq_forward->output(1), lstm_seq_reverse->output(1)}, 1);
        auto concat_2 =
            std::make_shared<op::v0::Concat>(OutputVector{lstm_seq_forward->output(2), lstm_seq_reverse->output(2)}, 1);

        model = std::make_shared<Model>(OutputVector{concat_0, concat_1, concat_2}, inputs.get_parameter_vector());
    }

    {
        auto inputs = get_inputs(params);
        auto constants = get_constants(params, 4);
        auto lstm_seq = create_lstm_sequence(params, inputs, constants);
        model_ref = std::make_shared<Model>(OutputVector{lstm_seq->output(0), lstm_seq->output(1), lstm_seq->output(2)},
                                            inputs.get_parameter_vector());
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::BidirectionalSequenceComposition>();
    pass_manager.run_passes(model);
    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}
}  // namespace testing
