// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bidirectional_lstm_sequence_composition.hpp"

#include <cuda_op_buffers_extractor.hpp>
#include <gsl/gsl_assert>
#include <gsl/span_ext>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/lstm_sequence.hpp>
#include <openvino/op/transpose.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/common_optimizations/nop_elimination.hpp>
#include <transformations/op_conversions/bidirectional_sequences_decomposition.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>
#include <transformations/op_conversions/convert_ti_to_sequences.hpp>
#include <transformer/nodes/concat_optimized.hpp>
#include <transformer/nodes/lstm_sequence_optimized.hpp>

#include "cuda_rt_info.hpp"

namespace ngraph::pass {

NGRAPH_RTTI_DEFINITION(ngraph::pass::Convert2LSTMSequenceToBidirectionalLSTMSequence,
                       "Convert2LSTMSequenceToBidirectionalLSTMSequence",
                       0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertBidirectionalLSTMSequenceToBidirectionalLSTMSequenceOptimized,
                       "ConvertBidirectionalLSTMSequenceToBidirectionalLSTMSequenceOptimized",
                       0);
namespace {

Node* findConcat(const Output<Node>& node, ov::op::v1::Transpose*& transpose) {
    for (const auto& in : node.get_target_inputs()) {
        if (auto tranNode = dynamic_cast<ov::op::v1::Transpose*>(in.get_node())) {
            if (!transpose) {
                transpose = tranNode;
            }
        }
        if (dynamic_cast<ov::op::v0::Concat*>(in.get_node())) {
            return in.get_node();
        } else if (dynamic_cast<CUDAPlugin::nodes::ConcatOptimized*>(in.get_node())) {
            return in.get_node();
        }
        for (const auto& out : in.get_node()->outputs()) {
            auto foundNode = findConcat(out, transpose);
            if (foundNode) {
                return foundNode;
            }
        }
    }
    return nullptr;
}

void getLastTransposeOrReshape(Node& node, Node*& foundNode) {
    if (node.outputs().size() == 1) {
        if (dynamic_cast<ov::op::v1::Transpose*>(&node)) {
            foundNode = &node;
        } else if (dynamic_cast<ov::op::v1::Reshape*>(&node)) {
            foundNode = &node;
        }

        for (const auto& in : node.get_output_target_inputs(0)) {
            if (dynamic_cast<ov::op::v1::Transpose*>(in.get_node())) {
                foundNode = in.get_node();
            } else if (dynamic_cast<ov::op::v1::Reshape*>(in.get_node())) {
                foundNode = in.get_node();
            }
            getLastTransposeOrReshape(*in.get_node(), foundNode);
        }
    }
}

auto transposePerm(const Node* tran) {
    if (tran) {
        std::vector<int64_t> perm;
        auto perm_const = dynamic_cast<ov::op::v0::Constant*>(tran->get_input_source_output(1).get_node());
        const int64_t* ptr = reinterpret_cast<const int64_t*>(perm_const->get_data_ptr());
        auto span = gsl::make_span(
            ptr, CUDAPlugin::OperationBuffersExtractor::GetTensorByteSize(perm_const->output(0)) / sizeof(int64_t));
        return std::vector<int64_t>{span.begin(), span.end()};
    }
    return std::vector<int64_t>{};
}

}  // namespace

bool bidirectional_lstm_sequence_composition(ngraph::pattern::Matcher& m) {
    auto transpose = std::dynamic_pointer_cast<ov::op::v1::Transpose>(m.get_match_root());
    if (!transpose) {
        return false;
    }

    std::vector<ov::Node*> outputsLSTMSequence;
    std::vector<ov::op::v5::LSTMSequence*> inputs;
    inputs.reserve(2);
    for (const auto& in : transpose->get_output_target_inputs(0)) {
        auto in_node = in.get_node();
        if (in_node->get_output_target_inputs(0).size() != 1) {
            return false;
        }
        ov::op::v5::LSTMSequence* lstm_sequence = nullptr;
        for (const auto& inn : in_node->get_output_target_inputs(0)) {
            lstm_sequence = dynamic_cast<ov::op::v5::LSTMSequence*>(inn.get_node());
            if (!lstm_sequence) {
                return false;
            }
        }

        if (lstm_sequence->get_direction() == ov::op::RecurrentSequenceDirection::FORWARD) {
            inputs.insert(inputs.begin(), lstm_sequence);
            outputsLSTMSequence.insert(outputsLSTMSequence.begin(), in_node);
        } else {
            outputsLSTMSequence.push_back(in_node);
            inputs.push_back(lstm_sequence);
        }
    }
    if (inputs.size() != 2) {
        return false;
    }
    auto lstm_sequence_forward = inputs[0];
    auto lstm_sequence_reverse = inputs[1];
    if (lstm_sequence_forward->get_hidden_size() != lstm_sequence_reverse->get_hidden_size() ||
        lstm_sequence_forward->get_activations_alpha() != lstm_sequence_reverse->get_activations_alpha() ||
        lstm_sequence_forward->get_activations_beta() != lstm_sequence_reverse->get_activations_beta() ||
        lstm_sequence_forward->get_activations() != lstm_sequence_reverse->get_activations() ||
        lstm_sequence_forward->get_clip() != lstm_sequence_reverse->get_clip()) {
        return false;
    }

    ov::op::v1::Transpose* y_transpose = nullptr;
    ov::op::v1::Transpose* ho_transpose = nullptr;
    ov::op::v1::Transpose* co_transpose = nullptr;
    auto y_replacement = findConcat(lstm_sequence_forward->output(0), y_transpose);
    auto ho_replacement = findConcat(lstm_sequence_forward->output(1), ho_transpose);
    auto co_replacement = findConcat(lstm_sequence_forward->output(2), co_transpose);
    if (!y_replacement || !ho_replacement || !co_replacement) {
        return false;
    }
    if (y_transpose && transposePerm(y_transpose) != std::vector<int64_t>{2, 1, 0, 3}) {
        return false;
    }

    const auto hidden_size = lstm_sequence_forward->get_hidden_size();
    const auto activations_alpha = lstm_sequence_forward->get_activations_alpha();
    const auto activations_beta = lstm_sequence_forward->get_activations_beta();
    const auto activations = lstm_sequence_forward->get_activations();
    const auto clip = lstm_sequence_forward->get_clip();

    constexpr auto axis_0 = 0;
    constexpr auto axis_1 = 1;

    auto x = outputsLSTMSequence[0]->output(0);

    auto initial_hidden_state = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{lstm_sequence_forward->input_value(1), lstm_sequence_reverse->input_value(1)}, axis_1);
    auto initial_cell_state = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{lstm_sequence_forward->input_value(2), lstm_sequence_reverse->input_value(2)}, axis_1);

    auto sequence_lengths = lstm_sequence_forward->input_value(3);

    auto weights = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{lstm_sequence_forward->input_value(4), lstm_sequence_reverse->input_value(4)}, axis_0);
    auto recurrent_weights = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{lstm_sequence_forward->input_value(5), lstm_sequence_reverse->input_value(5)}, axis_0);
    auto bias = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{lstm_sequence_forward->input_value(6), lstm_sequence_reverse->input_value(6)}, axis_0);

    auto lstm_sequence_bidirectional =
        std::make_shared<ov::op::v5::LSTMSequence>(x,
                                                   initial_hidden_state,
                                                   initial_cell_state,
                                                   sequence_lengths,
                                                   weights,
                                                   recurrent_weights,
                                                   bias,
                                                   hidden_size,
                                                   ov::op::RecurrentSequenceDirection::BIDIRECTIONAL,
                                                   activations_alpha,
                                                   activations_beta,
                                                   activations,
                                                   clip);

    auto y = lstm_sequence_bidirectional->output(0);
    auto ho = lstm_sequence_bidirectional->output(1);
    auto co = lstm_sequence_bidirectional->output(2);

    ov::copy_runtime_info({lstm_sequence_forward->shared_from_this(), lstm_sequence_reverse->shared_from_this()},
                          lstm_sequence_bidirectional);
    ov::replace_node(lstm_sequence_forward->shared_from_this(), lstm_sequence_bidirectional);
    ov::replace_node(lstm_sequence_reverse->shared_from_this(), lstm_sequence_bidirectional);

    if (y_transpose) {
        std::vector<int64_t> transpose_y_perm = {2, 1, 0, 3};
        std::vector<int64_t> transpose_ho_co_perm = {1, 0, 2};
        auto transpose_y_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, transpose_y_perm);
        auto transpose_ho_co_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, transpose_ho_co_perm);
        auto transpose_y = std::make_shared<ov::op::v1::Transpose>(y, transpose_y_const);
        auto transpose_ho = std::make_shared<ov::op::v1::Transpose>(ho, transpose_ho_co_const);
        auto transpose_co = std::make_shared<ov::op::v1::Transpose>(co, transpose_ho_co_const);
        transpose_y->set_friendly_name(y_replacement->get_friendly_name());
        transpose_ho->set_friendly_name(ho_replacement->get_friendly_name());
        transpose_co->set_friendly_name(co_replacement->get_friendly_name());
        ov::copy_runtime_info(y_replacement->shared_from_this(), transpose_y);
        ov::copy_runtime_info(ho_replacement->shared_from_this(), transpose_ho);
        ov::copy_runtime_info(co_replacement->shared_from_this(), transpose_co);
        ov::replace_node(y_replacement->shared_from_this(), transpose_y);
        ov::replace_node(ho_replacement->shared_from_this(), transpose_ho);
        ov::replace_node(co_replacement->shared_from_this(), transpose_co);
    } else {
        std::string original_names_mapping = "FUSED:";
        auto lstmSequenceOutputReplacer = [&original_names_mapping](const auto& replacement, const auto& new_node) {
            for (auto& out : replacement->outputs()) {
                for (const auto& in : out.get_target_inputs()) {
                    const auto& in_node = in.get_node();
                    if (dynamic_cast<ov::op::v0::Result*>(in_node)) {
                        original_names_mapping +=
                            in.get_node()->get_friendly_name() + "=" + out.get_node()->get_friendly_name() + ";";
                    }
                }
                out.replace(new_node);
            }
        };

        lstmSequenceOutputReplacer(y_replacement, y);
        lstmSequenceOutputReplacer(ho_replacement, ho);
        lstmSequenceOutputReplacer(co_replacement, co);

        auto& rt_info = lstm_sequence_bidirectional->get_rt_info();
        rt_info[CUDAPlugin::RtInfo::CUDA_FUSED_NAMES_MAPPING] = original_names_mapping;
    }

    return true;
}

bool bidirectional_lstm_sequence_cudnn_optimized(ngraph::pattern::Matcher& m) {
    // Original OpenVINO:
    //   in              - [batch_size, seq_length, input_size]
    //   out             - [batch_size, num_directions, seq_length, hidden_size]
    //   cell/hidden in  - [batch_size, num_directions, hidden_size]
    //   cell/hidden out - [batch_size, num_directions, hidden_size]
    //
    // cuDNN Optimized 0:
    //   in              - [batch_size, seq_length, input_size]
    //   out             - [batch_size, seq_length, num_directions, hidden_size]
    //   cell/hidden in  - [batch_size, num_directions, hidden_size]
    //   cell/hidden out - [num_directions, batch_size, hidden_size]
    //
    // cuDNN Optimized 1:
    //   in              - [seq_length, batch_size, input_size]
    //   out             - [seq_length, batch_size, num_directions, hidden_size]
    //   cell/hidden in  - [batch_size, num_directions, hidden_size]
    //   cell/hidden out - [num_directions, batch_size, hidden_size]

    using MajorFormat = CUDAPlugin::nodes::LSTMSequenceOptimized::MajorFormat;

    auto lstm_sequence_bidirectional = std::dynamic_pointer_cast<ov::op::v5::LSTMSequence>(m.get_match_root());
    if (!lstm_sequence_bidirectional ||
        lstm_sequence_bidirectional->get_direction() != ov::op::RecurrentSequenceDirection::BIDIRECTIONAL) {
        return false;
    }

    Node* y_replacement = nullptr;
    getLastTransposeOrReshape(*lstm_sequence_bidirectional->get_output_target_inputs(0).begin()->get_node(),
                              y_replacement);
    Node* ho_replacement = nullptr;
    getLastTransposeOrReshape(*lstm_sequence_bidirectional->get_output_target_inputs(1).begin()->get_node(),
                              ho_replacement);
    Node* co_replacement = nullptr;
    getLastTransposeOrReshape(*lstm_sequence_bidirectional->get_output_target_inputs(2).begin()->get_node(),
                              co_replacement);
    if (!y_replacement || !ho_replacement || !co_replacement) {
        return false;
    }

    const auto y_shape = lstm_sequence_bidirectional->output(0).get_shape();
    const auto y_replacement_shape = y_replacement->output(0).get_shape();
    if (y_replacement_shape.size() < 3) {
        return false;
    }
    std::optional<MajorFormat> major_format;
    std::shared_ptr<ov::op::v1::Transpose> x_transpose;
    if (y_replacement_shape[0] == y_shape[0] && y_replacement_shape[1] == y_shape[2]) {
        major_format = MajorFormat::BatchMajor;
    } else if (y_replacement_shape[0] == y_shape[2] && y_replacement_shape[1] == y_shape[0]) {
        const auto x_node = lstm_sequence_bidirectional->get_input_source_output(0).get_node();
        if (auto x_tran = dynamic_cast<ov::op::v1::Transpose*>(x_node)) {
            if (transposePerm(x_transpose.get()) == std::vector<int64_t>{1, 0, 2}) {
                x_transpose = std::dynamic_pointer_cast<ov::op::v1::Transpose>(x_tran->shared_from_this());
                major_format = MajorFormat::SequenceMajor;
            }
        }
    }
    if (!major_format) {
        return false;
    }

    auto new_lstm_sequence_bidirectional = std::make_shared<CUDAPlugin::nodes::LSTMSequenceOptimized>(
        x_transpose ? x_transpose : lstm_sequence_bidirectional->get_input_source_output(0),
        lstm_sequence_bidirectional->get_input_source_output(1),
        lstm_sequence_bidirectional->get_input_source_output(2),
        lstm_sequence_bidirectional->get_input_source_output(3),
        lstm_sequence_bidirectional->get_input_source_output(4),
        lstm_sequence_bidirectional->get_input_source_output(5),
        lstm_sequence_bidirectional->get_input_source_output(6),
        lstm_sequence_bidirectional->get_hidden_size(),
        ov::op::RecurrentSequenceDirection::BIDIRECTIONAL,
        major_format.value(),
        lstm_sequence_bidirectional->get_activations_alpha(),
        lstm_sequence_bidirectional->get_activations_beta(),
        lstm_sequence_bidirectional->get_activations(),
        lstm_sequence_bidirectional->get_clip());
    new_lstm_sequence_bidirectional->validate_and_infer_types();

    std::shared_ptr<ov::op::v1::Reshape> reshape;
    if (y_replacement_shape.size() == 3) {
        const auto& out = new_lstm_sequence_bidirectional->output(0);
        const auto& out_shape = out.get_shape();
        std::vector<int32_t> reshape_pattern_values = {static_cast<int32_t>(out_shape[0]),
                                                       static_cast<int32_t>(out_shape[1]),
                                                       static_cast<int32_t>(out_shape[2] * out_shape[3])};
        auto reshape_pattern =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, reshape_pattern_values);
        reshape = std::make_shared<ov::op::v1::Reshape>(out, reshape_pattern, true);
    }

    auto y = new_lstm_sequence_bidirectional->output(0);
    auto ho = new_lstm_sequence_bidirectional->output(1);
    auto co = new_lstm_sequence_bidirectional->output(2);

    if (x_transpose) {
        ov::copy_runtime_info(x_transpose->shared_from_this(), new_lstm_sequence_bidirectional);
        ov::replace_node(x_transpose->shared_from_this(), new_lstm_sequence_bidirectional);
    }
    ov::copy_runtime_info(lstm_sequence_bidirectional->shared_from_this(), new_lstm_sequence_bidirectional);
    ov::replace_node(lstm_sequence_bidirectional->shared_from_this(), new_lstm_sequence_bidirectional);

    std::string original_names_mapping = "FUSED:";
    auto lstmSequenceOutputReplacer = [&original_names_mapping](const auto& replacement, const auto& new_node) {
        for (auto& out : replacement->outputs()) {
            for (const auto& in : out.get_target_inputs()) {
                const auto& in_node = in.get_node();
                if (dynamic_cast<ov::op::v0::Result*>(in_node)) {
                    original_names_mapping +=
                        in.get_node()->get_friendly_name() + "=" + out.get_node()->get_friendly_name() + ";";
                }
            }
            out.replace(new_node);
        }
    };

    if (reshape) {
        reshape->set_friendly_name(y_replacement->get_friendly_name());
        ov::copy_runtime_info(y_replacement->shared_from_this(), reshape);
        ov::replace_node(y_replacement->shared_from_this(), reshape);
    } else {
        lstmSequenceOutputReplacer(y_replacement, y);
    }
    lstmSequenceOutputReplacer(ho_replacement, ho);
    lstmSequenceOutputReplacer(co_replacement, co);

    auto& rt_info = new_lstm_sequence_bidirectional->get_rt_info();
    rt_info[CUDAPlugin::RtInfo::CUDA_FUSED_NAMES_MAPPING] = original_names_mapping;

    return true;
}

Convert2LSTMSequenceToBidirectionalLSTMSequence::Convert2LSTMSequenceToBidirectionalLSTMSequence() {
    auto transpose = ngraph::pattern::wrap_type<ov::op::v1::Transpose>(pattern::consumers_count(2));

    ov::matcher_pass_callback callback = [](pattern::Matcher& m) { return bidirectional_lstm_sequence_composition(m); };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose, "Convert2LSTMSequenceToBidirectionalLSTMSequence");
    this->register_matcher(m, callback);
}

ConvertBidirectionalLSTMSequenceToBidirectionalLSTMSequenceOptimized::
    ConvertBidirectionalLSTMSequenceToBidirectionalLSTMSequenceOptimized() {
    auto transpose = ngraph::pattern::wrap_type<ov::op::v5::LSTMSequence>();

    ov::matcher_pass_callback callback = [](pattern::Matcher& m) {
        return bidirectional_lstm_sequence_cudnn_optimized(m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(
        transpose, "ConvertBidirectionalLSTMSequenceToBidirectionalLSTMSequenceOptimized");
    this->register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::BidirectionalSequenceComposition, "BidirectionalSequenceComposition", 0);

BidirectionalSequenceComposition::BidirectionalSequenceComposition(std::shared_ptr<PassConfig> pass_config)
    : pass_config_(std::move(pass_config)) {
    pass_config_->disable<pass::BidirectionalLSTMSequenceDecomposition>();
    pass_config_->disable<pass::BidirectionalGRUSequenceDecomposition>();
    // TODO: Uncomment when support for GRUSequence and RNNSequence will be added
    // pass_config_->disable<pass::BidirectionalRNNSequenceDecomposition>();

    pass_config_->disable<pass::ConvertLSTMSequenceToTensorIterator>();
    pass_config_->disable<pass::ConvertGRUSequenceToTensorIterator>();
    // TODO: Uncomment when support for GRUSequence and RNNSequence will be added
    // pass_config_->disable<pass::ConvertRNNSequenceToTensorIterator>();
}

bool BidirectionalSequenceComposition::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager manager{pass_config_};

    manager.register_pass<pass::ConvertTensorIteratorToLSTMSequence>();
    manager.register_pass<pass::ConvertTensorIteratorToGRUSequence>();
    manager.register_pass<pass::NopElimination>();
    manager.register_pass<Convert2LSTMSequenceToBidirectionalLSTMSequence>();
    manager.register_pass<pass::CommonOptimizations>();
    manager.register_pass<ConvertBidirectionalLSTMSequenceToBidirectionalLSTMSequenceOptimized>();
    manager.register_pass<pass::CommonOptimizations>();

    manager.run_passes(f);

    return false;
}

}  // namespace ngraph::pass
