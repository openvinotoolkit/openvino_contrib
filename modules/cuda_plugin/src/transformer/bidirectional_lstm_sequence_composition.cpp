// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bidirectional_lstm_sequence_composition.hpp"

#include <cuda_op_buffers_extractor.hpp>
#include <exec_graph_info.hpp>
#include <gsl/gsl_assert>
#include <gsl/span_ext>
#include <ngraph/op/concat.hpp>
#include <ngraph/op/lstm_sequence.hpp>
#include <ngraph/op/transpose.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/common_optimizations/nop_elimination.hpp>
#include <transformations/op_conversions/bidirectional_sequences_decomposition.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>
#include <transformations/op_conversions/convert_ti_to_sequences.hpp>
#include <transformer/nodes/concat_optimized.hpp>

#include "cuda_rt_info.hpp"

namespace ngraph::pass {

NGRAPH_RTTI_DEFINITION(ngraph::pass::Convert2LSTMSequenceToBidirectionalLSTMSequence,
                       "Convert2LSTMSequenceToBidirectionalLSTMSequence",
                       0);
namespace {

Node* findConcat(const Output<Node>& node) {
    for (const auto& in : node.get_target_inputs()) {
        if (dynamic_cast<ngraph::op::Concat*>(in.get_node())) {
            return in.get_node();
        } else if (dynamic_cast<CUDAPlugin::nodes::ConcatOptimized*>(in.get_node())) {
            return in.get_node();
        }
        for (const auto& out : in.get_node()->outputs()) {
            auto foundNode = findConcat(out);
            if (foundNode) {
                return foundNode;
            }
        }
    }
    return nullptr;
}

}  // namespace

bool bidirectional_lstm_sequence_composition(ngraph::pattern::Matcher& m) {
    auto transpose = std::dynamic_pointer_cast<ngraph::op::v1::Transpose>(m.get_match_root());
    if (!transpose) {
        return false;
    }

    std::vector<ngraph::Node*> outputsLSTMSequence;
    std::vector<ngraph::op::v5::LSTMSequence*> inputs;
    inputs.reserve(2);
    for (const auto& in : transpose->get_output_target_inputs(0)) {
        auto in_node = in.get_node();
        if (in_node->get_output_target_inputs(0).size() != 1) {
            return false;
        }
        ngraph::op::v5::LSTMSequence* lstm_sequence = nullptr;
        for (const auto& inn : in_node->get_output_target_inputs(0)) {
            lstm_sequence = dynamic_cast<ngraph::op::v5::LSTMSequence*>(inn.get_node());
            if (!lstm_sequence) {
                return false;
            }
        }

        if (lstm_sequence->get_direction() == ngraph::op::RecurrentSequenceDirection::FORWARD) {
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

    auto y_replacement = findConcat(lstm_sequence_forward->output(0));
    auto ho_replacement = findConcat(lstm_sequence_forward->output(1));
    auto co_replacement = findConcat(lstm_sequence_forward->output(2));
    if (!y_replacement || !ho_replacement || !co_replacement) {
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

    auto initial_hidden_state = std::make_shared<ngraph::op::v0::Concat>(
        ngraph::OutputVector{lstm_sequence_forward->input_value(1), lstm_sequence_reverse->input_value(1)}, axis_1);
    auto initial_cell_state = std::make_shared<ngraph::op::v0::Concat>(
        ngraph::OutputVector{lstm_sequence_forward->input_value(2), lstm_sequence_reverse->input_value(2)}, axis_1);

    auto sequence_lengths = lstm_sequence_forward->input_value(3);

    auto weights = std::make_shared<ngraph::op::v0::Concat>(
        ngraph::OutputVector{lstm_sequence_forward->input_value(4), lstm_sequence_reverse->input_value(4)}, axis_0);
    auto recurrent_weights = std::make_shared<ngraph::op::v0::Concat>(
        ngraph::OutputVector{lstm_sequence_forward->input_value(5), lstm_sequence_reverse->input_value(5)}, axis_0);
    auto bias = std::make_shared<ngraph::op::v0::Concat>(
        ngraph::OutputVector{lstm_sequence_forward->input_value(6), lstm_sequence_reverse->input_value(6)}, axis_0);

    auto lstm_sequence_bidirectional =
        std::make_shared<ngraph::op::v5::LSTMSequence>(x,
                                                       initial_hidden_state,
                                                       initial_cell_state,
                                                       sequence_lengths,
                                                       weights,
                                                       recurrent_weights,
                                                       bias,
                                                       hidden_size,
                                                       ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL,
                                                       activations_alpha,
                                                       activations_beta,
                                                       activations,
                                                       clip);

    auto y = lstm_sequence_bidirectional->output(0);
    auto ho = lstm_sequence_bidirectional->output(1);
    auto co = lstm_sequence_bidirectional->output(2);

    ngraph::copy_runtime_info({lstm_sequence_forward->shared_from_this(), lstm_sequence_reverse->shared_from_this()},
                              lstm_sequence_bidirectional);
    ngraph::replace_node(lstm_sequence_forward->shared_from_this(), lstm_sequence_bidirectional);
    ngraph::replace_node(lstm_sequence_reverse->shared_from_this(), lstm_sequence_bidirectional);

    std::string original_names_mapping = "FUSED:";
    auto lstmSequenceOutputReplacer = [&original_names_mapping](const auto& replacement, const auto& new_node) {
        for (auto& out : replacement->outputs()) {
            for (const auto& in : out.get_target_inputs()) {
                const auto& in_node = in.get_node();
                if (dynamic_cast<ngraph::op::Result*>(in_node)) {
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
    rt_info[CUDAPlugin::RtInfo::CUDA_FUSED_NAMES_MAPPING] =
        std::make_shared<ngraph::VariantWrapper<std::string>>(original_names_mapping);

    return true;
}

Convert2LSTMSequenceToBidirectionalLSTMSequence::Convert2LSTMSequenceToBidirectionalLSTMSequence() {
    auto transpose = ngraph::pattern::wrap_type<ngraph::op::v1::Transpose>(pattern::consumers_count(2));

    ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
        return bidirectional_lstm_sequence_composition(m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose, "BidirectionalLSTMSequenceDecomposition");
    this->register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::BidirectionalSequenceComposition, "BidirectionalSequenceComposition", 0);

BidirectionalSequenceComposition::BidirectionalSequenceComposition(std::shared_ptr<PassConfig> pass_config)
    : pass_config_(std::move(pass_config)) {
    pass_config_->disable<ngraph::pass::BidirectionalLSTMSequenceDecomposition>();
    // TODO: Uncomment when support for GRUSequence and RNNSequence will be added
    // pass_config_->disable<ngraph::pass::BidirectionalGRUSequenceDecomposition>();
    // pass_config_->disable<ngraph::pass::BidirectionalRNNSequenceDecomposition>();
    pass_config_->disable<ngraph::pass::ConvertLSTMSequenceToTensorIterator>();
    // TODO: Uncomment when support for GRUSequence and RNNSequence will be added
    // pass_config_->disable<ngraph::pass::ConvertGRUSequenceToTensorIterator>();
    // pass_config_->disable<ngraph::pass::ConvertRNNSequenceToTensorIterator>();
}

bool BidirectionalSequenceComposition::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager manager{pass_config_};

    manager.register_pass<ngraph::pass::ConvertTensorIteratorToLSTMSequence>();
    manager.register_pass<ngraph::pass::NopElimination>();
    manager.register_pass<Convert2LSTMSequenceToBidirectionalLSTMSequence>();
    manager.register_pass<ngraph::pass::CommonOptimizations>();

    manager.run_passes(f);

    return false;
}

}  // namespace ngraph::pass
