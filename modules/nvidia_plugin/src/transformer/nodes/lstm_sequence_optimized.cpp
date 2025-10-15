// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_sequence_optimized.hpp"

#include <openvino/core/except.hpp>

namespace ov::nvidia_gpu::nodes {

LSTMSequenceOptimized::LSTMSequenceOptimized(const ov::Output<Node>& X,
                                             const ov::Output<Node>& initial_hidden_state,
                                             const ov::Output<Node>& initial_cell_state,
                                             const ov::Output<Node>& sequence_lengths,
                                             const ov::Output<Node>& W,
                                             const ov::Output<Node>& R,
                                             const ov::Output<Node>& B,
                                             const std::int64_t hidden_size,
                                             const ov::op::RecurrentSequenceDirection lstm_direction,
                                             MajorFormat major_format,
                                             const std::vector<float>& activations_alpha,
                                             const std::vector<float>& activations_beta,
                                             const std::vector<std::string>& activations,
                                             const float clip)
        : RNNCellBase({X, initial_hidden_state, initial_cell_state, sequence_lengths, W, R, B},
                    hidden_size,
                    clip,
                    activations,
                    activations_alpha,
                    activations_beta),
            m_direction(lstm_direction),
            m_major_format{major_format} {
        constructor_validate_and_infer_types();
      }

std::shared_ptr<ov::Node> LSTMSequenceOptimized::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    if (new_args.size() == 7) {
        auto lstmSequence = std::make_shared<LSTMSequenceOptimized>(new_args.at(0),  // X
                                                                    new_args.at(1),  // initial_hidden_state
                                                                    new_args.at(2),  // initial_cell_state
                                                                    new_args.at(3),  // sequence_lengths
                                                                    new_args.at(4),  // W
                                                                    new_args.at(5),  // R
                                                                    new_args.at(6),  // B
                                                                    m_hidden_size,
                                                                    this->get_direction(),
                                                                    m_major_format,
                                                                    m_activations_alpha,
                                                                    m_activations_beta,
                                                                    m_activations,
                                                                    m_clip);
        lstmSequence->validate_and_infer_types();
        return lstmSequence;
    } else {
        OPENVINO_THROW("Incorrect number of new arguments");
    }
}

bool LSTMSequenceOptimized::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("hidden_size", m_hidden_size);
    visitor.on_attribute("activations", m_activations);
    visitor.on_attribute("activations_alpha", m_activations_alpha);
    visitor.on_attribute("activations_beta", m_activations_beta);
    visitor.on_attribute("clip", m_clip);
    visitor.on_attribute("direction", m_direction);
    visitor.on_attribute("major_format", m_major_format);
    return true;
}

void LSTMSequenceOptimized::validate_and_infer_types() {
    for (const auto& input : inputs()) {
        if (input.get_partial_shape().rank().is_dynamic()) {
            set_output_type(0, get_input_element_type(0), ov::PartialShape::dynamic());
            set_output_type(1, get_input_element_type(0), ov::PartialShape::dynamic());
            set_output_type(2, get_input_element_type(0), ov::PartialShape::dynamic());
            return;
        }
    }
    std::vector<ov::PartialShape> input_param{};

    auto lstm_seq_gates_count = 4;
    auto merged_batch_size = ov::Dimension::dynamic();
    auto merged_hidden_size = ov::Dimension::dynamic();
    auto merged_num_directions = ov::Dimension::dynamic();
    auto result_et = ov::element::dynamic;

    // Copy all inputs without initial_cell_state information for further validation
    for (size_t i = 0; i < get_input_size(); i++) {
        // exclude initial_cell_state from the loop
        if (i != 2) {
            input_param.push_back(get_input_partial_shape(i));
        }
    }

    // Get input partial shape for all inputs
    const auto& x_pshape = get_input_partial_shape(0);
    const auto& ht_pshape = get_input_partial_shape(1);
    const auto& ct_pshape = get_input_partial_shape(2);
    const auto& sl_pshape = get_input_partial_shape(3);
    const auto& w_pshape = get_input_partial_shape(4);
    const auto& r_pshape = get_input_partial_shape(5);
    const auto& b_pshape = get_input_partial_shape(6);

    // Validate rank and dimension for initial_cell_state input
    NODE_VALIDATION_CHECK(this,
                          (ct_pshape.rank().get_length() == 3),
                          "LSTMSequence input tensor initial_cell_state shall have dimension 3D.");

    // Validate input types and save result for output type
    NODE_VALIDATION_CHECK(this,
                          ov::element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
                              ov::element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
                              ov::element::Type::merge(result_et, result_et, get_input_element_type(2)) &&
                              ov::element::Type::merge(result_et, result_et, get_input_element_type(4)) &&
                              ov::element::Type::merge(result_et, result_et, get_input_element_type(5)) &&
                              ov::element::Type::merge(result_et, result_et, get_input_element_type(6)),
                          "Element types for X, initial_hidden_state, initial_cell_state, W, R and B inputs do "
                          "not "
                          "match.");

    // Merge batch_size dimension across all inputs to evaluate output[0] dimension
    if (m_major_format == MajorFormat::BatchMajor) {
        NODE_VALIDATION_CHECK(this,
                            ov::Dimension::merge(merged_batch_size, merged_batch_size, ht_pshape[0]) &&
                                ov::Dimension::merge(merged_batch_size, merged_batch_size, ct_pshape[0]) &&
                                ov::Dimension::merge(merged_batch_size, merged_batch_size, x_pshape[0]) &&
                                ov::Dimension::merge(merged_batch_size, merged_batch_size, sl_pshape[0]),
                            "Parameter batch_size not matched in LSTMSequence.");
    } else {
        NODE_VALIDATION_CHECK(this,
                            ov::Dimension::merge(merged_batch_size, merged_batch_size, ht_pshape[0]) &&
                                ov::Dimension::merge(merged_batch_size, merged_batch_size, ct_pshape[0]) &&
                                ov::Dimension::merge(merged_batch_size, merged_batch_size, x_pshape[1]) &&
                                ov::Dimension::merge(merged_batch_size, merged_batch_size, sl_pshape[0]),
                            "Parameter batch_size not matched in LSTMSequence.");
    }

    // Merge hidden_size dimension across all inputs to evaluate output dimension
    NODE_VALIDATION_CHECK(this,
                          ov::Dimension::merge(merged_hidden_size, merged_hidden_size, ht_pshape[2]) &&
                              ov::Dimension::merge(merged_hidden_size, merged_hidden_size, ct_pshape[2]) &&
                              ov::Dimension::merge(merged_hidden_size, merged_hidden_size, r_pshape[2]),
                          "Parameter hidden_size not matched LSTMSequence.");

    // Merge num_directions dimension across all inputs to evaluate output dimension
    NODE_VALIDATION_CHECK(this,
                          ov::Dimension::merge(merged_num_directions, merged_num_directions, ht_pshape[1]) &&
                              ov::Dimension::merge(merged_num_directions, merged_num_directions, ct_pshape[1]) &&
                              ov::Dimension::merge(merged_num_directions, merged_num_directions, w_pshape[0]) &&
                              ov::Dimension::merge(merged_num_directions, merged_num_directions, r_pshape[0]) &&
                              ov::Dimension::merge(merged_num_directions, merged_num_directions, b_pshape[0]),
                          "Parameter num_directions not matched in LSTMSequence.");

    // Validate hidden_size value for W, R, B inputs
    if (merged_hidden_size.is_static()) {
        if (w_pshape[1].is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  w_pshape[1].compatible(merged_hidden_size * lstm_seq_gates_count),
                                  "Parameter hidden_size mistmatched in W input. Current value is: ",
                                  w_pshape[1].get_length(),
                                  ", expected: ",
                                  merged_hidden_size.get_length() * lstm_seq_gates_count,
                                  ".");
        }

        if (r_pshape[1].is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  r_pshape[1].compatible(merged_hidden_size * lstm_seq_gates_count),
                                  "Parameter hidden_size mistmatched in R input. Current value is: ",
                                  r_pshape[1].get_length(),
                                  ", expected: ",
                                  merged_hidden_size.get_length() * lstm_seq_gates_count,
                                  ".");
        }

        if (b_pshape[1].is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  b_pshape[1].compatible(merged_hidden_size * lstm_seq_gates_count),
                                  "Parameter hidden_size mistmatched in B input. Current value is: ",
                                  b_pshape[1].get_length(),
                                  ", expected: ",
                                  merged_hidden_size.get_length() * lstm_seq_gates_count,
                                  ".");
        }
    }

    // Mark inputs which are relevant to output parameters
    for (size_t i = 0; i <= 6; ++i) {
        set_input_is_relevant_to_shape(i);
    }

    // Set output size, type and shape
    set_output_size(3);
    if (m_major_format == MajorFormat::BatchMajor) {
        // x shape [batch_size, seq_length, input_size]
        // y shape [batch_size, seq_length, num_directions, hidden_size]
        set_output_type(0, result_et, {merged_batch_size, x_pshape[1], merged_num_directions, merged_hidden_size});
    } else {
        // x shape [seq_length, batch_size, input_size]
        // y shape [seq_length, batch_size, num_directions, hidden_size]
        set_output_type(0, result_et, {x_pshape[0], merged_batch_size, merged_num_directions, merged_hidden_size});
    }
    set_output_type(1, result_et, {merged_num_directions, merged_batch_size, merged_hidden_size});
    set_output_type(2, result_et, {merged_num_directions, merged_batch_size, merged_hidden_size});
}

}  // namespace ov::nvidia_gpu::nodes


namespace ov {
std::ostream& operator<<(std::ostream& s, const nvidia_gpu::nodes::LSTMSequenceOptimized::MajorFormat& type) {
    return s << as_string(type);
}
template <>
EnumNames<nvidia_gpu::nodes::LSTMSequenceOptimized::MajorFormat>&
EnumNames<nvidia_gpu::nodes::LSTMSequenceOptimized::MajorFormat>::get() {
    static auto enum_names = EnumNames<nvidia_gpu::nodes::LSTMSequenceOptimized::MajorFormat>(
        "nvidia_gpu::nodes::LSTMSequenceOptimized::MajorFormat",
        {{"batch_major", nvidia_gpu::nodes::LSTMSequenceOptimized::MajorFormat::BatchMajor},
         {"sequence_major", nvidia_gpu::nodes::LSTMSequenceOptimized::MajorFormat::SequenceMajor}});
    return enum_names;
}
} // namespace ov