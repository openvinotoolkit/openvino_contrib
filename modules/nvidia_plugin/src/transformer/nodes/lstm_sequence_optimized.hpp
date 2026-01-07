// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/lstm_sequence.hpp"

namespace ov::nvidia_gpu::nodes {

class LSTMSequenceOptimized : public ov::op::util::RNNCellBase {
public:
    enum MajorFormat { BatchMajor, SequenceMajor };

    OPENVINO_OP("LSTMSequenceOptimized", "nvidia_gpu", ov::op::util::RNNCellBase);

    LSTMSequenceOptimized() = default;
    ~LSTMSequenceOptimized() = default;

    size_t get_default_output_index() const override { return no_default_index(); }

    explicit LSTMSequenceOptimized(const ov::Output<Node>& X,
                                   const ov::Output<Node>& initial_hidden_state,
                                   const ov::Output<Node>& initial_cell_state,
                                   const ov::Output<Node>& sequence_lengths,
                                   const ov::Output<Node>& W,
                                   const ov::Output<Node>& R,
                                   const ov::Output<Node>& B,
                                   const std::int64_t hidden_size,
                                   const ov::op::RecurrentSequenceDirection lstm_direction,
                                   MajorFormat major_format,
                                   const std::vector<float>& activations_alpha = {},
                                   const std::vector<float>& activations_beta = {},
                                   const std::vector<std::string>& activations = {"sigmoid", "tanh", "tanh"},
                                   const float clip = 0.f);

    MajorFormat get_major_format() const { return m_major_format; }

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::vector<float> get_activations_alpha() const { return m_activations_alpha; }
    std::vector<float> get_activations_beta() const { return m_activations_beta; }
    std::vector<std::string> get_activations() const { return m_activations; }
    float get_clip_threshold() const { return m_clip; }
    ov::op::RecurrentSequenceDirection get_direction() const { return m_direction; }
    void set_direction(const ov::op::RecurrentSequenceDirection& dir) { m_direction = dir; }
    std::size_t get_hidden_size() const { return m_hidden_size; }

private:
    ov::op::RecurrentSequenceDirection m_direction;
    MajorFormat m_major_format;
};
}  // namespace ov::nvidia_gpu::nodes

namespace ov {
template <>
class AttributeAdapter<nvidia_gpu::nodes::LSTMSequenceOptimized::MajorFormat>
    : public EnumAttributeAdapterBase<nvidia_gpu::nodes::LSTMSequenceOptimized::MajorFormat> {
public:
    AttributeAdapter(nvidia_gpu::nodes::LSTMSequenceOptimized::MajorFormat& value)
        : EnumAttributeAdapterBase<nvidia_gpu::nodes::LSTMSequenceOptimized::MajorFormat>(value) {}

    OPENVINO_RTTI("AttributeAdapter<nvidia_gpu::nodes::LSTMSequenceOptimized::MajorFormat>");
};

}  // namespace ov
