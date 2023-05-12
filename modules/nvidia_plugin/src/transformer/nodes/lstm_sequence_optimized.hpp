// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/lstm_sequence.hpp"

namespace ov::nvidia_gpu::nodes {

class LSTMSequenceOptimized : public ov::op::v5::LSTMSequence {
public:
    enum MajorFormat { BatchMajor, SequenceMajor };

    OPENVINO_OP("LSTMSequenceOptimized", "nvidia_gpu", ov::op::v5::LSTMSequence);

    LSTMSequenceOptimized() = default;
    ~LSTMSequenceOptimized() = default;

    explicit LSTMSequenceOptimized(const ov::Output<Node>& X,
                                   const ov::Output<Node>& initial_hidden_state,
                                   const ov::Output<Node>& initial_cell_state,
                                   const ov::Output<Node>& sequence_lengths,
                                   const ov::Output<Node>& W,
                                   const ov::Output<Node>& R,
                                   const ov::Output<Node>& B,
                                   const std::int64_t hidden_size,
                                   const direction lstm_direction,
                                   MajorFormat major_format,
                                   const std::vector<float>& activations_alpha = {},
                                   const std::vector<float>& activations_beta = {},
                                   const std::vector<std::string>& activations = {"sigmoid", "tanh", "tanh"},
                                   const float clip = 0.f);

    MajorFormat get_major_format() const { return m_major_format; }

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

private:
    MajorFormat m_major_format;
};
}  // namespace ov::nvidia_gpu::nodes


namespace ov {
template <>
class OPENVINO_API AttributeAdapter<nvidia_gpu::nodes::LSTMSequenceOptimized::MajorFormat>
    : public EnumAttributeAdapterBase<nvidia_gpu::nodes::LSTMSequenceOptimized::MajorFormat> {
public:
    AttributeAdapter(nvidia_gpu::nodes::LSTMSequenceOptimized::MajorFormat& value): EnumAttributeAdapterBase<nvidia_gpu::nodes::LSTMSequenceOptimized::MajorFormat>(value) {}

    OPENVINO_RTTI("AttributeAdapter<nvidia_gpu::nodes::LSTMSequenceOptimized::MajorFormat>");
};


} // namespace ov
