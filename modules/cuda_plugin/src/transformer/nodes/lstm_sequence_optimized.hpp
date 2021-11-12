// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/lstm_sequence.hpp>

namespace CUDAPlugin::nodes {

class LSTMSequenceOptimized : public ngraph::op::v5::LSTMSequence {
public:
    enum MajorFormat { BatchMajor, SequenceMajor };

    LSTMSequenceOptimized() = default;
    explicit LSTMSequenceOptimized(const ngraph::Output<Node>& X,
                                   const ngraph::Output<Node>& initial_hidden_state,
                                   const ngraph::Output<Node>& initial_cell_state,
                                   const ngraph::Output<Node>& sequence_lengths,
                                   const ngraph::Output<Node>& W,
                                   const ngraph::Output<Node>& R,
                                   const ngraph::Output<Node>& B,
                                   const std::int64_t hidden_size,
                                   const direction lstm_direction,
                                   MajorFormat major_format,
                                   const std::vector<float>& activations_alpha = {},
                                   const std::vector<float>& activations_beta = {},
                                   const std::vector<std::string>& activations = {"sigmoid", "tanh", "tanh"},
                                   const float clip = 0.f);

    inline static constexpr type_info_t type_info{"LSTMSequenceOptimized", 0};
    const type_info_t& get_type_info() const override { return type_info; }
    MajorFormat get_major_format() const { return m_major_format; }

    std::shared_ptr<Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

    void validate_and_infer_types() override;

private:
    MajorFormat m_major_format;
};

}  // namespace CUDAPlugin::nodes
