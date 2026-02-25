// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>

#include "openvino/op/op.hpp"

namespace ov::nvidia_gpu::nodes {

class FullyConnected : public ov::op::Op {
public:
    OPENVINO_OP("FullyConnected", "nvidia_gpu");

    FullyConnected() = default;
    ~FullyConnected() = default;

    FullyConnected(const ov::Output<Node>& A,
                   const ov::Output<Node>& B,
                   const ov::Output<Node>& C,
                   const bool& transpose_a,
                   const bool& transpose_b);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    void validate_and_infer_types() override;

    bool get_transpose_a() const { return m_transpose_a; }
    bool get_transpose_b() const { return m_transpose_b; }

private:
    bool m_transpose_a;
    bool m_transpose_b;
};

}  // namespace ov::nvidia_gpu::nodes
