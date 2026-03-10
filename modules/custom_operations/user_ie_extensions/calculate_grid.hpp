// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>

namespace TemplateExtension {

class CalculateGrid : public ov::op::Op {
public:
    OPENVINO_OP("CalculateGrid");

    CalculateGrid() = default;
    CalculateGrid(const ov::Output<ov::Node>& inp_pos);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

}  // namespace TemplateExtension
