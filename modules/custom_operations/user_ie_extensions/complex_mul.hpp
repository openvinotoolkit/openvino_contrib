// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>

namespace TemplateExtension {

class ComplexMultiplication : public ov::op::Op {
public:
    OPENVINO_OP("ComplexMultiplication");

    ComplexMultiplication() = default;
    ComplexMultiplication(const ov::OutputVector& args);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

}  // namespace TemplateExtension
