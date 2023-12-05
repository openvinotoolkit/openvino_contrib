// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>

class NormalizeUnicode : public ov::op::Op {
public:
    OPENVINO_OP("NormalizeUnicode");

    NormalizeUnicode () = default;

    NormalizeUnicode(const ov::OutputVector& arguments, const std::string& normalization_form = "NFD") :
        ov::op::Op(arguments),
        m_normalization_form(normalization_form) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<NormalizeUnicode>(inputs, m_normalization_form);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("normalization_form", m_normalization_form);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }

private:

    std::string m_normalization_form = "NFD";
};
