// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>

class OPENVINO_API RegexSplit : public ov::op::Op {
public:
    OPENVINO_OP("RegexSplit");

    RegexSplit () = default;

    RegexSplit(const ov::OutputVector& arguments, const std::string& behaviour = "remove", bool invert = false) :
        ov::op::Op(arguments),
        m_behaviour(behaviour),
        m_invert(invert) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<RegexSplit>(inputs, m_behaviour, m_invert);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("behaviour", m_behaviour);
        visitor.on_attribute("invert", m_invert);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const {
        return true;
    }

private:

    std::string m_behaviour = "remove";
    bool m_invert = false;
};
