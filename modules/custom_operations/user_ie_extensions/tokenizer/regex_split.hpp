// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>
#include "openvino/opsets/opset10.hpp"
#include "fast_tokenizer/pretokenizers/pretokenizers.h"

using namespace ov;
using namespace paddlenlp::fast_tokenizer;


class RegexSplit : public ov::op::Op {
public:
    OPENVINO_OP("RegexSplit");

    RegexSplit () = default;
    RegexSplit(const ov::OutputVector& arguments, const std::string& behaviour = "remove", bool invert = false);
    RegexSplit(
        const ov::OutputVector& arguments,
        const std::shared_ptr<pretokenizers::SplitPreTokenizer>& pretokenizer,
        const std::string& behaviour = "remove",
        bool invert = false
    );

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<RegexSplit>(inputs, m_pretokenizer, m_behaviour, m_invert);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("behaviour", m_behaviour);
        visitor.on_attribute("invert", m_invert);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }

private:
    std::shared_ptr<pretokenizers::SplitPreTokenizer> m_pretokenizer;
    std::string m_behaviour = "remove";
    bool m_invert = false;
};
