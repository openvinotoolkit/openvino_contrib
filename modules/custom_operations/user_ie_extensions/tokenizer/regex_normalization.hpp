// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "normalizer.h" // for absl::string_view

#include <openvino/op/op.hpp>
#include "openvino/opsets/opset10.hpp"
#include "fast_tokenizer/normalizers/normalizers.h"

using namespace ov;
using namespace ov::opset10;

class RegexNormalization : public ov::op::Op {
public:
    OPENVINO_OP("RegexNormalization");

    RegexNormalization () = default;
    RegexNormalization(const ov::OutputVector& arguments);  // not used
    RegexNormalization(
        const ov::OutputVector& arguments,
        const std::shared_ptr<re2::RE2>& search_pattern_re,
        const absl::string_view replace_pattern
    );

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<RegexNormalization>(inputs, m_search_pattern_re, m_replace_pattern);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }
private:
    std::shared_ptr<re2::RE2> m_search_pattern_re;
    absl::string_view m_replace_pattern;
};
