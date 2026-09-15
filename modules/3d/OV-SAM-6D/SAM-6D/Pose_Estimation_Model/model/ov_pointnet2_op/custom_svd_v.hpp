// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

//! [op:common_include]
#include <openvino/op/op.hpp>
//! [op:common_include]

//! [op:header]
namespace TemplateExtension {

class CustomSVDv : public ov::op::Op {
public:
    OPENVINO_OP("CustomSVDv");

    CustomSVDv() = default;
    CustomSVDv(const ov::Output<ov::Node>& x);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
//! [op:header]

}  // namespace TemplateExtension
