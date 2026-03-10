// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>

namespace TemplateExtension {

class FFT : public ov::op::Op {
public:
    OPENVINO_OP("FFT");

    FFT() = default;
    FFT(const ov::OutputVector& args, bool inverse, bool centered);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;

private:
    bool inverse = false;
    bool centered = false;
};

}  // namespace TemplateExtension
