// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>

// Having a decomposed representation for a tensor, converts it to a single string tensor for debugging purposes and to facilitate model conversion
// Base tensor on which this operation builds a ragged tensor can have any shape or type, this operation doesn't try to interpret it.
class RaggedTensorPack : public ov::op::Op {
public:
    OPENVINO_OP("RaggedTensorPack");

    RaggedTensorPack () = default;

    RaggedTensorPack(ov::OutputVector inputs)
        : ov::op::Op(inputs) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        auto result = std::make_shared<RaggedTensorPack>(inputs);
        return result;
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool has_evaluate() const override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
};
