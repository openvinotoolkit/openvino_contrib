// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <vector>
#include <openvino/op/op.hpp>


const std::array<std::vector<uint8_t>, 256> create_bytes_to_chars_map();

class BytesToChars : public ov::op::Op {
public:
    OPENVINO_OP("BytesToChars");

    BytesToChars () = default;

    BytesToChars(const ov::OutputVector& arguments) :
        ov::op::Op(arguments) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<BytesToChars>(inputs);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }

private:
    const std::array<std::vector<uint8_t>, 256> m_bytes_to_chars = create_bytes_to_chars_map();
};
