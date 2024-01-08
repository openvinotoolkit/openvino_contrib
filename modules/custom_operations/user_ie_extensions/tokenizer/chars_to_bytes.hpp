// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <vector>
#include <openvino/op/op.hpp>

class CharsToBytes : public ov::op::Op {
public:
    OPENVINO_OP("CharsToBytes");

    CharsToBytes () = default;

    CharsToBytes(const ov::OutputVector& arguments) :
        ov::op::Op(arguments) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CharsToBytes>(inputs);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }

    std::array<std::array<uint8_t, 64>, 4> create_pair_map();

private:
    const std::array<std::array<uint8_t, 64>, 4> m_pair_map = create_pair_map();
    const uint8_t m_one_byte_border = 128;  // if char > 128 => it is two byte char
    //
    const uint8_t m_first_byte_offset = 194;
    const uint8_t m_second_byte_offset = 128;
};
