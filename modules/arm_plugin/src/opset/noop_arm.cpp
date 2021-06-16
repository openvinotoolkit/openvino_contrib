// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "noop_arm.hpp"
#include "utils.hpp"

using namespace ngraph;
using namespace ArmPlugin;

NGRAPH_RTTI_DEFINITION(opset::ArmNoOp, "ArmNoOp", 0);

opset::ArmNoOp::ArmNoOp(const Output<Node>& arg) : Op({arg}) {
    constructor_validate_and_infer_types();
}

bool opset::ArmNoOp::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

void opset::ArmNoOp::validate_and_infer_types() {
    auto validate_and_infer_elementwise_args = [] (Node* node) {
        NGRAPH_CHECK(node != nullptr, "nGraph node is empty! Cannot validate eltwise arguments.");
        element::Type element_type = node->get_input_element_type(0);
        PartialShape pshape = node->get_input_partial_shape(0);

        if (node->get_input_size() > 1) {
            for (size_t i = 1; i < node->get_input_size(); ++i) {
                NODE_VALIDATION_CHECK(
                    node,
                    element::Type::merge(element_type, element_type, node->get_input_element_type(i)),
                    "Argument element types are inconsistent.");
            }
        }

        return std::make_tuple(element_type, pshape);
    };
    auto args_et_pshape = validate_and_infer_elementwise_args(this);
    element::Type& args_et = std::get<0>(args_et_pshape);
    PartialShape& args_pshape = std::get<1>(args_et_pshape);

    set_output_type(0, args_et, args_pshape);
}

std::shared_ptr<Node> opset::ArmNoOp::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<opset::ArmNoOp>(new_args.at(0));
}
