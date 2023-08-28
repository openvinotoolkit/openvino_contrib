// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/core/type/element_type.hpp"

inline std::shared_ptr<ov::Model> create_matmul_test_model() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 2, 10, 10});
    auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{3, 2, 10, 20});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(param, constant, false, false);
    auto result = std::make_shared<ov::op::v0::Result>(matmul);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "MatMul");
}

class SuperDummyOp : public ov::op::Op {
public:
    inline static constexpr type_info_t type_info{"SuperOperation", "opset0"};
    const type_info_t& get_type_info() const override { return type_info; }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        check_new_args_count(this, new_args);
        return std::make_shared<SuperDummyOp>(new_args.at(0), new_args.at(1));
    }

    void validate_and_infer_types() override {
        ov::element::Type result_et;

        NODE_VALIDATION_CHECK(this,
                              ov::element::Type::merge(result_et, get_input_element_type(0), get_input_element_type(1)),
                              "Arguments do not have the same element type (arg0 element type: ",
                              get_input_element_type(0),
                              ", arg1 element type: ",
                              get_input_element_type(1),
                              ").");

        const auto& A_partial_shape = get_input_partial_shape(0);
        const auto& B_partial_shape = get_input_partial_shape(1);

        if (A_partial_shape.rank().is_static() && B_partial_shape.rank().is_static()) {
            ov::PartialShape output_shape;
            set_output_type(0, result_et, A_partial_shape);
        } else {
            set_output_type(0, result_et, ov::PartialShape::dynamic());
        }
    }

    SuperDummyOp(const ov::Output<Node>& A, const ov::Output<Node>& B) : ov::op::Op(ov::OutputVector{A, B}) {
        constructor_validate_and_infer_types();
    }
};

inline std::shared_ptr<ov::Model> create_super_operation_test_model() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 2, 10, 10});
    auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{3, 2, 10, 20});
    auto super_op = std::make_shared<SuperDummyOp>(param, constant);
    auto result = std::make_shared<ov::op::v0::Result>(super_op);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "SuperOperation");
}