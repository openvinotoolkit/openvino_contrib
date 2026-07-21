// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "custom_det.hpp"
#include <Eigen/Dense>
#include <stdexcept>

using namespace TemplateExtension;

//! [op:ctor]
CustomDet::CustomDet(const ov::Output<ov::Node>& x) : Op({x}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void CustomDet::validate_and_infer_types() {
    // inputshape: (batch_size, m, n)
    const auto& det_input = input(0);
    auto det_shape = det_input.get_partial_shape();
    auto elem_type = get_input_element_type(0);

    if (det_shape.rank().is_static() && det_shape.rank().get_length() == 3) {
        auto n1 = det_shape[1];
        auto n2 = det_shape[2];
        if (n1 != n2) {
            throw std::runtime_error("The last two dimensions must be equal (square matrices)");
        }
        // output shape : (batch_size)
        set_output_type(0, elem_type, ov::PartialShape{det_shape[0]});
    } else {
        throw std::runtime_error("Input must be a 3D tensor of shape (b, n, n)");
    }
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> CustomDet::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<CustomDet>(new_args.at(0));
}
//! [op:copy]

//! [op:visit_attributes]
bool CustomDet::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool CustomDet::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const auto& in = inputs[0];
    auto shape = in.get_shape();
    if (shape.size() != 3 || shape[1] != shape[2])
        throw std::runtime_error("Input must be a 3D tensor with square matrices");

    size_t batch = shape[0];
    size_t n = shape[1];
    const float* data = in.data<const float>();

    auto& out = outputs[0];
    out.set_shape({batch});
    float* out_data = out.data<float>();

    for (size_t b = 0; b < batch; ++b) {
        // Each batch's starting pointer
        const float* batch_data = data + b * n * n;
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A(batch_data, n, n);
        out_data[b] = A.determinant();
    }
    return true;
}

bool CustomDet::has_evaluate() const {
    return true;
}
//! [op:evaluate]
