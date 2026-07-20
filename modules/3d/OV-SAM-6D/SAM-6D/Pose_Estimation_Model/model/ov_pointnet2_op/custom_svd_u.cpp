// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "custom_svd_u.hpp"
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

using namespace TemplateExtension;

//! [op:ctor]
CustomSVDu::CustomSVDu(const ov::Output<ov::Node>& x) : Op({x}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void CustomSVDu::validate_and_infer_types() {
    // Support arbitrary batch dimensions, the last two dimensions are (M, N)
    const auto& svd_input = input(0);
    auto svd_shape = svd_input.get_partial_shape();
    set_output_type(0, get_input_element_type(0), ov::PartialShape(svd_shape)); // U
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> CustomSVDu::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<CustomSVDu>(new_args.at(0));
}
//! [op:copy]

//! [op:visit_attributes]
bool CustomSVDu::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

// JacobiSVD already returns non-negative singular values in descending order.
// Canonicalize the paired singular vectors so U/V use a deterministic sign
// convention across backends while preserving A = U * S * V^T.
static void ensure_svd_signs(Eigen::MatrixXf& U, Eigen::MatrixXf& V) {
    const int cols = std::min(U.cols(), V.cols());
    for (int i = 0; i < cols; ++i) {
        Eigen::Index max_abs_row = 0;
        U.col(i).cwiseAbs().maxCoeff(&max_abs_row);
        if (U(max_abs_row, i) < 0.0f) {
            U.col(i) = -U.col(i);
            V.col(i) = -V.col(i);
        }
    }
}

//! [op:evaluate]
bool CustomSVDu::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    // Support batch SVD, input shape: [batch..., M, N]
    const auto& in = inputs[0];
    auto shape = in.get_shape();
    if (shape.size() < 2)
        throw std::runtime_error("CustomSVDu input must have at least 2 dimensions");
    size_t rank = shape.size();
    size_t m = shape[rank - 2], n = shape[rank - 1];
    size_t batch = 1;
    for (size_t i = 0; i < rank - 2; ++i) batch *= shape[i];
    const float* data = in.data<const float>();
    
    // Output shape
    std::vector<size_t> u_shape = shape; u_shape[rank - 1] = m; // (batch..., M, M)
    outputs[0].set_shape(u_shape);
    float* u_data = outputs[0].data<float>();
    size_t in_mat_size = m * n;
    size_t u_mat_size = m * m;
    size_t s_vec_size = std::min(m, n);
    size_t v_mat_size = n * n;
    
    for (size_t b = 0; b < batch; ++b) {
        const float* batch_data = data + b * in_mat_size;
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A(batch_data, m, n);
        
        // Use Eigen's JacobiSVD and canonicalize the paired singular vectors.
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        
        // Get SVD results
        Eigen::MatrixXf U = svd.matrixU();
        Eigen::MatrixXf V = svd.matrixV();
        
        // Keep U/V sign choices deterministic for the downstream rotation path.
        ensure_svd_signs(U, V);
        
        // Write outputs
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(u_data + b * u_mat_size, m, m) = U;
    }
    return true;
}

bool CustomSVDu::has_evaluate() const {
    return true;
}
//! [op:evaluate]
