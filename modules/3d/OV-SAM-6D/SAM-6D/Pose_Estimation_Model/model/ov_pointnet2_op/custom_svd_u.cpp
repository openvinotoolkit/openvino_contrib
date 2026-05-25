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

// Helper function to ensure proper SVD signs (similar to PyTorch)
void ensure_svd_u_signs(Eigen::MatrixXf& U, Eigen::VectorXf& S, Eigen::MatrixXf& V) {
    // Ensure singular values are non-negative and sorted in descending order
   for (int i = 0; i < S.size(); ++i) {
        if (S(i) < 0) {
            S(i) = -S(i);
            U.col(i) = -U.col(i);
        }
    }
    
    // Sort singular values in descending order
    std::vector<std::pair<float, int>> s_indices;
    for (int i = 0; i < S.size(); ++i) {
        s_indices.push_back({S(i), i});
    }
    std::sort(s_indices.begin(), s_indices.end(), 
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                  return a.first > b.first;
              });
    
    // Reorder U, S, V according to sorted singular values
    Eigen::MatrixXf U_new = U;
    Eigen::MatrixXf V_new = V;
    Eigen::VectorXf S_new = S;
    
    for (int i = 0; i < S.size(); ++i) {
        int old_idx = s_indices[i].second;
        S_new(i) = s_indices[i].first;
        U_new.col(i) = U.col(old_idx);
        V_new.col(i) = V.col(old_idx);
    }
    
    U = U_new;
    S = S_new;
    V = V_new;
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
        
        // Use Eigen's BDCSVD for better numerical stability (similar to LAPACK)
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        
        // Get SVD results
        Eigen::MatrixXf U = svd.matrixU();
        Eigen::VectorXf S = svd.singularValues();
        Eigen::MatrixXf V = svd.matrixV();
        
        // Ensure proper signs and ordering (similar to PyTorch)
        ensure_svd_u_signs(U, S, V);
        
        // Write outputs
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(u_data + b * u_mat_size, m, m) = U;
    }
    return true;
}

bool CustomSVDu::has_evaluate() const {
    return true;
}
//! [op:evaluate]
