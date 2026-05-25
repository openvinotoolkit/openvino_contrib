// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "custom_svd_v.hpp"
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

using namespace TemplateExtension;

//! [op:ctor]
CustomSVDv::CustomSVDv(const ov::Output<ov::Node>& x) : Op({x}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void CustomSVDv::validate_and_infer_types() {
    // Support arbitrary batch dimensions, the last two dimensions are (M, N)
    const auto& svd_input = input(0);
    auto svd_shape = svd_input.get_partial_shape();
    set_output_type(0, get_input_element_type(0), ov::PartialShape(svd_shape)); // V
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> CustomSVDv::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<CustomSVDv>(new_args.at(0));
}
//! [op:copy]

//! [op:visit_attributes]
bool CustomSVDv::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

// Helper function to ensure proper SVD signs (similar to PyTorch)
void ensure_svd_v_signs(Eigen::MatrixXf& U, Eigen::VectorXf& S, Eigen::MatrixXf& V) {
    // Ensure singular values are non-negative and sorted in descending order
    for (int i = 0; i < S.size(); ++i) {
        if (S(i) < 0) {
	    std::cout << "\n we got hiy S(i) < 0";
            S(i) = -S(i);
            U.col(i) = -V.col(i);
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
bool CustomSVDv::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    // Support batch SVD, input shape: [batch..., M, N]
    const auto& in = inputs[0];
    auto shape = in.get_shape();
    if (shape.size() < 2)
        throw std::runtime_error("CustomSVDv input must have at least 2 dimensions");
    size_t rank = shape.size();
    size_t m = shape[rank - 2], n = shape[rank - 1];
    size_t batch = 1;
    for (size_t i = 0; i < rank - 2; ++i) batch *= shape[i];
    const float* data = in.data<const float>();
    
    std::vector<size_t> v_shape = shape; v_shape[rank - 2] = n; v_shape[rank - 1] = n;
    outputs[0].set_shape(v_shape);
    float* v_data = outputs[0].data<float>();
    size_t in_mat_size = m * n;
    size_t u_mat_size = m * m;
    size_t s_vec_size = std::min(m, n);
    size_t v_mat_size = n * n;
    
    for (size_t b = 0; b < batch; ++b) {
        const float* batch_data = data + b * in_mat_size;
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A(batch_data, m, n);
        
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        
        // Get SVD results
        Eigen::MatrixXf U = svd.matrixU();
        Eigen::VectorXf S = svd.singularValues();
        Eigen::MatrixXf V = svd.matrixV();
        
        // Ensure proper signs and ordering (similar to PyTorch)
        ensure_svd_v_signs(U, S, V);
        
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(v_data + b * v_mat_size, n, n) = V;
    }
    return true;
}

bool CustomSVDv::has_evaluate() const {
    return true;
}
//! [op:evaluate]
