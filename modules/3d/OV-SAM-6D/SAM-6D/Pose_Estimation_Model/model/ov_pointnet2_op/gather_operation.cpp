// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
// Portions of this file are adapted from the MIT-licensed PointNet2 kernels
// by Erik Wijmans and Charles R. Qi. See third-party-programs.txt.

#include "gather_operation.hpp"
#include <cmath>

using namespace TemplateExtension;

//! [op:ctor]
GatherOperation::GatherOperation(const ov::Output<ov::Node>& features, const ov::Output<ov::Node>& idx) : Op({features, idx}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void GatherOperation::validate_and_infer_types() {
    // Operation doesn't change shapes end element type
    /*
    Parameters
    ----------
    features : torch.Tensor
        (B, C, N) tensor

    idx : torch.Tensor
        (B, npoint) tensor of the features to gather

    Returns
    -------
    torch.Tensor
        (B, C, npoint) tensor
    */
    const auto& features_input = input(0);
    const auto& idx_input = input(1);

    auto features_shape = features_input.get_partial_shape();
    auto idx_shape = idx_input.get_partial_shape();
    ov::PartialShape output_shape = {features_shape[0], features_shape[1], idx_shape[1]};
    set_output_type(0, features_input.get_element_type(), output_shape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> GatherOperation::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    // OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");

    return std::make_shared<GatherOperation>(new_args.at(0), new_args.at(1));
}
//! [op:copy]

//! [op:visit_attributes]
bool GatherOperation::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool GatherOperation::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const float* features = inputs[0].data<const float>();
    const int* idx = inputs[1].data<const int>();

    int b = inputs[0].get_shape()[0]; // batch size
    int c = inputs[0].get_shape()[1]; // channels
    int n = inputs[0].get_shape()[2]; // number of points
    int npoints = inputs[1].get_shape()[1]; // number of points to gather

    ov::PartialShape output_shape = {b, c, npoints};
    outputs[0].set_shape(output_shape.to_shape());
    auto* out_tensor = outputs[0].data<float>();

    // Initialize output to 0, to avoid uninitialized memory
    int output_total = b * c * npoints;
    for (int i = 0; i < output_total; ++i) {
        out_tensor[i] = 0.0f;
    }
    
    for (int i = 0; i < b; ++i) {
        // For each channel c
        for (int l = 0; l < c; ++l) {
            // For each sample point m
            for (int j = 0; j < npoints; ++j) {
                // Get the index of the corresponding original point
                int a = idx[i * npoints + j];
                if(a >= 0 && a < n) { // Ensure the index is valid
                    // Correct memory layout calculation
                    // Input: features[batch][channel][point] = features[i * c * n + l * n + a]
                    // Output: out_tensor[batch][channel][point] = out_tensor[i * c * npoints + l * npoints + j]
                    float input_val = features[i * c * n + l * n + a];
                    
                    // Check if the input value is NaN or Inf, if so, set to 0
                    if (std::isnan(input_val) || std::isinf(input_val)) {
                        out_tensor[i * c * npoints + l * npoints + j] = 0.0f;
                    } else {
                        out_tensor[i * c * npoints + l * npoints + j] = input_val;
                    }
                } else {
                    // If the index is invalid, set to 0.0
                    out_tensor[i * c * npoints + l * npoints + j] = 0.0f;
                }
            }
        }
    }
    return true;
}

bool GatherOperation::has_evaluate() const {
    return true;
}
//! [op:evaluate]
