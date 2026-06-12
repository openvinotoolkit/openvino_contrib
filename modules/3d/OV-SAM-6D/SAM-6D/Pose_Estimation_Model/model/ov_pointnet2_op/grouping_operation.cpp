// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "grouping_operation.hpp"

using namespace TemplateExtension;

//! [op:ctor]
GroupingOperation::GroupingOperation(const ov::Output<ov::Node>& features, const ov::Output<ov::Node>& idx) : Op({features, idx}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void GroupingOperation::validate_and_infer_types() {
    // Operation doesn't change shapes end element type
    /*
    Parameters
    ----------
    features : torch.Tensor
        (B, C, N) tensor of features to group
    idx : torch.Tensor
        (B, npoint, nsample) tensor containing the indicies of features to group with

    Returns
    -------
    torch.Tensor
        (B, C, npoint, nsample) tensor
    */
    const auto& features_input = input(0);
    const auto& idx_input = input(1);

    auto features_shape = features_input.get_partial_shape();
    auto idx_shape = idx_input.get_partial_shape();
    // Dynamic inference output shape
    ov::PartialShape output_shape = {features_shape[0], features_shape[1], idx_shape[1], idx_shape[2]};
    set_output_type(0, features_input.get_element_type(), output_shape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> GroupingOperation::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    // OPENVINO_ASSERT(new_args.size() == 2, "Incorrect number of new arguments");
    return std::make_shared<GroupingOperation>(new_args.at(0), new_args.at(1));
}
//! [op:copy]

//! [op:visit_attributes]
bool GroupingOperation::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool GroupingOperation::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const float* features = inputs[0].data<const float>();
    const int* idx = inputs[1].data<const int>();

    int b = inputs[0].get_shape()[0]; // batch size
    int c = inputs[0].get_shape()[1]; // number of channels
    int n = inputs[0].get_shape()[2]; // number of points in features
    int npoint = inputs[1].get_shape()[1]; // number of points in idx
    int nsample = inputs[1].get_shape()[2]; // number of samples in idx

    ov::PartialShape output_shape = {b, c, npoint, nsample};
    outputs[0].set_shape(output_shape.to_shape());
    auto& out_tensor = outputs[0];

    for (int batch_index = 0; batch_index < b; ++batch_index) {
        const float *current_features = features + batch_index * c * n;
        const int *current_idx = idx + batch_index * npoint * nsample;
        float *current_out = out_tensor.data<float>() + batch_index * c * npoint * nsample;
        for (int i = 0; i < c * npoint * nsample; ++i) {
            current_out[i] = 0.0f;
        }

        for (int l = 0; l < c; ++l) {
            for (int j = 0; j < npoint; ++j) {
                for (int k = 0; k < nsample; ++k) {
                    int ii = current_idx[j * nsample + k];
                    if(ii >= 0 && ii < n) {
                        current_out[(l * npoint + j) * nsample + k] = current_features[l * n + ii];
                    }
                }
            }
        }
    }
    
    return true;
}

bool GroupingOperation::has_evaluate() const {
    return true;
}
