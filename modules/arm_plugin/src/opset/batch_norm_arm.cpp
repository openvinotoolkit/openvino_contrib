// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_norm_arm.hpp"
#include <sstream>

using namespace std;
using namespace ov;

ArmPlugin::opset::v5::ArmBatchNormInference::ArmBatchNormInference(const Output<Node>& input,
                                                                   const Output<Node>& gamma,
                                                                   const Output<Node>& beta,
                                                                   const Output<Node>& mean,
                                                                   const Output<Node>& variance,
                                                                   double epsilon,
                                                                   PartialShape output_shape)
        : m_output_shape{std::move(output_shape)} {
    set_arguments({input, gamma, beta, mean, variance});
    set_eps_value(epsilon);
    constructor_validate_and_infer_types();
}

void ArmPlugin::opset::v5::ArmBatchNormInference::validate_and_infer_types() {
    if (m_output_shape == PartialShape{}) {
        ov::op::v5::BatchNormInference::validate_and_infer_types();
    } else {
        set_output_type(0, get_input_element_type(0), m_output_shape);
    }
}

std::shared_ptr<ov::Node>
ArmPlugin::opset::v5::ArmBatchNormInference::clone_with_new_inputs(const OutputVector &new_args) const {
    enum BatchNormInput {Features, Gamma, Beta, Mean, Variance};
    return std::make_shared<ArmBatchNormInference>(new_args.at(BatchNormInput::Features),
                                                   new_args.at(BatchNormInput::Gamma),
                                                   new_args.at(BatchNormInput::Beta),
                                                   new_args.at(BatchNormInput::Mean),
                                                   new_args.at(BatchNormInput::Variance),
                                                   get_eps_value(),
                                                   m_output_shape);
}
