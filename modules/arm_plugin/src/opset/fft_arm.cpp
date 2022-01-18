// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fft_arm.hpp"

using namespace ngraph;
using namespace ArmPlugin;

opset::ArmFFT::ArmFFT(const ngraph::Output<ngraph::Node>& data, ArmFFT::Axis axis, bool inverse)
    : Op({data}), m_axis(axis == opset::ArmFFT::Axis::axisY ? 1u : 0u), m_inverse(inverse) {
    constructor_validate_and_infer_types();
}

bool opset::ArmFFT::visit_attributes(ngraph::AttributeVisitor& visitor) {
    visitor.on_attribute("axis", m_axis);
    visitor.on_attribute("inverse", m_inverse);
    return true;
}

void opset::ArmFFT::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 1,
                          "ArmFFT op must have 1 input.");

    NODE_VALIDATION_CHECK(this, get_input_element_type(0) == element::f32,
                          "ArmFFT op input element type must be f32");

    const auto& input_shape = get_input_partial_shape(0);

    if (input_shape.rank().is_static()) {
        const auto input_rank = input_shape.rank().get_length();
        NODE_VALIDATION_CHECK(this,
                              input_rank >= 2,
                              "The input rank must be greater or equal to 2. Got input rank: ",
                              input_rank);

        auto last_dim_with_two = input_shape[input_rank - 1] & Dimension(2);
        NODE_VALIDATION_CHECK(this,
                              !last_dim_with_two.get_interval().empty(),
                              "The last dimension of input data must be 2. Got: ",
                              input_shape[input_rank - 1]);

        NODE_VALIDATION_CHECK(this, m_axis == 0 || input_rank > 2,
                              "ArmFFT op axis cannot be the last axis.");
    }

    set_output_type(0, get_input_element_type(0), input_shape);
}

std::shared_ptr<ngraph::Node> ArmPlugin::opset::ArmFFT::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    auto num_args = new_args.size();
    if (num_args == 1) {
        return std::make_shared<ArmFFT>(new_args.at(0), m_axis ? opset::ArmFFT::Axis::axisY : opset::ArmFFT::Axis::axisX, m_inverse);
    } else {
        throw ngraph_error("Unsupported number of arguments for ArmFFT operation");
    }
}
