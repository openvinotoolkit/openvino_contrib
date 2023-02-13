// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pool_arm.hpp"

using namespace ngraph;
using namespace ArmPlugin;

opset::v1::ArmMaxPool::ArmMaxPool(const ov::Output<Node>& arg,
                                  const ov::Strides& strides,
                                  const ov::Shape& pads_begin,
                                  const ov::Shape& pads_end,
                                  const ov::Shape& kernel,
                                  const ov::op::RoundingType& rounding_type,
                                  const ov::op::PadType& auto_pad,
                                  const PartialShape& output_shape) : m_output_shape{output_shape} {
    set_arguments({arg});
    set_strides(strides);
    set_pads_begin(pads_begin);
    set_adding_above(pads_end);
    set_kernel(kernel);
    set_rounding_type(rounding_type);
    set_auto_pad(auto_pad);
    constructor_validate_and_infer_types();
}


std::shared_ptr<ov::Node> opset::v1::ArmMaxPool::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<ArmMaxPool>(new_args.at(0), m_strides, m_pads_begin, m_pads_end, m_kernel, m_rounding_type, m_auto_pad, m_output_shape);
}

void opset::v1::ArmMaxPool::validate_and_infer_types() {
    if (m_output_shape == PartialShape{}) {
        ov::op::v1::MaxPool::validate_and_infer_types();
    } else {
        set_output_type(0, get_input_element_type(0), m_output_shape);
    }
}

opset::v8::ArmMaxPool::ArmMaxPool(const ov::Output<Node>& arg,
                                  const ov::Strides& strides,
                                  const ov::Strides& dilations,
                                  const ov::Shape& pads_begin,
                                  const ov::Shape& pads_end,
                                  const ov::Shape& kernel,
                                  const ov::op::RoundingType& rounding_type,
                                  const ov::op::PadType& auto_pad,
                                  const ov::element::Type& index_element_type,
                                  int64_t axis,
                                  const PartialShape& output_shape) : m_output_shape{output_shape} {
    set_arguments({arg});
    set_strides(strides);
    set_dilations(dilations);
    set_pads_begin(pads_begin);
    set_adding_above(pads_end);
    set_kernel(kernel);
    set_rounding_type(rounding_type);
    set_auto_pad(auto_pad);
    set_index_element_type(index_element_type);
    set_axis(axis);
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> opset::v8::ArmMaxPool::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<ArmMaxPool>(new_args.at(0), m_strides, get_dilations(), m_pads_begin, m_pads_end, m_kernel, m_rounding_type, m_auto_pad, get_index_element_type(), get_axis(), m_output_shape);
}

void opset::v8::ArmMaxPool::validate_and_infer_types() {
    if (m_output_shape == PartialShape{}) {
        ov::op::v8::MaxPool::validate_and_infer_types();
    } else {
        set_output_type(0, get_input_element_type(0), m_output_shape);
        set_output_type(1, get_index_element_type(), m_output_shape);
    }
}

opset::v1::ArmAvgPool::ArmAvgPool(const ov::Output<Node>& arg,
                                  const ov::Strides& strides,
                                  const ov::Shape& pads_begin,
                                  const ov::Shape& pads_end,
                                  const ov::Shape& kernel,
                                  bool exclude_pad,
                                  const ov::op::RoundingType& rounding_type,
                                  const ov::op::PadType& auto_pad,
                                  const PartialShape& output_shape) : m_output_shape{output_shape} {
    set_arguments({arg});
    set_strides(strides);
    set_pads_begin(pads_begin);
    set_pads_end(pads_end);
    set_kernel(kernel);
    set_exclude_pad(exclude_pad);
    set_rounding_type(rounding_type);
    set_auto_pad(auto_pad);
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> opset::v1::ArmAvgPool::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<ArmAvgPool>(new_args.at(0), m_strides, m_pads_begin, m_pads_end, m_kernel, m_exclude_pad, m_rounding_type, m_auto_pad, m_output_shape);
}

void opset::v1::ArmAvgPool::validate_and_infer_types() {
    if (m_output_shape == PartialShape{}) {
        ov::op::v1::AvgPool::validate_and_infer_types();
    } else {
        set_output_type(0, get_input_element_type(0), m_output_shape);
    }
}
