// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_opset.hpp"
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {
namespace v1 {

class ArmMaxPool : public ov::op::v1::MaxPool {
public:
    OPENVINO_OP("ArmMaxPool", "arm_opset", ov::op::v1::MaxPool);

    ArmMaxPool(const ov::Output<Node>& arg,
               const ov::Strides& strides,
               const ov::Shape& pads_begin,
               const ov::Shape& pads_end,
               const ov::Shape& kernel,
               const ov::op::RoundingType& rounding_type = ov::op::RoundingType::FLOOR,
               const ov::op::PadType& auto_pad = ov::op::PadType::EXPLICIT,
               const ov::PartialShape& output_shape = {});

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    void validate_and_infer_types() override;

private:
    ov::PartialShape m_output_shape;
};

class ArmAvgPool : public ov::op::v1::AvgPool {
public:
    OPENVINO_OP("ArmAvgPool", "arm_opset", ov::op::v1::AvgPool);

    ArmAvgPool(const ov::Output<Node>& arg,
               const ov::Strides& strides,
               const ov::Shape& pads_begin,
               const ov::Shape& pads_end,
               const ov::Shape& kernel,
               bool exclude_pad,
               const ov::op::RoundingType& rounding_type = ov::op::RoundingType::FLOOR,
               const ov::op::PadType& auto_pad = ov::op::PadType::EXPLICIT,
               const ov::PartialShape& output_shape = {});

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    void validate_and_infer_types() override;

private:
    ov::PartialShape m_output_shape;
};
} // namespace v1

namespace v8 {
class ArmMaxPool : public ov::op::v8::MaxPool {
public:
    OPENVINO_OP("ArmMaxPool", "arm_opset", ov::op::v8::MaxPool);

    ArmMaxPool(const ov::Output<Node>& arg,
               const ov::Strides& strides,
               const ov::Strides& dilations,
               const ov::Shape& pads_begin,
               const ov::Shape& pads_end,
               const ov::Shape& kernel,
               const ov::op::RoundingType& rounding_type = ov::op::RoundingType::FLOOR,
               const ov::op::PadType& auto_pad = ov::op::PadType::EXPLICIT,
               const ov::element::Type& index_element_type = ov::element::i64,
               int64_t axis = 0,
               const ov::PartialShape& output_shape = {});

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    void validate_and_infer_types() override;

private:
    ov::PartialShape m_output_shape;
};
} // namespace v8

}  // namespace opset
}  // namespace ArmPlugin
