// Copyright (C) 2022 Intel Corporation
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
    OPENVINO_OP("ArmMaxPool", "arm_opset", ov::op::v1::MaxPool, 1);

    ArmMaxPool(const ov::Output<Node> &arg,
                 const ov::Strides &strides,
                 const ov::Shape &pads_begin,
                 const ov::Shape &pads_end,
                 const ov::Shape &kernel,
                 ov::op::RoundingType rounding_type = ov::op::RoundingType::FLOOR,
                 ov::op::PadType auto_pad = ov::op::PadType::EXPLICIT);
};

class ArmAvgPool : public ov::op::v1::AvgPool {
public:
    OPENVINO_OP("ArmAvgPool", "arm_opset", ov::op::v1::AvgPool, 1);
    ArmAvgPool(const ov::Output<Node>& arg,
               const ov::Strides& strides,
               const ov::Shape& pads_begin,
               const ov::Shape& pads_end,
               const ov::Shape& kernel,
               bool exclude_pad,
               ov::op::RoundingType rounding_type = ov::op::RoundingType::FLOOR,
               const ov::op::PadType& auto_pad = ov::op::PadType::EXPLICIT);
};
}

namespace v8 {
class ArmMaxPool : public ov::op::v8::MaxPool {
public:
    OPENVINO_OP("ArmMaxPool", "arm_opset", ov::op::v8::MaxPool, 8);

    ArmMaxPool(const ov::Output<Node> &arg,
                 const ov::Strides &strides,
                 const ov::Strides &dilations,
                 const ov::Shape &pads_begin,
                 const ov::Shape &pads_end,
                 const ov::Shape &kernel,
                 ov::op::RoundingType rounding_type = ov::op::RoundingType::FLOOR,
                 ov::op::PadType auto_pad = ov::op::PadType::EXPLICIT,
                 ov::element::Type index_element_type = ov::element::i64,
                 int64_t axis = 0);
};
}



}  // namespace opset
}  // namespace ArmPlugin
