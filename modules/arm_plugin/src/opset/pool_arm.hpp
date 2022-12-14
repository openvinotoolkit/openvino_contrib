// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_opset.hpp"
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {

class ArmMaxPoolV1 : public ov::op::v1::MaxPool {
public:
    OPENVINO_OP("ArmMaxPoolV1", "arm_opset", ov::op::v1::MaxPool);
    ArmMaxPoolV1(const ov::Output<Node>& arg,
                const ov::Strides& strides,
                const ov::Shape& pads_begin,
                const ov::Shape& pads_end,
                const ov::Shape& kernel,
                ov::op::RoundingType rounding_type = ov::op::RoundingType::FLOOR,
                ov::op::PadType auto_pad = ov::op::PadType::EXPLICIT);
};

class ArmMaxPoolV8 : public ov::op::v8::MaxPool {
public:
    OPENVINO_OP("ArmMaxPoolV8", "arm_opset", ov::op::v8::MaxPool);
    ArmMaxPoolV8(const ov::Output<Node>& arg,
                 const ov::Strides& strides,
                 const ov::Strides& dilations,
                 const ov::Shape& pads_begin,
                 const ov::Shape& pads_end,
                 const ov::Shape& kernel,
                 ov::op::RoundingType rounding_type = ov::op::RoundingType::FLOOR,
                 ov::op::PadType auto_pad = ov::op::PadType::EXPLICIT,
                 ov::element::Type index_element_type = ov::element::i64,
                 int64_t axis = 0);
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
                  ov::op::RoundingType rounding_type = ov::op::RoundingType::FLOOR,
                  const ov::op::PadType& auto_pad = ov::op::PadType::EXPLICIT);
};

}  // namespace opset
}  // namespace ArmPlugin
