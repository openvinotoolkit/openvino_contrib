// Copyright (C) 2020-2022 Intel Corporation
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
                                        const ov::op::RoundingType rounding_type,
                                        const ov::op::PadType auto_pad)
        : ov::op::v1::MaxPool{arg, strides, pads_begin, pads_end, kernel, rounding_type, auto_pad} {}

opset::v8::ArmMaxPool::ArmMaxPool(const ov::Output<Node>& arg,
                                  const ov::Strides& strides,
                                  const ov::Strides& dilations,
                                  const ov::Shape& pads_begin,
                                  const ov::Shape& pads_end,
                                  const ov::Shape& kernel,
                                  const ov::op::RoundingType rounding_type,
                                  const ov::op::PadType auto_pad,
                                  const ov::element::Type index_element_type,
                                  const int64_t axis)
        : ov::op::v8::MaxPool{arg, strides, dilations, pads_begin, pads_end, kernel, rounding_type, auto_pad, index_element_type, axis} {}

opset::v1::ArmAvgPool::ArmAvgPool(const ov::Output<Node>& arg,
                              const ov::Strides& strides,
                              const ov::Shape& pads_begin,
                              const ov::Shape& pads_end,
                              const ov::Shape& kernel,
                              bool exclude_pad,
                              ov::op::RoundingType rounding_type,
                              const ov::op::PadType& auto_pad)
        : ov::op::v1::AvgPool{arg, strides, pads_begin, pads_end, kernel, exclude_pad, rounding_type, auto_pad} {}
