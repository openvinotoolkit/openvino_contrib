// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0



#include <arm_compute/runtime/NEON/functions/NEReductionOperation.h>
#include <arm_compute/runtime/NEON/functions/NEReduceMean.h>
#include <ngraph/runtime/reference/logical_reduction.hpp>
#include "arm_converter/arm_converter.hpp"
#include "opset/utils.hpp"

namespace ArmPlugin {
template<typename Reduce>
static auto ConvertReduce(const Reduce& node, const arm_compute::ReductionOperation& op, Converter* converter) {
    auto axes = safe_cast<opset::Constant>(node.input_value(1).get_node())->template cast_vector<std::int64_t>();
    if (axes.size() != 1) {
        IE_THROW() << "Arm Plugin: Multiple reduction axes aren't supported";
    }
    unsigned int axis = AxisCast(axes[0], node.get_input_shape(0).size());
    return converter->MakeConversion<arm_compute::NEReductionOperation>(node.input(0), node.output(0), axis, op, node.get_keep_dims());
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ReduceProd& node) {
    return ConvertReduce(node, arm_compute::ReductionOperation::PROD, this);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ReduceMin& node) {
    return ConvertReduce(node, arm_compute::ReductionOperation::MIN, this);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ReduceMax& node) {
    return ConvertReduce(node, arm_compute::ReductionOperation::MAX, this);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ReduceSum& node) {
    return ConvertReduce(node, arm_compute::ReductionOperation::SUM, this);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ReduceMean& node) {
    arm_compute::Coordinates axes;
    auto reduction_axes = safe_cast<opset::Constant>(node.input_value(1).get_node())->cast_vector<std::int64_t>();
    for (size_t i = 0; i < reduction_axes.size(); ++i) {
        auto pos = AxisCast(i, reduction_axes.size());
        axes.set(pos, reduction_axes[i]);
    }
    return MakeConversion<arm_compute::NEReduceMean>(node.input(0), axes, node.get_keep_dims(), node.output(0));
}

static void wrap_reduce_logical_and(const std::uint8_t* arg,
                                    std::uint8_t* out,
                                    const ngraph::Shape& input_shape,
                                    const ngraph::AxisSet& reduction_axes) {
    ngraph::runtime::reference::reduce_logical_and(reinterpret_cast<const char*>(arg),
                                                   reinterpret_cast<char*>(out),
                                                   input_shape,
                                                   reduction_axes);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ReduceLogicalAnd& node) {
    if (node.get_input_element_type(0) != ngraph::element::u8) {
        IE_THROW() << "Arm Plugin: Unsupported Type: " << node.get_input_element_type(0);
    }

    return MakeConversion(wrap_reduce_logical_and,
                          node.input(0),
                          node.output(0),
                          node.get_input_shape(0),
                          node.get_reduction_axes());
}

static void wrap_reduce_logical_or(const std::uint8_t* arg,
                                   std::uint8_t* out,
                                   const ngraph::Shape& input_shape,
                                   const ngraph::AxisSet& reduction_axes) {
    ngraph::runtime::reference::reduce_logical_or(reinterpret_cast<const char*>(arg),
                                                  reinterpret_cast<char*>(out),
                                                  input_shape,
                                                  reduction_axes);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ReduceLogicalOr& node) {
    if (node.get_input_element_type(0) != ngraph::element::u8) {
        IE_THROW() << "Arm Plugin: Unsupported Type: " << node.get_input_element_type(0);
    }

    return MakeConversion(wrap_reduce_logical_or,
                          node.input(0),
                          node.output(0),
                          node.get_input_shape(0),
                          node.get_reduction_axes());
}

} // namespace ArmPlugin
