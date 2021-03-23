// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0



#include <arm_compute/runtime/NEON/functions/NEReductionOperation.h>
#include <arm_compute/runtime/NEON/functions/NEReduceMean.h>
#include <ngraph/runtime/reference/logical_reduction.hpp>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<typename Reduce>
static auto ConvertReduce(const Reduce& node, const arm_compute::ReductionOperation& op, Converter* converter) {
    auto axes = dynamic_cast<const opset::Constant&>(*(node.input_value(1).get_node())).cast_vector<int64_t>();
    if (axes.size() != 1) {
        IE_THROW() << "Multiple reduction axes aren't supported";
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
    auto reduction_axes = dynamic_cast<const opset::Constant&>(*(node.input_value(1).get_node())).cast_vector<int64_t>();
    for (size_t i = 0; i < reduction_axes.size(); ++i) {
        auto pos = AxisCast(i, reduction_axes.size());
        axes.set(pos, reduction_axes[i]);
    }
    return MakeConversion<arm_compute::NEReduceMean>(node.input(0), axes, node.get_keep_dims(), node.output(0));
}

template <typename T>
void wrap_reduce_logical_and(const T* arg,
                             T* out,
                             const ngraph::Shape& input_shape,
                             const ngraph::AxisSet& reduction_axes,
                             bool keep_dims) {
    ngraph::runtime::reference::reduce_logical_and(reinterpret_cast<const char*>(arg),
                                                   reinterpret_cast<char*>(out),
                                                   input_shape,
                                                   reduction_axes,
                                                   keep_dims);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ReduceLogicalAnd& node) {
    if (node.get_input_element_type(0) != ngraph::element::u8) {
        IE_THROW() << "Unsupported Type: " << node.get_input_element_type(0);
    }

    auto func = wrap_reduce_logical_and<std::uint8_t>;
    return MakeConversion(func,
                          node.input(0),
                          node.output(0),
                          node.get_input_shape(0),
                          node.get_reduction_axes(),
                          node.get_keep_dims());
}

template <typename T>
void wrap_reduce_logical_or(const T* arg,
                            T* out,
                            const ngraph::Shape& input_shape,
                            const ngraph::AxisSet& reduction_axes,
                            bool keep_dims) {
    ngraph::runtime::reference::reduce_logical_or(reinterpret_cast<const char*>(arg),
                                                  reinterpret_cast<char*>(out),
                                                  input_shape,
                                                  reduction_axes,
                                                  keep_dims);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ReduceLogicalOr& node) {
    if (node.get_input_element_type(0) != ngraph::element::u8) {
        IE_THROW() << "Unsupported Type: " << node.get_input_element_type(0);
    }

    auto func = wrap_reduce_logical_or<std::uint8_t>;
    return MakeConversion(func,
                          node.input(0),
                          node.output(0),
                          node.get_input_shape(0),
                          node.get_reduction_axes(),
                          node.get_keep_dims());
}

} // namespace ArmPlugin
