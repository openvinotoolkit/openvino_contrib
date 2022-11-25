// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0



#include <arm_compute/runtime/NEON/functions/NESlice.h>
#include <ngraph/runtime/reference/slice.hpp>
#include "ngraph/slice_plan.hpp"
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {

template <typename T, typename U>
void wrap_slice(const T* arg,
                        const U* begin,
                        const U* steps,
                        const U* axes,
                        T* out,
                        const ngraph::Shape& arg_shape,
                        const ngraph::Shape& out_shape,
                        const ngraph::Shape& begin_shape,
                        const ngraph::Shape& step_shape,
                        const ngraph::Shape& axes_shape,
                        size_t elem_size) {
    std::vector<int64_t> begin_const(begin, begin + ngraph::shape_size(begin_shape));
    std::vector<int64_t> step_const(steps, steps + ngraph::shape_size(step_shape));
    std::vector<int64_t> axes_const(axes, axes + ngraph::shape_size(axes_shape));

    ngraph::runtime::reference::slice(reinterpret_cast<const char*>(arg),
                                      arg_shape,
                                      reinterpret_cast<char*>(out),
                                      out_shape,
                                      elem_size,
                                      begin_const,
                                      step_const,
                                      axes_const);
}

template <> Converter::Conversion::Ptr Converter::Convert(const ngraph::op::v8::Slice& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.input(1),
                                    node.input(3),
                                    node.input(4),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_output_shape(0),
                                    node.get_input_shape(1),
                                    node.get_input_shape(3),
                                    node.get_input_shape(4),
                                    node.get_element_type().size());
    };

    return CallSwitch(
        AP_WRAP(make, wrap_slice),
        node.input(0), allTypes,
        node.input(1), indexTypes);
}

template <> Converter::Conversion::Ptr Converter::Convert(const opset::ArmSlice& node) {
    auto begin  = safe_cast<ngraph::op::Constant>(
                        node.input_value(1).get_node_shared_ptr())->cast_vector<int>();
    auto end    = safe_cast<ngraph::op::Constant>(
                        node.input_value(2).get_node_shared_ptr())->cast_vector<int>();
    auto axes = safe_cast<ngraph::op::Constant>(
                        node.input_value(4).get_node_shared_ptr())->cast_vector<int>();
    auto shape  = node.get_input_shape(0);
    const auto rank = shape.size();

    arm_compute::Coordinates starts, finishes;
    starts.set_num_dimensions(rank);
    finishes.set_num_dimensions(rank);
    for (size_t i = 0; i < rank; ++i) {
        const auto axis = AxisCast(i, rank);
        starts.set(axis, 0);
        finishes.set(axis, shape[i]);
    }

    for (size_t i = 0; i < begin.size(); ++i) {
        const auto axis = AxisCast(
            axes[i] >= 0 ?
            axes[i] : axes[i] + rank, rank);

        starts.set(axis, begin[i]);
        finishes.set(axis, end[i] >= 0 ?
            end[i] : shape[axes[i]] + end[i]);
    }

    return MakeConversion<arm_compute::NESlice>(node.input(0), node.output(0), starts, finishes);
}
}  //  namespace ArmPlugin