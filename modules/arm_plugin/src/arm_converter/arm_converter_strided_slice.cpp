// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0



#include <arm_compute/runtime/NEON/functions/NEStridedSlice.h>
#include <ngraph/runtime/reference/strided_slice.hpp>
#include "ngraph/slice_plan.hpp"
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
static ngraph::AxisSet convert_mask_to_axis_set(const std::vector<int64_t>& mask) {
    ngraph::AxisSet axis_set{};
    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i] == 1) {
            axis_set.emplace(i);
        }
    }
    return axis_set;
}

template <typename T, typename U>
void wrap_strided_slice(const T* arg,
                        const U* begin,
                        const U* end,
                        const U* strides,
                        T* out,
                        const ngraph::Shape& arg_shape,
                        const ngraph::Shape& begin_shape,
                        const ngraph::Shape& end_shape,
                        const ngraph::Shape& strides_shape,
                        const std::vector<int64_t>& begin_mask,
                        const std::vector<int64_t>& end_mask,
                        const std::vector<int64_t>& new_axis_mask,
                        const std::vector<int64_t>& shrink_axis_mask,
                        const std::vector<int64_t>& ellipsis_mask,
                        size_t elem_size) {
    std::vector<int64_t> begin_const(begin, begin + ngraph::shape_size(begin_shape));
    std::vector<int64_t> end_const(end, end + ngraph::shape_size(end_shape));
    std::vector<int64_t> stride_const(strides, strides + ngraph::shape_size(strides_shape));
    ngraph::SlicePlan slice_plan = ngraph::make_slice_plan(arg_shape,
                                                           begin_const,
                                                           end_const,
                                                           stride_const,
                                                           convert_mask_to_axis_set(begin_mask),
                                                           convert_mask_to_axis_set(end_mask),
                                                           convert_mask_to_axis_set(new_axis_mask),
                                                           convert_mask_to_axis_set(shrink_axis_mask),
                                                           convert_mask_to_axis_set(ellipsis_mask));

    ngraph::runtime::reference::strided_slice(reinterpret_cast<const char*>(arg),
                                              reinterpret_cast<char*>(out),
                                              arg_shape,
                                              slice_plan,
                                              elem_size);
}

template <> Converter::Conversion::Ptr Converter::Convert(const opset::StridedSlice& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.input(1),
                                    node.input(2),
                                    node.input(3),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_input_shape(1),
                                    node.get_input_shape(2),
                                    node.get_input_shape(3),
                                    node.get_begin_mask(),
                                    node.get_end_mask(),
                                    node.get_new_axis_mask(),
                                    node.get_shrink_axis_mask(),
                                    node.get_ellipsis_mask(),
                                    node.get_element_type().size());
    };

    return CallSwitch(
        AP_WRAP(make, wrap_strided_slice),
        node.input(0), allTypes,
        node.input(1), indexTypes);
}

template <> Converter::Conversion::Ptr Converter::Convert(const opset::ArmStridedSlice& node) {
    auto begin  = safe_cast<ngraph::op::Constant>(
                        node.input_value(1).get_node_shared_ptr())->cast_vector<int>();
    auto end    = safe_cast<ngraph::op::Constant>(
                        node.input_value(2).get_node_shared_ptr())->cast_vector<int>();
    auto stride = safe_cast<ngraph::op::Constant>(
                        node.input_value(3).get_node_shared_ptr())->cast_vector<int>();

    arm_compute::Coordinates starts, finishes, deltas;
    for (size_t i = 0; i < begin.size(); ++i) {
        auto axis = AxisCast(i, begin.size());
        starts.set(axis, begin[i]);
        finishes.set(axis, end[i]);
        deltas.set(axis, stride[i]);
    }

    return MakeConversion<arm_compute::NEStridedSlice>(node.input(0), node.output(0), starts, finishes, deltas);
}
}  //  namespace ArmPlugin