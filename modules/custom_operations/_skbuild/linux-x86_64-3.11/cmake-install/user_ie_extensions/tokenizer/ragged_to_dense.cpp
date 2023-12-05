// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/constant.hpp>

#include "ragged_to_dense.hpp"
#include "utils.hpp"

using namespace ov;
using op::v0::Constant;

void RaggedToDense::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() == 3 + 1 + 1);

    // Input ragged tensor
    check_ragged_input(this, 0);

    // Target size along ragged dimension
    OPENVINO_ASSERT(get_input_element_type(3).is_integral_number());
    auto rank = get_input_partial_shape(3).rank();
    OPENVINO_ASSERT(
        rank.is_dynamic() ||
        rank.get_length() == 0 ||
        rank.get_length() == 1 && get_input_partial_shape(3)[0].compatible(1),
        "Target dense dimension size for RaggedToDense should be a 0D or 1D tensor with a single element");

    // Default value to fill out of ragged range elements in output tensor
    OPENVINO_ASSERT(get_input_element_type(4).compatible(get_input_element_type(2)));
    auto input4_rank = get_input_partial_shape(4).rank();
    OPENVINO_ASSERT(input4_rank.compatible(0));

    set_input_is_relevant_to_shape(3);

    if(get_input_partial_shape(0).rank().is_dynamic()) {
        set_output_type(0, get_input_element_type(2), PartialShape::dynamic());
        set_output_type(1, element::boolean, PartialShape::dynamic());
    } else {
        auto shape = get_input_partial_shape(0);
        if(auto target_dim = dynamic_cast<Constant*>(get_input_node_ptr(3))) {
            shape.push_back(target_dim->cast_vector<int64_t>()[0]);
        } else {
            shape.push_back(Dimension());
        }
            set_output_type(0, get_input_element_type(2), shape);
            set_output_type(1, element::boolean, shape);
    }
}


bool RaggedToDense::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    // FIXME: Works for POD types only (not for strings!)
    // FIXME: Output mask is calculated even if there are no consumers
    auto begins = inputs[0].data<const int32_t>();
    auto ends   = inputs[1].data<const int32_t>();
    auto nelems = inputs[0].get_size();
    auto elems  = reinterpret_cast<const char*>(inputs[2].data());
    auto elem_size = inputs[2].get_element_type().size();
    auto default_value = reinterpret_cast<const char*>(inputs[4].data());

    // Suppose validate was called and set correct output shape
    // Take a target shape value for ragged dimension
    size_t target_dim = outputs[0].get_shape().back();

    auto out_elems = reinterpret_cast<char*>(outputs[0].data());
    auto out_mask = outputs[1].data<char>();

    auto out_elem_orig = out_elems;
    auto out_mask_orig = out_mask;

    for(size_t i = 0; i < nelems; ++i) {
        auto begin = elems + elem_size*begins[i];
        auto len = std::min(size_t(ends[i] - begins[i]), target_dim);  // truncation
        auto end = begin + elem_size*len;
        out_elems = std::copy(begin, end, out_elems);
        out_mask = std::fill_n(out_mask, len, char(1));
        if(len < target_dim)
            out_mask = std::fill_n(out_mask, target_dim - len, char(0));
        while(len < target_dim) {
            out_elems = std::copy(default_value, default_value + elem_size, out_elems);
            ++len;
        }
    }

    OPENVINO_ASSERT(out_elems == out_elem_orig + outputs[0].get_byte_size());
    OPENVINO_ASSERT(out_mask == out_mask_orig + outputs[1].get_byte_size());
    return true;
}
