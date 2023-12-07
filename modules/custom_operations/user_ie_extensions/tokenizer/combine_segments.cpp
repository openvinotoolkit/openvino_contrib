// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "combine_segments.hpp"
#include "utils.hpp"

using namespace ov;

void CombineSegments::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() > 0);
    OPENVINO_ASSERT((get_input_size() - 1)%3 == 0);

    // First come several ragged tensors each represented as 3 regular tesors
    size_t num_inputs = (get_input_size() - 1)/3;
    PartialShape ps = PartialShape::dynamic();
    element::Type et = element::dynamic;
    for (size_t i = 0; i < num_inputs; ++i) {
        check_ragged_input(this, 3*i);
        // Check limited broadcast
        // Limited means that we support only two shapes on inputs: scalar and not scalars,
        // and all not-scalars should have the same shape
        auto rank = get_input_partial_shape(3*i).rank();
        if(rank.is_static() && rank.get_length()) {
            OPENVINO_ASSERT(ps.merge_into(ps, get_input_partial_shape(3*i)));
        }
        OPENVINO_ASSERT(element::Type::merge(et, et, get_input_element_type(3*i)));
        OPENVINO_ASSERT(element::Type::merge(et, et, get_input_element_type(3*i + 1)));
    }

    set_ragged_output(this, 0, ps, et);
    // TODO: Avoid emitting ragged indices for the second ragged tensor, they should be identical to the first output ragged tensor
    set_ragged_output(this, 3, ps, get_input_element_type(get_input_size() - 1));
}

bool CombineSegments::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    // FIXME: Works for POD types only (not for strings!)
    size_t num_of_ragged = (inputs.size() - 1)/3;
    OPENVINO_ASSERT(num_of_ragged == inputs.back().get_size());
    std::vector<const int32_t*> begins;
    std::vector<const int32_t*> ends;
    std::vector<size_t> nelems;
    std::vector<const char*> elems;
    auto element_type = inputs[2].get_element_type();
    auto elem_size = element_type.size();
    size_t max_nelems = 0;
    size_t flat_out_size = 0;
    Shape ps;

    for(size_t i = 0; i < num_of_ragged; ++i) {
        OPENVINO_ASSERT(inputs[3*i + 2].get_element_type() == element_type);
        begins.push_back(inputs[3*i + 0].data<const int32_t>());
        ends.push_back(inputs[3*i + 1].data<const int32_t>());
        nelems.push_back(inputs[3*i + 0].get_size());
        elems.push_back(reinterpret_cast<const char*>(inputs[3*i + 2].data()));
        // TODO: Get rank from a tensor instead of partial_shape. This is a WA for CPU bug that gives 1D tensors instead of 0D tensors.
        if(get_input_partial_shape(3*i + 0).rank().get_length() > 0) {
            ps = inputs[3*i + 0].get_shape();
        }
        max_nelems = std::max(max_nelems, nelems.back());
    }

    // flat_out_size is going to be an estimation of the final size
    // This is only an estimation, not the exact output size, because ragged tensor may have gaps in the representation

    for(size_t i = 0; i < num_of_ragged; ++i) {
        if(nelems[i] == 1) {
            flat_out_size += max_nelems * inputs[3*i + 2].get_size(); // broadcast
        } else {
            flat_out_size += inputs[3*i + 2].get_size();    // FIXME: doesn't work for overlapped ragged regions
        }
    }

    auto ids = reinterpret_cast<const char*>(inputs.back().data());
    size_t id_type_size = inputs.back().get_element_type().size();

    outputs[3*0 + 0].set_shape(ps);
    outputs[3*0 + 1].set_shape(ps);
    OPENVINO_ASSERT(max_nelems == outputs[3*0 + 0].get_size());
    OPENVINO_ASSERT(max_nelems == outputs[3*0 + 1].get_size());
    outputs[3*0 + 2].set_shape({flat_out_size});

    outputs[3*1 + 0].set_shape(ps);
    outputs[3*1 + 1].set_shape(ps);
    OPENVINO_ASSERT(max_nelems == outputs[3*1 + 0].get_size());
    OPENVINO_ASSERT(max_nelems == outputs[3*1 + 1].get_size());
    outputs[3*1 + 2].set_shape({flat_out_size});

    auto out_elem_begins = outputs[3*0 + 0].data<int32_t>();
    auto out_elem_ends = outputs[3*0 + 1].data<int32_t>();
    auto out_elems = reinterpret_cast<char*>(outputs[3*0 + 2].data());
    auto out_id_begins = outputs[3*1 + 0].data<int32_t>();
    auto out_id_ends = outputs[3*1 + 1].data<int32_t>();
    auto out_ids = reinterpret_cast<char*>(outputs[3*1 + 2].data());

    auto out_elems_orig = out_elems;
    auto out_ids_orig = out_ids;
    size_t out_offset = 0;

    for(size_t i = 0; i < max_nelems; ++i) {
        out_elem_begins[i] = out_offset;
        out_id_begins[i] = out_offset;

        for(size_t j = 0; j < num_of_ragged; ++j) {
            const char* begin;
            size_t len;
            if(nelems[j] == 1) {
                begin = elems[j] + elem_size*begins[j][0];
                len = ends[j][0] - begins[j][0];
            } else {
                begin = elems[j] + elem_size*begins[j][i];
                len = ends[j][i] - begins[j][i];
            }
            auto end = begin + elem_size*len;
            out_elems = std::copy(begin, end, out_elems);
            for(size_t k = 0; k < len; ++k) {
                out_ids = std::copy(ids + id_type_size*j, ids + id_type_size*(j + 1), out_ids);
            }
            out_offset += len;
        }

        out_elem_ends[i] = out_offset;
        out_id_ends[i] = out_offset;
    }

    OPENVINO_ASSERT(out_offset <= flat_out_size);

    outputs[3*0 + 2].set_shape({out_offset});
    outputs[3*1 + 2].set_shape({out_offset});

    OPENVINO_ASSERT(out_elems == out_elems_orig + outputs[3*0 + 2].get_byte_size());
    OPENVINO_ASSERT(out_ids == out_ids_orig + outputs[3*1 + 2].get_byte_size());
    return true;
}

