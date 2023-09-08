// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ragged_tensor_pack.hpp"
#include "utils.hpp"

using namespace ov;


void RaggedTensorPack::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() == 3);
    OPENVINO_ASSERT(get_input_element_type(0) == element::i32);
    OPENVINO_ASSERT(get_input_element_type(1) == element::i32);

    // Pass through the base tensor which is used to build ragged dimensions
    // TODO: Provide correct implementation that saves information about ragged structure
    // TODO: Requires single-tensor packed representation for ragged tensor
    set_output_type(0, get_input_element_type(2), get_input_partial_shape(2));
}


bool RaggedTensorPack::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto input_shape = inputs[0].get_shape();
    auto begins = inputs[0].data<const int32_t>();
    auto ends   = inputs[1].data<const int32_t>();
    auto num_elements = shape_size(input_shape);

    inputs[2].copy_to(outputs[0]);

    return true;
}
