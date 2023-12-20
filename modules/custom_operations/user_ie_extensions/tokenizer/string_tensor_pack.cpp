// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "string_tensor_pack.hpp"
#include "utils.hpp"

using namespace ov;


void StringTensorPack::validate_and_infer_types() {
    OPENVINO_ASSERT(m_mode == "begins_ends", "StringTensorPack supports only 'begins_ends' mode, but get " + m_mode);
    check_string_input(this, 0);
    set_output_type(0, element::string, get_input_partial_shape(0));
}

bool StringTensorPack::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto num_strings = outputs[0].get_size();
    OPENVINO_ASSERT(inputs[0].get_size() == num_strings);
    OPENVINO_ASSERT(inputs[1].get_size() == num_strings);

    auto begins = inputs[0].data<const int32_t>();
    auto ends   = inputs[1].data<const int32_t>();
    auto chars  = reinterpret_cast<const char*>(inputs[2].data<const uint8_t>());

    auto strings = outputs[0].data<std::string>();

    for(size_t i = 0; i < num_strings; ++i) {
        strings[i].assign(chars + begins[i], chars + ends[i]);
    }

    return true;
}
