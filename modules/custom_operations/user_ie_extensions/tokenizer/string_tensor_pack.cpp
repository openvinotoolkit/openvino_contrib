// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "string_tensor_pack.hpp"
#include "utils.hpp"

using namespace ov;


void StringTensorPack::validate_and_infer_types() {
    OPENVINO_ASSERT(m_mode == "begins_ends", "StringTensorPack supports only 'begins_ends' mode, but get " + m_mode);
    check_string_input(this, 0);
    #if USE_STRING_TENSORS
    set_output_type(0, element::string, get_input_partial_shape(0));
    #else
    set_output_type(0, element::u8, PartialShape{Dimension()});
    #endif
}

bool StringTensorPack::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
#if USE_STRING_TENSORS
    // TODO
    return false;
#else
    auto rank = inputs[0].get_shape().size();
    if (rank != 1) {
        std::cerr << "[ WARNING ] StringTensorPack ignores the rank " << rank << " of input tensor and set rank=1 in the output\n";
    }

    auto num_elements = shape_size(inputs[0].get_shape());
    auto num_chars = shape_size(inputs[2].get_shape());
    auto num_output_elements = 4*(1 + 1 + num_elements) + num_chars;
    outputs[0].set_shape(Shape{num_output_elements});

    // FIXME: Do the repacking, otherwise cannot handle string tensors with gaps between strings
    //auto begins = inputs[0].data<const int32_t>();    // this is not needed as no repacking happens in this version of code
    auto ends   = inputs[1].data<const int32_t>();
    auto chars  = inputs[2].data<const uint8_t>();

    auto output = outputs[0].data<uint8_t>();
    auto output_int32 = reinterpret_cast<int32_t*>(output);

    *output_int32++ = num_elements;
    *output_int32++ = 0;
    output_int32 = std::copy(ends, ends + num_elements, output_int32);
    output = reinterpret_cast<uint8_t*>(output_int32);
    output = std::copy(chars, chars + num_chars, output);

    OPENVINO_ASSERT(num_output_elements == output - outputs[0].data<uint8_t>(), "[ INTERNAL ERROR ] StringTensorPack output tensor is corrupted");

    // WARNING! Chars are not repacked. If there are gaps between strings, they will remain.

    return true;
#endif
}
