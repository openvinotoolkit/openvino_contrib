// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "string_tensor_unpack.hpp"
#include "utils.hpp"

using namespace ov;


void StringTensorUnpack::validate_and_infer_types() {
    OPENVINO_ASSERT(
        get_input_size() == 1,
        "Number of inputs for StringTensorUnpack is not equal to 1");

    auto output_shape = PartialShape::dynamic();
    auto input_element_type = get_input_element_type(0);
    auto input_partial_shape = get_input_partial_shape(0);

    if(input_element_type == element::string) {
        output_shape = input_partial_shape;
    } else if (input_element_type == element::u8) {
        // Legacy u8 packed format
        OPENVINO_ASSERT(
            input_partial_shape.rank().is_dynamic() || input_partial_shape.rank().get_length() == 1,
            "StringTensorUnpack expects a string tensor or a u8 tensor with rank 1 that holds "
            "packed batched string tensor as an input, but observes type " +
                input_element_type.get_type_name() + " and shape " + input_partial_shape.to_string());

        output_shape = PartialShape({Dimension()});  // [?]
    } else if (input_element_type != element::dynamic) {
        OPENVINO_THROW(
            "StringTensorUnpack expects a tensor with string or u8 elements, got a tensor with " +
            input_element_type.get_type_name() + " elements");
    }

    if (m_mode == "begins_ends") {
        set_string_output(this, 0, output_shape);
    } else {
        OPENVINO_ASSERT(m_mode == "begins_ends", "StringTensorUnpack supporst only 'begins_ends' mode, but get " + m_mode);
    }
}


bool StringTensorUnpack::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto tensor = inputs[0];
    auto input_element_type = tensor.get_element_type();

    if(input_element_type == element::string) {
        Shape input_shape = tensor.get_shape();
        const std::string* input_strings = tensor.data<std::string>();
        unpack_strings_to_tensors(input_strings, input_shape, outputs[0], outputs[1], outputs[2]);
        return true;
    } else if(input_element_type == element::u8) {
        int32_t batch_size;
        const int32_t* begin_ids;
        const int32_t* end_ids;
        const uint8_t* data;
        parse_packed_strings(tensor, batch_size, begin_ids, end_ids, data);
        auto num_chars = end_ids[batch_size - 1];

        outputs[0].set_shape(Shape{static_cast<unsigned long>(batch_size)});
        outputs[1].set_shape(Shape{static_cast<unsigned long>(batch_size)});
        outputs[2].set_shape(Shape{static_cast<unsigned long>(num_chars)});
        auto begins = outputs[0].data<int32_t>();
        auto ends = outputs[1].data<int32_t>();
        auto chars = outputs[2].data<uint8_t>();
        std::copy(begin_ids, begin_ids + batch_size, begins);
        std::copy(end_ids, end_ids + batch_size, ends);
        std::copy(data, data + num_chars, chars);

        return true;
    } else {
        OPENVINO_THROW(
            "StringTensorUnpack::evaluate expects a tensor with string or u8 elements, got a tensor with " +
            input_element_type.get_type_name() + " elements");
    }
}
