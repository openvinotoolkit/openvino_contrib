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

    // In case of explicit string tensors the shape is carried by input tensor itself
//     OPENVINO_ASSERT(
//         input_shape == PartialShape::dynamic(),
//         "Excplicitly set shape for a string tensor in the unpacking is not supported");

    // There are three cases that affect expected element type of the input tensor:
    // - when string tensor is passed and we are before the hack is applied (element::string) and
    // - when string tensor is passed and we are after the hack in CPU (element::u8) and
    // - when stirng tensor is not really used, and we expect a packed string tensor in this case (element::u8)

    OPENVINO_ASSERT(
#if OPENVINO_ELEMENT_STRING_SUPPORTED
        get_input_element_type(0) == element::string ||
#endif
#if OPENVINO_USE_INPUT_OUTPUT_STRING_TENSOR_HACK || !USE_STRING_TENSORS
        get_input_element_type(0) == element::u8 ||
#endif
        get_input_element_type(0) == element::dynamic,
        "Type of StringTensorUnpack input is expected to be element::string before a model compilation or element::u8 after the compilation or when element::string is not supported");

#if OPENVINO_ELEMENT_STRING_SUPPORTED
    if(get_input_element_type(0) == element::string) {
        output_shape = get_input_partial_shape(0);
    }
#endif

#if OPENVINO_USE_INPUT_OUTPUT_STRING_TENSOR_HACK || !USE_STRING_TENSORS
    if(get_input_element_type(0) == element::u8)
    {
        #if OPENVINO_USE_INPUT_OUTPUT_STRING_TENSOR_HACK
        // After the plugin hack, a tensor is represented as a wrapping u8 tensor that will hold a pointer to a string tensor.
        // The original shape of a string tensor is stored in RT attribute of a tensor descriptor.
        const auto& rt_info = get_input_tensor(0).get_rt_info();
        auto it = rt_info.find("__original_partial_shape");

        // StringTensorUnpack expects __original_partial_shape attribute of type PartialShape in the input tensor.
        // If it is not found that means that model compilation wasn't pass the expected transformation where a string tensor
        // is wrapped to a u8 tensor holding a pointer, or because evaluation of this node is in progress and tensor attributes aren't preserved.
        if(it != rt_info.end() && it->second.is<PartialShape>()) {
            output_shape = it->second.as<PartialShape>();
        } else {
        #endif
            #if !USE_STRING_TENSORS
            // If string tensors shouldn't be used, then the packed u8 format is also expected
            // as an input, but in this case only rank is known
                OPENVINO_ASSERT(
                    get_input_partial_shape(0).rank().is_dynamic() || get_input_partial_shape(0).rank().get_length() == 1,
                    "StringTensorUnpack expects a u8 tensor with rank 1 that holds packed batched string tensor as an input, but observes type " +
                        get_input_element_type(0).get_type_name() + " and shape " + get_input_partial_shape(0).to_string());

            output_shape = PartialShape({Dimension()});  // [?]
            #endif
        #if OPENVINO_USE_INPUT_OUTPUT_STRING_TENSOR_HACK
        }
        #endif
    }
#endif

    OPENVINO_ASSERT(m_mode == "begins_ends", "StringTensorUnpack supporst only 'begins_ends' mode, but get " + m_mode);

    if (m_mode == "begins_ends") {
        set_string_output(this, 0, output_shape);
    }
}

bool StringTensorUnpack::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto ptensor = &inputs[0];
    #if OPENVINO_USE_INPUT_OUTPUT_STRING_TENSOR_HACK
    if(ptensor->get_element_type() == element::u8 && ptensor->get_byte_size() == sizeof(void*)) {
        auto data = *reinterpret_cast<const void* const*>(ptensor->data());
        if(data != nullptr) {
            ptensor = reinterpret_cast<const ov::Tensor*>(data);
        }
    }
    #endif

    auto tensor = *ptensor;

#if OPENVINO_ELEMENT_STRING_SUPPORTED
    if(tensor.get_element_type() == element::string) {
        Shape input_shape = tensor.get_shape();
        const std::string* input_strings = tensor.data<std::string>();
        unpack_strings_to_tensors(input_strings, input_shape, outputs[0], outputs[1], outputs[2]);
        return true;
    } else {
#endif

#if USE_STRING_TENSORS
    OPENVINO_ASSERT(false, "Detected a u8 tensor but element::string tensor should be provided");
#endif

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

#if OPENVINO_ELEMENT_STRING_SUPPORTED
    }
#endif
}
