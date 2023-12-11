// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/framework_node.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"
#include "string_tensor_pack.hpp"
#include "string_tensor_unpack.hpp"
#include "ragged_tensor_pack.hpp"

using namespace ov;
using namespace ov::frontend;
using namespace ov::opset10;

void parse_packed_strings (const Tensor& packed, int32_t& batch_size, const int32_t*& begin_ids, const int32_t*& end_ids, const uint8_t*& symbols) {
    auto strings = packed.data<const uint8_t>();
    auto bitstream_size = packed.get_byte_size();
    // check the format of the input bitstream representing the string tensor
    FRONT_END_GENERAL_CHECK(bitstream_size >= 4, "Incorrect packed string tensor format: no batch size in the packed string tensor");
    batch_size = *reinterpret_cast<const int32_t*>(strings + 0);
    FRONT_END_GENERAL_CHECK(bitstream_size >= 4 + 4 + 4 * batch_size,
        "Incorrect packed string tensor format: the packed string tensor must contain first string offset and end indices");
    begin_ids = reinterpret_cast<const int32_t*>(strings + 4);
    end_ids = begin_ids + 1;
    symbols = strings + 4 + 4 + 4 * batch_size;
}

void check_string_input(const Node* node, size_t input_index) {
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+0) == element::i32, "Expected an i32 tensor as the first part of the decomposed string representation");
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+1) == element::i32, "Expected an i32 tensor as the second part of the decomposed string representation");
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+2) == element::u8,  "Expected a u8 tensor as the third part of the decomposed string representation");
}

void check_string_scalar_input(const Node* node, size_t input_index) {
    auto shape = node->get_input_partial_shape(input_index);
    auto element_type = node->get_input_element_type(input_index);

    #if false && USE_STRING_TENSORS
    // This block is not used when we convert ops to decomposed representation (and we really do)

    OPENVINO_ASSERT(
        (element_type == element::dynamic || element_type == element::string) &&
        (shape.rank().is_dynamic() || shape.rank().get_length() == 0),
        "string/0D tensor is expected, but observed: " + element_type.get_type_name() + shape.to_string());

    #else

    OPENVINO_ASSERT(
        (element_type == element::dynamic || element_type == element::u8) &&
        (shape.rank().is_dynamic() || shape.rank().get_length() == 1),
        "u8/1D tensor is expected");

    #endif
}

void check_ragged_input(const Node* node, size_t input_index) {
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+0) == element::i32, "Expected an i32 tensor as the first part of the decomposed ragged representation");
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+1) == element::i32, "Expected an i32 tensor as the second part of the decomposed ragged representation");
    auto rank = node->get_input_partial_shape(input_index+2).rank();
    FRONT_END_GENERAL_CHECK(rank.is_dynamic() || rank.get_length() == 1, "The last tensor in ragged tensor representation should be a 1D tensor");
}

void check_ragged_string_input(const Node* node, size_t input_index) {
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+0) == element::i32, "Expected an i32 tensor as the first part of the decomposed ragged string representation");
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+1) == element::i32, "Expected an i32 tensor as the second part of the decomposed ragged string representation");
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+2) == element::i32, "Expected an i32 tensor as the third part of the decomposed ragged string representation");
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+3) == element::i32, "Expected an i32 tensor as the forth part of the decomposed ragged string representation");
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+4) == element::u8,  "Expected a u8 tensor as the fifth part of the decomposed ragged string representation");
}

void set_string_output(Node* node, size_t output_index, const PartialShape& shape) {
    node->set_output_type(output_index+0, element::i32, shape);     // byte offset in output[+2] -- begin of each string
    node->set_output_type(output_index+1, element::i32, shape);     // byte offset in output[+2] -- end of each string
    node->set_output_type(output_index+2, element::u8,  PartialShape{Dimension()});     // symbols from all strings concatenated
}

void set_ragged_string_output(Node* node, size_t output_index, const PartialShape& shape) {
    node->set_output_type(output_index+0, element::i32, shape);     // element offset in output[+2] -- begin of each ragged dimension elements
    node->set_output_type(output_index+1, element::i32, shape);     // element offset in output[+3] -- end of each ragged dimension elements
    node->set_output_type(output_index+2, element::i32, PartialShape{Dimension()}); // byte offset in output[+4] -- begin of each string
    node->set_output_type(output_index+3, element::i32, PartialShape{Dimension()}); // byte offset in output[+4] -- end of each string
    node->set_output_type(output_index+4, element::u8,  PartialShape{Dimension()}); // symbols from all strings cnocatenated
}

void set_ragged_output(Node* node, size_t output_index, const PartialShape& shape, element::Type type) {
    node->set_output_type(output_index+0, element::i32, shape);     // element offset in output[+2] -- begin of each ragged dimension elements
    node->set_output_type(output_index+1, element::i32, shape);     // element offset in output[+2] -- end of each ragged dimension elements
    node->set_output_type(output_index+2, type, PartialShape{Dimension()}); // flatten elements
}


void unpack_strings_to_tensors (const std::string* strings, const Shape shape, ov::Tensor& begins, ov::Tensor& ends, ov::Tensor& chars) {
    auto nelements = shape_size(shape);

    size_t total = 0;
    for(size_t i = 0; i < nelements; ++i)
        total += strings[i].length();

    begins.set_shape(shape);
    ends.set_shape(shape);
    chars.set_shape(Shape{total});

    auto pbegins = begins.data<int32_t>();
    auto pends = ends.data<int32_t>();
    auto poutput_symbols = reinterpret_cast<char*>(chars.data<uint8_t>());
    size_t offset = 0;

    for(size_t i = 0; i < nelements; ++i)
    {
        pbegins[i] = offset;
        poutput_symbols = std::copy(strings[i].begin(), strings[i].end(), poutput_symbols);
        offset += strings[i].length();
        pends[i] = offset;
    }
}

void override_parameter (std::shared_ptr<ov::Node> node, element::Type type, const PartialShape& shape) {
    if (auto parameter = std::dynamic_pointer_cast<Parameter>(node)) {
        // TODO: Apply this change conditionally based on real Parameter value
        std::cerr << "Overriding Parameter element_type to " << type << " and shape " << shape << "\n";
        parameter->set_partial_shape(shape);
        parameter->set_element_type(type);
        parameter->validate_and_infer_types();
    }
}

// TODO: replace NodeContext and input_index by a single input
OutputVector pre_translate_string_tensor_input(ov::Output<ov::Node> input) {
    auto input_node = input.get_node_shared_ptr();

#if !USE_STRING_TENSORS
    override_parameter(input_node, element::u8, PartialShape{Dimension()});
#endif

    if (auto struct_pack = std::dynamic_pointer_cast<StringTensorPack>(input_node)) {
        FRONT_END_GENERAL_CHECK(struct_pack->get_input_size() == 3, "Expected 3 inputs to StringTensorPack which represents a string tensor");
        return struct_pack->input_values();
    } else {
        #if USE_STRING_TENSORS || true     // always
        return std::make_shared<StringTensorUnpack>(OutputVector{input}, "begins_ends")->outputs();
        #else
        // Suppose this is u8 packed string tensor with a single batch dimension
        // Unpack this tensor using standard operations

        // Cannot do that because there is not ReinterprectCast operation in OV
        // TODO: Find a way to make it without reinterpretation operation or introduce it as an extension (easy)
        #endif
    }
}

OutputVector pre_translate_ragged_tensor_input(ov::Output<ov::Node> input) {
    auto ragged_pack = dynamic_cast<RaggedTensorPack*>(input.get_node());
    OPENVINO_ASSERT(ragged_pack, "Expected RaggedTensorPack but didn't find it");
    return ragged_pack->input_values();
}

OutputVector pre_translate_ragged_string_tensor_input(ov::Output<ov::Node> input) {
    auto ragged_inputs = pre_translate_ragged_tensor_input(input);
    auto string_inputs = pre_translate_string_tensor_input(ragged_inputs[2]);
    ragged_inputs.pop_back();
    ragged_inputs.insert(ragged_inputs.end(), string_inputs.begin(), string_inputs.end());
    return ragged_inputs;
}

ov::Output<ov::Node> post_translate_string_tensor_output(const OutputVector& outputs) {
    FRONT_END_GENERAL_CHECK(outputs.size() == 3, "Expected 3 tensors in decomposed string tensor representation");
    return std::make_shared<StringTensorPack>(outputs, "begins_ends");
}

ov::Output<ov::Node> post_translate_ragged_tensor_output(const OutputVector& outputs) {
    FRONT_END_GENERAL_CHECK(outputs.size() == 3, "Expected 3 tensors in decomposed string tensor representation");
    return std::make_shared<RaggedTensorPack>(outputs);
}

bool evaluate_normalization_helper (ov::TensorVector& outputs, const ov::TensorVector& inputs, std::function<std::string(const std::string&)> normalizer) {
    auto begins = inputs[0].data<const int32_t>();
    auto ends   = inputs[1].data<const int32_t>();
    auto chars  = inputs[2].data<const uint8_t>();

    // Set output shapes
    outputs[0].set_shape(inputs[0].get_shape());
    outputs[1].set_shape(inputs[1].get_shape());
    const size_t num_elements = inputs[0].get_size();

    // TODO: How to avoid copying from this temporary buffer?
    // TODO: It can be possible to collect output symbols directly in the output tensor memory if `normalizer` has reasonable estimation for the final size.
    std::deque<uint8_t> buffer;

    // For the whole implementation below the input shapes can be ignored, we are working with the flatten representaions
    // and only number of elements in the original tensors matter

    // Get pointers in the output tensors
    auto new_begins = outputs[0].data<int32_t>();
    auto new_ends   = outputs[1].data<int32_t>();

    for(size_t i = 0; i < num_elements; ++i) {
        new_begins[i] = buffer.size();
        std::string new_str = normalizer(std::string(chars + begins[i], chars + ends[i]));
        buffer.insert(buffer.end(), new_str.begin(), new_str.end());
        new_ends[i] = buffer.size();
    }

    // Copy collected symbols to the target output tensor

    outputs[2].set_shape(Shape{buffer.size()});
    auto new_chars  = outputs[2].data<uint8_t>();
    std::copy(buffer.begin(), buffer.end(), new_chars);

    return true;
}

std::shared_ptr<Node> string_attribute_to_constant (const ov::frontend::NodeContext& node, const std::string& name) {
    auto value = node.get_attribute<std::string>(name);

    // TODO: How to translate attribute `replace_global`?

    #if USE_STRING_TENSORS
    return std::make_shared<Constant>(element::string, Shape{}, &value);
    #else
    return std::make_shared<Constant>(element::u8, Shape{value.length()}, (const void*)value.data());
    #endif
}
