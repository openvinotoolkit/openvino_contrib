// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "normalizer.h"
#include "sentence_piece.hpp"

#include "openvino/op/util/framework_node.hpp"
#include "openvino/opsets/opset10.hpp"

#include "fast_tokenizer/normalizers/normalizers.h"
#include "fast_tokenizer/models/models.h"
#include "fast_tokenizer/pretokenizers/pretokenizers.h"

// TODO: Replace shape_size(t.get_shape()) by t.get_size(), where t is ov::Tensor

#ifndef OPENVINO_ELEMENT_STRING_SUPPORTED
    #define OPENVINO_ELEMENT_STRING_SUPPORTED 0
#endif

#ifndef OPENVINO_USE_INPUT_OUTPUT_STRING_TENSOR_HACK
    #define OPENVINO_USE_INPUT_OUTPUT_STRING_TENSOR_HACK 0
#endif

#define USE_STRING_TENSORS 0    // modify this depending on willingness to use explicit string tensors

#if USE_STRING_TENSORS && !OPENVINO_ELEMENT_STRING_SUPPORTED
    #error "USE_STRING_TENSORS = 1 can be used only when OpenVINO supports element::string that is determined by OPENVINO_ELEMENT_STRING_SUPPORTED == 1"
#endif

#define SENTENCE_PIECE_EXTENSION_DECOMPOSED_STRINGS 0

using sentencepiece::SentencePieceProcessor;
using namespace TemplateExtension;
using namespace ov;
using namespace ov::frontend;
using namespace ov::opset10;

namespace {
    template<typename T>
    T extract_scalar_const_value(const std::shared_ptr<Node>& node, const std::string& const_name) {
        auto const_node = as_type_ptr<Constant>(node);
        FRONT_END_GENERAL_CHECK(const_node, "Conversion expects " + const_name + " to be constant.");
        std::vector<T> const_value = const_node->cast_vector<T>();
        FRONT_END_GENERAL_CHECK(const_value.size() == 1, "Conversion expects " + const_name + " to be a scalar.");
        return const_value[0];
    }
}  // namespace

SentencepieceTokenizer::SentencepieceTokenizer(const OutputVector& args, int32_t nbest_size, float alpha,
    bool add_bos, bool add_eos, bool reverse) : m_sp(std::make_shared<SentencePieceProcessor>()),
    m_nbest_size(nbest_size), m_alpha(alpha), m_add_bos(add_bos), m_add_eos(add_eos),
    m_reverse(reverse), Op(args) {
    auto sp_model_const = as_type_ptr<Constant>(args[0].get_node_shared_ptr());
    FRONT_END_GENERAL_CHECK(sp_model_const, "SentencepieceTokenizer expects SentencePiece model to be constant.");
    auto spm_model = static_cast<const char*>(sp_model_const->get_data_ptr());
    auto spm_model_size = sp_model_const->get_byte_size();

    // configure SentencePieceProcessor
    std::string model_proto(spm_model, spm_model_size);
    CHECK_OK(m_sp->LoadFromSerializedProto(model_proto));

    // form extra options to configure SentencePieceProcessor
    std::string extra_options = "";
    if (m_add_bos) {
        extra_options += "bos";
    }
    if (m_add_eos) {
        extra_options = extra_options.empty() ? extra_options : extra_options + ":";
        extra_options += "eos";
    }
    /* TODO: TF ignores this option, so we are ignoring it as well; need to understand what should we do
    if (m_reverse) {
        extra_options = extra_options.empty() ? extra_options : extra_options + ":";
        extra_options += "reverse";
    }
    */
    // example of extra_options, if "bos:eos:reverse"
    CHECK_OK(m_sp->SetEncodeExtraOptions(extra_options));
    constructor_validate_and_infer_types();
}

SentencepieceTokenizer::SentencepieceTokenizer(const OutputVector& args, const std::shared_ptr<sentencepiece::SentencePieceProcessor>& sp,
    int32_t nbest_size, float alpha, bool add_bos, bool add_eos, bool reverse) : m_sp(sp),
    m_nbest_size(nbest_size), m_alpha(alpha), m_add_bos(add_bos), m_add_eos(add_eos),
    m_reverse(reverse), Op(args) {
    constructor_validate_and_infer_types();
}

void SentencepieceTokenizer::validate_and_infer_types() {

    #if SENTENCE_PIECE_EXTENSION_DECOMPOSED_STRINGS

    FRONT_END_GENERAL_CHECK(get_input_size() == 1 + 3, "SentencepieceTokenizer expects 4 inputs: sp model and input sentences represented as 3 decomposed tensors (begins, ends, sybols)");
    FRONT_END_GENERAL_CHECK(get_input_element_type(0) == element::u8, "SentencepieceTokenizer accepts sp model as the first input and it should be of type u8 tensor");
    FRONT_END_GENERAL_CHECK(get_input_element_type(1) == element::i32, "SentencepieceTokenizer accepts begins offsets as the second and it should be of type i32 tensor");
    FRONT_END_GENERAL_CHECK(get_input_element_type(2) == element::i32, "SentencepieceTokenizer accepts ends offsets as the third and it should be of type i32 tensor");
    FRONT_END_GENERAL_CHECK(get_input_element_type(3) == element::u8, "SentencepieceTokenizer accepts sentence symbols as the fourth input and it should be of type u8 tensor");

    #else

    FRONT_END_GENERAL_CHECK(get_input_size() == 2, "SentencepieceTokenizer expects two inputs: sp model and input sentences");
    FRONT_END_GENERAL_CHECK(get_input_element_type(0) == element::u8, "SentencepieceTokenizer accepts sp model as the first input and it should be of type u8 tensor");

    #if USE_STRING_TENSORS

        #if OPENVINO_USE_INPUT_OUTPUT_STRING_TENSOR_HACK
        FRONT_END_GENERAL_CHECK(
            get_input_element_type(1) == element::string || get_input_element_type(1) == element::u8,
            "SentencepieceTokenizer accepts sentences as the second input and it should be of type u8 or string depending on the current stage of model preparation");
        #else
        FRONT_END_GENERAL_CHECK(
            get_input_element_type(1) == element::string,
            "SentencepieceTokenizer accepts sentences as the second input and it should be of type string tensor");
        #endif

    #else

#if 0   // change to 0 when compiled with master and the bug with data propagation from within inline context is not solved
    FRONT_END_GENERAL_CHECK(
        get_input_element_type(1) == element::u8,
        "SentencepieceTokenizer accepts sentences as the second input and it should be of type u8 tensor, but got " +
            get_input_element_type(1).get_type_name());
#endif

    #endif

    #endif

    // The operation SentencepieceTokenizerExtensionOp has three outputs: sparse indices, sparse values
    // and dense shape
    set_output_type(0, element::i64, PartialShape{ Dimension(), Dimension(2) });
    set_output_type(1, element::i32, PartialShape{ Dimension() });
    set_output_type(2, element::i64, PartialShape{ Dimension(2) });
}

bool SentencepieceTokenizer::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("nbest_size", m_nbest_size);
    visitor.on_attribute("alpha", m_alpha);
    visitor.on_attribute("add_bos", m_add_bos);
    visitor.on_attribute("add_eos", m_add_eos);
    visitor.on_attribute("reverse", m_reverse);
    return true;
}

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

bool SentencepieceTokenizer::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    std::vector<int64_t> sparse_indices;
    std::vector<int32_t> sparse_values;
    std::vector<int64_t> sparse_dense_shape;

#if SENTENCE_PIECE_EXTENSION_DECOMPOSED_STRINGS

    auto begin_ids = inputs[1].data<const int32_t>();
    auto end_ids = inputs[2].data<const int32_t>();
    auto data = inputs[3].data<const uint8_t>();

    auto batch_size = shape_size(inputs[1].get_shape());

#else

#if USE_STRING_TENSORS

    #if OPENVINO_USE_INPUT_OUTPUT_STRING_TENSOR_HACK
    const ov::Tensor& strings_tensor = **reinterpret_cast<ov::Tensor**>(inputs[1].data<uint8_t>());
    #else
    const ov::Tensor& strings_tensor = inputs[1];
    #endif

    const std::string* strings = strings_tensor.data<std::string>();
    size_t batch_size = ov::shape_size(strings_tensor.get_shape());

#else

    // const uint8_t* strings = inputs[1].data<const uint8_t>();
    // auto bitstream_size = inputs[1].get_byte_size();

    // // check the format of the input bitstream representing the string tensor
    // FRONT_END_GENERAL_CHECK(bitstream_size >= 4, "Incorrect packed string tensor format: no batch size in the packed string tensor");
    // auto batch_size = *reinterpret_cast<const int32_t*>(strings + 0);
    // FRONT_END_GENERAL_CHECK(bitstream_size >= 4 + 4 + 4 * batch_size,
    //     "Incorrect packed string tensor format: the packed string tensor must contain first string offset and end indices");
    // auto begin_ids = reinterpret_cast<const int32_t*>(strings + 4);
    // auto end_ids = begin_ids + 1;
    // auto data = strings + 4 + 4 + 4 * batch_size;
    int32_t batch_size;
    const int32_t* begin_ids;
    const int32_t* end_ids;
    const uint8_t* data;
    parse_packed_strings(inputs[1], batch_size, begin_ids, end_ids, data);

#endif

#endif
    //std::cerr << "    Batch size: " << batch_size << "\n";

    size_t max_token_id = 0;
    for (size_t batch_ind = 0; batch_ind < batch_size; ++batch_ind) {
#if USE_STRING_TENSORS && !SENTENCE_PIECE_EXTENSION_DECOMPOSED_STRINGS
        const std::string& sentence = strings[batch_ind];
        //std::cerr << "    sentence: " << sentence << "\n";
#else
        auto begin_ind = begin_ids[batch_ind];
        auto end_ind = end_ids[batch_ind];
        //std::string sentence(data + begin_ind, data + end_ind);
        absl::string_view sentence((const char*)data + begin_ind, end_ind - begin_ind);
        //std::cerr << "string: " << sentence << "\n";
#endif
        std::vector<int32_t> ids;
        CHECK_OK(m_sp->SampleEncode(sentence, m_nbest_size, m_alpha, &ids));
        // put into resulted vectors
        for (size_t token_id = 0; token_id < ids.size(); ++token_id) {
            sparse_indices.push_back(static_cast<int64_t>(batch_ind));
            sparse_indices.push_back(static_cast<int64_t>(token_id));
            sparse_values.push_back(static_cast<int32_t>(ids[token_id]));
        }
        max_token_id = max_token_id < ids.size() ? ids.size() : max_token_id;
    }
    sparse_dense_shape.push_back(static_cast<int64_t>(batch_size));
    sparse_dense_shape.push_back(static_cast<int64_t>(max_token_id));

    outputs[0].set_shape({ sparse_indices.size() / 2, 2 });
    memcpy(outputs[0].data(), sparse_indices.data(), sizeof(int64_t) * sparse_indices.size());
    outputs[1].set_shape({ sparse_values.size() });
    memcpy(outputs[1].data(), sparse_values.data(), sizeof(int32_t) * sparse_values.size());
    outputs[2].set_shape({ 2 });
    memcpy(outputs[2].data(), sparse_dense_shape.data(), sizeof(int64_t) * sparse_dense_shape.size());
    return true;
}

bool SentencepieceTokenizer::has_evaluate() const {
    return true;
}

std::shared_ptr<Node> SentencepieceTokenizer::clone_with_new_inputs(const OutputVector& new_args) const {
    return std::make_shared<SentencepieceTokenizer>(new_args, m_sp, m_nbest_size, m_alpha, m_add_bos, m_add_eos, m_reverse);
}

OutputVector translate_sentencepiece_op(const NodeContext& node) {
    // extract model to configure SentencePieceTokenizer
    auto sp_model_ov_any = node.get_attribute_as_any("model");
    FRONT_END_GENERAL_CHECK(sp_model_ov_any.is<std::string>(),
        "SentencePieceOp configuration model is in incorrect format");
    auto str_spm_model = sp_model_ov_any.as<std::string>();
    auto sp_model_const = std::make_shared<Constant>(element::u8, Shape{ str_spm_model.size() }, str_spm_model.data());
    return { sp_model_const };
}




void check_string_input(const Node* node, size_t input_index) {
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+0) == element::i32, "Expected an i32 tensor as the first part of the decomposed string representation");
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+1) == element::i32, "Expected an i32 tensor as the second part of the decomposed string representation");
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+2) == element::u8,  "Expected a u8 tensor as the third part of the decomposed string representation");
}

void check_string_scalar_input(const Node* node, size_t input_index) {
    auto shape = node->get_input_partial_shape(input_index);
    auto element_type = node->get_input_element_type(input_index);

    #if USE_STRING_TENSORS

    OPENVINO_ASSERT(
        (element_type == element::dynamic || element_type == element::string) &&
        (shape.rank().is_dynamic() || shape.rank().get_length() == 0),
        "string/0D tensor is expected");

    #else

    OPENVINO_ASSERT(
        (element_type == element::dynamic || element_type == element::u8) &&
        (shape.rank().is_dynamic() || shape.rank().get_length() == 1),
        "u8/1D tensor is expected");

    #endif
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


void StringTensorPack::validate_and_infer_types() {
    OPENVINO_ASSERT(m_mode == "begins_ends", "StringTensorPack supporst only 'begins_ends' mode, but get " + m_mode);
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
    // Implementation for debuggin purposes: directly print ragged indices to std::cout and pass the base tensor with elements throug.

    auto input_shape = inputs[0].get_shape();
    std::cout << "[ DEBUG ] RaggedTensorPack: shape = " << input_shape << "\n";
    auto begins = inputs[0].data<const int32_t>();
    auto ends   = inputs[1].data<const int32_t>();
    auto num_elements = shape_size(input_shape);

    for(size_t i = 0; i < num_elements; ++i) {
        std::cout << "[ DEBUG ]     [" << i << "] " << begins[i] << ":" << ends[i] << " with size = " << ends[i] - begins[i] << "\n";
    }

    inputs[2].copy_to(outputs[0]);

    return true;
}


void StringTensorUnpack::validate_and_infer_types() {
    OPENVINO_ASSERT(
        get_input_size() == 1,
        "Number of inputs for StringTensorUnpack is not equal to 1");

    auto output_shape = PartialShape::dynamic();


    // In case of explicit string tensors the shape is carried by input tensor itself
    // OPENVINO_ASSERT(
    //     input_shape == PartialShape::dynamic(),
    //     "Excplicitly set shape for a string tensor in the unpacking is not supported");

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

void unpack_strings (const std::string* strings, const Shape shape, ov::Tensor& begins, ov::Tensor& ends, ov::Tensor& chars) { // TODO: no need for a reference to a ov::Tensor?
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
        unpack_strings(input_strings, input_shape, outputs[0], outputs[1], outputs[2]);
        return true;
    } else {
#endif

#if USE_STRING_TENSORS
    OPENVINO_ASSERT(false, "Detected a u8 tensor but element::string tensor should be provided")
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
    // auto ragged_pack = dynamic_cast<RaggedTensorPack*>(node.get_input(input_index).get_node());
    // OPENVINO_ASSERT(ragged_pack, "Expected RaggedTensorPack but didn't find it");
    auto ragged_inputs = pre_translate_ragged_tensor_input(input);
    auto string_inputs = pre_translate_string_tensor_input(ragged_inputs[2]);
    ragged_inputs.pop_back();
    ragged_inputs.insert(ragged_inputs.end(), string_inputs.begin(), string_inputs.end());
    // auto string_pack = dynamic_cast<StringTensorPack*>(ragged_pack->get_input_node_ptr(2));
    // OPENVINO_ASSERT(string_pack, "Expected StringTensorPack as a base for RaggedTensorPack but didn't find it");
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

NamedOutputVector translate_sentencepiece_tokenizer(const NodeContext& node) {
    // this is custom translator that converts a sub-graph with SentencePieceOp, SentencePieceTokenizer,
    // and RaggedTensorToSparse operation- into a custom operation SentencepieceTokenizerExtensionOp
    FRONT_END_GENERAL_CHECK(node.get_input_size() > 0, "RaggedTensorToSparse expects at least one input.");
    auto node_name = node.get_name();

    // check that producers of RaggedTensorToSparse is SentencePieceTokenizer
    auto sp_tokenize_op = node.get_input(0).get_node_shared_ptr();
    FRONT_END_GENERAL_CHECK(sp_tokenize_op->get_input_size() > 6,
        "SentencepieceTokenizeOp expects at least six inputs");

    // prepare inputs that go to custom operation
    // prepare input 0 - SentencePieceTokenizer configuration model
    auto sp_model_const = as_type_ptr<Constant>(sp_tokenize_op->input_value(0).get_node_shared_ptr());
    FRONT_END_GENERAL_CHECK(sp_model_const, "Conversion expects SentencePiece model to be constant.");

    // prepare input six inputs
    auto inputs = sp_tokenize_op->input_value(1);

    // extract values for nbest_size, alpha, add_bos, add_eos, reverse attributes
    auto nbest_size = extract_scalar_const_value<int32_t>(sp_tokenize_op->input_value(2).get_node_shared_ptr(), "nbest_size");
    auto alpha = extract_scalar_const_value<float>(sp_tokenize_op->input_value(3).get_node_shared_ptr(), "alpha");
    auto add_bos = extract_scalar_const_value<bool>(sp_tokenize_op->input_value(4).get_node_shared_ptr(), "add_bos");
    auto add_eos = extract_scalar_const_value<bool>(sp_tokenize_op->input_value(5).get_node_shared_ptr(), "add_eos");
    auto reverse = extract_scalar_const_value<bool>(sp_tokenize_op->input_value(6).get_node_shared_ptr(), "reverse");

#if !USE_STRING_TENSORS
    // Override type of input tensor if this is a Parameter
    if (auto parameter = std::dynamic_pointer_cast<Parameter>(inputs.get_node_shared_ptr())) {
        parameter->set_partial_shape(PartialShape{ Dimension() });
        parameter->set_element_type(element::u8);
        parameter->validate_and_infer_types();
    }
#endif

#if SENTENCE_PIECE_EXTENSION_DECOMPOSED_STRINGS

    OutputVector inputs_vector = OutputVector{ sp_model_const };
    auto unpacked_outputs = std::make_shared<StringTensorUnpack>(OutputVector{inputs}, "begins_ends")->outputs();
    inputs_vector.insert(inputs_vector.end(), unpacked_outputs.begin(), unpacked_outputs.end());

#else

    OutputVector inputs_vector = OutputVector{ sp_model_const, inputs };

#endif

    // create a node with custom operation
    auto sp_tokenizer_ext = std::make_shared<SentencepieceTokenizer>(inputs_vector, nbest_size, alpha, add_bos, add_eos, reverse);
    FRONT_END_GENERAL_CHECK(sp_tokenizer_ext->get_output_size() == 3,
        "Internal error: SentencepieceTokenizer operation extension must have three outputs.");

    // set tensor names
    sp_tokenizer_ext->output(0).add_names({ node_name + ":0" });
    sp_tokenizer_ext->output(1).add_names({ node_name + ":1" });
    sp_tokenizer_ext->output(2).add_names({ node_name + ":2" });

    // create named outputs for the conversion extension
    NamedOutputVector named_results;
    named_results.push_back({ "sparse_indices", sp_tokenizer_ext->output(0) });
    named_results.push_back({ "sparse_values", sp_tokenizer_ext->output(1) });
    named_results.push_back({ "sparse_dense_shape", sp_tokenizer_ext->output(2) });

    return named_results;
}


void CaseFold::validate_and_infer_types() {
    check_string_input(this, 0);
    set_string_output(this, 0, get_input_partial_shape(0));
}

bool CaseFold::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto begins = inputs[0].data<const int32_t>();
    auto ends   = inputs[1].data<const int32_t>();
    auto chars  = inputs[2].data<const uint8_t>();

#if 1
    // Set output shapes
    outputs[0].set_shape(inputs[0].get_shape());
    outputs[1].set_shape(inputs[1].get_shape());
    const size_t num_elements = inputs[0].get_size();

    // TODO: Provide more accurate heuristics to estimate output shape
    const size_t new_len = 2*inputs[2].get_size();

    outputs[2].set_shape(Shape{new_len});

    // For the whole implementation below the input shapes can be ignored, we are working with the flatten representaions
    // and only number of elements in the original tensors matter

    // Get pointers in the output tensors
    auto new_begins = outputs[0].data<int32_t>();
    auto new_ends   = outputs[1].data<int32_t>();
    auto new_chars  = outputs[2].data<uint8_t>();
    int32_t char_offset = 0;

    for(size_t i = 0; i < num_elements; ++i) {
        new_begins[i] = char_offset;

        using namespace paddlenlp::fast_tokenizer;
        normalizers::NormalizedString str(std::string(chars + begins[i], chars + ends[i]));

        // Do the job
        str.Lowercase();

        const std::string& new_str = str.GetStr();
        std::copy(new_str.data(), new_str.data() + new_str.length(), new_chars + char_offset);
        char_offset += new_str.length();
        new_ends[i] = char_offset;
    }
    return true;
#else
    // Stub implementation that transforms each input string "X" to "CaseFold(X)" for debugging purposes
    {
        // Set output shapes
        outputs[0].set_shape(inputs[0].get_shape());
        outputs[1].set_shape(inputs[1].get_shape());
        const std::string left_side = "CaseFold(", right_side = ")";
        const size_t num_elements = inputs[0].get_size();
        const size_t new_len = inputs[2].get_size() + (left_side.length() + right_side.length())*num_elements;
        outputs[2].set_shape(Shape{new_len});

        // For the whole implementation below the input shapes can be ignored, we are working with the flatten representaions
        // and only number of elements in the original tensors matter

        // Get pointers in the output tensors
        auto new_begins = outputs[0].data<int32_t>();
        auto new_ends   = outputs[1].data<int32_t>();
        auto new_chars  = outputs[2].data<uint8_t>();
        int32_t char_offset = 0;

        for(size_t i = 0; i < num_elements; ++i) {
            new_begins[i] = char_offset;
            std::string new_str = left_side + std::string(chars + begins[i], chars + ends[i]) + right_side;
            std::copy(new_str.data(), new_str.data() + new_str.length(), new_chars + char_offset);
            char_offset += new_str.length();
            new_ends[i] = char_offset;
        }
        return true;
    }
    // End of stub implementation
#endif
}


ov::OutputVector translate_case_fold_utf8(const ov::frontend::NodeContext& node) {
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 1, "CaseFold expects only 1 input");
    return { post_translate_string_tensor_output(std::make_shared<CaseFold>(
        pre_translate_string_tensor_input(node.get_input(0)))->outputs()) };
}



void NormalizeUnicode::validate_and_infer_types() {
    check_string_input(this, 0);
    set_string_output(this, 0, get_input_partial_shape(0));
}

bool NormalizeUnicode::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto begins = inputs[0].data<const int32_t>();
    auto ends   = inputs[1].data<const int32_t>();
    auto chars  = inputs[2].data<const uint8_t>();

#if 0
    // TODO: Complete implementation
#else
    // Stub implementation that transforms each input string "X" to "NormalizeUnicode(X, normalization_form)" for debugging purposes
    {
        // Set output shapes
        outputs[0].set_shape(inputs[0].get_shape());
        outputs[1].set_shape(inputs[1].get_shape());
        const std::string left_side = "NormalizeUnicode(", right_side = ")", delimeter = ", ";
        const size_t num_elements = inputs[0].get_size();
        const size_t new_len = inputs[2].get_size() + (left_side.length() + right_side.length() + delimeter.length() + m_normalization_form.length())*num_elements;
        outputs[2].set_shape(Shape{new_len});

        // For the whole implementation below the input shapes can be ignored, we are working with the flatten representaions
        // and only number of elements in the original tensors matter

        // Get pointers in the output tensors
        auto new_begins = outputs[0].data<int32_t>();
        auto new_ends   = outputs[1].data<int32_t>();
        auto new_chars  = outputs[2].data<uint8_t>();
        int32_t char_offset = 0;

        for(size_t i = 0; i < num_elements; ++i) {
            new_begins[i] = char_offset;
            std::string new_str = left_side + std::string(chars + begins[i], chars + ends[i]) + delimeter + m_normalization_form + right_side;
            std::copy(new_str.data(), new_str.data() + new_str.length(), new_chars + char_offset);
            char_offset += new_str.length();
            new_ends[i] = char_offset;
        }
        return true;
    }
    // End of stub implementation
#endif
}


ov::OutputVector translate_normalize_utf8(const ov::frontend::NodeContext& node) {
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 1, "NormalizeUTF8 expects only 1 input");
    return { post_translate_string_tensor_output(std::make_shared<NormalizeUnicode>(
        pre_translate_string_tensor_input(node.get_input(0)),
        node.get_attribute<std::string>("normalization_form"))->outputs()) };
}


void RegexNormalization::validate_and_infer_types() {
    check_string_input(this, 0);
    check_string_scalar_input(this, 3);
    check_string_scalar_input(this, 4);
    set_string_output(this, 0, get_input_partial_shape(0));
}

bool RegexNormalization::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto begins = inputs[0].data<const int32_t>();
    auto ends   = inputs[1].data<const int32_t>();
    auto chars  = inputs[2].data<const uint8_t>();

    auto search_pattern_buf  = inputs[3].data<const uint8_t>();
    auto replace_pattern_buf  = inputs[4].data<const uint8_t>();
    auto search_pattern = absl::string_view((const char*)search_pattern_buf, shape_size(inputs[3].get_shape()) - 1);   // FIXME: -1 is a complementary change to a WA applied in string_attribute_to_constant
    auto replace_pattern = absl::string_view((const char*)replace_pattern_buf, shape_size(inputs[4].get_shape()) - 1);   // FIXME: -1 is a complementary change to a WA applied in string_attribute_to_constant

#if 0
    // TODO: Complete implementation
#else
    // Stub implementation that transforms each input string "X" to "RegexNormalization(X, search_pattern, replace_pattern)" for debugging purposes
    {
        // Set output shapes
        outputs[0].set_shape(inputs[0].get_shape());
        outputs[1].set_shape(inputs[1].get_shape());
        const std::string left_side = "RegexNormalization(", right_side = ")", delimeter = ", ";
        const size_t num_elements = inputs[0].get_size();
        const size_t new_len = inputs[2].get_size() + (left_side.length() + right_side.length() + 2*delimeter.length() + search_pattern.length() + replace_pattern.length())*num_elements;
        outputs[2].set_shape(Shape{new_len});

        // For the whole implementation below the input shapes can be ignored, we are working with the flatten representaions
        // and only number of elements in the original tensors matter

        // Get pointers in the output tensors
        auto new_begins = outputs[0].data<int32_t>();
        auto new_ends   = outputs[1].data<int32_t>();
        auto new_chars  = outputs[2].data<uint8_t>();
        int32_t char_offset = 0;

        for(size_t i = 0; i < num_elements; ++i) {
            new_begins[i] = char_offset;

            std::string new_str =
                left_side + std::string(chars + begins[i], chars + ends[i]) + delimeter +
                std::string(search_pattern) + delimeter +
                std::string(replace_pattern) + right_side;

            std::copy(new_str.data(), new_str.data() + new_str.length(), new_chars + char_offset);
            char_offset += new_str.length();
            new_ends[i] = char_offset;
        }
        return true;
    }
    // End of stub implementation
#endif
}


std::shared_ptr<Node> string_attribute_to_constant (const ov::frontend::NodeContext& node, const std::string& name) {
    // FIXME: using space to pad the value to work-around CPU issue with empty constants
    auto value = node.get_attribute<std::string>(name) + " ";

    #if USE_STRING_TENSORS
    return std::make_shared<Constant>(element::string, {}, value);
    #else
    return std::make_shared<Constant>(element::u8, Shape{value.length()}, (const void*)value.data());
    #endif
}


ov::OutputVector translate_static_regex_replace(const ov::frontend::NodeContext& node) {
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 1, "StaticRegexReplace expects only 1 input");
    ov::OutputVector inputs = pre_translate_string_tensor_input(node.get_input(0));
    inputs.push_back(string_attribute_to_constant(node, "pattern"));
    inputs.push_back(string_attribute_to_constant(node, "rewrite"));
    return { post_translate_string_tensor_output(std::make_shared<RegexNormalization>(inputs)->outputs()) };
}



void RegexSplit::validate_and_infer_types() {
    check_string_input(this, 0);
    check_string_scalar_input(this, 3);
    set_ragged_string_output(this, 0, get_input_partial_shape(0));
}

bool RegexSplit::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto begins = inputs[0].data<const int32_t>();
    auto ends   = inputs[1].data<const int32_t>();
    auto chars  = inputs[2].data<const uint8_t>();

    auto split_pattern_buf  = inputs[3].data<const uint8_t>();
    auto split_pattern = absl::string_view((const char*)split_pattern_buf, shape_size(inputs[3].get_shape())/* - 1*/);   // Shouldn't be applied FIXME: -1 is a complementary change to a WA applied in string_attribute_to_constant

#if 0
    // TODO: Complete implementation
#else
    // Stub implementation that transforms each input string "X" to multiple "RegexSplit(X, split_pattern) = part(X)" for debugging purposes
    // Where part(X) is a part of original X divided by predefined length with some reminder
    // So each element X is divided into multiple output elements along ragged dimension, and the number of elements depends on the input X length and
    // can vary for different X. For example, let the length = 2 and input X = "words", the output would consist of 3 elements along corresponding
    // ragged dimension in the output with values:
    //  - "RegexSplit(word, search_pattern, replace_pattern) = wo",
    //  - "RegexSplit(word, search_pattern, replace_pattern) = rd",
    //  - "RegexSplit(word, search_pattern, replace_pattern) = s"
    // split_pattern is cut for the sake of readability of ouput
    {
        const size_t part_length = 30;   // any positive number, defines the length of each part in bytes

        std::string split_pattern_part = std::string(split_pattern.substr(0, part_length));

        // Set output shapes
        outputs[0].set_shape(inputs[0].get_shape());
        outputs[1].set_shape(inputs[1].get_shape());

        const std::string left_side = "RegexSplit(", right_side = ")", delimeter = ", ";
        const size_t num_elements = inputs[0].get_size();
        size_t num_parts = 0;   // will count the number of all parts
        size_t num_additional_chars = 0;  //
        // Count the resulting number of part that we are going to obtain
        for(size_t i = 0; i < num_elements; ++i) {
            auto length = ends[i] - begins[i];
            auto num_of_whole_parts = length/part_length;
            auto remainder = length%part_length;
            auto num_local_parts = num_of_whole_parts + int(bool(remainder));
            num_parts += num_local_parts;
            num_additional_chars += length*num_local_parts;
        }

        size_t num_chars = inputs[2].get_size();

        // FIXME: Overestimation
        const size_t new_num_chars = num_chars + num_parts*30/*!*/ + (left_side.length() + right_side.length() + delimeter.length() + split_pattern_part.length())*num_elements;
        outputs[2].set_shape(Shape{num_parts});
        outputs[3].set_shape(Shape{num_parts});
        outputs[4].set_shape(Shape{new_num_chars});

        // For the whole implementation below the input shapes can be ignored, we are working with the flatten representaions
        // and only number of elements in the original tensors matter

        // Get pointers in the output tensors
        auto new_ragged_begins = outputs[0].data<int32_t>();
        auto new_ragged_ends   = outputs[1].data<int32_t>();
        auto new_begins = outputs[2].data<int32_t>();
        auto new_ends   = outputs[3].data<int32_t>();
        auto new_chars  = outputs[4].data<uint8_t>();
        int32_t ragged_offset = 0;
        int32_t char_offset = 0;

        for(size_t i = 0; i < num_elements; ++i) {
            new_ragged_begins[i] = ragged_offset;
            auto old_str = std::string(chars + begins[i], chars + ends[i]);
            auto new_str_part_base = left_side + old_str + delimeter + split_pattern_part + right_side;

            for(size_t j = 0; j < old_str.length(); j += part_length) {
                new_begins[ragged_offset] = char_offset;
                //auto new_str_part = new_str_part_base + old_str.substr(j, part_length);
                std::string new_str_part = j == 0 ? new_str_part_base : "part[" + std::to_string(i) + "," + std::to_string(j) + "]";
                std::copy(new_str_part.data(), new_str_part.data() + new_str_part.length(), new_chars + char_offset);
                char_offset += new_str_part.length();
                new_ends[ragged_offset] = char_offset;
                ++ragged_offset;
            }

            new_ragged_ends[i] = ragged_offset;
        }

        outputs[4].set_shape({char_offset});

        //OPENVINO_ASSERT(char_offset == new_num_chars, "Internal error in RegexSplit::evaluate: out of range for chars");
        OPENVINO_ASSERT(ragged_offset == num_parts, "Internal error in RegexSplit::evaluate: out of range for ragged parts");

        return true;
    }
    // End of stub implementation
#endif
}


ov::OutputVector translate_regex_split_with_offsets(const ov::frontend::NodeContext& node) {
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 3, "RegexSplitWithOffsets expects 3 inputs");
    ov::OutputVector inputs = pre_translate_string_tensor_input(node.get_input(0));
    auto delim_regex_pattern = node.get_input(1).get_node()->input_value(2);    // use u8 part of packed string tensor as we are expecting a scalar string: TODO: verify it is really there
    inputs.push_back(delim_regex_pattern);
    auto outputs = std::make_shared<RegexSplit>(inputs)->outputs();
    auto flatten_string_tensor = post_translate_string_tensor_output({outputs[2], outputs[3], outputs[4]});
    return { post_translate_ragged_tensor_output({outputs[0], outputs[1], flatten_string_tensor}) };
}



void WordpieceTokenizer::validate_and_infer_types() {
    check_ragged_string_input(this, 0);
    check_string_input(this, 5);
    set_ragged_output(this, 0, get_input_partial_shape(0), element::i32);
}

bool WordpieceTokenizer::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto ragged_begins = inputs[0].data<const int32_t>();
    auto ragged_ends   = inputs[1].data<const int32_t>();
    auto begins = inputs[2].data<const int32_t>();
    auto ends   = inputs[3].data<const int32_t>();
    auto chars  = inputs[4].data<const uint8_t>();

    auto vocab_begins = inputs[5].data<const int32_t>();
    auto vocab_ends   = inputs[6].data<const int32_t>();
    auto vocab_chars  = inputs[7].data<const uint8_t>();

    OPENVINO_ASSERT(inputs.size() == 9, "Too few inputs passed to WordpieceTokenizer, it means it is not converted properly or it is not used in the supported pattern");

    auto unk_token_id  = *inputs[8].data<const int32_t>();
#if 0
    // TODO: Complete implementation
#else
    // Stub implementation that transforms each input string to its length duplicating element if the length is odd
    {
        std::cout << "[ DEBUG ] WordpieceTokenizer\n";
        std::cout << "[ DEBUG ]     vocab size: " << inputs[5].get_size() << "\n";
        std::cout << "[ DEBUG ]     unk_token_id: " << unk_token_id << "\n";

        // Set output shapes
        outputs[0].set_shape(inputs[0].get_shape());
        outputs[1].set_shape(inputs[1].get_shape());
        const size_t num_elems = inputs[0].get_size();

        const size_t num_parts = inputs[2].get_size();
        size_t new_num_parts = num_parts;
        // Count number of output elements
        for(size_t i = 0; i < num_parts; ++i) {
            auto length = ends[i] - begins[i];
            new_num_parts += length % 2;
        }

        outputs[2].set_shape({new_num_parts});

        // Get pointers in the output tensors
        auto new_begins = outputs[0].data<int32_t>();
        auto new_ends   = outputs[1].data<int32_t>();
        auto new_elems  = outputs[2].data<int32_t>();
        int32_t offset = 0;

        for(size_t j = 0; j < num_elems; ++j) {
            new_begins[j] = offset;

            for(size_t i = ragged_begins[j]; i < ragged_ends[j]; ++i) {

                auto length = ends[i] - begins[i];
                new_elems[offset++] = length;

                if(length % 2) {
                    new_elems[offset++] = length;
                }
            }

            new_ends[j] = offset;
        }

        OPENVINO_ASSERT(offset == outputs[2].get_size(), "Internal error in RegexSplit::evaluate: out of range for ragged parts");
        return true;
    }
    // End of stub implementation
#endif
}


ov::OutputVector translate_wordpiece_tokenize_with_offsets(const ov::frontend::NodeContext& node) {
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 2, "WordpieceTokenizeWithOffsets expects 2 inputs");
    ov::OutputVector inputs = pre_translate_ragged_string_tensor_input(node.get_input(0));

    #if USE_STRING_TENSORS
    // It may seem enough to call pre_translate_string_tensor_input that will override Parameter element
    // type in case if string tensors are not used.
    // But a Parameter is still required to be overridden even if string tensors are used because in TF model
    // it is represented not as a string tensor, but as a resource with hash table for lookup that we cannot interpret
    // and have to replace by 1D string tensor.
    override_parameter(node.get_input(1).get_node_shared_ptr(), element::string, PartialShape{Dimension()});
    #endif

    auto vocab = pre_translate_string_tensor_input(node.get_input(1));
    inputs.insert(inputs.end(), vocab.begin(), vocab.end());
    // FIXME: Cannot set real value for unk_token_id from attributes because it is not known in this operation
    // TODO: Set other attributes.
    auto wp_tokenizer = std::make_shared<WordpieceTokenizer>(
        inputs,
        node.get_attribute<std::string>("suffix_indicator"),
        node.get_attribute<long>("max_bytes_per_word")
    );
    return { post_translate_ragged_tensor_output(wp_tokenizer->outputs()) };
}


ov::OutputVector translate_lookup_table_find_v2(const ov::frontend::NodeContext& node) {
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 3, "LookupTableFindV2 expects 3 inputs");

    // Check if this node is used in a combination with already converted WordpieceTokenizeWithOffsets
    auto wp_tokenizer_outputs = pre_translate_ragged_tensor_input(node.get_input(1));
    auto wp_tokenizer = dynamic_cast<WordpieceTokenizer*>(wp_tokenizer_outputs[0].get_node());
    OPENVINO_ASSERT(wp_tokenizer, "Conversion of LookupTableFindV2 without coupled WordpieceTokenizer is not yet supported");

    // TODO: Check vocab matching for LookupTableFindV2 and WordpieceTokenizer

    // TODO: Check if overflow really happens in real models due to i64 to i32 conversion
    auto unk_token_id = std::make_shared<opset10::Convert>(node.get_input(2), element::i32);

    auto wp_tokenizer_inputs = wp_tokenizer->input_values();
    wp_tokenizer_inputs.push_back(unk_token_id);
    //std::cerr << "Added extra input, total number of inputs is " << wp_tokenizer_inputs.size() << "\n";

    auto new_wp_tokenizer = wp_tokenizer->clone_with_new_inputs(wp_tokenizer_inputs);
    return { post_translate_ragged_tensor_output(new_wp_tokenizer->outputs()) };
}


ov::OutputVector translate_reshape(const ov::frontend::NodeContext& node) {
    // This is a copied-and-pasted and adopted fragment of TF reshape translator from OV.
    // It checks if the input tensor has string type, and then perform custom tranlation.
    // Otherwise it should operate identically to the stock version of Reshape translator in TF FE.
    // TODO: Introduce an API to call original translators from an extension without copying the code to an extension.

    FRONT_END_GENERAL_CHECK(node.get_input_size() == 2, "Tensorflow Reshape op should have two inputs");
    auto tensor = node.get_input(0);
    auto shape = node.get_input(1);
    if(auto pack = dynamic_cast<StringTensorPack*>(tensor.get_node())) {
        // TODO: If it is a beginning of the graph, how to detect strings? It falls in 'else' branch in this case.
        // FIXME: Needs extension for a Parameter to prepare it first
        auto begins = std::make_shared<Reshape>(pack->input_value(0), shape, false);
        auto ends = std::make_shared<Reshape>(pack->input_value(1), shape, false);
        auto chars = pack->input_value(2);
        auto reshape = post_translate_string_tensor_output({begins, ends, chars});
        return {reshape};
    } else {
        auto reshape = std::make_shared<Reshape>(tensor, shape, false);
        return {reshape};
    }
    // set_node_name(node.get_name(), reshape); // TODO: requires dependencies from TF FE internals
}


// Copied and pasted from TF FE and adopted to not use internal TF FE operation classes
ov::OutputVector translate_const(const ov::frontend::NodeContext& node) {
    auto ov_type = node.get_attribute_as_any("dtype");
    std::shared_ptr<Node> const_node;
    if (!ov_type.is<ov::element::Type>() || ov_type.as<ov::element::Type>() == ov::element::dynamic ||
        ov_type.as<ov::element::Type>() == ov::element::undefined) {
        if (ov_type.is<std::string>() && ov_type.as<std::string>() == "DT_STRING") {
            auto value_as_any = node.get_attribute_as_any("value");
            const auto& values = value_as_any.as<std::vector<std::string>>();
            ov::Tensor begins(element::i32, {}), ends(element::i32, {}), chars(element::u8, {});
            unpack_strings(&values[0], {values.size()}, begins, ends, chars);
            const_node = std::make_shared<StringTensorPack>(OutputVector{
                std::make_shared<Constant>(begins),
                std::make_shared<Constant>(ends),
                std::make_shared<Constant>(chars)
            });
        } else {
            const_node = std::make_shared<ov::op::util::FrameworkNode>(OutputVector{});
        }
    } else {
        //static std::vector<ov::Tensor> tensors;
        auto tensor = node.get_attribute<ov::Tensor>("value");
        //tensors.push_back(tensor);
        const_node = std::make_shared<Constant>(tensor);
        #if OPENVINO_ELEMENT_STRING_SUPPORTED
        if (const_node->get_element_type() == element::string) {
            if(shape_size(tensor.get_shape()) > 0) {
                auto strings = std::dynamic_pointer_cast<Constant>(const_node)->get_data_ptr<std::string>();
            }
            const_node = std::make_shared<StringTensorUnpack>(const_node->outputs());
            const_node = std::make_shared<StringTensorPack>(const_node->outputs());
        }
        #endif
    }
    //set_node_name(node.get_name(), const_node);   // TODO: Provide alternative to internal function set_node_name
    return {const_node};
}

