// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "normalizer.h"
#include "sentence_piece.hpp"

#include "openvino/opsets/opset10.hpp"

//#define USE_STRING_TENSORS

#ifdef USE_STRING_TENSORS

// A plugin can support a string tensor on inputs and outputs via the hack which wraps such tensor to
// a u8 tensor holding a pointer to the original string tensor. The hack lets us avoid more deep
// plugin modifications by pre-transform a model where string tensor parameters and results are replaced
// by the described wrapping tensors. Such a hack requires some pre/post processing in operations
// that handle such wrapping tensors on the edge of a model.
#define USE_INPUT_OUTPUT_STRING_TENSOR_HACK

#endif

#define SENTENCE_PIECE_EXTENSION_DECOMPOSED_STRINGS

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

    #ifdef SENTENCE_PIECE_EXTENSION_DECOMPOSED_STRINGS

    FRONT_END_GENERAL_CHECK(get_input_size() == 1 + 3, "SentencepieceTokenizer expects 4 inputs: sp model and input sentences represented as 3 decomposed tensors (begins, ends, sybols)");
    FRONT_END_GENERAL_CHECK(get_input_element_type(0) == element::u8, "SentencepieceTokenizer accepts sp model as the first input and it should be of type u8 tensor");
    FRONT_END_GENERAL_CHECK(get_input_element_type(1) == element::i32, "SentencepieceTokenizer accepts begins offsets as the second and it should be of type i32 tensor");
    FRONT_END_GENERAL_CHECK(get_input_element_type(2) == element::i32, "SentencepieceTokenizer accepts ends offsets as the third and it should be of type i32 tensor");
    FRONT_END_GENERAL_CHECK(get_input_element_type(3) == element::u8, "SentencepieceTokenizer accepts sentence symbols as the fourth input and it should be of type u8 tensor");

    #else

    FRONT_END_GENERAL_CHECK(get_input_size() == 2, "SentencepieceTokenizer expects two inputs: sp model and input sentences");
    FRONT_END_GENERAL_CHECK(get_input_element_type(0) == element::u8, "SentencepieceTokenizer accepts sp model as the first input and it should be of type u8 tensor");

    #ifdef USE_STRING_TENSORS

        #ifdef USE_INPUT_OUTPUT_STRING_TENSOR_HACK
        FRONT_END_GENERAL_CHECK(
            get_input_element_type(1) == element::string || get_input_element_type(1) == element::u8,
            "SentencepieceTokenizer accepts sentences as the second input and it should be of type u8 or string depending on the current stage of model preparation");
        #else
        FRONT_END_GENERAL_CHECK(
            get_input_element_type(1) == element::string,
            "SentencepieceTokenizer accepts sentences as the second input and it should be of type string tensor");
        #endif

    #else

    if(get_input_element_type(1) != element::u8) {
        std::cout << "Stopped\n";
        std::cin.get();
    }

    FRONT_END_GENERAL_CHECK(
        get_input_element_type(1) == element::u8,
        "SentencepieceTokenizer accepts sentences as the second input and it should be of type u8 tensor, but got " +
            get_input_element_type(1).get_type_name());

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

bool SentencepieceTokenizer::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    std::vector<int64_t> sparse_indices;
    std::vector<int32_t> sparse_values;
    std::vector<int64_t> sparse_dense_shape;

#ifdef SENTENCE_PIECE_EXTENSION_DECOMPOSED_STRINGS

    auto begin_ids = inputs[1].data<const int32_t>();
    auto end_ids = inputs[2].data<const int32_t>();
    auto data = inputs[3].data<const uint8_t>();

    auto batch_size = shape_size(inputs[1].get_shape());

#else

#ifdef USE_STRING_TENSORS

    #ifdef USE_INPUT_OUTPUT_STRING_TENSOR_HACK
    const ov::Tensor& strings_tensor = **reinterpret_cast<ov::Tensor**>(inputs[1].data<uint8_t>());
    #else
    const ov::Tensor& strings_tensor = inputs[1];
    #endif

    const std::string* strings = strings_tensor.data<std::string>();
    size_t batch_size = ov::shape_size(strings_tensor.get_shape());

#else

    const uint8_t* strings = inputs[1].data<uint8_t>();
    auto bitstream_size = inputs[1].get_byte_size();

    // check the format of the input bitstream representing the string tensor
    FRONT_END_GENERAL_CHECK(bitstream_size >= 4, "Incorrect packed string tensor format: no batch size in the packed string tensor");
    auto batch_size = *reinterpret_cast<const int32_t*>(strings + 0);
    FRONT_END_GENERAL_CHECK(bitstream_size >= 4 + 4 + 4 * batch_size,
        "Incorrect packed string tensor format: the packed string tensor must contain first string offset and end indices");
    auto begin_ids = reinterpret_cast<const int32_t*>(strings + 4);
    auto end_ids = begin_ids + 1;
    auto data = strings + 4 + 4 + 4 * batch_size;

#endif

#endif
    //std::cerr << "    Batch size: " << batch_size << "\n";

    size_t max_token_id = 0;
    for (size_t batch_ind = 0; batch_ind < batch_size; ++batch_ind) {
#if defined(USE_STRING_TENSORS) && !defined(SENTENCE_PIECE_EXTENSION_DECOMPOSED_STRINGS)
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


// Unpack a string tensor representation regardless of the source format, which
// can be an OV tensor with element::string element type (if supported) or u8
// packed representation, to a decompose tensor representation that may potentially
// consist of multiple tensors. The destination format is defined by `mode` attribute.
// Shape of the output tensor is compitelly recognized from the input (if supported)
// or defined partially by a dedicated input attribute `shape`. If `shape` is not set,
// which default to completelly dynamic `shape`, then output shape is defined
// by an input tensor.
class StringTensorUnpack : public ov::op::Op {
public:
    OPENVINO_OP("StringTensorUnpack");

    StringTensorUnpack(OutputVector inputs, const std::string& mode = "begins_ends"
        /*const std::string* _data = nullptr, PartialShape _input_shape = PartialShape::dynamic()*/)
        : ov::op::Op(inputs), m_mode(mode) {
        constructor_validate_and_infer_types();
    }
    //const std::string* data = nullptr;
    //PartialShape input_shape;

    void validate_and_infer_types() override {
        OPENVINO_ASSERT(
            get_input_size() == 1,
            "Number of inputs for StringTensorUnpack is not equal to 1");

        OPENVINO_ASSERT(
            #ifdef USE_STRING_TENSORS
            get_input_element_type(0) == element::string ||
            #endif
            get_input_element_type(0) == element::dynamic ||
            get_input_element_type(0) == element::u8,
            "Unsupported input element type for StringTensorUnpack");

        OPENVINO_ASSERT(
            get_input_partial_shape(0).rank().is_static(),
            "StringTensorUnpack supports only static input rank");

#if 0
        // Obtain shape from rt_info.
        auto& rt_info = get_input_node_shared_ptr(0)->get_rt_info();
        auto ops = rt_info.find("original_partial_shape");
        if(ops != rt_info.end()) {
            input_shape = ops->second.as<PartialShape>();
            std::cerr << "StringTensorUnpack: orig_partial_shape: " << input_shape << "\n";
        } else {
            std::cerr << "Impossible\n";
            std::cerr << get_input_node_shared_ptr(0) << "\n";
        }
#endif

        auto output_shape = PartialShape::dynamic();

#ifdef USE_STRING_TENSORS

        // In case of explicit string tensors the shape is carried by input tensor itself
        // OPENVINO_ASSERT(
        //     input_shape == PartialShape::dynamic(),
        //     "Excplicitly set shape for a string tensor in the unpacking is not supported");

        #ifdef USE_INPUT_OUTPUT_STRING_TENSOR_HACK

        // There are two cases that affect expected element type of the input tensor:
        // before the hack is applied (element::string) and after it (element::u8).

        OPENVINO_ASSERT(
            get_input_element_type(0) == element::string
            || get_input_element_type(0) == element::u8,
            "Type of StringTensorUnpack input is expected to be element::string before a model compilation or element::u8 after the compilation");

        if(get_input_element_type(0) == element::string) {
            output_shape = get_input_partial_shape(0);
        }

        if(get_input_element_type(0) == element::u8)
        {
            // After the plugin hack, a tensor is represented as a wrapping u8 tensor that will hold a pointer to a string tensor.
            // The original shape of a string tensor is stored in RT attribute of a tensor descriptor.
            const auto& rt_info = get_input_tensor(0).get_rt_info();
            auto it = rt_info.find("__original_partial_shape");

            // StringTensorUnpack expects __original_partial_shape attribute of type PartialShape in the input tensor.
            // If it is not found that means that model compilation wasn't pass the expected transformation where a string tensor
            // is wrapped to a u8 tensor holding a pointer, or because evaluation of this node is in progress and tensor attributes aren't preserved.
            if(it != rt_info.end() && it->second.is<PartialShape>()) {
                output_shape = it->second.as<PartialShape>();
            }
        }

        #else

        OPENVINO_ASSERT(
            get_input_element_type(0) == element::string,
            "StringTensorUnpack expects element::string in an input tensor, but it is " + std::string(get_input_element_type(0)));

        output_shape = get_input_partial_shape(0);

        #endif

#else
        // Expect packed string tensor represenation which can carry only a string tensors of shape [?]
        // Shape is not known in advance and only rank of the output can be set

        OPENVINO_ASSERT(
            get_input_element_type(0) == element::u8 &&
            get_input_partial_shape(0).rank().is_static() && get_input_partial_shape(0).rank().get_length() == 1,
            "StringTensorUnpack expects a u8 tensor with rank 1 that holds packed batched string tensor as an input, but observes type " +
                get_input_element_type(0).get_type_name() + " and shape " + get_input_partial_shape(0).to_string());

        output_shape = PartialShape({Dimension()});  // [?]

        #if 0

        if(get_input_element_type(0) == element::u8) {
            if(all_inputs_are_constants(this)) {
                std::cerr << "StringTensorUnpack: u8/const\n";
                // HACK: Tensor of strings is passed by a raw pointer to a tensor
                auto constant = std::dynamic_pointer_cast<ov::opset1::Constant>(get_input_node_shared_ptr(0));
                size_t raw_size = constant->get_shape()[0];
                if(raw_size == 0) {
                    // means empty input
                    std::cerr << "StringTensorUnpack: empty\n";
                    data = nullptr;
                    input_shape = PartialShape({0});
                } else if(raw_size == sizeof(void*)) {
                    std::cerr << "StringTensorUnpack: not empty, tensor HACK\n";
                    auto tensor = *reinterpret_cast<const ov::Tensor* const *>(constant->get_data_ptr<uint8_t>());
                    std::cerr << "Pointer to tensor from op: " << tensor << "\n";
                    input_shape = tensor->get_shape();
                    data = tensor->data<std::string>();
                } else {

                    OPENVINO_ASSERT(
                        false,
                        "Unexpected size for hacked Tensor<string> input. Something went wrong.");
                }
            } else {
                std::cerr << "StringTensorUnpack: u8/not constant\n";
            }
        } else {
            std::cerr << "StringTensorUnpack: string\n";
            input_shape = get_input_partial_shape(0);
            if(all_inputs_are_constants(this)) {
                auto constant = std::dynamic_pointer_cast<ov::opset1::Constant>(get_input_node_shared_ptr(0));
                data = constant->get_data_ptr<std::string>();
            } else {
                input_shape = get_input_partial_shape(0);
            }
        }

        #endif

#endif

        OPENVINO_ASSERT(m_mode == "begins_ends", "StringTensorUnpack supporst only 'begins_ends' mode, but get " + m_mode);

        if (m_mode == "begins_ends") {
            set_output_type(0, element::i32, output_shape);
            set_output_type(1, element::i32, output_shape);
            set_output_type(2, element::u8, PartialShape{Dimension()});
        }
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        auto result = std::make_shared<StringTensorUnpack>(inputs, m_mode);
        return result;
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        // FIXME: Serialization only, there is no deserialization
        visitor.on_attribute("mode", m_mode);
        return true;
    }

    bool has_evaluate() const {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {


#ifdef USE_STRING_TENSORS

        #ifdef USE_INPUT_OUTPUT_STRING_TENSOR_HACK
        auto tensor = *reinterpret_cast<const ov::Tensor* const *>(inputs[0].data<uint8_t>());
        #else
        auto tensor = inputs[0];
        #endif

        //std::cerr << "Pointer to tensor from op evaluate: " << tensor << "\n";
        Shape input_shape = tensor->get_shape();
        const std::string* input_strings = tensor->data<std::string>();
        std::cerr << "input_shape = " << input_shape << "\n";
        //std::cerr << data << "\n";

        auto nelements = shape_size(input_shape);
        size_t total = 0;
        for(size_t i = 0; i < nelements; ++i)
            total += input_strings[i].length();

        outputs[0].set_shape(input_shape);
        outputs[1].set_shape(input_shape);
        outputs[2].set_shape(Shape{total});

        auto begins = outputs[0].data<int32_t>();
        auto ends = outputs[1].data<int32_t>();
        auto output_symbols = reinterpret_cast<char*>(outputs[2].data<uint8_t>());
        size_t offset = 0;

        for(size_t i = 0; i < nelements; ++i)
        {
            begins[i] = offset;
            output_symbols = std::copy(input_strings[i].begin(), input_strings[i].end(), output_symbols);
            offset += input_strings[i].length();
            ends[i] = offset;
        }

        return true;

#else

        OPENVINO_ASSERT(false, "StringTensorUnpack supporst only element::string representation");
        return false;

#endif
    }

    std::string m_mode;
};

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

#ifndef USE_STRING_TENSORS
    // Override type of input tensor if this is a Parameter
    if (auto parameter = std::dynamic_pointer_cast<Parameter>(inputs.get_node_shared_ptr())) {
        parameter->set_partial_shape(PartialShape{ Dimension() });
        parameter->set_element_type(element::u8);
        parameter->validate_and_infer_types();
    }
#endif

#ifdef SENTENCE_PIECE_EXTENSION_DECOMPOSED_STRINGS

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
