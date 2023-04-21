// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "normalizer.h"
#include "sentence_piece.hpp"

#include "openvino/opsets/opset10.hpp"

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
    FRONT_END_GENERAL_CHECK(args.size() == 2, "SentencepieceTokenizer expects two inputs: sp model and input sentences");
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

    FRONT_END_GENERAL_CHECK(inputs.size() == 2, "SentencepieceTokenizer expects two inputs: sp model and input sentences");

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

    size_t max_token_id = 0;
    for (size_t batch_ind = 0; batch_ind < batch_size; ++batch_ind) {
        auto begin_ind = begin_ids[batch_ind];
        auto end_ind = end_ids[batch_ind];
        std::vector<int32_t> ids;
        std::string sentence(data + begin_ind, data + end_ind);
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

    OutputVector inputs_vector = OutputVector{ sp_model_const, inputs };

    // Override type of input tensor if this is a Parameter
    if (auto parameter = std::dynamic_pointer_cast<Parameter>(inputs.get_node_shared_ptr())) {
        parameter->set_partial_shape(PartialShape{ Dimension() });
        parameter->set_element_type(element::u8);
        parameter->validate_and_infer_types();
    }

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
