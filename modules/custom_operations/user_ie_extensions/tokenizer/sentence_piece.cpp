// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>

#include "normalizer.h"

#include "openvino/op/util/framework_node.hpp"
#include "openvino/opsets/opset10.hpp"

#include "sentence_piece.hpp"
#include "utils.hpp"

using sentencepiece::SentencePieceProcessor;
using namespace TemplateExtension;
using namespace ov;
using namespace ov::frontend;
using namespace ov::opset10;

// TODO: Replace shape_size(t.get_shape()) by t.get_size(), where t is ov::Tensor

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