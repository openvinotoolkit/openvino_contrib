// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>

#include "normalizer.h"
#include "model_interface.h"

#include "openvino/op/util/framework_node.hpp"
#include "openvino/opsets/opset10.hpp"

#include "sentence_piece.hpp"
#include "utils.hpp"

using sentencepiece::SentencePieceProcessor;
using sentencepiece::util::Status;
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
    int32_t nbest_size, float alpha, bool add_bos, bool add_eos, bool reverse) :
    m_sp((sp == nullptr) ? std::make_shared<SentencePieceProcessor>(): sp),
    m_nbest_size(nbest_size), m_alpha(alpha), m_add_bos(add_bos), m_add_eos(add_eos),
    m_reverse(reverse), Op(args) {
    // constructor above without sp argument never called when the node is created with python factory, so need to init and cache m_sp here
    if (!m_sp->status().ok()) {
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
        if (m_reverse) {
            extra_options = extra_options.empty() ? extra_options : extra_options + ":";
            extra_options += "reverse";
        }
        // example of extra_options, if "bos:eos:reverse"
        CHECK_OK(m_sp->SetEncodeExtraOptions(extra_options));
    };
    constructor_validate_and_infer_types();
}

void SentencepieceTokenizer::validate_and_infer_types() {
    FRONT_END_GENERAL_CHECK(get_input_element_type(0) == element::u8, "SentencepieceTokenizer accepts sp model as the first input and it should be of type u8 tensor");

    auto input_size = get_input_size();
    if(input_size == 2) {
        FRONT_END_GENERAL_CHECK(
            // WA: f32 appeared as a placeholder for unknown type during intermediate conversion steps
            get_input_element_type(1) == element::string || get_input_element_type(1) == element::f32,
            "SentencepieceTokenizer accepts sentences as the second input and it should be of type string tensor");
    } else if (input_size == 4) {
        FRONT_END_GENERAL_CHECK(get_input_element_type(1) == element::i32, "SentencepieceTokenizer accepts begins offsets as the second and it should be of type i32 tensor");
        FRONT_END_GENERAL_CHECK(get_input_element_type(2) == element::i32, "SentencepieceTokenizer accepts ends offsets as the third and it should be of type i32 tensor");
        FRONT_END_GENERAL_CHECK(get_input_element_type(3) == element::u8, "SentencepieceTokenizer accepts sentence symbols as the fourth input and it should be of type u8 tensor");
    } else {
        OPENVINO_THROW("Unexpected input format. SentencepieceTokenizer accepts one string input or three decomposed string inputs (begins, ends, symbols)");
    };

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

    auto input_size = get_input_size();
    int32_t batch_size;

    // used in case of string tensors
    const std::string* strings;

    // used in case of u8 packed representation
    const int32_t* begin_ids;
    const int32_t* end_ids;
    const uint8_t* data;

    if (input_size == 2) {
        auto input_element_type = get_input_element_type(1);
        if(input_element_type == ov::element::string) {
            strings = inputs[1].data<const std::string>();
            batch_size = static_cast<int32_t>(ov::shape_size(inputs[1].get_shape()));
        } else {
            OPENVINO_THROW("Unexpected input type during inference. SentencepieceTokenizer accepts element::u8 or element::string.");
        }
    } else {
        auto begin_ids = inputs[1].data<const int32_t>();
        auto end_ids = inputs[2].data<const int32_t>();
        auto data = inputs[3].data<const uint8_t>();
        batch_size = shape_size(inputs[1].get_shape());
    };

    size_t max_token_id = 0;
    for (size_t batch_ind = 0; batch_ind < batch_size; ++batch_ind) {
        absl::string_view sentence;
        if (input_size == 2) {
            sentence = strings[batch_ind];
        } else {
            auto begin_ind = begin_ids[batch_ind];
            auto end_ind = end_ids[batch_ind];
            sentence = absl::string_view((const char*)data + begin_ind, end_ind - begin_ind);
        };

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


// Detokenizer

SentencepieceDetokenizer::SentencepieceDetokenizer(const OutputVector& args) :
    m_sp(std::make_shared<SentencePieceProcessor>()), Op(args) {
    auto sp_model_const = as_type_ptr<Constant>(args[0].get_node_shared_ptr());
    OPENVINO_ASSERT(sp_model_const, "SentencepieceDetokenizer expects SentencePiece model to be constant.");
    auto spm_model = static_cast<const char*>(sp_model_const->get_data_ptr());
    auto spm_model_size = sp_model_const->get_byte_size();

    // configure SentencePieceProcessor
    std::string model_proto(spm_model, spm_model_size);
    CHECK_OK(m_sp->LoadFromSerializedProto(model_proto));
    constructor_validate_and_infer_types();
}

SentencepieceDetokenizer::SentencepieceDetokenizer(const OutputVector& args, const std::shared_ptr<sentencepiece::SentencePieceProcessor>& sp) :
    m_sp((sp == nullptr) ? std::make_shared<SentencePieceProcessor>(): sp), Op(args) {
    // constructor above without sp argument never called when the node is created with python factory, so need to init and cache m_sp here
    if (!m_sp->status().ok()) {
        auto sp_model_const = as_type_ptr<Constant>(args[0].get_node_shared_ptr());
        OPENVINO_ASSERT(sp_model_const, "SentencepieceDetokenizer expects SentencePiece model to be constant.");
        auto spm_model = static_cast<const char*>(sp_model_const->get_data_ptr());
        auto spm_model_size = sp_model_const->get_byte_size();

        // configure SentencePieceProcessor
        std::string model_proto(spm_model, spm_model_size);
        CHECK_OK(m_sp->LoadFromSerializedProto(model_proto));
    };
    constructor_validate_and_infer_types();
}

void SentencepieceDetokenizer::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() == 2, "SentencepieceDetokenizer expects two inputs: sp model and token ids");
    OPENVINO_ASSERT(get_input_element_type(0) == element::u8, "SentencepieceDetokenizer accepts sp model as the first input and it should be of type u8 tensor");
    OPENVINO_ASSERT(get_input_partial_shape(1).size() == 2, "SentencepieceDetokenizer expects 2D tensor as second input");

    auto batch_size = PartialShape({get_input_partial_shape(1)[0]});
    set_string_output(this, 0, batch_size);
}

bool SentencepieceDetokenizer::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

bool SentencepieceDetokenizer::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    auto batch_size = inputs[1].get_shape()[0];
    auto seq_len    = inputs[1].get_shape()[1];
    auto input_data = inputs[1].data<const int32_t>();

    outputs[0].set_shape({batch_size});
    outputs[1].set_shape({batch_size});
    outputs[2].set_shape({batch_size * seq_len * 100});  // 100 chars - max token length

    auto begins = outputs[0].data<int32_t>();
    auto ends   = outputs[1].data<int32_t>();
    auto chars  = outputs[2].data<uint8_t>();
    uint32_t char_offset = 0;

    for(size_t batch = 0; batch < batch_size; ++batch) {
        auto start = batch * seq_len;

        std::vector<int32_t> token_ids(seq_len);
        std::memcpy(&token_ids[0], &input_data[start], sizeof(int32_t) * seq_len);

        std::string detokenized;
        CHECK_OK(m_sp->Decode(token_ids, &detokenized));
        std::copy(detokenized.begin(), detokenized.end(), &chars[char_offset]);

        begins[batch] = char_offset;
        char_offset += detokenized.size();
        ends[batch] = char_offset;
    }
    outputs[2].set_shape({char_offset});
    return true;
}

bool SentencepieceDetokenizer::has_evaluate() const {
    return true;
}

std::shared_ptr<Node> SentencepieceDetokenizer::clone_with_new_inputs(const OutputVector& new_args) const {
    return std::make_shared<SentencepieceDetokenizer>(new_args, m_sp);
}


// Stream Detokenizer

SentencepieceStreamDetokenizer::SentencepieceStreamDetokenizer(const OutputVector& args) :
    m_sp(std::make_shared<SentencePieceProcessor>()), Op(args) {
    auto sp_model_const = as_type_ptr<Constant>(args[0].get_node_shared_ptr());
    OPENVINO_ASSERT(sp_model_const, "SentencepieceDetokenizer expects SentencePiece model to be constant.");
    auto spm_model = static_cast<const char*>(sp_model_const->get_data_ptr());
    auto spm_model_size = sp_model_const->get_byte_size();

    // configure SentencePieceProcessor
    std::string model_proto(spm_model, spm_model_size);
    CHECK_OK(m_sp->LoadFromSerializedProto(model_proto));
    constructor_validate_and_infer_types();
}

SentencepieceStreamDetokenizer::SentencepieceStreamDetokenizer(const OutputVector& args, const std::shared_ptr<sentencepiece::SentencePieceProcessor>& sp) :
    m_sp((sp == nullptr) ? std::make_shared<SentencePieceProcessor>(): sp), Op(args) {
    // constructor above without sp argument never called when the node is created with python factory, so need to init and cache m_sp here
    if (!m_sp->status().ok()) {
        auto sp_model_const = as_type_ptr<Constant>(args[0].get_node_shared_ptr());
        OPENVINO_ASSERT(sp_model_const, "SentencepieceDetokenizer expects SentencePiece model to be constant.");
        auto spm_model = static_cast<const char*>(sp_model_const->get_data_ptr());
        auto spm_model_size = sp_model_const->get_byte_size();

        // configure SentencePieceProcessor
        std::string model_proto(spm_model, spm_model_size);
        CHECK_OK(m_sp->LoadFromSerializedProto(model_proto));
    };
    constructor_validate_and_infer_types();
}

void SentencepieceStreamDetokenizer::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() == 2, "SentencepieceDetokenizer expects two inputs: sp model and token ids");
    OPENVINO_ASSERT(get_input_element_type(0) == element::u8, "SentencepieceDetokenizer accepts sp model as the first input and it should be of type u8 tensor");
    OPENVINO_ASSERT(get_input_partial_shape(1).size() == 2, "SentencepieceDetokenizer expects 2D tensor as second input");

    auto batch_size = PartialShape({get_input_partial_shape(1)[0]});
    set_string_output(this, 0, batch_size);
}

bool SentencepieceStreamDetokenizer::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

bool SentencepieceStreamDetokenizer::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    auto batch_size = inputs[1].get_shape()[0];
    auto seq_len    = inputs[1].get_shape()[1];
    auto input_data = inputs[1].data<const int32_t>();

    outputs[0].set_shape({batch_size});
    outputs[1].set_shape({batch_size});
    outputs[2].set_shape({batch_size * seq_len * 100});  // 100 chars - max token length

    auto begins = outputs[0].data<int32_t>();
    auto ends   = outputs[1].data<int32_t>();
    auto chars  = outputs[2].data<uint8_t>();
    uint32_t char_offset = 0;

    for(size_t batch = 0; batch < batch_size; ++batch) {
        const auto start = batch * seq_len;

        begins[batch] = char_offset;
        for(size_t seq = start; seq < start + seq_len; ++seq) {
            const auto token_id = input_data[seq];
            const auto token = m_sp->IdToPiece(token_id);

            if(token.rfind("<") == 0 && token.rfind(">") == 5) {
                // convert "byte tokens" into bytes
                int ch = sentencepiece::PieceToByte(token);
                chars[char_offset++] = ch;
            } else {
                std::copy(token.begin(), token.end(), &chars[char_offset]);
                char_offset += token.size();
            };
        };
        ends[batch] = char_offset;
    }
    outputs[2].set_shape({char_offset});
    return true;
}

bool SentencepieceStreamDetokenizer::has_evaluate() const {
    return true;
}

std::shared_ptr<Node> SentencepieceStreamDetokenizer::clone_with_new_inputs(const OutputVector& new_args) const {
    return std::make_shared<SentencepieceStreamDetokenizer>(new_args, m_sp);
}
