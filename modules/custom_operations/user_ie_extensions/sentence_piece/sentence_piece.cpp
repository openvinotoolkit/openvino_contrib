// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builder.h"
#include "common.h"
#include "filesystem.h"
#include "init.h"
#include "normalizer.h"
#include "sentencepiece.pb.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "sentencepiece_trainer.h"
#include "third_party/absl/flags/flag.h"
#include "sentence_piece.hpp"

#include "openvino/frontend/decoder.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/op/util/framework_node.hpp"

using sentencepiece::ModelProto;
using sentencepiece::NormalizerSpec;
using sentencepiece::SentencePieceProcessor;
using sentencepiece::SentencePieceTrainer;
using sentencepiece::normalizer::Builder;
using sentencepiece::normalizer::Normalizer;
using namespace TemplateExtension;
using namespace ov;
using namespace ov::opset10;

namespace {
    bool evaluate_helper(const ov::TensorVector& inputs,
        std::vector<int32_t>& sparse_indices,
        std::vector<int32_t>& sparse_values,
        std::vector<int32_t>& sparse_dense_shape) {
        // inputs should have at least 3 tensors for input strings
        // [0] i32 tensor of begin indices, indices are offsets in [2]
        // [1] i32 tensor of end indices, indices are offsets in [2]
        // [2] 1D u8 tensor of bytes where all strings are concatenated

        // the operation has the following inputs:
        // 0. spm_model
        // data inputs
        // 1. [0] i32 tensor of begin indices, indices are offsets in [2]
        // 2. [1] i32 tensor of end indices, indices are offsets in [2]
        // 3. [2] 1D u8 tensor of bytes where all strings are concatenated
        // 4. nbest_size
        // 5. alpha
        // 6. add_bos
        // 7. add_eos
        // 8. reverse
        auto spm_model = static_cast<char*>(inputs[0].data());
        auto spm_model_size = inputs[0].get_byte_size();

        const uint8_t* strings = inputs[1].data<uint8_t>();
        auto batch_size = *reinterpret_cast<const int32_t*>(strings + 0);
        auto begin_ids = reinterpret_cast<const int32_t*>(strings + 4);
        auto end_ids = begin_ids + 1;
        auto data = strings + 4 + 4 + 4*batch_size;

        auto nbest_size = *static_cast<int32_t*>(inputs[2].data());
        auto alpha = *static_cast<float*>(inputs[3].data());
        auto add_bos = *static_cast<bool*>(inputs[4].data());
        auto add_eos = *static_cast<bool*>(inputs[5].data());
        auto reverse = *static_cast<bool*>(inputs[6].data());

        SentencePieceProcessor sp;
        std::string model_proto(spm_model, spm_model_size);
        CHECK_OK(sp.LoadFromSerializedProto(model_proto));

        // form extra options to configure SentencePieceProcessor
        std::string extra_options = "";
        if (add_bos) {
            extra_options += "bos";
        }
        if (add_eos) {
            extra_options = extra_options.empty() ? extra_options : extra_options + ":";
            extra_options += "eos";
        }
        /* TODO: TF ignores this option, so we are ignoring it as well; need to understand what should we do
        if (reverse) {
            extra_options = extra_options.empty() ? extra_options : extra_options + ":";
            extra_options += "reverse";
        }
        */
        // example of extra_options, if "bos:eos:reverse"
        CHECK_OK(sp.SetEncodeExtraOptions(extra_options));

        size_t max_token_id = 0;
        for (size_t batch_ind = 0; batch_ind < batch_size; ++batch_ind) {
            auto begin_ind = begin_ids[batch_ind];
            auto end_ind = end_ids[batch_ind];
            std::vector<int32_t> ids;
            std::string sentence(data + begin_ind, data + end_ind);
            CHECK_OK(sp.SampleEncode(sentence, nbest_size, alpha, &ids));
            // put into resulted vectors
            for (size_t token_id = 0; token_id < ids.size(); ++token_id) {
                sparse_indices.push_back(static_cast<int32_t>(batch_ind));
                sparse_indices.push_back(static_cast<int32_t>(token_id));
                sparse_values.push_back(static_cast<int32_t>(ids[token_id]));
            }
            max_token_id = max_token_id < ids.size() ? ids.size() : max_token_id;
        }
        sparse_dense_shape.push_back(static_cast<int32_t>(batch_size));
        sparse_dense_shape.push_back(static_cast<int32_t>(max_token_id));

        return true;
    }

}  // namespace

SentencepieceTokenizer::SentencepieceTokenizer(const ov::OutputVector& args)
    : Op(args) {
    constructor_validate_and_infer_types();
}

void SentencepieceTokenizer::validate_and_infer_types() {
    // The operation SentencepieceTokenizerExtensionOp has three outputs: sparse indices, sparse values
    // and dense shape
    set_output_type(0, element::i32, PartialShape{ Dimension(), Dimension(2) });  // FIXME: change to i64 after CPU fix
    set_output_type(1, element::i32, PartialShape{ Dimension() });
    set_output_type(2, element::i32, PartialShape{ Dimension(2) });  // FIXME: change to i64 after CPU fix
}

bool SentencepieceTokenizer::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    std::vector<int32_t> m_sparse_indices;
    std::vector<int32_t> m_sparse_values;
    std::vector<int32_t> m_sparse_dense_shape;

    evaluate_helper(inputs, m_sparse_indices, m_sparse_values, m_sparse_dense_shape);

    outputs[0].set_shape({ m_sparse_indices.size() / 2, 2 });
    memcpy(outputs[0].data(), m_sparse_indices.data(), sizeof(int32_t) * m_sparse_indices.size());
    outputs[1].set_shape({ m_sparse_values.size() });
    memcpy(outputs[1].data(), m_sparse_values.data(), sizeof(int32_t) * m_sparse_values.size());
    outputs[2].set_shape({ 2 });
    memcpy(outputs[2].data(), m_sparse_dense_shape.data(), sizeof(int32_t) * m_sparse_dense_shape.size());
    return true;
}

bool SentencepieceTokenizer::has_evaluate() const {
    return true;
}

std::shared_ptr<ov::Node> SentencepieceTokenizer::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<SentencepieceTokenizer>(new_args);
}

OutputVector translate_sentencepiece_op(const ov::frontend::NodeContext& node) {
    // extract model to configure SentencePieceTokenizer
    auto sp_model_ov_any = node.get_attribute_as_any("model");
    FRONT_END_GENERAL_CHECK(sp_model_ov_any.is<std::string>(),
        "SentencePieceOp configuration model is in incorrect format");
    auto str_spm_model = sp_model_ov_any.as<std::string>();
    auto sp_model_const = std::make_shared<Constant>(element::u8, Shape{ str_spm_model.size() }, str_spm_model.data());
    return { sp_model_const };
}

OutputVector translate_sentencepiece_tokenizer(const ov::frontend::NodeContext& node) {
    // this is custom translator that converts a sub-graph with SentencePieceOp, SentencePieceTokenizer,
    // and RaggedTensorToSparse operation- into a custom operation SentencepieceTokenizerExtensionOp
    FRONT_END_GENERAL_CHECK(node.get_input_size() > 0, "RaggedTensorToSparse expects at least one input.");

    // check that producers of RaggedTensorToSparse is SentencePieceTokenizer
    auto sp_tokenize_op = node.get_input(0).get_node_shared_ptr();
    FRONT_END_GENERAL_CHECK(sp_tokenize_op->get_input_size() > 6,
        "SentencepieceTokenizeOp expects at least six inputs");

    // prepare inputs that go to custom operation
    // prepare input 0 - SentencePieceTokenizer configuration model
    auto sp_model_const = sp_tokenize_op->input_value(0).get_node_shared_ptr();

    // prepare input six inputs
    auto inputs = sp_tokenize_op->input_value(1);
    auto nbest_size = sp_tokenize_op->input_value(2);
    auto alpha = sp_tokenize_op->input_value(3);
    auto add_bos = sp_tokenize_op->input_value(4);
    auto add_eos = sp_tokenize_op->input_value(5);
    auto reverse = sp_tokenize_op->input_value(6);

    OutputVector inputs_vector = OutputVector{ sp_model_const, inputs, nbest_size, alpha, add_bos, add_eos, reverse };

    // Override type of input tensor if this is a Parameter
    if(auto parameter = std::dynamic_pointer_cast<ov::opset10::Parameter>(inputs.get_node_shared_ptr())) {
        parameter->set_partial_shape(ov::PartialShape{Dimension()});
        parameter->set_element_type(ov::element::u8);
        parameter->validate_and_infer_types();
    }

    // create a node with custom operation
    auto sp_tokenizer_ext = std::make_shared<SentencepieceTokenizer>(inputs_vector);

    return sp_tokenizer_ext->outputs();
}
