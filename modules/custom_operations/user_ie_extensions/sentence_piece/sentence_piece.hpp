// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>
#include <openvino/frontend/node_context.hpp>

namespace sentencepiece {
    class SentencePieceProcessor;
}

namespace TemplateExtension {
    class SentencepieceTokenizer : public ov::op::Op {
    public:
        OPENVINO_OP("SentencepieceTokenizer");

        SentencepieceTokenizer() = default;
        SentencepieceTokenizer(const ov::OutputVector& args, const std::vector<char>& sp_model, int32_t nbest_size, float alpha,
            bool add_bos, bool add_eos, bool reverse);

        void validate_and_infer_types() override;

        std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

        bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

        bool has_evaluate() const override;

    private:
        std::shared_ptr<sentencepiece::SentencePieceProcessor> m_sp;
        std::vector<char> m_sp_model;
        int32_t m_nbest_size;
        float m_alpha;
        bool m_add_bos;
        bool m_add_eos;
        bool m_reverse;
    };
}  // namespace TemplateExtension

ov::OutputVector translate_sentencepiece_op(const ov::frontend::NodeContext& node);

ov::OutputVector translate_sentencepiece_tokenizer(const ov::frontend::NodeContext& node);
