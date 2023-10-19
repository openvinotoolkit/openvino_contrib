// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>

namespace sentencepiece {
    class SentencePieceProcessor;
}

namespace TemplateExtension {
    class SentencepieceTokenizer : public ov::op::Op {
    public:
        OPENVINO_OP("SentencepieceTokenizer");

        SentencepieceTokenizer() = default;
        SentencepieceTokenizer(const ov::OutputVector& args, int32_t nbest_size, float alpha, bool add_bos, bool add_eos, bool reverse);
        SentencepieceTokenizer(const ov::OutputVector& args, const std::shared_ptr<sentencepiece::SentencePieceProcessor>& sp, int32_t nbest_size, float alpha,
            bool add_bos, bool add_eos, bool reverse);

        bool visit_attributes(ov::AttributeVisitor& visitor) override;

        void validate_and_infer_types() override;

        std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

        bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

        bool has_evaluate() const override;

    private:
        std::shared_ptr<sentencepiece::SentencePieceProcessor> m_sp;
        int32_t m_nbest_size;
        float m_alpha;
        bool m_add_bos;
        bool m_add_eos;
        bool m_reverse;
    };


    class SentencepieceDetokenizer : public ov::op::Op {
    public:
        OPENVINO_OP("SentencepieceDetokenizer");

        SentencepieceDetokenizer() = default;
        SentencepieceDetokenizer(const ov::OutputVector& args);
        SentencepieceDetokenizer(const ov::OutputVector& args,
                                 const std::shared_ptr<sentencepiece::SentencePieceProcessor>& sp);

        bool visit_attributes(ov::AttributeVisitor& visitor) override;

        void validate_and_infer_types() override;

        std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

        bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

        bool has_evaluate() const override;

    private:
        std::shared_ptr<sentencepiece::SentencePieceProcessor> m_sp;
    };


    class SentencepieceStreamDetokenizer : public ov::op::Op {
    public:
        OPENVINO_OP("SentencepieceStreamDetokenizer");

        SentencepieceStreamDetokenizer() = default;
        SentencepieceStreamDetokenizer(const ov::OutputVector& args);
        SentencepieceStreamDetokenizer(const ov::OutputVector& args,
                                 const std::shared_ptr<sentencepiece::SentencePieceProcessor>& sp);

        bool visit_attributes(ov::AttributeVisitor& visitor) override;

        void validate_and_infer_types() override;

        std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

        bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

        bool has_evaluate() const override;

    private:
        std::shared_ptr<sentencepiece::SentencePieceProcessor> m_sp;
    };
}  // namespace TemplateExtension
