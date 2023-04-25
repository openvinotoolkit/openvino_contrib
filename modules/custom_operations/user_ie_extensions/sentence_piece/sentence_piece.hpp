// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>
#include <openvino/frontend/node_context.hpp>


namespace TemplateExtension {

    class SentencepieceTokenizer : public ov::op::Op {
    public:
        OPENVINO_OP("SentencepieceTokenizer");

        SentencepieceTokenizer() = default;
        SentencepieceTokenizer(const ov::OutputVector& args);

        void validate_and_infer_types() override;

        std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

        bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

        bool has_evaluate() const override;

    };

}  // namespace TemplateExtension

ov::OutputVector translate_sentencepiece_op(const ov::frontend::NodeContext& node);

ov::frontend::NamedOutputVector translate_sentencepiece_tokenizer(const ov::frontend::NodeContext& node);
