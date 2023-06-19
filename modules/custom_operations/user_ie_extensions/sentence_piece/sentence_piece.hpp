// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>
#include <openvino/frontend/node_context.hpp>

namespace sentencepiece {
    class SentencePieceProcessor;
}

// Having a decomposed representation for a tensor, converts it to a single string tensor
// (packed u8 or natively supported element::string depending on whether or not USE_STRING_TENSORS defined).
class StringTensorPack : public ov::op::Op {
public:
    OPENVINO_OP("StringTensorPack");

    StringTensorPack () = default;

    StringTensorPack(ov::OutputVector inputs, const std::string& mode = "begins_ends")
        : ov::op::Op(inputs), m_mode(mode) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        auto result = std::make_shared<StringTensorPack>(inputs, m_mode);
        return result;
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("mode", m_mode);
        return true;
    }

    bool has_evaluate() const {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const;

private:

    std::string m_mode = "begins_ends";
};


// Having a decomposed representation for a tensor, converts it to a single string tensor for debugging purposes and to facilitate model conversion
// Base tensor on which this operation builds a ragged tensor can have any shape or type, this operation doesn't try to interpret it.
class RaggedTensorPack : public ov::op::Op {
public:
    OPENVINO_OP("RaggedTensorPack");

    RaggedTensorPack () = default;

    RaggedTensorPack(ov::OutputVector inputs)
        : ov::op::Op(inputs) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        auto result = std::make_shared<RaggedTensorPack>(inputs);
        return result;
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool has_evaluate() const {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const;
};


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

    StringTensorUnpack () = default;

    StringTensorUnpack(ov::OutputVector inputs, const std::string& mode = "begins_ends")
        : ov::op::Op(inputs), m_mode(mode) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        auto result = std::make_shared<StringTensorUnpack>(inputs, m_mode);
        return result;
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("mode", m_mode);
        return true;
    }

    bool has_evaluate() const {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const;

private:

    std::string m_mode = "begins_ends";
};



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
}  // namespace TemplateExtension

ov::OutputVector translate_sentencepiece_op(const ov::frontend::NodeContext& node);

ov::frontend::NamedOutputVector translate_sentencepiece_tokenizer(const ov::frontend::NodeContext& node);


class OPENVINO_API CaseFold  : public ov::op::Op {
public:
    OPENVINO_OP("CaseFold");

    CaseFold () = default;

    CaseFold (const ov::OutputVector& arguments) : ov::op::Op(arguments) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CaseFold >(inputs);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const {
        return true;
    }
};


ov::OutputVector translate_case_fold_utf8(const ov::frontend::NodeContext& node);


class OPENVINO_API NormalizeUnicode : public ov::op::Op {
public:
    OPENVINO_OP("NormalizeUnicode");

    NormalizeUnicode () = default;

    NormalizeUnicode(const ov::OutputVector& arguments, const std::string& normalization_form = "NFD") :
        ov::op::Op(arguments),
        m_normalization_form(normalization_form) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<NormalizeUnicode>(inputs, m_normalization_form);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("normalization_form", m_normalization_form);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const {
        return true;
    }

private:

    std::string m_normalization_form = "NFD";
};

ov::OutputVector translate_normalize_utf8(const ov::frontend::NodeContext& node);


class OPENVINO_API RegexNormalization : public ov::op::Op {
public:
    OPENVINO_OP("RegexNormalization");

    RegexNormalization () = default;

    RegexNormalization(const ov::OutputVector& arguments) :
        ov::op::Op(arguments) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<RegexNormalization>(inputs);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const {
        return true;
    }
};

ov::OutputVector translate_static_regex_replace(const ov::frontend::NodeContext& node);

class OPENVINO_API RegexSplit : public ov::op::Op {
public:
    OPENVINO_OP("RegexSplit");

    RegexSplit () = default;

    RegexSplit(const ov::OutputVector& arguments, const std::string& behaviour = "removed", bool invert = false) :
        ov::op::Op(arguments),
        m_behaviour(behaviour),
        m_invert(invert) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<RegexSplit>(inputs, m_behaviour, m_invert);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("behaviour", m_behaviour);
        visitor.on_attribute("invert", m_invert);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const {
        return true;
    }

private:

    std::string m_behaviour = "remove";
    bool m_invert = false;
};

ov::OutputVector translate_regex_split_with_offsets(const ov::frontend::NodeContext& node);

class OPENVINO_API WordpieceTokenizer : public ov::op::Op {
public:
    OPENVINO_OP("WordpieceTokenizer");

    WordpieceTokenizer () = default;

    WordpieceTokenizer(const ov::OutputVector& arguments, const std::string& suffix_indicator = "##", int max_bytes_per_word = 100) :
        ov::op::Op(arguments),
        m_suffix_indicator(suffix_indicator),
        m_max_bytes_per_word(max_bytes_per_word) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<WordpieceTokenizer>(inputs, m_suffix_indicator, m_max_bytes_per_word);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("suffix_indicator", m_suffix_indicator);
        visitor.on_attribute("max_bytes_per_word", m_max_bytes_per_word);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const {
        return true;
    }

private:

    std::string m_suffix_indicator = "##";
    int m_max_bytes_per_word = 100;   // TODO: Can it be done outside the op as preprocessing of the input?
};

ov::OutputVector translate_wordpiece_tokenize_with_offsets(const ov::frontend::NodeContext& node);
ov::OutputVector translate_lookup_table_find_v2(const ov::frontend::NodeContext& node);

class OPENVINO_API BPETokenizer : public ov::op::Op {
public:
    OPENVINO_OP("BPETokenizer");

    BPETokenizer () = default;

    BPETokenizer(
        const ov::OutputVector& arguments,
        const std::string& unk_token = "",
        bool fuse_unk = false,
        const std::string& suffix_indicator = "",
        const std::string& end_suffix = "",
        bool byte_fallback = false
    ) :
        ov::op::Op(arguments),
        m_unk_token(unk_token),
        m_fuse_unk(fuse_unk),
        m_suffix_indicator(suffix_indicator),
        m_end_suffix(end_suffix),
        m_byte_fallback(byte_fallback) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<BPETokenizer>(inputs, m_unk_token, m_fuse_unk, m_suffix_indicator, m_end_suffix, m_byte_fallback);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("unk_token", m_unk_token);
        visitor.on_attribute("fuse_unk", m_fuse_unk);
        visitor.on_attribute("suffix_indicator", m_suffix_indicator);
        visitor.on_attribute("end_suffix", m_end_suffix);
        visitor.on_attribute("byte_fallback", m_byte_fallback);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const {
        return true;
    }

private:
    std::string m_unk_token;
    bool m_fuse_unk = false;
    std::string m_suffix_indicator;
    std::string m_end_suffix;
    bool m_byte_fallback = false;
};


class OPENVINO_API CombineSegments : public ov::op::Op {
public:
    OPENVINO_OP("CombineSegments");

    CombineSegments () = default;

    CombineSegments(const ov::OutputVector& arguments) :
        ov::op::Op(arguments) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CombineSegments>(inputs);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const {
        return true;
    }
};

// Takes a ragged tensor with one ragged right-most dimension and produces a normal tensor
class OPENVINO_API RaggedToDense : public ov::op::Op {
public:
    OPENVINO_OP("RaggedToDense");

    RaggedToDense () = default;

    RaggedToDense(const ov::OutputVector& arguments) :
        ov::op::Op(arguments) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<RaggedToDense>(inputs);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const {
        return true;
    }
};

ov::OutputVector translate_reshape(const ov::frontend::NodeContext& node);
ov::OutputVector translate_const(const ov::frontend::NodeContext& node);
