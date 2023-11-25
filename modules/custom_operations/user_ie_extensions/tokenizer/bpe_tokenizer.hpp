// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>

#ifdef _MSC_VER
#    pragma warning(disable : 4251)
#    pragma warning(disable : 4275)
#endif

#include "fast_tokenizer/models/models.h"

using namespace paddlenlp::fast_tokenizer;

#undef tokenizer
#undef m_tokenizer

class BPETokenizer : public ov::op::Op {
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
    );
    BPETokenizer(
        const ov::OutputVector& arguments,
        const std::shared_ptr<models::BPE>& tokenizer,
        const std::string& unk_token = "",
        bool fuse_unk = false,
        const std::string& suffix_indicator = "",
        const std::string& end_suffix = "",
        bool byte_fallback = false
    );

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<BPETokenizer>(inputs, m_tokenizer, m_unk_token, m_fuse_unk, m_suffix_indicator, m_end_suffix, m_byte_fallback);
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

    bool has_evaluate() const override {
        return true;
    }

private:
    std::shared_ptr<models::BPE> m_tokenizer;
    std::string m_unk_token;
    bool m_fuse_unk = false;
    std::string m_suffix_indicator;
    std::string m_end_suffix;
    bool m_byte_fallback = false;
};
