// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace gfx_plugin {
namespace op {

class GfxSDPAWithCausalMask final : public ov::op::Op {
public:
    OPENVINO_OP("GfxSDPAWithCausalMask", "gfx_plugin_opset", ov::op::Op);

    GfxSDPAWithCausalMask() = default;

    explicit GfxSDPAWithCausalMask(const ov::OutputVector& args)
        : ov::op::Op(args) {
        constructor_validate_and_infer_types();
    }

    bool visit_attributes(ov::AttributeVisitor& /*visitor*/) override {
        return true;
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        check_new_args_count(this, new_args);
        return std::make_shared<GfxSDPAWithCausalMask>(new_args);
    }

    void validate_and_infer_types() override {
        NODE_VALIDATION_CHECK(this,
                              get_input_size() == 6,
                              "GfxSDPAWithCausalMask expects Q, K, V, attention_mask, cache_positions and scale");

        const auto& q_pshape = get_input_partial_shape(0);
        const auto& v_pshape = get_input_partial_shape(2);
        ov::PartialShape out = q_pshape;
        if (out.rank().is_static() && v_pshape.rank().is_static() &&
            out.rank().get_length() == 4 && v_pshape.rank().get_length() == 4) {
            out[3] = v_pshape[3];
        }
        set_output_type(0, get_input_element_type(0), out);
    }
};

}  // namespace op
}  // namespace gfx_plugin
}  // namespace ov
