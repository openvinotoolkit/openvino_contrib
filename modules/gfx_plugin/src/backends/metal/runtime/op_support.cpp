// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_support.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"
#include "transforms/gfx_llm_ops.hpp"

namespace ov {
namespace gfx_plugin {

bool metal_supports_node(const std::shared_ptr<const ov::Node>& node) {
    if (ov::is_type<const ov::op::v1::Split>(node) ||
        ov::is_type<const ov::op::v1::VariadicSplit>(node)) {
        return node->get_input_size() >= 2 &&
               ov::is_type<const ov::op::v0::Constant>(node->input_value(1).get_node_shared_ptr()) &&
               (ov::is_type<const ov::op::v1::Split>(node) ||
                (node->get_input_size() >= 3 &&
                 ov::is_type<const ov::op::v0::Constant>(node->input_value(2).get_node_shared_ptr())));
    }
    if (ov::is_type<const ov::op::v13::ScaledDotProductAttention>(node)) {
        const auto et = node->get_output_element_type(0);
        const auto q_rank = node->get_input_partial_shape(0).rank();
        const auto k_rank = node->get_input_partial_shape(1).rank();
        const auto v_rank = node->get_input_partial_shape(2).rank();
        return (et == ov::element::f16 || et == ov::element::f32) &&
               q_rank.is_static() && q_rank.get_length() == 4 &&
               k_rank.is_static() && k_rank.get_length() == 4 &&
               v_rank.is_static() && v_rank.get_length() == 4 &&
               node->get_input_size() >= 3 && node->get_input_size() <= 5;
    }
    if (ov::is_type<const ov::gfx_plugin::op::GfxSDPAWithCausalMask>(node)) {
        const auto et = node->get_output_element_type(0);
        const auto q_rank = node->get_input_partial_shape(0).rank();
        const auto k_rank = node->get_input_partial_shape(1).rank();
        const auto v_rank = node->get_input_partial_shape(2).rank();
        const auto mask_rank = node->get_input_partial_shape(3).rank();
        const auto pos_rank = node->get_input_partial_shape(4).rank();
        return (et == ov::element::f16 || et == ov::element::f32) &&
               q_rank.is_static() && q_rank.get_length() == 4 &&
               k_rank.is_static() && k_rank.get_length() == 4 &&
               v_rank.is_static() && v_rank.get_length() == 4 &&
               mask_rank.is_static() && mask_rank.get_length() == 2 &&
               pos_rank.is_static() && pos_rank.get_length() == 1 &&
               node->get_input_element_type(3) == ov::element::i64 &&
               node->get_input_element_type(4) == ov::element::i64 &&
               node->get_input_size() == 6;
    }
    return mlir_supports_node(node);
}

}  // namespace gfx_plugin
}  // namespace ov
