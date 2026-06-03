// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/tensor_layout.hpp"

#include <cstddef>

#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

bool is_identity_permutation(const std::shared_ptr<const ov::Node>& node) {
    auto transpose = ov::as_type_ptr<const ov::op::v1::Transpose>(node);
    if (!transpose || transpose->get_input_size() != 2) {
        return false;
    }
    auto perm_const =
        ov::as_type_ptr<const ov::op::v0::Constant>(transpose->input_value(1).get_node_shared_ptr());
    if (!perm_const) {
        return false;
    }
    const auto perm = perm_const->cast_vector<int64_t>();
    const auto& in_pshape = transpose->get_input_partial_shape(0);
    const auto& out_pshape = transpose->get_output_partial_shape(0);
    if (!in_pshape.rank().is_static() || !out_pshape.rank().is_static()) {
        return false;
    }
    const auto in_rank = static_cast<size_t>(in_pshape.rank().get_length());
    const auto out_rank = static_cast<size_t>(out_pshape.rank().get_length());
    if (perm.size() != in_rank || perm.size() != out_rank) {
        return false;
    }
    for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] != static_cast<int64_t>(i)) {
            return false;
        }
    }
    return true;
}

}  // namespace

std::string_view tensor_layout_kind_to_string(TensorLayoutKind kind) noexcept {
    switch (kind) {
        case TensorLayoutKind::Materialized:
            return "materialized";
        case TensorLayoutKind::ViewOnly:
            return "view_only";
        case TensorLayoutKind::Unknown:
        default:
            return "logical";
    }
}

TensorLayoutPlan tensor_layout_plan_from_contract(std::string_view layout_contract,
                                                  bool view_only_hint) noexcept {
    TensorLayoutPlan plan{};
    if (layout_contract == "view_only" || view_only_hint) {
        plan.kind = TensorLayoutKind::ViewOnly;
        plan.view_only = true;
        return plan;
    }
    if (layout_contract == "materialized") {
        plan.kind = TensorLayoutKind::Materialized;
        plan.view_only = false;
        return plan;
    }
    plan.kind = TensorLayoutKind::Unknown;
    plan.view_only = false;
    return plan;
}

TensorLayoutPlan select_tensor_layout_plan(const std::string& stage_type,
                                           const std::shared_ptr<const ov::Node>& node) {
    TensorLayoutPlan plan{};
    if (stage_type == "ReadValue") {
        plan.kind = TensorLayoutKind::Materialized;
        plan.view_only = false;
        return plan;
    }
    if (stage_type == "Reshape" || stage_type == "Squeeze" ||
        stage_type == "Unsqueeze") {
        plan.kind = TensorLayoutKind::ViewOnly;
        plan.view_only = true;
        return plan;
    }
    if (stage_type == "Transpose" && is_identity_permutation(node)) {
        plan.kind = TensorLayoutKind::ViewOnly;
        plan.view_only = true;
        return plan;
    }
    if (stage_type == "Transpose" || stage_type == "Reshape" || stage_type == "Squeeze" ||
        stage_type == "Unsqueeze") {
        plan.kind = TensorLayoutKind::Materialized;
    }
    return plan;
}

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
