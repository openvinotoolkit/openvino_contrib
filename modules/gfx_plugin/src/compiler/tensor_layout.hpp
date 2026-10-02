// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "openvino/core/node.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

enum class TensorLayoutKind {
    Unknown,
    Materialized,
    ViewOnly,
};

struct TensorLayoutPlan {
    TensorLayoutKind kind = TensorLayoutKind::Unknown;
    bool view_only = false;
};

TensorLayoutPlan select_tensor_layout_plan(const std::string& stage_type,
                                           const std::shared_ptr<const ov::Node>& node);
std::string_view tensor_layout_kind_to_string(TensorLayoutKind kind) noexcept;
TensorLayoutPlan tensor_layout_plan_from_contract(std::string_view layout_contract,
                                                  bool view_only_hint = false) noexcept;

}  // namespace compiler

using GfxTensorLayoutKind = compiler::TensorLayoutKind;
using GfxTensorLayoutPlan = compiler::TensorLayoutPlan;

}  // namespace gfx_plugin
}  // namespace ov
