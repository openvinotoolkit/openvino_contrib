// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string_view>

namespace ov {
namespace gfx_plugin {

enum class RuntimeShapeRuleKind {
    Unsupported,
    StaticOrDescriptor,
    Concat,
    Broadcast,
    Select,
    ShapeOf,
    Slice,
    Range,
    Tile,
};

std::string_view runtime_shape_rule_name(RuntimeShapeRuleKind kind) noexcept;

RuntimeShapeRuleKind runtime_shape_rule_kind_from_name(
    std::string_view runtime_shape_rule) noexcept;

bool runtime_shape_rule_known(std::string_view runtime_shape_rule) noexcept;

bool runtime_shape_rule_requires_materializer(RuntimeShapeRuleKind kind) noexcept;

bool runtime_shape_rule_requires_materializer(
    std::string_view runtime_shape_rule) noexcept;

bool descriptor_owns_runtime_shape_rule(std::string_view op_family,
                                        std::string_view runtime_shape_rule) noexcept;

}  // namespace gfx_plugin
}  // namespace ov
