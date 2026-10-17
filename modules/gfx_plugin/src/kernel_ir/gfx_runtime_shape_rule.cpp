// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/gfx_runtime_shape_rule.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

constexpr std::string_view kStaticOrDescriptor = "static_or_descriptor";
constexpr std::string_view kConcat = "concat";
constexpr std::string_view kBroadcast = "broadcast";
constexpr std::string_view kSelect = "select";
constexpr std::string_view kShapeOf = "shape_of";
constexpr std::string_view kSlice = "slice";
constexpr std::string_view kRange = "range";
constexpr std::string_view kTile = "tile";

}  // namespace

std::string_view runtime_shape_rule_name(RuntimeShapeRuleKind kind) noexcept {
    switch (kind) {
    case RuntimeShapeRuleKind::StaticOrDescriptor:
        return kStaticOrDescriptor;
    case RuntimeShapeRuleKind::Concat:
        return kConcat;
    case RuntimeShapeRuleKind::Broadcast:
        return kBroadcast;
    case RuntimeShapeRuleKind::Select:
        return kSelect;
    case RuntimeShapeRuleKind::ShapeOf:
        return kShapeOf;
    case RuntimeShapeRuleKind::Slice:
        return kSlice;
    case RuntimeShapeRuleKind::Range:
        return kRange;
    case RuntimeShapeRuleKind::Tile:
        return kTile;
    case RuntimeShapeRuleKind::Unsupported:
        return {};
    }
    return {};
}

RuntimeShapeRuleKind runtime_shape_rule_kind_from_name(
    std::string_view runtime_shape_rule) noexcept {
    if (runtime_shape_rule == kStaticOrDescriptor) {
        return RuntimeShapeRuleKind::StaticOrDescriptor;
    }
    if (runtime_shape_rule == kConcat) {
        return RuntimeShapeRuleKind::Concat;
    }
    if (runtime_shape_rule == kBroadcast) {
        return RuntimeShapeRuleKind::Broadcast;
    }
    if (runtime_shape_rule == kSelect) {
        return RuntimeShapeRuleKind::Select;
    }
    if (runtime_shape_rule == kShapeOf) {
        return RuntimeShapeRuleKind::ShapeOf;
    }
    if (runtime_shape_rule == kSlice) {
        return RuntimeShapeRuleKind::Slice;
    }
    if (runtime_shape_rule == kRange) {
        return RuntimeShapeRuleKind::Range;
    }
    if (runtime_shape_rule == kTile) {
        return RuntimeShapeRuleKind::Tile;
    }
    return RuntimeShapeRuleKind::Unsupported;
}

bool runtime_shape_rule_known(std::string_view runtime_shape_rule) noexcept {
    return runtime_shape_rule_kind_from_name(runtime_shape_rule) !=
           RuntimeShapeRuleKind::Unsupported;
}

bool runtime_shape_rule_requires_materializer(RuntimeShapeRuleKind kind) noexcept {
    return kind != RuntimeShapeRuleKind::Unsupported &&
           kind != RuntimeShapeRuleKind::StaticOrDescriptor;
}

bool runtime_shape_rule_requires_materializer(
    std::string_view runtime_shape_rule) noexcept {
    return runtime_shape_rule_requires_materializer(
        runtime_shape_rule_kind_from_name(runtime_shape_rule));
}

bool descriptor_owns_runtime_shape_rule(std::string_view op_family,
                                        std::string_view runtime_shape_rule) noexcept {
    switch (runtime_shape_rule_kind_from_name(runtime_shape_rule)) {
    case RuntimeShapeRuleKind::StaticOrDescriptor:
        return true;
    case RuntimeShapeRuleKind::Concat:
        return op_family == "Concat";
    case RuntimeShapeRuleKind::Broadcast:
        return op_family == "Broadcast";
    case RuntimeShapeRuleKind::Select:
        return op_family == "Select";
    case RuntimeShapeRuleKind::ShapeOf:
        return op_family == "ShapeOf";
    case RuntimeShapeRuleKind::Slice:
        return op_family == "Slice" || op_family == "StridedSlice";
    case RuntimeShapeRuleKind::Range:
        return op_family == "Range";
    case RuntimeShapeRuleKind::Tile:
        return op_family == "Tile";
    case RuntimeShapeRuleKind::Unsupported:
        return false;
    }
    return false;
}

}  // namespace gfx_plugin
}  // namespace ov
