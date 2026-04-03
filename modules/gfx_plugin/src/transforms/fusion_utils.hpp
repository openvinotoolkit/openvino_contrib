// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "runtime/gfx_activation.hpp"

namespace ov {
namespace gfx_plugin {
namespace fusion_utils {

inline bool has_single_user(mlir::Value value, mlir::Operation*& user) {
    if (!value || !value.hasOneUse()) {
        return false;
    }
    user = *value.getUsers().begin();
    return user != nullptr;
}

inline bool has_single_forward_user(mlir::Value value, mlir::Operation*& user) {
    if (!value) {
        return false;
    }
    user = nullptr;
    for (mlir::Operation* candidate : value.getUsers()) {
        if (!candidate || candidate->getName().getStringRef() == "func.return") {
            continue;
        }
        if (user && user != candidate) {
            user = nullptr;
            return false;
        }
        user = candidate;
    }
    return user != nullptr;
}

inline bool is_constant_like_value(mlir::Value value) {
    if (!value) {
        return false;
    }

    auto* def = value.getDefiningOp();
    if (!def) {
        return false;
    }

    const auto name = def->getName().getStringRef();
    if (name == "gfx.Constant") {
        return true;
    }

    // Allow trivial compile-time wrappers around constants so fusion still
    // triggers after common canonicalization inserts Convert/Reshape-style ops.
    const bool constant_wrapper = (name == "gfx.Convert" || name == "gfx.Reshape" || name == "gfx.Squeeze" ||
                                   name == "gfx.Unsqueeze" || name == "gfx.Transpose" || name == "gfx.Broadcast");
    if (!constant_wrapper || def->getNumOperands() == 0) {
        return false;
    }

    for (mlir::Value operand : def->getOperands()) {
        if (!is_constant_like_value(operand)) {
            return false;
        }
    }
    return true;
}

inline bool is_supported_activation_op(mlir::Operation* op, ActivationKind& kind) {
    if (!op) {
        return false;
    }
    const auto name = op->getName().getStringRef();
    if (name == "gfx.Relu") {
        kind = ActivationKind::Relu;
        return true;
    }
    if (name == "gfx.Sigmoid") {
        kind = ActivationKind::Sigmoid;
        return true;
    }
    if (name == "gfx.Tanh") {
        kind = ActivationKind::Tanh;
        return true;
    }
    if (name == "gfx.Gelu") {
        kind = ActivationKind::Gelu;
        return true;
    }
    if (name == "gfx.Elu") {
        kind = ActivationKind::Elu;
        return op->hasAttr("gfx.activation_alpha");
    }
    if (name == "gfx.PRelu") {
        kind = ActivationKind::Prelu;
        return op->hasAttr("gfx.activation_alpha");
    }
    if (name == "gfx.Swish") {
        kind = ActivationKind::Swish;
        return true;
    }
    if (name == "gfx.HSwish") {
        kind = ActivationKind::HSwish;
        return true;
    }
    if (name == "gfx.HSigmoid") {
        kind = ActivationKind::HSigmoid;
        return true;
    }
    if (name == "gfx.Abs") {
        kind = ActivationKind::Abs;
        return true;
    }
    if (name == "gfx.Sign") {
        kind = ActivationKind::Sign;
        return true;
    }
    return false;
}

inline bool get_activation_alpha(mlir::Operation* op, float& alpha) {
    if (!op) {
        return false;
    }
    if (auto attr = op->getAttrOfType<mlir::FloatAttr>("gfx.activation_alpha")) {
        alpha = static_cast<float>(attr.getValueAsDouble());
        return true;
    }
    return false;
}

inline float activation_alpha_or(mlir::Operation* op, float fallback = 0.0f) {
    float alpha = fallback;
    (void)get_activation_alpha(op, alpha);
    return alpha;
}

inline const char* activation_kind_name(ActivationKind kind) {
    switch (kind) {
        case ActivationKind::Relu:
            return "Relu";
        case ActivationKind::Sigmoid:
            return "Sigmoid";
        case ActivationKind::Tanh:
            return "Tanh";
        case ActivationKind::Elu:
            return "Elu";
        case ActivationKind::Prelu:
            return "Prelu";
        case ActivationKind::Gelu:
            return "Gelu";
        case ActivationKind::Swish:
            return "Swish";
        case ActivationKind::HSwish:
            return "HSwish";
        case ActivationKind::HSigmoid:
            return "HSigmoid";
        case ActivationKind::Abs:
            return "Abs";
        case ActivationKind::Sign:
            return "Sign";
        default:
            return "Relu";
    }
}

}  // namespace fusion_utils
}  // namespace gfx_plugin
}  // namespace ov
