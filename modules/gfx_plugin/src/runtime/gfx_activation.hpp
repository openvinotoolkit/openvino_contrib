// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

namespace ov {
namespace gfx_plugin {

// Backend-neutral activation kinds used across MLIR and runtime.
enum class ActivationKind {
    Relu,
    Sigmoid,
    Tanh,
    Elu,
    Prelu,
    Gelu,
    Swish,
    HSwish,
    HSigmoid,
    SoftPlus,
    Mish,
    SoftSign,
    Abs,
    Sign,
    Identity,
    Clamp,
    LogicalNot,
    Exp,
    Log,
    Sqrt,
    Floor,
    Ceil,
    Negative,
    Sin,
    Cos,
    Tan,
    Erf,
    Asin,
    Acos,
    Atan,
    Asinh,
    Acosh,
    Atanh,
    Sinh,
    Cosh,
    RoundEven,
    RoundAway
};

}  // namespace gfx_plugin
}  // namespace ov
