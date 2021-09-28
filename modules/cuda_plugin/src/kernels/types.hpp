// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace CUDAPlugin {
namespace kernel {

enum class Type_t {
    undefined,
    dynamic,
    boolean,
    bf16,
    f16,
    f32,
    f64,
    i4,
    i8,
    i16,
    i32,
    i64,
    u1,
    u4,
    u8,
    u16,
    u32,
    u64
};

}  // namespace kernel
}  // namespace CUDAPlugin
