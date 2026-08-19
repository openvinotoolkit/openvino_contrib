// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <metal_stdlib>
using namespace metal;

struct MpsrtTopKI64PackParams {
    uint rows;
    uint k;
    uint matrix_count;
    uint src_matrix_stride_u32;
    uint dst_row_stride_i32;
    uint dst_matrix_stride_i32;
};

kernel void gfx_mpsrt_topk_pack_u32_to_i64(device const uint* src [[buffer(0)]],
                                           device int* dst [[buffer(1)]],
                                           constant MpsrtTopKI64PackParams& p [[buffer(2)]],
                                           uint gid [[thread_position_in_grid]]) {
    const uint row_items = p.rows * p.k;
    const uint total = row_items * p.matrix_count;
    if (gid >= total || p.k == 0) {
        return;
    }
    const uint matrix = gid / row_items;
    const uint rem = gid - matrix * row_items;
    const uint row = rem / p.k;
    const uint col = rem - row * p.k;
    const uint v = src[matrix * p.src_matrix_stride_u32 + rem];
    const uint dst_base = matrix * p.dst_matrix_stride_i32 + row * p.dst_row_stride_i32 + col * 2u;
    dst[dst_base] = static_cast<int>(v);
    dst[dst_base + 1u] = 0;
}
