// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <metal_stdlib>
using namespace metal;

struct MpsrtTopKStableIndexParams {
    uint rows;
    uint k;
    uint source_columns;
    uint matrix_count;
    uint input_matrix_stride;
    uint input_row_stride;
    uint values_matrix_stride;
    uint values_row_stride;
    uint fallback_matrix_stride;
    uint dst_row_stride_i32;
    uint dst_matrix_stride_i32;
};

kernel void gfx_mpsrt_topk_stable_i64_indices(device const float* input [[buffer(0)]],
                                              device const float* values [[buffer(1)]],
                                              device const uint* fallback_indices [[buffer(2)]],
                                              device int* dst [[buffer(3)]],
                                              constant MpsrtTopKStableIndexParams& p [[buffer(4)]],
                                              uint gid [[thread_position_in_grid]]) {
    const uint row_items = p.rows * p.k;
    const uint total = row_items * p.matrix_count;
    if (gid >= total || p.k == 0 || p.source_columns == 0) {
        return;
    }
    const uint matrix = gid / row_items;
    const uint rem = gid - matrix * row_items;
    const uint row = rem / p.k;
    const uint col = rem - row * p.k;
    const uint values_row_base = matrix * p.values_matrix_stride + row * p.values_row_stride;
    const auto target = values[values_row_base + col];
    uint duplicate_rank = 0;
    for (uint prev = 0; prev < col; ++prev) {
        if (values[values_row_base + prev] == target) {
            ++duplicate_rank;
        }
    }
    uint selected = fallback_indices[matrix * p.fallback_matrix_stride + rem];
    uint seen = 0;
    const uint input_row_base = matrix * p.input_matrix_stride + row * p.input_row_stride;
    for (uint source_col = 0; source_col < p.source_columns; ++source_col) {
        if (input[input_row_base + source_col] == target) {
            if (seen == duplicate_rank) {
                selected = source_col;
                break;
            }
            ++seen;
        }
    }
    const uint dst_base = matrix * p.dst_matrix_stride_i32 + row * p.dst_row_stride_i32 + col * 2u;
    dst[dst_base] = static_cast<int>(selected);
    dst[dst_base + 1u] = 0;
}
