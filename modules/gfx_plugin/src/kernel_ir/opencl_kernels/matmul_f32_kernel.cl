// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

__kernel void gfx_opencl_generated_matmul_f32(__global const float* lhs,
                                              __global const float* rhs,
                                              __global float* dst,
                                              uint count,
                                              uint m,
                                              uint n,
                                              uint k_dim,
                                              uint lhs_batch_stride,
                                              uint rhs_batch_stride,
                                              uint lhs_row_stride,
                                              uint lhs_col_stride,
                                              uint rhs_row_stride,
                                              uint rhs_col_stride) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint mn = m * n;
    const uint batch_idx = gid / mn;
    const uint rem = gid - batch_idx * mn;
    const uint row = rem / n;
    const uint col = rem - row * n;
    const uint lhs_base = lhs_batch_stride == 0u ? 0u : batch_idx * lhs_batch_stride;
    const uint rhs_base = rhs_batch_stride == 0u ? 0u : batch_idx * rhs_batch_stride;

    float acc = 0.0f;
    for (uint k = 0u; k < k_dim; ++k) {
        const uint lhs_idx = lhs_base + row * lhs_row_stride + k * lhs_col_stride;
        const uint rhs_idx = rhs_base + k * rhs_row_stride + col * rhs_col_stride;
        acc += lhs[lhs_idx] * rhs[rhs_idx];
    }
    dst[gid] = acc;
}
