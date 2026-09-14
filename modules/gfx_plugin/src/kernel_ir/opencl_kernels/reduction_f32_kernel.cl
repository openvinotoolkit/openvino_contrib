// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

static inline uint gfx_reduce_output_coord(uint input_axis,
                                           uint out_axis0,
                                           uint out_axis1,
                                           uint out_axis2,
                                           uint out_axis3,
                                           uint o0,
                                           uint o1,
                                           uint o2,
                                           uint o3) {
    if (out_axis0 == input_axis) {
        return o0;
    }
    if (out_axis1 == input_axis) {
        return o1;
    }
    if (out_axis2 == input_axis) {
        return o2;
    }
    if (out_axis3 == input_axis) {
        return o3;
    }
    return 0u;
}

static inline float gfx_reduce_f32_initial(uint op) {
    if (op == 68u) {
        return -3.4028234663852886e38f;
    }
    if (op == 69u) {
        return 3.4028234663852886e38f;
    }
    if (op == 70u) {
        return 1.0f;
    }
    return 0.0f;
}

static inline float gfx_reduce_f32_accumulate(float acc, float value, uint op) {
    if (op == 68u) {
        return fmax(acc, value);
    }
    if (op == 69u) {
        return fmin(acc, value);
    }
    if (op == 70u) {
        return acc * value;
    }
    if (op == 71u) {
        return acc + fabs(value);
    }
    if (op == 72u) {
        return acc + value * value;
    }
    return acc + value;
}

static inline float gfx_reduce_f32_finalize(float acc,
                                            uint op,
                                            uint reduction_count) {
    if (op == 67u) {
        return acc / (float)reduction_count;
    }
    if (op == 72u) {
        return sqrt(acc);
    }
    return acc;
}

static inline float gfx_reduce_f32_at(__global const float* src,
                                      uint out_idx,
                                      uint op,
                                      uint out_rank,
                                      uint in_dim0,
                                      uint in_dim1,
                                      uint in_dim2,
                                      uint in_dim3,
                                      uint out_dim1,
                                      uint out_dim2,
                                      uint out_dim3,
                                      uint reduce_mask,
                                      uint out_axis0,
                                      uint out_axis1,
                                      uint out_axis2,
                                      uint out_axis3) {
    uint o0 = 0u;
    uint o1 = 0u;
    uint o2 = 0u;
    uint o3 = 0u;
    if (out_rank == 1u) {
        o0 = out_idx;
    } else if (out_rank == 2u) {
        o0 = out_idx / out_dim1;
        o1 = out_idx - o0 * out_dim1;
    } else if (out_rank == 3u) {
        const uint plane0 = out_dim1 * out_dim2;
        const uint rem0 = out_idx - (out_idx / plane0) * plane0;
        o0 = out_idx / plane0;
        o1 = rem0 / out_dim2;
        o2 = rem0 - o1 * out_dim2;
    } else if (out_rank == 4u) {
        const uint plane0 = out_dim1 * out_dim2 * out_dim3;
        const uint rem0 = out_idx - (out_idx / plane0) * plane0;
        const uint plane1 = out_dim2 * out_dim3;
        const uint rem1 = rem0 - (rem0 / plane1) * plane1;
        o0 = out_idx / plane0;
        o1 = rem0 / plane1;
        o2 = rem1 / out_dim3;
        o3 = rem1 - o2 * out_dim3;
    }

    const uint base0 = gfx_reduce_output_coord(0u, out_axis0, out_axis1, out_axis2, out_axis3,
                                               o0, o1, o2, o3);
    const uint base1 = gfx_reduce_output_coord(1u, out_axis0, out_axis1, out_axis2, out_axis3,
                                               o0, o1, o2, o3);
    const uint base2 = gfx_reduce_output_coord(2u, out_axis0, out_axis1, out_axis2, out_axis3,
                                               o0, o1, o2, o3);
    const uint base3 = gfx_reduce_output_coord(3u, out_axis0, out_axis1, out_axis2, out_axis3,
                                               o0, o1, o2, o3);

    const uint r0_count = (reduce_mask & 1u) != 0u ? in_dim0 : 1u;
    const uint r1_count = (reduce_mask & 2u) != 0u ? in_dim1 : 1u;
    const uint r2_count = (reduce_mask & 4u) != 0u ? in_dim2 : 1u;
    const uint r3_count = (reduce_mask & 8u) != 0u ? in_dim3 : 1u;
    const uint reduction_count = r0_count * r1_count * r2_count * r3_count;
    float acc = gfx_reduce_f32_initial(op);
    for (uint r0 = 0u; r0 < r0_count; ++r0) {
        const uint c0 = (reduce_mask & 1u) != 0u ? r0 : base0;
        for (uint r1 = 0u; r1 < r1_count; ++r1) {
            const uint c1 = (reduce_mask & 2u) != 0u ? r1 : base1;
            for (uint r2 = 0u; r2 < r2_count; ++r2) {
                const uint c2 = (reduce_mask & 4u) != 0u ? r2 : base2;
                for (uint r3 = 0u; r3 < r3_count; ++r3) {
                    const uint c3 = (reduce_mask & 8u) != 0u ? r3 : base3;
                    const uint input_offset = ((c0 * in_dim1 + c1) * in_dim2 + c2) * in_dim3 + c3;
                    acc = gfx_reduce_f32_accumulate(acc, src[input_offset], op);
                }
            }
        }
    }
    return gfx_reduce_f32_finalize(acc, op, reduction_count);
}

__kernel void gfx_opencl_generated_reduction_f32(__global const float* src,
                                                 __global float* dst,
                                                 uint count,
                                                 uint op,
                                                 uint rank,
                                                 uint out_rank,
                                                 uint in_dim0,
                                                 uint in_dim1,
                                                 uint in_dim2,
                                                 uint in_dim3,
                                                 uint out_dim0,
                                                 uint out_dim1,
                                                 uint out_dim2,
                                                 uint out_dim3,
                                                 uint reduce_mask,
                                                 uint out_axis0,
                                                 uint out_axis1,
                                                 uint out_axis2,
                                                 uint out_axis3) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    (void)rank;
    (void)out_dim0;
    dst[gid] = gfx_reduce_f32_at(src, gid, op, out_rank,
                                 in_dim0, in_dim1, in_dim2, in_dim3,
                                 out_dim1, out_dim2, out_dim3,
                                 reduce_mask, out_axis0, out_axis1,
                                 out_axis2, out_axis3);
}
