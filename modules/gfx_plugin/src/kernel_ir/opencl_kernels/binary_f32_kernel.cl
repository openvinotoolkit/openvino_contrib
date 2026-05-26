// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

static inline float gfx_binary_f32(float lhs, float rhs, uint op) {
    switch (op) {
    case 1u: return lhs + rhs;
    case 2u: return lhs - rhs;
    case 3u: return lhs * rhs;
    case 4u: return lhs / rhs;
    case 5u: return fmax(lhs, rhs);
    case 6u: return fmin(lhs, rhs);
    case 7u: return pow(lhs, rhs);
    case 8u: {
        const float diff = lhs - rhs;
        return diff * diff;
    }
    case 9u: {
        const float rem = fmod(lhs, rhs);
        return fabs(rem) >= fabs(rhs) ? 0.0f : rem;
    }
    case 10u: {
        const float rem = lhs - floor(lhs / rhs) * rhs;
        return fabs(rem) >= fabs(rhs) ? 0.0f : rem;
    }
    default: return lhs;
    }
}

__kernel void gfx_opencl_baseline_binary_f32(__global const float* lhs,
                                             __global const float* rhs,
                                             __global float* dst,
                                             uint count,
                                             uint op) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    dst[gid] = gfx_binary_f32(lhs[gid], rhs[gid], op);
}
