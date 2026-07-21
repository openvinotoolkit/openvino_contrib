// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

static inline float gfx_eltwise_f32(float lhs, float rhs, uint op) {
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

static inline int gfx_pow_i32_exact(int base, int exp) {
    if (exp < 0) {
        return (int)pow((float)base, (float)exp);
    }
    int result = 1;
    int factor = base;
    uint e = (uint)exp;
    while (e != 0u) {
        if ((e & 1u) != 0u) {
            result *= factor;
        }
        e >>= 1u;
        if (e != 0u) {
            factor *= factor;
        }
    }
    return result;
}

static inline int gfx_eltwise_i32(int lhs, int rhs, uint op) {
    switch (op) {
    case 1u: return lhs + rhs;
    case 2u: return lhs - rhs;
    case 3u: return lhs * rhs;
    case 4u: return lhs / rhs;
    case 5u: return lhs > rhs ? lhs : rhs;
    case 6u: return lhs < rhs ? lhs : rhs;
    case 7u: return gfx_pow_i32_exact(lhs, rhs);
    case 8u: {
        const int diff = lhs - rhs;
        return diff * diff;
    }
    case 9u: return lhs % rhs;
    case 10u: {
        const int rem = lhs % rhs;
        const int fix = ((rhs < 0) != (rem < 0)) ? rhs : 0;
        return rem + fix;
    }
    default: return lhs;
    }
}

static inline uint gfx_eltwise_broadcast_offset(uint idx,
                                                uint rank,
                                                uint out_dim1,
                                                uint out_dim2,
                                                uint out_dim3,
                                                uint stride0,
                                                uint stride1,
                                                uint stride2,
                                                uint stride3) {
    uint coord0 = 0u;
    uint coord1 = 0u;
    uint coord2 = 0u;
    uint coord3 = 0u;
    if (rank == 1u) {
        coord0 = idx;
    } else if (rank == 2u) {
        coord0 = idx / out_dim1;
        coord1 = idx - coord0 * out_dim1;
    } else if (rank == 3u) {
        const uint plane0 = out_dim1 * out_dim2;
        const uint rem0 = idx - (idx / plane0) * plane0;
        coord0 = idx / plane0;
        coord1 = rem0 / out_dim2;
        coord2 = rem0 - coord1 * out_dim2;
    } else {
        const uint plane0 = out_dim1 * out_dim2 * out_dim3;
        const uint rem0 = idx - (idx / plane0) * plane0;
        const uint plane1 = out_dim2 * out_dim3;
        const uint rem1 = rem0 - (rem0 / plane1) * plane1;
        coord0 = idx / plane0;
        coord1 = rem0 / plane1;
        coord2 = rem1 / out_dim3;
        coord3 = rem1 - coord2 * out_dim3;
    }
    return coord0 * stride0 + coord1 * stride1 + coord2 * stride2 + coord3 * stride3;
}

__kernel void gfx_opencl_generated_eltwise_binary_f32(__global const float* lhs,
                                                      __global const float* rhs,
                                                      __global float* dst,
                                                      uint count,
                                                      uint op) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    dst[gid] = gfx_eltwise_f32(lhs[gid], rhs[gid], op);
}

__kernel void gfx_opencl_generated_eltwise_scalar_f32(__global const float* lhs,
                                                      __global const float* rhs,
                                                      __global float* dst,
                                                      uint count,
                                                      uint op,
                                                      uint input_mode) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    const float l = input_mode == 2u ? lhs[0] : lhs[gid];
    const float r = input_mode == 1u ? rhs[0] : rhs[gid];
    dst[gid] = gfx_eltwise_f32(l, r, op);
}

__kernel void gfx_opencl_generated_eltwise_const_f32(__global const float* tensor,
                                                     __global float* dst,
                                                     uint count,
                                                     uint op,
                                                     uint input_mode,
                                                     float scalar_value) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    const float t = tensor[gid];
    const float lhs = input_mode == 4u ? scalar_value : t;
    const float rhs = input_mode == 4u ? t : scalar_value;
    dst[gid] = gfx_eltwise_f32(lhs, rhs, op);
}

__kernel void gfx_opencl_generated_eltwise_broadcast_f32(__global const float* lhs,
                                                         __global const float* rhs,
                                                         __global float* dst,
                                                         uint count,
                                                         uint op,
                                                         uint rank,
                                                         uint out_dim0,
                                                         uint out_dim1,
                                                         uint out_dim2,
                                                         uint out_dim3,
                                                         uint lhs_stride0,
                                                         uint lhs_stride1,
                                                         uint lhs_stride2,
                                                         uint lhs_stride3,
                                                         uint rhs_stride0,
                                                         uint rhs_stride1,
                                                         uint rhs_stride2,
                                                         uint rhs_stride3) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    (void)out_dim0;
    const uint lhs_offset = gfx_eltwise_broadcast_offset(gid, rank, out_dim1, out_dim2, out_dim3,
                                                         lhs_stride0, lhs_stride1, lhs_stride2, lhs_stride3);
    const uint rhs_offset = gfx_eltwise_broadcast_offset(gid, rank, out_dim1, out_dim2, out_dim3,
                                                         rhs_stride0, rhs_stride1, rhs_stride2, rhs_stride3);
    dst[gid] = gfx_eltwise_f32(lhs[lhs_offset], rhs[rhs_offset], op);
}

#define GFX_ELTWISE_LOAD_F16_BITS(src, idx) \
    (((idx) & 1u) == 0u ? ((src)[(idx) >> 1u] & 65535u) : (((src)[(idx) >> 1u] >> 16u) & 65535u))
#define GFX_ELTWISE_STORE_F16_PAIR(dst, word_idx, lo, hi) \
    ((dst)[(word_idx)] = ((lo) & 65535u) | (((hi) & 65535u) << 16u))

static inline float gfx_eltwise_f16_bits_to_f32(uint bits) {
    const uint sign = (bits & 32768u) << 16u;
    uint exp = (bits >> 10u) & 31u;
    uint mant = bits & 1023u;
    uint out = sign;
    if (exp == 0u) {
        if (mant == 0u) {
            return as_float(out);
        }
        int normalized_exp = -14;
        while ((mant & 1024u) == 0u) {
            mant <<= 1u;
            --normalized_exp;
        }
        mant &= 1023u;
        out |= (uint)(normalized_exp + 127) << 23u;
        out |= mant << 13u;
        return as_float(out);
    }
    if (exp == 31u) {
        out |= 2139095040u | (mant << 13u);
        return as_float(out);
    }
    out |= (exp + 112u) << 23u;
    out |= mant << 13u;
    return as_float(out);
}

static inline uint gfx_eltwise_f32_to_f16_bits(float value) {
    const uint bits = as_uint(value);
    const uint sign = (bits >> 16u) & 32768u;
    const uint exp_bits = (bits >> 23u) & 255u;
    const uint mant = bits & 8388607u;
    if (exp_bits == 255u) {
        return sign | 31744u | (mant != 0u ? 512u : 0u);
    }
    int exp = (int)exp_bits - 127 + 15;
    if (exp <= 0) {
        if (exp < -10) {
            return sign;
        }
        uint sub = (mant | 8388608u) >> (uint)(1 - exp);
        return sign | ((sub + 4096u) >> 13u);
    }
    if (exp >= 31) {
        return sign | 31744u;
    }
    uint half_mant = (mant + 4096u) >> 13u;
    if (half_mant == 1024u) {
        half_mant = 0u;
        ++exp;
        if (exp >= 31) {
            return sign | 31744u;
        }
    }
    return sign | ((uint)exp << 10u) | half_mant;
}

static inline uint gfx_eltwise_f16_bits(uint lhs_bits, uint rhs_bits, uint op) {
    return gfx_eltwise_f32_to_f16_bits(
        gfx_eltwise_f32(gfx_eltwise_f16_bits_to_f32(lhs_bits),
                        gfx_eltwise_f16_bits_to_f32(rhs_bits),
                        op));
}

__kernel void gfx_opencl_generated_eltwise_binary_f16(__global const uint* lhs,
                                                      __global const uint* rhs,
                                                      __global uint* dst,
                                                      uint count,
                                                      uint op) {
    const uint word_idx = (uint)get_global_id(0);
    const uint elem0 = word_idx * 2u;
    if (elem0 >= count) {
        return;
    }
    const uint lo = gfx_eltwise_f16_bits(GFX_ELTWISE_LOAD_F16_BITS(lhs, elem0),
                                         GFX_ELTWISE_LOAD_F16_BITS(rhs, elem0),
                                         op);
    uint hi = 0u;
    if (elem0 + 1u < count) {
        hi = gfx_eltwise_f16_bits(GFX_ELTWISE_LOAD_F16_BITS(lhs, elem0 + 1u),
                                  GFX_ELTWISE_LOAD_F16_BITS(rhs, elem0 + 1u),
                                  op);
    }
    GFX_ELTWISE_STORE_F16_PAIR(dst, word_idx, lo, hi);
}

__kernel void gfx_opencl_generated_eltwise_scalar_f16(__global const uint* lhs,
                                                      __global const uint* rhs,
                                                      __global uint* dst,
                                                      uint count,
                                                      uint op,
                                                      uint input_mode) {
    const uint word_idx = (uint)get_global_id(0);
    const uint elem0 = word_idx * 2u;
    if (elem0 >= count) {
        return;
    }
    const uint lhs0 = input_mode == 2u ? GFX_ELTWISE_LOAD_F16_BITS(lhs, 0u) : GFX_ELTWISE_LOAD_F16_BITS(lhs, elem0);
    const uint rhs0 = input_mode == 1u ? GFX_ELTWISE_LOAD_F16_BITS(rhs, 0u) : GFX_ELTWISE_LOAD_F16_BITS(rhs, elem0);
    const uint lo = gfx_eltwise_f16_bits(lhs0, rhs0, op);
    uint hi = 0u;
    if (elem0 + 1u < count) {
        const uint lhs1 = input_mode == 2u ? GFX_ELTWISE_LOAD_F16_BITS(lhs, 0u) : GFX_ELTWISE_LOAD_F16_BITS(lhs, elem0 + 1u);
        const uint rhs1 = input_mode == 1u ? GFX_ELTWISE_LOAD_F16_BITS(rhs, 0u) : GFX_ELTWISE_LOAD_F16_BITS(rhs, elem0 + 1u);
        hi = gfx_eltwise_f16_bits(lhs1, rhs1, op);
    }
    GFX_ELTWISE_STORE_F16_PAIR(dst, word_idx, lo, hi);
}

__kernel void gfx_opencl_generated_eltwise_broadcast_f16(__global const uint* lhs,
                                                         __global const uint* rhs,
                                                         __global uint* dst,
                                                         uint count,
                                                         uint op,
                                                         uint rank,
                                                         uint out_dim0,
                                                         uint out_dim1,
                                                         uint out_dim2,
                                                         uint out_dim3,
                                                         uint lhs_stride0,
                                                         uint lhs_stride1,
                                                         uint lhs_stride2,
                                                         uint lhs_stride3,
                                                         uint rhs_stride0,
                                                         uint rhs_stride1,
                                                         uint rhs_stride2,
                                                         uint rhs_stride3) {
    const uint word_idx = (uint)get_global_id(0);
    const uint elem0 = word_idx * 2u;
    if (elem0 >= count) {
        return;
    }
    (void)out_dim0;
    const uint lhs0 = gfx_eltwise_broadcast_offset(elem0, rank, out_dim1, out_dim2, out_dim3,
                                                   lhs_stride0, lhs_stride1, lhs_stride2, lhs_stride3);
    const uint rhs0 = gfx_eltwise_broadcast_offset(elem0, rank, out_dim1, out_dim2, out_dim3,
                                                   rhs_stride0, rhs_stride1, rhs_stride2, rhs_stride3);
    const uint lo = gfx_eltwise_f16_bits(GFX_ELTWISE_LOAD_F16_BITS(lhs, lhs0),
                                         GFX_ELTWISE_LOAD_F16_BITS(rhs, rhs0),
                                         op);
    uint hi = 0u;
    if (elem0 + 1u < count) {
        const uint elem1 = elem0 + 1u;
        const uint lhs1 = gfx_eltwise_broadcast_offset(elem1, rank, out_dim1, out_dim2, out_dim3,
                                                       lhs_stride0, lhs_stride1, lhs_stride2, lhs_stride3);
        const uint rhs1 = gfx_eltwise_broadcast_offset(elem1, rank, out_dim1, out_dim2, out_dim3,
                                                       rhs_stride0, rhs_stride1, rhs_stride2, rhs_stride3);
        hi = gfx_eltwise_f16_bits(GFX_ELTWISE_LOAD_F16_BITS(lhs, lhs1),
                                  GFX_ELTWISE_LOAD_F16_BITS(rhs, rhs1),
                                  op);
    }
    GFX_ELTWISE_STORE_F16_PAIR(dst, word_idx, lo, hi);
}

__kernel void gfx_opencl_generated_eltwise_binary_i32(__global const int* lhs,
                                                      __global const int* rhs,
                                                      __global int* dst,
                                                      uint count,
                                                      uint op) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    dst[gid] = gfx_eltwise_i32(lhs[gid], rhs[gid], op);
}

__kernel void gfx_opencl_generated_eltwise_scalar_i32(__global const int* lhs,
                                                      __global const int* rhs,
                                                      __global int* dst,
                                                      uint count,
                                                      uint op,
                                                      uint input_mode) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    const int l = input_mode == 2u ? lhs[0] : lhs[gid];
    const int r = input_mode == 1u ? rhs[0] : rhs[gid];
    dst[gid] = gfx_eltwise_i32(l, r, op);
}

__kernel void gfx_opencl_generated_eltwise_broadcast_i32(__global const int* lhs,
                                                         __global const int* rhs,
                                                         __global int* dst,
                                                         uint count,
                                                         uint op,
                                                         uint rank,
                                                         uint out_dim0,
                                                         uint out_dim1,
                                                         uint out_dim2,
                                                         uint out_dim3,
                                                         uint lhs_stride0,
                                                         uint lhs_stride1,
                                                         uint lhs_stride2,
                                                         uint lhs_stride3,
                                                         uint rhs_stride0,
                                                         uint rhs_stride1,
                                                         uint rhs_stride2,
                                                         uint rhs_stride3) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    (void)out_dim0;
    const uint lhs_offset = gfx_eltwise_broadcast_offset(gid, rank, out_dim1, out_dim2, out_dim3,
                                                         lhs_stride0, lhs_stride1, lhs_stride2, lhs_stride3);
    const uint rhs_offset = gfx_eltwise_broadcast_offset(gid, rank, out_dim1, out_dim2, out_dim3,
                                                         rhs_stride0, rhs_stride1, rhs_stride2, rhs_stride3);
    dst[gid] = gfx_eltwise_i32(lhs[lhs_offset], rhs[rhs_offset], op);
}
