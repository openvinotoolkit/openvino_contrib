// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define GFX_LOAD_F16_BITS(src, idx) \
    (((idx) & 1u) == 0u ? ((src)[(idx) >> 1u] & 65535u) : (((src)[(idx) >> 1u] >> 16u) & 65535u))
#define GFX_STORE_F16_PAIR(dst, word_idx, lo, hi) \
    ((dst)[(word_idx)] = ((lo) & 65535u) | (((hi) & 65535u) << 16u))

static inline float gfx_f16_bits_to_f32(uint bits) {
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

static inline uint gfx_f32_to_f16_bits(float value) {
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

static inline float gfx_interpolate_source_coord(uint out_idx,
                                                 uint in_size,
                                                 uint out_size,
                                                 uint align_corners,
                                                 uint use_half_pixel) {
    if (align_corners != 0u && out_size > 1u) {
        return (float)out_idx * (float)(in_size - 1u) / (float)(out_size - 1u);
    }
    const float scale = out_size != 0u ? (float)in_size / (float)out_size : 1.0f;
    if (use_half_pixel != 0u) {
        return ((float)out_idx + 0.5f) * scale - 0.5f;
    }
    return (float)out_idx * scale;
}

static inline int gfx_interpolate_nearest_index(float coord,
                                                uint in_size,
                                                uint nearest_mode) {
    if (nearest_mode == 1u) {
        return clamp((int)floor(coord), 0, (int)in_size - 1);
    }
    if (nearest_mode == 2u) {
        return clamp((int)ceil(coord), 0, (int)in_size - 1);
    }
    return clamp((int)round(coord), 0, (int)in_size - 1);
}

static inline uint gfx_interpolate_f16_bits(__global const uint* src,
                                            uint elem,
                                            uint count,
                                            uint nearest,
                                            uint align_corners,
                                            uint use_half_pixel,
                                            uint nearest_mode,
                                            uint n_total,
                                            uint channels,
                                            uint h_in,
                                            uint w_in,
                                            uint h_out,
                                            uint w_out) {
    if (elem >= count) {
        return 0u;
    }

    uint tmp = elem;
    const uint out_w = tmp % w_out;
    tmp /= w_out;
    const uint out_h = tmp % h_out;
    tmp /= h_out;
    const uint channel = tmp % channels;
    tmp /= channels;
    const uint batch = tmp;
    if (batch >= n_total) {
        return 0u;
    }

    const float src_h = gfx_interpolate_source_coord(out_h,
                                                     h_in,
                                                     h_out,
                                                     align_corners,
                                                     use_half_pixel);
    const float src_w = gfx_interpolate_source_coord(out_w,
                                                     w_in,
                                                     w_out,
                                                     align_corners,
                                                     use_half_pixel);
    const uint base = ((batch * channels + channel) * h_in) * w_in;
    if (nearest != 0u) {
        const int in_h = gfx_interpolate_nearest_index(src_h, h_in, nearest_mode);
        const int in_w = gfx_interpolate_nearest_index(src_w, w_in, nearest_mode);
        return GFX_LOAD_F16_BITS(src, base + (uint)in_h * w_in + (uint)in_w);
    }

    const float floor_h = floor(src_h);
    const float floor_w = floor(src_w);
    const int h0 = clamp((int)floor_h, 0, (int)h_in - 1);
    const int w0 = clamp((int)floor_w, 0, (int)w_in - 1);
    const int h1 = min(h0 + 1, (int)h_in - 1);
    const int w1 = min(w0 + 1, (int)w_in - 1);
    const float dh = src_h - floor_h;
    const float dw = src_w - floor_w;

    const float v00 = gfx_f16_bits_to_f32(GFX_LOAD_F16_BITS(src, base + (uint)h0 * w_in + (uint)w0));
    const float v01 = gfx_f16_bits_to_f32(GFX_LOAD_F16_BITS(src, base + (uint)h0 * w_in + (uint)w1));
    const float v10 = gfx_f16_bits_to_f32(GFX_LOAD_F16_BITS(src, base + (uint)h1 * w_in + (uint)w0));
    const float v11 = gfx_f16_bits_to_f32(GFX_LOAD_F16_BITS(src, base + (uint)h1 * w_in + (uint)w1));
    const float v0 = v00 + (v01 - v00) * dw;
    const float v1 = v10 + (v11 - v10) * dw;
    return gfx_f32_to_f16_bits(v0 + (v1 - v0) * dh);
}

__kernel void gfx_opencl_generated_interpolate_f16(__global const uint* src,
                                                   __global uint* dst,
                                                   uint count,
                                                   uint nearest,
                                                   uint align_corners,
                                                   uint use_half_pixel,
                                                   uint nearest_mode,
                                                   uint n_total,
                                                   uint channels,
                                                   uint h_in,
                                                   uint w_in,
                                                   uint h_out,
                                                   uint w_out) {
    const uint word_idx = (uint)get_global_id(0);
    const uint elem0 = word_idx * 2u;
    if (elem0 >= count) {
        return;
    }
    const uint lo = gfx_interpolate_f16_bits(src,
                                             elem0,
                                             count,
                                             nearest,
                                             align_corners,
                                             use_half_pixel,
                                             nearest_mode,
                                             n_total,
                                             channels,
                                             h_in,
                                             w_in,
                                             h_out,
                                             w_out);
    uint hi = 0u;
    if (elem0 + 1u < count) {
        hi = gfx_interpolate_f16_bits(src,
                                      elem0 + 1u,
                                      count,
                                      nearest,
                                      align_corners,
                                      use_half_pixel,
                                      nearest_mode,
                                      n_total,
                                      channels,
                                      h_in,
                                      w_in,
                                      h_out,
                                      w_out);
    }
    GFX_STORE_F16_PAIR(dst, word_idx, lo, hi);
}
