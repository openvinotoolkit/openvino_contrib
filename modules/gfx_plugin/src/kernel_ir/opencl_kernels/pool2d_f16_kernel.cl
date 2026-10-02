// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

static inline uint gfx_pool_load_f16_bits(__global const uint* src, uint idx) {
    const uint packed = src[idx >> 1u];
    return (idx & 1u) == 0u ? (packed & 65535u) : ((packed >> 16u) & 65535u);
}

static inline void gfx_pool_store_f16_pair(__global uint* dst,
                                           uint word_idx,
                                           uint lo,
                                           uint hi) {
    dst[word_idx] = (lo & 65535u) | ((hi & 65535u) << 16u);
}

static inline float gfx_pool_f16_bits_to_f32(uint bits) {
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

static inline uint gfx_pool_f32_to_f16_bits(float value) {
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

static inline uint gfx_pool2d_f16_value(__global const uint* src,
                                        uint elem,
                                        uint count,
                                        uint n_total,
                                        uint channels,
                                        uint h_in,
                                        uint w_in,
                                        uint k_h,
                                        uint k_w,
                                        uint stride_h,
                                        uint stride_w,
                                        uint dilation_h,
                                        uint dilation_w,
                                        uint pad_top,
                                        uint pad_left,
                                        uint h_out,
                                        uint w_out,
                                        uint is_avg,
                                        uint exclude_pad) {
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

    float acc = is_avg != 0u ? 0.0f : -3.402823466e+38f;
    uint sample_count = 0u;
    const int base_h = (int)out_h * (int)stride_h - (int)pad_top;
    const int base_w = (int)out_w * (int)stride_w - (int)pad_left;
    for (uint kh = 0u; kh < k_h; ++kh) {
        const int in_h = base_h + (int)kh * (int)dilation_h;
        for (uint kw = 0u; kw < k_w; ++kw) {
            const int in_w = base_w + (int)kw * (int)dilation_w;
            const uint inside = in_h >= 0 && in_w >= 0 &&
                                in_h < (int)h_in && in_w < (int)w_in;
            if (inside == 0u) {
                if (is_avg != 0u && exclude_pad == 0u) {
                    ++sample_count;
                }
                continue;
            }

            const uint src_index =
                ((batch * channels + channel) * h_in + (uint)in_h) * w_in +
                (uint)in_w;
            const float value =
                gfx_pool_f16_bits_to_f32(gfx_pool_load_f16_bits(src, src_index));
            if (is_avg != 0u) {
                acc += value;
                ++sample_count;
            } else {
                acc = fmax(acc, value);
            }
        }
    }

    if (is_avg != 0u) {
        acc = sample_count == 0u ? 0.0f : acc / (float)sample_count;
    }
    return gfx_pool_f32_to_f16_bits(acc);
}

__kernel void gfx_opencl_generated_pool2d_f16(__global const uint* src,
                                              __global uint* dst,
                                              uint count,
                                              uint n_total,
                                              uint channels,
                                              uint h_in,
                                              uint w_in,
                                              uint k_h,
                                              uint k_w,
                                              uint stride_h,
                                              uint stride_w,
                                              uint dilation_h,
                                              uint dilation_w,
                                              uint pad_top,
                                              uint pad_left,
                                              uint pad_bottom,
                                              uint pad_right,
                                              uint h_out,
                                              uint w_out,
                                              uint is_avg,
                                              uint exclude_pad) {
    (void)pad_bottom;
    (void)pad_right;

    const uint word_idx = (uint)get_global_id(0);
    const uint elem0 = word_idx * 2u;
    if (elem0 >= count) {
        return;
    }

    const uint lo = gfx_pool2d_f16_value(src,
                                         elem0,
                                         count,
                                         n_total,
                                         channels,
                                         h_in,
                                         w_in,
                                         k_h,
                                         k_w,
                                         stride_h,
                                         stride_w,
                                         dilation_h,
                                         dilation_w,
                                         pad_top,
                                         pad_left,
                                         h_out,
                                         w_out,
                                         is_avg,
                                         exclude_pad);
    uint hi = 0u;
    if (elem0 + 1u < count) {
        hi = gfx_pool2d_f16_value(src,
                                  elem0 + 1u,
                                  count,
                                  n_total,
                                  channels,
                                  h_in,
                                  w_in,
                                  k_h,
                                  k_w,
                                  stride_h,
                                  stride_w,
                                  dilation_h,
                                  dilation_w,
                                  pad_top,
                                  pad_left,
                                  h_out,
                                  w_out,
                                  is_avg,
                                  exclude_pad);
    }
    gfx_pool_store_f16_pair(dst, word_idx, lo, hi);
}
