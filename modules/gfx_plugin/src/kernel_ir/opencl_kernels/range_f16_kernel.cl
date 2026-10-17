// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define GFX_RANGE_LOAD_F16_BITS(src, idx)                                      \
  (((idx) & 1u) == 0u ? ((src)[(idx) >> 1u] & 65535u)                          \
                      : (((src)[(idx) >> 1u] >> 16u) & 65535u))
#define GFX_RANGE_STORE_F16_PAIR(dst, word_idx, lo, hi)                       \
  ((dst)[(word_idx)] = ((lo) & 65535u) | (((hi) & 65535u) << 16u))

static inline float gfx_range_f16_bits_to_f32(uint bits) {
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

static inline uint gfx_range_f32_to_f16_bits(float value) {
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

__kernel void gfx_opencl_generated_range_f16(__global const uint *start_words,
                                             __global const uint *stop_words,
                                             __global const uint *step_words,
                                             __global uint *dst,
                                             uint count) {
  const uint word_idx = (uint)get_global_id(0);
  const uint elem0 = word_idx * 2u;
  if (elem0 >= count) {
    return;
  }
  (void)stop_words;
  const float start =
      gfx_range_f16_bits_to_f32(GFX_RANGE_LOAD_F16_BITS(start_words, 0u));
  const float step =
      gfx_range_f16_bits_to_f32(GFX_RANGE_LOAD_F16_BITS(step_words, 0u));
  const uint lo = gfx_range_f32_to_f16_bits(start + (float)elem0 * step);
  uint hi = 0u;
  if (elem0 + 1u < count) {
    hi = gfx_range_f32_to_f16_bits(start + (float)(elem0 + 1u) * step);
  }
  GFX_RANGE_STORE_F16_PAIR(dst, word_idx, lo, hi);
}
