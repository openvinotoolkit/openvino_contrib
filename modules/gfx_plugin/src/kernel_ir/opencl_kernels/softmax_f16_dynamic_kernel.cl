// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

static inline uint gfx_load_f16_bits(__global const uint *src, uint idx) {
  return (idx & 1u) == 0u ? (src[idx >> 1u] & 65535u)
                          : ((src[idx >> 1u] >> 16u) & 65535u);
}

static inline void gfx_store_f16_pair(__global uint *dst, uint word_idx,
                                      uint lo, uint hi) {
  dst[word_idx] = (lo & 65535u) | ((hi & 65535u) << 16u);
}

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

static inline uint gfx_softmax_f16_bits(__global const uint *src, uint elem,
                                        uint outer, uint axis_dim, uint inner) {
  const uint plane = axis_dim * inner;
  const uint outer_idx = elem / plane;
  if (outer_idx >= outer) {
    return 0u;
  }
  const uint inner_idx = elem % inner;
  const uint base = outer_idx * plane + inner_idx;

  float max_value = gfx_f16_bits_to_f32(gfx_load_f16_bits(src, base));
  for (uint axis_idx = 1u; axis_idx < axis_dim; ++axis_idx) {
    max_value = fmax(max_value, gfx_f16_bits_to_f32(gfx_load_f16_bits(
                                    src, base + axis_idx * inner)));
  }

  float denom = 0.0f;
  for (uint axis_idx = 0u; axis_idx < axis_dim; ++axis_idx) {
    denom += exp(
        gfx_f16_bits_to_f32(gfx_load_f16_bits(src, base + axis_idx * inner)) -
        max_value);
  }
  const float value = gfx_f16_bits_to_f32(gfx_load_f16_bits(src, elem));
  return gfx_f32_to_f16_bits(exp(value - max_value) / denom);
}

__kernel void gfx_opencl_generated_softmax_dynamic_f16(
    __global const uint *src, __global uint *dst, uint count, uint rank,
    uint axis, uint dim0, uint dim1, uint dim2, uint dim3, uint dim4, uint dim5,
    uint dim6, uint dim7) {
  const uint dim[8] = {dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7};
  uint outer = 1u;
  for (uint i = 0u; i < axis; ++i) {
    outer *= dim[i];
  }
  const uint axis_dim = dim[axis];
  uint inner = 1u;
  for (uint i = axis + 1u; i < rank; ++i) {
    inner *= dim[i];
  }
  const uint word_idx = (uint)get_global_id(0);
  const uint elem0 = word_idx * 2u;
  if (elem0 >= count) {
    return;
  }
  const uint lo = gfx_softmax_f16_bits(src, elem0, outer, axis_dim, inner);
  uint hi = 0u;
  if (elem0 + 1u < count) {
    hi = gfx_softmax_f16_bits(src, elem0 + 1u, outer, axis_dim, inner);
  }
  gfx_store_f16_pair(dst, word_idx, lo, hi);
}
