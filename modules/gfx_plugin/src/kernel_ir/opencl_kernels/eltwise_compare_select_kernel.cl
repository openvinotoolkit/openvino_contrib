// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

static inline uchar gfx_compare_f32(float lhs, float rhs, uint op) {
  uint result = 0u;
  if (op == 32u) {
    result = lhs == rhs ? 1u : 0u;
  } else if (op == 33u) {
    result = lhs != rhs ? 1u : 0u;
  } else if (op == 34u) {
    result = lhs > rhs ? 1u : 0u;
  } else if (op == 35u) {
    result = lhs >= rhs ? 1u : 0u;
  } else if (op == 36u) {
    result = lhs < rhs ? 1u : 0u;
  } else if (op == 37u) {
    result = lhs <= rhs ? 1u : 0u;
  }
  return (uchar)result;
}

static inline uint
gfx_compare_select_broadcast_offset(uint idx, uint rank, uint out_dim1,
                                    uint out_dim2, uint out_dim3, uint stride0,
                                    uint stride1, uint stride2, uint stride3) {
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
  return coord0 * stride0 + coord1 * stride1 + coord2 * stride2 +
         coord3 * stride3;
}

static inline uint gfx_load_bool_mask(__global const uint *src, uint idx) {
  const uint word = src[idx >> 2u];
  const uint lane = (word >> ((idx & 3u) * 8u)) & 255u;
  return 0u - (lane != 0u);
}

static inline uint gfx_load_f16_bits(__global const uint *src, uint idx) {
  const uint word = src[idx >> 1u];
  if ((idx & 1u) == 0u) {
    return word & 65535u;
  }
  return (word >> 16u) & 65535u;
}

static inline uint gfx_pack_f16_pair(uint lo, uint hi) {
  return (lo & 65535u) | ((hi & 65535u) << 16u);
}

static inline uint gfx_select_f16_bits(uint mask, uint then_bits,
                                       uint else_bits) {
  return (else_bits & ~mask) | (then_bits & mask);
}

__kernel void gfx_opencl_generated_eltwise_compare_f32(
    __global const float *lhs, __global const float *rhs, __global uchar *dst,
    uint count, uint op) {
  const uint word_idx = (uint)get_global_id(0);
  const uint base = word_idx * 4u;
  if (base >= count) {
    return;
  }
  uint packed = 0u;
  if (base < count) {
    packed |= ((uint)gfx_compare_f32(lhs[base], rhs[base], op)) << 0u;
  }
  if (base + 1u < count) {
    packed |= ((uint)gfx_compare_f32(lhs[base + 1u], rhs[base + 1u], op)) << 8u;
  }
  if (base + 2u < count) {
    packed |= ((uint)gfx_compare_f32(lhs[base + 2u], rhs[base + 2u], op))
              << 16u;
  }
  if (base + 3u < count) {
    packed |= ((uint)gfx_compare_f32(lhs[base + 3u], rhs[base + 3u], op))
              << 24u;
  }
  ((__global uint *)dst)[word_idx] = packed;
}

__kernel void gfx_opencl_generated_eltwise_compare_broadcast_f32(
    __global const float *lhs, __global const float *rhs, __global uchar *dst,
    uint count, uint op, uint rank, uint out_dim0, uint out_dim1, uint out_dim2,
    uint out_dim3, uint lhs_stride0, uint lhs_stride1, uint lhs_stride2,
    uint lhs_stride3, uint rhs_stride0, uint rhs_stride1, uint rhs_stride2,
    uint rhs_stride3) {
  const uint word_idx = (uint)get_global_id(0);
  const uint base = word_idx * 4u;
  if (base >= count) {
    return;
  }
  (void)out_dim0;

  uint packed = 0u;
  if (base < count) {
    const uint lhs_offset = gfx_compare_select_broadcast_offset(
        base, rank, out_dim1, out_dim2, out_dim3, lhs_stride0, lhs_stride1,
        lhs_stride2, lhs_stride3);
    const uint rhs_offset = gfx_compare_select_broadcast_offset(
        base, rank, out_dim1, out_dim2, out_dim3, rhs_stride0, rhs_stride1,
        rhs_stride2, rhs_stride3);
    packed |= ((uint)gfx_compare_f32(lhs[lhs_offset], rhs[rhs_offset], op))
              << 0u;
  }
  if (base + 1u < count) {
    const uint idx = base + 1u;
    const uint lhs_offset = gfx_compare_select_broadcast_offset(
        idx, rank, out_dim1, out_dim2, out_dim3, lhs_stride0, lhs_stride1,
        lhs_stride2, lhs_stride3);
    const uint rhs_offset = gfx_compare_select_broadcast_offset(
        idx, rank, out_dim1, out_dim2, out_dim3, rhs_stride0, rhs_stride1,
        rhs_stride2, rhs_stride3);
    packed |= ((uint)gfx_compare_f32(lhs[lhs_offset], rhs[rhs_offset], op))
              << 8u;
  }
  if (base + 2u < count) {
    const uint idx = base + 2u;
    const uint lhs_offset = gfx_compare_select_broadcast_offset(
        idx, rank, out_dim1, out_dim2, out_dim3, lhs_stride0, lhs_stride1,
        lhs_stride2, lhs_stride3);
    const uint rhs_offset = gfx_compare_select_broadcast_offset(
        idx, rank, out_dim1, out_dim2, out_dim3, rhs_stride0, rhs_stride1,
        rhs_stride2, rhs_stride3);
    packed |= ((uint)gfx_compare_f32(lhs[lhs_offset], rhs[rhs_offset], op))
              << 16u;
  }
  if (base + 3u < count) {
    const uint idx = base + 3u;
    const uint lhs_offset = gfx_compare_select_broadcast_offset(
        idx, rank, out_dim1, out_dim2, out_dim3, lhs_stride0, lhs_stride1,
        lhs_stride2, lhs_stride3);
    const uint rhs_offset = gfx_compare_select_broadcast_offset(
        idx, rank, out_dim1, out_dim2, out_dim3, rhs_stride0, rhs_stride1,
        rhs_stride2, rhs_stride3);
    packed |= ((uint)gfx_compare_f32(lhs[lhs_offset], rhs[rhs_offset], op))
              << 24u;
  }
  ((__global uint *)dst)[word_idx] = packed;
}

__kernel void gfx_opencl_generated_eltwise_select_f32(
    __global const uchar *cond, __global const float *then_data,
    __global const float *else_data, __global float *dst, uint count) {
  const uint gid = (uint)get_global_id(0);
  if (gid >= count) {
    return;
  }
  const float then_value = then_data[gid];
  const float else_value = else_data[gid];
  const float mask = convert_float(cond[gid]);
  dst[gid] = else_value + mask * (then_value - else_value);
}

__kernel void gfx_opencl_generated_eltwise_select_broadcast_f32(
    __global const uchar *cond, __global const float *then_data,
    __global const float *else_data, __global float *dst, uint count, uint rank,
    uint out_dim0, uint out_dim1, uint out_dim2, uint out_dim3,
    uint cond_stride0, uint cond_stride1, uint cond_stride2, uint cond_stride3,
    uint then_stride0, uint then_stride1, uint then_stride2, uint then_stride3,
    uint else_stride0, uint else_stride1, uint else_stride2,
    uint else_stride3) {
  const uint gid = (uint)get_global_id(0);
  if (gid >= count) {
    return;
  }
  (void)out_dim0;

  const uint cond_offset = gfx_compare_select_broadcast_offset(
      gid, rank, out_dim1, out_dim2, out_dim3, cond_stride0, cond_stride1,
      cond_stride2, cond_stride3);
  const uint then_offset = gfx_compare_select_broadcast_offset(
      gid, rank, out_dim1, out_dim2, out_dim3, then_stride0, then_stride1,
      then_stride2, then_stride3);
  const uint else_offset = gfx_compare_select_broadcast_offset(
      gid, rank, out_dim1, out_dim2, out_dim3, else_stride0, else_stride1,
      else_stride2, else_stride3);
  const float then_value = then_data[then_offset];
  const float else_value = else_data[else_offset];
  const float mask = convert_float(cond[cond_offset]);
  dst[gid] = else_value + mask * (then_value - else_value);
}

__kernel void gfx_opencl_generated_eltwise_select_f16_dynamic(
    __global const uint *cond, __global const uint *then_data,
    __global const uint *else_data, __global uint *dst, uint count) {
  const uint word_idx = (uint)get_global_id(0);
  const uint elem0 = word_idx * 2u;
  if (elem0 >= count) {
    return;
  }
  const uint lo_mask = gfx_load_bool_mask(cond, elem0);
  const uint lo =
      gfx_select_f16_bits(lo_mask, gfx_load_f16_bits(then_data, elem0),
                          gfx_load_f16_bits(else_data, elem0));
  uint hi = 0u;
  if (elem0 + 1u < count) {
    const uint hi_mask = gfx_load_bool_mask(cond, elem0 + 1u);
    hi = gfx_select_f16_bits(hi_mask, gfx_load_f16_bits(then_data, elem0 + 1u),
                             gfx_load_f16_bits(else_data, elem0 + 1u));
  }
  dst[word_idx] = gfx_pack_f16_pair(lo, hi);
}
