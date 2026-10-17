// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

static inline uint gfx_load_bool(__global const uchar *src, uint idx) {
  const uint word = ((__global const uint *)src)[idx >> 2u];
  return ((word >> ((idx & 3u) * 8u)) & 255u) == 0u ? 0u : 1u;
}

static inline uint gfx_logical_unary_bool(uint value, uint op) {
  if (op == 48u) {
    return value == 0u ? 1u : 0u;
  }
  return value == 0u ? 0u : 1u;
}

static inline uint gfx_logical_binary_bool(uint l, uint r, uint op) {
  uint result = 0u;
  if (op == 49u) {
    result = l & r;
  } else if (op == 50u) {
    result = l | r;
  } else if (op == 51u) {
    result = l ^ r;
  }
  return result;
}

static inline uint
gfx_logical_bool_broadcast_offset(uint idx, uint rank, uint out_dim1,
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

__kernel void gfx_opencl_generated_eltwise_logical_unary_bool(
    __global const uchar *src, __global uchar *dst, uint count, uint op) {
  const uint word_idx = (uint)get_global_id(0);
  const uint base = word_idx * 4u;
  if (base >= count) {
    return;
  }
  uint packed = 0u;
  if (base < count) {
    packed |= gfx_logical_unary_bool(gfx_load_bool(src, base), op) << 0u;
  }
  if (base + 1u < count) {
    packed |= gfx_logical_unary_bool(gfx_load_bool(src, base + 1u), op) << 8u;
  }
  if (base + 2u < count) {
    packed |= gfx_logical_unary_bool(gfx_load_bool(src, base + 2u), op) << 16u;
  }
  if (base + 3u < count) {
    packed |= gfx_logical_unary_bool(gfx_load_bool(src, base + 3u), op) << 24u;
  }
  ((__global uint *)dst)[word_idx] = packed;
}

__kernel void gfx_opencl_generated_eltwise_logical_binary_bool(
    __global const uchar *lhs, __global const uchar *rhs, __global uchar *dst,
    uint count, uint op) {
  const uint word_idx = (uint)get_global_id(0);
  const uint base = word_idx * 4u;
  if (base >= count) {
    return;
  }
  uint packed = 0u;
  if (base < count) {
    packed |= gfx_logical_binary_bool(gfx_load_bool(lhs, base),
                                      gfx_load_bool(rhs, base), op)
              << 0u;
  }
  if (base + 1u < count) {
    packed |= gfx_logical_binary_bool(gfx_load_bool(lhs, base + 1u),
                                      gfx_load_bool(rhs, base + 1u), op)
              << 8u;
  }
  if (base + 2u < count) {
    packed |= gfx_logical_binary_bool(gfx_load_bool(lhs, base + 2u),
                                      gfx_load_bool(rhs, base + 2u), op)
              << 16u;
  }
  if (base + 3u < count) {
    packed |= gfx_logical_binary_bool(gfx_load_bool(lhs, base + 3u),
                                      gfx_load_bool(rhs, base + 3u), op)
              << 24u;
  }
  ((__global uint *)dst)[word_idx] = packed;
}

__kernel void gfx_opencl_generated_eltwise_logical_binary_broadcast_bool(
    __global const uchar *lhs, __global const uchar *rhs, __global uchar *dst,
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
    const uint lhs_offset = gfx_logical_bool_broadcast_offset(
        base, rank, out_dim1, out_dim2, out_dim3, lhs_stride0, lhs_stride1,
        lhs_stride2, lhs_stride3);
    const uint rhs_offset = gfx_logical_bool_broadcast_offset(
        base, rank, out_dim1, out_dim2, out_dim3, rhs_stride0, rhs_stride1,
        rhs_stride2, rhs_stride3);
    packed |= gfx_logical_binary_bool(gfx_load_bool(lhs, lhs_offset),
                                      gfx_load_bool(rhs, rhs_offset), op)
              << 0u;
  }
  if (base + 1u < count) {
    const uint idx = base + 1u;
    const uint lhs_offset = gfx_logical_bool_broadcast_offset(
        idx, rank, out_dim1, out_dim2, out_dim3, lhs_stride0, lhs_stride1,
        lhs_stride2, lhs_stride3);
    const uint rhs_offset = gfx_logical_bool_broadcast_offset(
        idx, rank, out_dim1, out_dim2, out_dim3, rhs_stride0, rhs_stride1,
        rhs_stride2, rhs_stride3);
    packed |= gfx_logical_binary_bool(gfx_load_bool(lhs, lhs_offset),
                                      gfx_load_bool(rhs, rhs_offset), op)
              << 8u;
  }
  if (base + 2u < count) {
    const uint idx = base + 2u;
    const uint lhs_offset = gfx_logical_bool_broadcast_offset(
        idx, rank, out_dim1, out_dim2, out_dim3, lhs_stride0, lhs_stride1,
        lhs_stride2, lhs_stride3);
    const uint rhs_offset = gfx_logical_bool_broadcast_offset(
        idx, rank, out_dim1, out_dim2, out_dim3, rhs_stride0, rhs_stride1,
        rhs_stride2, rhs_stride3);
    packed |= gfx_logical_binary_bool(gfx_load_bool(lhs, lhs_offset),
                                      gfx_load_bool(rhs, rhs_offset), op)
              << 16u;
  }
  if (base + 3u < count) {
    const uint idx = base + 3u;
    const uint lhs_offset = gfx_logical_bool_broadcast_offset(
        idx, rank, out_dim1, out_dim2, out_dim3, lhs_stride0, lhs_stride1,
        lhs_stride2, lhs_stride3);
    const uint rhs_offset = gfx_logical_bool_broadcast_offset(
        idx, rank, out_dim1, out_dim2, out_dim3, rhs_stride0, rhs_stride1,
        rhs_stride2, rhs_stride3);
    packed |= gfx_logical_binary_bool(gfx_load_bool(lhs, lhs_offset),
                                      gfx_load_bool(rhs, rhs_offset), op)
              << 24u;
  }
  ((__global uint *)dst)[word_idx] = packed;
}
