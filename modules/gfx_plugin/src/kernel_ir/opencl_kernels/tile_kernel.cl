// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

static inline uint gfx_tile_load_f16_bits(__global const uint *src, uint idx) {
  const uint word = src[idx >> 1u];
  if ((idx & 1u) == 0u) {
    return word & 65535u;
  }
  return (word >> 16u) & 65535u;
}

static inline void gfx_tile_store_f16_pair(__global uint *dst, uint word_idx,
                                           uint lo, uint hi) {
  dst[word_idx] = (lo & 65535u) | ((hi & 65535u) << 16u);
}

static inline uint gfx_tile_static_src_offset(
    uint elem, uint rank, uint out_dim0, uint out_dim1, uint out_dim2,
    uint out_dim3, uint in_dim0, uint in_dim1, uint in_dim2, uint in_dim3,
    uint out_stride0, uint out_stride1, uint out_stride2, uint out_stride3,
    uint in_stride0, uint in_stride1, uint in_stride2, uint in_stride3) {
  const uint out_dim[4] = {out_dim0, out_dim1, out_dim2, out_dim3};
  const uint in_dim[4] = {in_dim0, in_dim1, in_dim2, in_dim3};
  const uint out_stride[4] = {out_stride0, out_stride1, out_stride2,
                              out_stride3};
  const uint in_stride[4] = {in_stride0, in_stride1, in_stride2, in_stride3};
  uint src_offset = 0u;
  for (uint axis = 0u; axis < rank; ++axis) {
    const uint out_axis_coord =
        out_stride[axis] == 0u ? 0u : (elem / out_stride[axis]) % out_dim[axis];
    const uint in_axis_coord =
        in_dim[axis] == 0u ? 0u : out_axis_coord % in_dim[axis];
    src_offset += in_axis_coord * in_stride[axis];
  }
  return src_offset;
}

static inline uint gfx_tile_dynamic_src_offset(uint elem, uint rank,
                                               uint out_dim0, uint out_dim1,
                                               uint out_dim2, uint out_dim3,
                                               uint in_dim0, uint in_dim1,
                                               uint in_dim2, uint in_dim3) {
  const uint out_dim[4] = {out_dim0, out_dim1, out_dim2, out_dim3};
  const uint in_dim[4] = {in_dim0, in_dim1, in_dim2, in_dim3};
  uint rem = elem;
  uint src_offset = 0u;
  for (uint axis = 0u; axis < rank; ++axis) {
    uint out_suffix = 1u;
    uint in_suffix = 1u;
    for (uint inner_axis = axis + 1u; inner_axis < rank; ++inner_axis) {
      out_suffix *= out_dim[inner_axis];
      in_suffix *= in_dim[inner_axis];
    }
    const uint coord = out_suffix == 0u ? 0u : rem / out_suffix;
    rem = out_suffix == 0u ? 0u : rem - coord * out_suffix;
    const uint in_coord = in_dim[axis] == 0u ? 0u : coord % in_dim[axis];
    src_offset += in_coord * in_suffix;
  }
  return src_offset;
}

__kernel void gfx_opencl_generated_tile_f32(
    __global const float *src, __global float *dst, uint count, uint rank,
    uint out_dim0, uint out_dim1, uint out_dim2, uint out_dim3, uint in_dim0,
    uint in_dim1, uint in_dim2, uint in_dim3, uint out_stride0,
    uint out_stride1, uint out_stride2, uint out_stride3, uint in_stride0,
    uint in_stride1, uint in_stride2, uint in_stride3) {
  const uint gid = (uint)get_global_id(0);
  if (gid >= count) {
    return;
  }
  const uint src_offset = gfx_tile_static_src_offset(
      gid, rank, out_dim0, out_dim1, out_dim2, out_dim3, in_dim0, in_dim1,
      in_dim2, in_dim3, out_stride0, out_stride1, out_stride2, out_stride3,
      in_stride0, in_stride1, in_stride2, in_stride3);
  dst[gid] = src[src_offset];
}

__kernel void gfx_opencl_generated_tile_dynamic_f32(
    __global const float *src, __global float *dst, uint count, uint rank,
    uint out_dim0, uint out_dim1, uint out_dim2, uint out_dim3, uint in_dim0,
    uint in_dim1, uint in_dim2, uint in_dim3) {
  const uint gid = (uint)get_global_id(0);
  if (gid >= count) {
    return;
  }
  const uint src_offset =
      gfx_tile_dynamic_src_offset(gid, rank, out_dim0, out_dim1, out_dim2,
                                  out_dim3, in_dim0, in_dim1, in_dim2, in_dim3);
  dst[gid] = src[src_offset];
}

__kernel void gfx_opencl_generated_tile_f16(
    __global const uint *src, __global uint *dst, uint count, uint rank,
    uint out_dim0, uint out_dim1, uint out_dim2, uint out_dim3, uint in_dim0,
    uint in_dim1, uint in_dim2, uint in_dim3, uint out_stride0,
    uint out_stride1, uint out_stride2, uint out_stride3, uint in_stride0,
    uint in_stride1, uint in_stride2, uint in_stride3) {
  const uint word_idx = (uint)get_global_id(0);
  const uint elem0 = word_idx * 2u;
  if (elem0 >= count) {
    return;
  }
  const uint src_offset0 = gfx_tile_static_src_offset(
      elem0, rank, out_dim0, out_dim1, out_dim2, out_dim3, in_dim0, in_dim1,
      in_dim2, in_dim3, out_stride0, out_stride1, out_stride2, out_stride3,
      in_stride0, in_stride1, in_stride2, in_stride3);
  const uint lo = gfx_tile_load_f16_bits(src, src_offset0);
  uint hi = 0u;
  if (elem0 + 1u < count) {
    const uint src_offset1 = gfx_tile_static_src_offset(
        elem0 + 1u, rank, out_dim0, out_dim1, out_dim2, out_dim3, in_dim0,
        in_dim1, in_dim2, in_dim3, out_stride0, out_stride1, out_stride2,
        out_stride3, in_stride0, in_stride1, in_stride2, in_stride3);
    hi = gfx_tile_load_f16_bits(src, src_offset1);
  }
  gfx_tile_store_f16_pair(dst, word_idx, lo, hi);
}

__kernel void gfx_opencl_generated_tile_dynamic_f16(
    __global const uint *src, __global uint *dst, uint count, uint rank,
    uint out_dim0, uint out_dim1, uint out_dim2, uint out_dim3, uint in_dim0,
    uint in_dim1, uint in_dim2, uint in_dim3) {
  const uint word_idx = (uint)get_global_id(0);
  const uint elem0 = word_idx * 2u;
  if (elem0 >= count) {
    return;
  }
  const uint src_offset0 =
      gfx_tile_dynamic_src_offset(elem0, rank, out_dim0, out_dim1, out_dim2,
                                  out_dim3, in_dim0, in_dim1, in_dim2, in_dim3);
  const uint lo = gfx_tile_load_f16_bits(src, src_offset0);
  uint hi = 0u;
  if (elem0 + 1u < count) {
    const uint src_offset1 = gfx_tile_dynamic_src_offset(
        elem0 + 1u, rank, out_dim0, out_dim1, out_dim2, out_dim3, in_dim0,
        in_dim1, in_dim2, in_dim3);
    hi = gfx_tile_load_f16_bits(src, src_offset1);
  }
  gfx_tile_store_f16_pair(dst, word_idx, lo, hi);
}
