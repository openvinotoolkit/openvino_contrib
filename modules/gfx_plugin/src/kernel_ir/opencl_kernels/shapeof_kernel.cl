// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

static inline uint gfx_shapeof_dim(uint axis, uint dim0, uint dim1, uint dim2,
                                   uint dim3, uint dim4, uint dim5, uint dim6,
                                   uint dim7) {
  if (axis == 0u) {
    return dim0;
  }
  if (axis == 1u) {
    return dim1;
  }
  if (axis == 2u) {
    return dim2;
  }
  if (axis == 3u) {
    return dim3;
  }
  if (axis == 4u) {
    return dim4;
  }
  if (axis == 5u) {
    return dim5;
  }
  if (axis == 6u) {
    return dim6;
  }
  return dim7;
}

__kernel void gfx_opencl_generated_shapeof_i32(__global const uchar *src,
                                               __global int *dst, uint count,
                                               uint dim0, uint dim1, uint dim2,
                                               uint dim3, uint dim4, uint dim5,
                                               uint dim6, uint dim7) {
  (void)src;
  const uint gid = (uint)get_global_id(0);
  if (gid >= count) {
    return;
  }
  dst[gid] =
      (int)gfx_shapeof_dim(gid, dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7);
}

__kernel void gfx_opencl_generated_shapeof_i64(__global const uchar *src,
                                               __global uint *dst, uint count,
                                               uint dim0, uint dim1, uint dim2,
                                               uint dim3, uint dim4, uint dim5,
                                               uint dim6, uint dim7) {
  (void)src;
  const uint gid = (uint)get_global_id(0);
  if (gid >= count) {
    return;
  }
  const uint word = gid * 2u;
  dst[word] =
      gfx_shapeof_dim(gid, dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7);
  dst[word + 1u] = 0u;
}
