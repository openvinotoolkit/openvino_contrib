// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

static inline float gfx_softmax_f32_value(__global const float *src, uint elem,
                                          uint outer, uint axis_dim,
                                          uint inner) {
  const uint plane = axis_dim * inner;
  const uint outer_idx = elem / plane;
  if (outer_idx >= outer) {
    return 0.0f;
  }
  const uint inner_idx = elem % inner;
  const uint base = outer_idx * plane + inner_idx;

  float max_value = src[base];
  for (uint axis_idx = 1u; axis_idx < axis_dim; ++axis_idx) {
    max_value = fmax(max_value, src[base + axis_idx * inner]);
  }

  float denom = 0.0f;
  for (uint axis_idx = 0u; axis_idx < axis_dim; ++axis_idx) {
    denom += exp(src[base + axis_idx * inner] - max_value);
  }
  return exp(src[elem] - max_value) / denom;
}

__kernel void gfx_opencl_generated_softmax_f32(__global const float *src,
                                               __global float *dst, uint count,
                                               uint outer, uint axis_dim,
                                               uint inner) {
  const uint gid = (uint)get_global_id(0);
  if (gid >= count) {
    return;
  }
  dst[gid] = gfx_softmax_f32_value(src, gid, outer, axis_dim, inner);
}
