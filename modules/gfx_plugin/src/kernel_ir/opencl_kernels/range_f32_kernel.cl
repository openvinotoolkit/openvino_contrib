// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

__kernel void gfx_opencl_generated_range_f32(__global const float *start,
                                             __global const float *stop,
                                             __global const float *step,
                                             __global float *dst,
                                             uint count) {
  const uint gid = (uint)get_global_id(0);
  if (gid >= count) {
    return;
  }
  (void)stop;
  dst[gid] = start[0] + (float)gid * step[0];
}
