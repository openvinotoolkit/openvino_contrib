// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

__kernel void gfx_opencl_generated_range_i64(__global const long *start,
                                             __global const long *stop,
                                             __global const long *step,
                                             __global long *dst,
                                             uint count) {
  const uint gid = (uint)get_global_id(0);
  if (gid >= count) {
    return;
  }
  (void)stop;
  dst[gid] = start[0] + (long)gid * step[0];
}
