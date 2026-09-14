// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

__kernel void gfx_opencl_generated_range_i64_unit(__global const uint *stop_words,
                                                  __global uint *dst,
                                                  uint count) {
  (void)stop_words;
  const uint gid = (uint)get_global_id(0);
  if (gid >= count) {
    return;
  }
  const uint word = gid * 2u;
  dst[word] = gid;
  dst[word + 1u] = 0u;
}
