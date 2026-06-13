// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <metal_stdlib>
using namespace metal;

kernel void gfx_metal_generated_logsoftmax_f32(
    device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
    constant uint &rows [[buffer(2)]], constant uint &cols [[buffer(3)]],
    constant uint &inner [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  const uint row = gid / cols;
  const uint col = gid - row * cols;
  if (row >= rows || col >= cols) {
    return;
  }

  const uint outer = row / inner;
  const uint inner_i = row - outer * inner;
  const uint base_outer = outer * cols * inner;

  float max_value = -INFINITY;
  for (uint c = 0; c < cols; ++c) {
    const uint idx = base_outer + c * inner + inner_i;
    max_value = max(max_value, input[idx]);
  }

  float denom = 0.0f;
  for (uint c = 0; c < cols; ++c) {
    const uint idx = base_outer + c * inner + inner_i;
    denom += exp(input[idx] - max_value);
  }

  const uint out_idx = base_outer + col * inner + inner_i;
  output[out_idx] = (input[out_idx] - max_value) - log(denom);
}
