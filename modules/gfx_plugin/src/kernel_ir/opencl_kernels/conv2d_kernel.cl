// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

__kernel void gfx_opencl_generated_conv2d_f32(
    __global const float *src, __global const float *weights,
    __global float *dst, uint count, uint batches, uint input_channels,
    uint input_h, uint input_w, uint output_channels, uint kernel_h,
    uint kernel_w, uint output_h, uint output_w, uint stride_h, uint stride_w,
    uint dilation_h, uint dilation_w, uint pad_top, uint pad_left) {
  const uint gid = (uint)get_global_id(0);
  if (gid >= count) {
    return;
  }

  uint tmp = gid;
  const uint out_w = tmp % output_w;
  tmp /= output_w;
  const uint out_h = tmp % output_h;
  tmp /= output_h;
  const uint out_c = tmp % output_channels;
  tmp /= output_channels;
  const uint batch = tmp;
  if (batch >= batches) {
    return;
  }

  float acc = 0.0f;
  const int base_h = (int)out_h * (int)stride_h - (int)pad_top;
  const int base_w = (int)out_w * (int)stride_w - (int)pad_left;
  for (uint in_c = 0u; in_c < input_channels; ++in_c) {
    for (uint kh = 0u; kh < kernel_h; ++kh) {
      const int in_h = base_h + (int)kh * (int)dilation_h;
      if (in_h < 0 || in_h >= (int)input_h) {
        continue;
      }
      for (uint kw = 0u; kw < kernel_w; ++kw) {
        const int in_w = base_w + (int)kw * (int)dilation_w;
        if (in_w < 0 || in_w >= (int)input_w) {
          continue;
        }
        const uint src_index =
            ((batch * input_channels + in_c) * input_h + (uint)in_h) * input_w +
            (uint)in_w;
        const uint weight_index =
            ((out_c * input_channels + in_c) * kernel_h + kh) * kernel_w + kw;
        acc += src[src_index] * weights[weight_index];
      }
    }
  }
  dst[gid] = acc;
}

__kernel void gfx_opencl_generated_group_conv2d_f32(
    __global const float *src, __global const float *weights,
    __global float *dst, uint count, uint batches, uint input_channels,
    uint input_h, uint input_w, uint groups, uint output_channels_per_group,
    uint input_channels_per_group, uint kernel_h, uint kernel_w, uint output_h,
    uint output_w, uint stride_h, uint stride_w, uint dilation_h,
    uint dilation_w, uint pad_top, uint pad_left) {
  const uint gid = (uint)get_global_id(0);
  if (gid >= count) {
    return;
  }

  const uint output_channels = groups * output_channels_per_group;
  uint tmp = gid;
  const uint out_w = tmp % output_w;
  tmp /= output_w;
  const uint out_h = tmp % output_h;
  tmp /= output_h;
  const uint out_c = tmp % output_channels;
  tmp /= output_channels;
  const uint batch = tmp;
  if (batch >= batches || output_channels == 0u) {
    return;
  }

  const uint group = out_c / output_channels_per_group;
  const uint out_c_in_group = out_c - group * output_channels_per_group;
  float acc = 0.0f;
  const int base_h = (int)out_h * (int)stride_h - (int)pad_top;
  const int base_w = (int)out_w * (int)stride_w - (int)pad_left;
  for (uint in_c_in_group = 0u; in_c_in_group < input_channels_per_group;
       ++in_c_in_group) {
    const uint in_c = group * input_channels_per_group + in_c_in_group;
    if (in_c >= input_channels) {
      continue;
    }
    for (uint kh = 0u; kh < kernel_h; ++kh) {
      const int in_h = base_h + (int)kh * (int)dilation_h;
      if (in_h < 0 || in_h >= (int)input_h) {
        continue;
      }
      for (uint kw = 0u; kw < kernel_w; ++kw) {
        const int in_w = base_w + (int)kw * (int)dilation_w;
        if (in_w < 0 || in_w >= (int)input_w) {
          continue;
        }
        const uint src_index =
            ((batch * input_channels + in_c) * input_h + (uint)in_h) * input_w +
            (uint)in_w;
        const uint weight_index =
            (((group * output_channels_per_group + out_c_in_group) *
                  input_channels_per_group +
              in_c_in_group) *
                 kernel_h +
             kh) *
                kernel_w +
            kw;
        acc += src[src_index] * weights[weight_index];
      }
    }
  }
  dst[gid] = acc;
}
