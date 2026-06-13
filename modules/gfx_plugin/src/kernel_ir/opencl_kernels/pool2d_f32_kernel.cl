// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

__kernel void gfx_opencl_generated_pool2d_f32(__global const float* src,
                                              __global float* dst,
                                              uint count,
                                              uint n_total,
                                              uint channels,
                                              uint h_in,
                                              uint w_in,
                                              uint k_h,
                                              uint k_w,
                                              uint stride_h,
                                              uint stride_w,
                                              uint dilation_h,
                                              uint dilation_w,
                                              uint pad_top,
                                              uint pad_left,
                                              uint pad_bottom,
                                              uint pad_right,
                                              uint h_out,
                                              uint w_out,
                                              uint is_avg,
                                              uint exclude_pad) {
    (void)pad_bottom;
    (void)pad_right;

    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }

    uint tmp = gid;
    const uint out_w = tmp % w_out;
    tmp /= w_out;
    const uint out_h = tmp % h_out;
    tmp /= h_out;
    const uint channel = tmp % channels;
    tmp /= channels;
    const uint batch = tmp;
    if (batch >= n_total) {
        return;
    }

    float acc = is_avg != 0u ? 0.0f : -3.402823466e+38f;
    uint sample_count = 0u;
    const int base_h = (int)out_h * (int)stride_h - (int)pad_top;
    const int base_w = (int)out_w * (int)stride_w - (int)pad_left;
    for (uint kh = 0u; kh < k_h; ++kh) {
        const int in_h = base_h + (int)kh * (int)dilation_h;
        for (uint kw = 0u; kw < k_w; ++kw) {
            const int in_w = base_w + (int)kw * (int)dilation_w;
            const uint inside = in_h >= 0 && in_w >= 0 &&
                                in_h < (int)h_in && in_w < (int)w_in;
            if (inside == 0u) {
                if (is_avg != 0u && exclude_pad == 0u) {
                    ++sample_count;
                }
                continue;
            }

            const uint src_index =
                ((batch * channels + channel) * h_in + (uint)in_h) * w_in +
                (uint)in_w;
            const float value = src[src_index];
            if (is_avg != 0u) {
                acc += value;
                ++sample_count;
            } else {
                acc = fmax(acc, value);
            }
        }
    }

    if (is_avg != 0u) {
        acc = sample_count == 0u ? 0.0f : acc / (float)sample_count;
    }
    dst[gid] = acc;
}
