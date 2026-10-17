// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

static inline float gfx_interpolate_source_coord(uint out_idx,
                                                 uint in_size,
                                                 uint out_size,
                                                 uint align_corners,
                                                 uint use_half_pixel) {
    if (align_corners != 0u && out_size > 1u) {
        return (float)out_idx * (float)(in_size - 1u) / (float)(out_size - 1u);
    }
    const float scale = out_size != 0u ? (float)in_size / (float)out_size : 1.0f;
    if (use_half_pixel != 0u) {
        return ((float)out_idx + 0.5f) * scale - 0.5f;
    }
    return (float)out_idx * scale;
}

static inline int gfx_interpolate_nearest_index(float coord,
                                                uint in_size,
                                                uint nearest_mode) {
    if (nearest_mode == 1u) {
        return clamp((int)floor(coord), 0, (int)in_size - 1);
    }
    if (nearest_mode == 2u) {
        return clamp((int)ceil(coord), 0, (int)in_size - 1);
    }
    return clamp((int)round(coord), 0, (int)in_size - 1);
}

__kernel void gfx_opencl_generated_interpolate_f32(__global const float* src,
                                                   __global float* dst,
                                                   uint count,
                                                   uint nearest,
                                                   uint align_corners,
                                                   uint use_half_pixel,
                                                   uint nearest_mode,
                                                   uint n_total,
                                                   uint channels,
                                                   uint h_in,
                                                   uint w_in,
                                                   uint h_out,
                                                   uint w_out) {
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

    const float src_h = gfx_interpolate_source_coord(out_h,
                                                     h_in,
                                                     h_out,
                                                     align_corners,
                                                     use_half_pixel);
    const float src_w = gfx_interpolate_source_coord(out_w,
                                                     w_in,
                                                     w_out,
                                                     align_corners,
                                                     use_half_pixel);
    const uint base = ((batch * channels + channel) * h_in) * w_in;
    if (nearest != 0u) {
        const int in_h = gfx_interpolate_nearest_index(src_h, h_in, nearest_mode);
        const int in_w = gfx_interpolate_nearest_index(src_w, w_in, nearest_mode);
        dst[gid] = src[base + (uint)in_h * w_in + (uint)in_w];
        return;
    }

    const float floor_h = floor(src_h);
    const float floor_w = floor(src_w);
    const int h0 = clamp((int)floor_h, 0, (int)h_in - 1);
    const int w0 = clamp((int)floor_w, 0, (int)w_in - 1);
    const int h1 = min(h0 + 1, (int)h_in - 1);
    const int w1 = min(w0 + 1, (int)w_in - 1);
    const float dh = src_h - floor_h;
    const float dw = src_w - floor_w;

    const float v00 = src[base + (uint)h0 * w_in + (uint)w0];
    const float v01 = src[base + (uint)h0 * w_in + (uint)w1];
    const float v10 = src[base + (uint)h1 * w_in + (uint)w0];
    const float v11 = src[base + (uint)h1 * w_in + (uint)w1];
    const float v0 = v00 + (v01 - v00) * dw;
    const float v1 = v10 + (v11 - v10) * dw;
    dst[gid] = v0 + (v1 - v0) * dh;
}
