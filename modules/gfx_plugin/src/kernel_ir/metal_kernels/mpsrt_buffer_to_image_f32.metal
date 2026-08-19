// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <metal_stdlib>
using namespace metal;

struct GfxMpsrtImageBridgeParams {
    uint width;
    uint height;
    uint channels;
    uint batch;
};

inline uint gfx_mpsrt_image_bridge_slices(uint channels) {
    return (channels + 3u) / 4u;
}

inline uint gfx_mpsrt_image_bridge_nchw_index(constant GfxMpsrtImageBridgeParams& p,
                                              uint n,
                                              uint c,
                                              uint y,
                                              uint x) {
    return ((n * p.channels + c) * p.height + y) * p.width + x;
}

kernel void gfx_mpsrt_buffer_to_image_f32(device const float* src [[buffer(0)]],
                                          texture2d_array<float, access::write> dst [[texture(0)]],
                                          constant GfxMpsrtImageBridgeParams& p [[buffer(1)]],
                                          uint3 gid [[thread_position_in_grid]]) {
    const uint x = gid.x;
    const uint y = gid.y;
    const uint plane = gid.z;
    const uint slices = gfx_mpsrt_image_bridge_slices(p.channels);
    const uint n = plane / slices;
    const uint slice = plane - n * slices;
    if (x >= p.width || y >= p.height || n >= p.batch) {
        return;
    }
    float4 value = float4(0.0f);
    const uint c0 = slice * 4u;
    if (c0 + 0u < p.channels) value.x = src[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 0u, y, x)];
    if (c0 + 1u < p.channels) value.y = src[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 1u, y, x)];
    if (c0 + 2u < p.channels) value.z = src[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 2u, y, x)];
    if (c0 + 3u < p.channels) value.w = src[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 3u, y, x)];
    dst.write(value, uint2(x, y), plane);
}
