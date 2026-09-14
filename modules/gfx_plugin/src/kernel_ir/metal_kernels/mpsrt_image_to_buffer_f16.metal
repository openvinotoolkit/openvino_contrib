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

kernel void gfx_mpsrt_image_to_buffer_f16(texture2d_array<half, access::read> src [[texture(0)]],
                                          device half* dst [[buffer(0)]],
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
    const half4 value = src.read(uint2(x, y), plane);
    const uint c0 = slice * 4u;
    if (c0 + 0u < p.channels) dst[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 0u, y, x)] = value.x;
    if (c0 + 1u < p.channels) dst[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 1u, y, x)] = value.y;
    if (c0 + 2u < p.channels) dst[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 2u, y, x)] = value.z;
    if (c0 + 3u < p.channels) dst[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 3u, y, x)] = value.w;
}
