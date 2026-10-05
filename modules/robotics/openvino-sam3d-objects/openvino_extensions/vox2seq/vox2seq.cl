/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */


// SAM 3D Objects — Vox2Seq OpenCL Kernel
// Per-voxel parallel Z-order (Morton) encoding on GPU.

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

inline long spread_bits_3(uint v) {
    long x = (long)(v & 0x1FFFFF);
    x = (x | (x << 32)) & 0x1F00000000FFFF;
    x = (x | (x << 16)) & 0x1F0000FF0000FF;
    x = (x | (x << 8))  & 0x100F00F00F00F00F;
    x = (x | (x << 4))  & 0x10C30C30C30C30C3;
    x = (x | (x << 2))  & 0x1249249249249249;
    return x;
}

__kernel void morton_encode_3d(
    __global const int* restrict coords,  // [N, 3]
    int N,
    __global long* restrict codes)        // [N]
{
    int i = get_global_id(0);
    if (i >= N) return;

    int x = coords[i * 3 + 0];
    int y = coords[i * 3 + 1];
    int z = coords[i * 3 + 2];

    codes[i] = spread_bits_3((uint)x) |
               (spread_bits_3((uint)y) << 1) |
               (spread_bits_3((uint)z) << 2);
}
