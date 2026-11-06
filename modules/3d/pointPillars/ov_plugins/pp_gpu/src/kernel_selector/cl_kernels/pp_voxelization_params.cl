// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// Voxelization stage 4: emit the pillar count.
// ---------------------------------------------------------------------------

__kernel void pp_voxelization_params(__global const INPUT0_TYPE* meta_storage,
                                     __global OUTPUT0_TYPE* params) {
    __global const int* meta = (__global const int*)meta_storage;
    int num_pillars = meta[META_COUNTER];
    if (num_pillars > MAX_VOXELS) num_pillars = MAX_VOXELS;
    params[0] = (OUTPUT0_TYPE)num_pillars;
}