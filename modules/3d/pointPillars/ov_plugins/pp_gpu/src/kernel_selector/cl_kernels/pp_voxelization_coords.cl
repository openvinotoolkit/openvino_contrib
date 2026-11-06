// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// Voxelization stage 3: emit pillar coordinates.
// ---------------------------------------------------------------------------
// One work-item copies the [batch, z, y, x] coordinate record for a pillar.

__kernel void pp_voxelization_coords(__global const INPUT0_TYPE* meta_storage,
                                     __global OUTPUT0_TYPE* coords) {
    __global const int* meta = (__global const int*)meta_storage;

    const int pillar = get_global_id(0);
    int num_pillars = meta[META_COUNTER];
    if (num_pillars > MAX_VOXELS) num_pillars = MAX_VOXELS;
    if (pillar >= num_pillars) return;

    coords[pillar * COORD_VALUES + COORD_BATCH] = (OUTPUT0_TYPE)meta[META_COORD + pillar * COORD_VALUES + COORD_BATCH];
    coords[pillar * COORD_VALUES + COORD_Z] = (OUTPUT0_TYPE)meta[META_COORD + pillar * COORD_VALUES + COORD_Z];
    coords[pillar * COORD_VALUES + COORD_Y] = (OUTPUT0_TYPE)meta[META_COORD + pillar * COORD_VALUES + COORD_Y];
    coords[pillar * COORD_VALUES + COORD_X] = (OUTPUT0_TYPE)meta[META_COORD + pillar * COORD_VALUES + COORD_X];
}
