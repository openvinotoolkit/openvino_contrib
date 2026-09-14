// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// Voxelization stage 5: emit the cell-to-pillar lookup.
// ---------------------------------------------------------------------------
// One work-item processes each cell. Zero denotes an empty cell; other values
// contain the pillar index plus one.

__kernel void pp_voxelization_cells(__global const INPUT0_TYPE* meta_storage,
                                    __global OUTPUT0_TYPE* table) {
    __global const int* meta = (__global const int*)meta_storage;

    const int cell = get_global_id(0);
    if (cell >= GRID_SIZE) return;

    int num_pillars = meta[META_COUNTER];
    if (num_pillars > MAX_VOXELS) num_pillars = MAX_VOXELS;

    const int slot = meta[META_G2P + cell];
    table[cell] = (OUTPUT0_TYPE)((slot > 0 && slot <= num_pillars) ? slot : 0);
}
