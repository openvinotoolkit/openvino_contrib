// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// Voxelization stage 1: scatter points and compact pillars.
// ---------------------------------------------------------------------------
// Phase 1 clears the counter and cell-to-pillar lookup.
// Phase 2 assigns pillar ids to occupied cells.
// Phase 3 appends source point indices to pillar slots.

__kernel __attribute__((reqd_work_group_size(SCATTER_WG, 1, 1)))
void pp_voxelization_scatter(__global const INPUT0_TYPE* points,
                             __global const INPUT1_TYPE* num_points_in,
                             __global OUTPUT0_TYPE* meta_storage) {
    __global int* meta = (__global int*)meta_storage;
    volatile __global int* counter = (volatile __global int*)meta_storage;

    const int lid = get_local_id(0);
    const int num_points = min((int)num_points_in[0], MAX_POINTS);

    // Phase 1: zero the counter and cell-to-pillar lookup.
    if (lid == 0) *counter = 0;
    for (int grid = lid; grid < GRID_SIZE; grid += SCATTER_WG) {
        meta[META_G2P + grid] = 0;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Phase 2: create one pillar per occupied grid cell.
    // The lookup stores 0 for empty, -1 while claimed, and pillar+1 when ready.
    for (int pid = lid; pid < num_points; pid += SCATTER_WG) {
        const int base = pid * POINT_FEATURES;
        int ix, iy, iz;
        if (!point_grid((float)points[base + 0], (float)points[base + 1], (float)points[base + 2], &ix, &iy, &iz))
            continue;
        const int grid = cell_index(ix, iy);

        volatile __global int* cell = (volatile __global int*)&meta[META_G2P + grid];
        if (atomic_cmpxchg(cell, 0, -1) == 0) {
            const int pillar = atomic_inc(counter);
            if (pillar < MAX_VOXELS) {
                const int coord = META_COORD + pillar * COORD_VALUES;
                meta[META_COUNT + pillar] = 0;
                meta[coord + COORD_BATCH] = 0;
                meta[coord + COORD_Z] = iz;
                meta[coord + COORD_Y] = iy;
                meta[coord + COORD_X] = ix;
            }
            *cell = pillar + 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Phase 3: append each point's index into its pillar.
    for (int pid = lid; pid < num_points; pid += SCATTER_WG) {
        const int base = pid * POINT_FEATURES;
        int ix, iy, iz;
        if (!point_grid((float)points[base + 0], (float)points[base + 1], (float)points[base + 2], &ix, &iy, &iz))
            continue;

        const int pillar = meta[META_G2P + cell_index(ix, iy)] - 1;
        if (pillar < 0 || pillar >= MAX_VOXELS) continue;

        // Limit each pillar to POINTS_PER_VOXEL entries.
        const int slot = atomic_inc((volatile __global int*)&meta[META_COUNT + pillar]);
        if (slot < POINTS_PER_VOXEL) {
            meta[META_RAW + pillar * META_RAW_STRIDE + slot] = pid;
        }
    }
}
