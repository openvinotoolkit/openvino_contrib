// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// BEV stage: dense scatter.
// ---------------------------------------------------------------------------
// Output layout: NCHW [1, BEV_C, BEV_H, BEV_W].
//
// One work-group owns row y = group_id:
//   Phase 1  clear the row's grid_to_pillar lookup in local memory (BEV_W ints).
//   Phase 2  scan the pillar list and record the pillars landing in this row.
//   Phase 3  write the row for all BEV_C channels, emitting either a pillar
//            feature or zero.
//
// Each work-group owns one row and writes every output cell in that row.

__kernel __attribute__((reqd_work_group_size(BEV_ROW_WG, 1, 1)))
void pp_scatter_bev(__global const INPUT0_TYPE* features,
                    __global const INPUT1_TYPE* coords,
                    __global const INPUT2_TYPE* params,
                    __global OUTPUT0_TYPE* bev) {
    __local int row_pillar[BEV_W];

    const int lid = get_local_id(0);
    const int y = (int)get_group_id(0);

    // Phase 1: mark every cell of this row empty.
    for (int x = lid; x < BEV_W; x += BEV_ROW_WG) row_pillar[x] = -1;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: map row cells to pillars.
    int num_pillars = (int)params[0];
    num_pillars = min(num_pillars, BEV_MAX_VOXELS);

    for (int pillar = lid; pillar < num_pillars; pillar += BEV_ROW_WG) {
        if ((int)coords[pillar * COORD_VALUES + COORD_Y] != y) continue;
        const int x = (int)coords[pillar * COORD_VALUES + COORD_X];
        if (x < 0 || x >= BEV_W) continue;
        row_pillar[x] = pillar;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 3: write this row for every channel.
    __global OUTPUT0_TYPE* row = bev + (long)y * BEV_W;
    for (int job = lid; job < BEV_C * BEV_X_CHUNKS; job += BEV_ROW_WG) {
        const int channel = job / BEV_X_CHUNKS;
        const int x0 = (job - channel * BEV_X_CHUNKS) * BEV_VEC;
        __global OUTPUT0_TYPE* dst = row + (long)channel * BEV_PLANE + x0;

        __attribute__((opencl_unroll_hint(BEV_VEC)))
        for (int k = 0; k < BEV_VEC; ++k) {
            const int pillar = row_pillar[x0 + k];
            dst[k] = (pillar >= 0) ? (OUTPUT0_TYPE)features[(long)pillar * BEV_C + channel]
                                   : (OUTPUT0_TYPE)0;
        }
    }
}
