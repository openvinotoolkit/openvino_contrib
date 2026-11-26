// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// Voxelization stage 1: scatter points and compact pillars.
// ---------------------------------------------------------------------------
// With VOX_DETERMINISTIC the kept points and pillars follow input order, as on
// the CPU: each pillar keeps its first POINTS_PER_VOXEL points, and above
// MAX_VOXELS pillars the pillars with the earliest first point are kept. Slot
// order is free; the finalize stage sorts each pillar's point indices.
// Otherwise the first atomic arrivals are kept.
// Phase 1 clears the counter and cell-to-pillar lookup.
// Phase 2 assigns pillar ids to occupied cells, renumbered in input order on overflow.
// Phase 3 appends source point indices to pillar slots.
// Phase 4 puts the earliest points into pillars that got more than POINTS_PER_VOXEL.

#ifndef VOX_DETERMINISTIC
#error "VOX_DETERMINISTIC must come from pp_tuning_config_generated.cl"
#endif

inline bool point_cell(__global const INPUT0_TYPE* points, int pid, int* cell, int* ix, int* iy, int* iz) {
    const int base = pid * POINT_FEATURES;
    if (!point_grid((float)points[base + 0], (float)points[base + 1], (float)points[base + 2], ix, iy, iz)) {
        return false;
    }
    *cell = cell_index(*ix, *iy);
    return true;
}

#if VOX_DETERMINISTIC
#define EMPTY_SLOT INT_MAX
// Late points one work-item holds for phase 4; more falls back to a full point pass.
#define LATE_POINT_QUEUE 16
// While renumbering, the lookup holds pid - MAX_POINTS (< 0) for the cell's first point.
#define FIRST_POINT_KEY(pid) ((pid) - MAX_POINTS)

// Exclusive prefix sum over the work-group; total receives the sum of all values.
inline int exclusive_sum(__local int* scratch, int value, int* total) {
    const int lid = get_local_id(0);
    scratch[lid] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = 1; offset < SCATTER_WG; offset <<= 1) {
        const int addend = lid >= offset ? scratch[lid - offset] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        scratch[lid] += addend;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const int inclusive = scratch[lid];
    *total = scratch[SCATTER_WG - 1];
    barrier(CLK_LOCAL_MEM_FENCE);
    return inclusive - value;
}

// Number the pillars in the order of their first point. Each work-item owns a
// contiguous point range, so a prefix sum of first-point counts gives the rank.
// Phase 2 already set the count and batch of every pillar id below MAX_VOXELS.
inline void renumber_pillars(__global const INPUT0_TYPE* points, __global int* meta, int num_points,
                             __local int* scratch) {
    const int lid = get_local_id(0);
    for (int grid = lid; grid < GRID_SIZE; grid += SCATTER_WG) {
        meta[META_G2P + grid] = 0;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int pid = lid; pid < num_points; pid += SCATTER_WG) {
        int cell, ix, iy, iz;
        if (point_cell(points, pid, &cell, &ix, &iy, &iz)) {
            atomic_min((volatile __global int*)&meta[META_G2P + cell], FIRST_POINT_KEY(pid));
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    const int range = (num_points + SCATTER_WG - 1) / SCATTER_WG;
    const int range_begin = min(lid * range, num_points);
    const int range_end = min(range_begin + range, num_points);
    int first_count = 0;
    for (int pid = range_begin; pid < range_end; pid++) {
        int cell, ix, iy, iz;
        if (point_cell(points, pid, &cell, &ix, &iy, &iz) && meta[META_G2P + cell] == FIRST_POINT_KEY(pid)) {
            first_count++;
        }
    }
    int pillar_count;
    int pillar = exclusive_sum(scratch, first_count, &pillar_count);
    for (int pid = range_begin; pid < range_end; pid++) {
        int cell, ix, iy, iz;
        if (!point_cell(points, pid, &cell, &ix, &iy, &iz) || meta[META_G2P + cell] != FIRST_POINT_KEY(pid)) {
            continue;
        }
        if (pillar < MAX_VOXELS) {
            const int coord = META_COORD + pillar * COORD_VALUES;
            meta[coord + COORD_Z] = iz;
            meta[coord + COORD_Y] = iy;
            meta[coord + COORD_X] = ix;
        }
        meta[META_G2P + cell] = pillar + 1;
        pillar++;
    }
    if (lid == 0) meta[META_COUNTER] = pillar_count;
    barrier(CLK_GLOBAL_MEM_FENCE);
}

// Insert a point index into an ascending slot list. atomic_min keeps the smaller
// index and carries the larger one on, so concurrent inserts leave the
// POINTS_PER_VOXEL smallest indices. Slots only decrease, so a slot that is
// already below the carried index can be skipped.
inline void insert_point(volatile __global int* slots, int value) {
    if (slots[POINTS_PER_VOXEL - 1] < value) {
        return;
    }
    for (int slot = 0; slot < POINTS_PER_VOXEL; slot++) {
        if (slots[slot] < value) {
            continue;
        }
        const int previous = atomic_min(&slots[slot], value);
        if (previous == EMPTY_SLOT) {
            return;
        }
        value = max(previous, value);
    }
}
#endif

__kernel __attribute__((reqd_work_group_size(SCATTER_WG, 1, 1)))
void pp_voxelization_scatter(__global const INPUT0_TYPE* points,
                             __global const INPUT1_TYPE* num_points_in,
                             __global OUTPUT0_TYPE* meta_storage) {
    __global int* meta = (__global int*)meta_storage;
    volatile __global int* counter = (volatile __global int*)meta_storage;
#if VOX_DETERMINISTIC
    __local int scratch[SCATTER_WG];
    __local int queue_full;
#endif

    const int lid = get_local_id(0);
    const int num_points = min((int)num_points_in[0], MAX_POINTS);

    // Phase 1: zero the counter and cell-to-pillar lookup.
    if (lid == 0) {
        *counter = 0;
#if VOX_DETERMINISTIC
        queue_full = 0;
#endif
    }
    for (int grid = lid; grid < GRID_SIZE; grid += SCATTER_WG) {
        meta[META_G2P + grid] = 0;
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    // Phase 2: create one pillar per occupied grid cell.
    // The lookup stores 0 for empty, -1 while claimed, and pillar+1 when ready.
    for (int pid = lid; pid < num_points; pid += SCATTER_WG) {
        int grid, ix, iy, iz;
        if (!point_cell(points, pid, &grid, &ix, &iy, &iz)) {
            continue;
        }

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

#if VOX_DETERMINISTIC
    // Arrival order picks the pillars on overflow; renumber them in input order.
    if (meta[META_COUNTER] > MAX_VOXELS) renumber_pillars(points, meta, num_points, scratch);

    // Phase 3: append each point's index into its pillar. Points that arrive
    // after the pillar is full are held for phase 4.
    int late_points[LATE_POINT_QUEUE];
    int late_count = 0;
#endif
    for (int pid = lid; pid < num_points; pid += SCATTER_WG) {
        int cell, ix, iy, iz;
        if (!point_cell(points, pid, &cell, &ix, &iy, &iz)) {
            continue;
        }

        const int pillar = meta[META_G2P + cell] - 1;
        if (pillar < 0 || pillar >= MAX_VOXELS) {
            continue;
        }

        const int slot = atomic_inc((volatile __global int*)&meta[META_COUNT + pillar]);
        if (slot < POINTS_PER_VOXEL) {
            meta[META_RAW + pillar * META_RAW_STRIDE + slot] = pid;
        }
#if VOX_DETERMINISTIC
        else if (late_count < LATE_POINT_QUEUE) {
            late_points[late_count++] = pid;
        } else {
            queue_full = 1;
        }
#endif
    }
#if VOX_DETERMINISTIC
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    // Phase 4: in each pillar that got more than POINTS_PER_VOXEL points, keep
    // the smallest point indices.
    const int num_pillars = min(meta[META_COUNTER], MAX_VOXELS);
    if (queue_full) {
        // Rebuild those pillars from every point.
        for (int pillar = lid; pillar < num_pillars; pillar += SCATTER_WG) {
            if (meta[META_COUNT + pillar] <= POINTS_PER_VOXEL) {
                continue;
            }
            for (int slot = 0; slot < POINTS_PER_VOXEL; slot++) {
                meta[META_RAW + pillar * META_RAW_STRIDE + slot] = EMPTY_SLOT;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        for (int pid = lid; pid < num_points; pid += SCATTER_WG) {
            int cell, ix, iy, iz;
            if (!point_cell(points, pid, &cell, &ix, &iy, &iz)) {
                continue;
            }
            const int pillar = meta[META_G2P + cell] - 1;
            if (pillar < 0 || pillar >= MAX_VOXELS || meta[META_COUNT + pillar] <= POINTS_PER_VOXEL) {
                continue;
            }
            insert_point((volatile __global int*)&meta[META_RAW + pillar * META_RAW_STRIDE], pid);
        }
        return;
    }

    // Sort the full slot lists, then insert the held points.
    for (int pillar = lid; pillar < num_pillars; pillar += SCATTER_WG) {
        if (meta[META_COUNT + pillar] <= POINTS_PER_VOXEL) {
            continue;
        }
        __global int* slots = &meta[META_RAW + pillar * META_RAW_STRIDE];
        int point_ids[POINTS_PER_VOXEL];
        for (int slot = 0; slot < POINTS_PER_VOXEL; slot++) point_ids[slot] = slots[slot];
        sort_point_ids(point_ids, POINTS_PER_VOXEL);
        for (int slot = 0; slot < POINTS_PER_VOXEL; slot++) slots[slot] = point_ids[slot];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int late = 0; late < late_count; late++) {
        const int pid = late_points[late];
        int cell, ix, iy, iz;
        point_cell(points, pid, &cell, &ix, &iy, &iz);
        const int pillar = meta[META_G2P + cell] - 1;
        insert_point((volatile __global int*)&meta[META_RAW + pillar * META_RAW_STRIDE], pid);
    }
#endif
}
