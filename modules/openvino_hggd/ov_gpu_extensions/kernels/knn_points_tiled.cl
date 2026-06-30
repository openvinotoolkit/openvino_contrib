/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * KNN Points - Tiled GPU kernel with local memory optimization
 *
 * Optimizations vs v2:
 *   1. Tile p2 (source points) into __local memory for cooperative loading
 *      → reduces global memory reads by TILE_SIZE× (256×)
 *   2. Max-tracking MinK (matches pytorch3d CUDA approach)
 *      → fixed O(K) scan on insert, no data shifting
 *   3. Explicit local work size (256,1,1) for workgroup cooperation
 *
 * Inputs:  p1 [B, N1, 3] - query points
 *          p2 [B, N2, 3] - source points
 * Output:  [B, N1, K*2] - packed [dists_0..K-1, idx_0..K-1]
 *
 * WorkSize: global=(N1, B, 1) local=(256, 1, 1)
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_SIZE 256
#define MAX_K 64

__kernel void knn_points_tiled_kernel(
    const __global INPUT0_TYPE* p1,
    const __global INPUT1_TYPE* p2,
          __global OUTPUT0_TYPE* output)
{
    const int local_id = get_local_id(0);
    const int query_idx = get_global_id(0);
    const int batch = get_global_id(1);

    if (batch >= INPUT0_DIMS[0]) return;

    const int N1 = INPUT0_DIMS[1];
    const int N2 = INPUT1_DIMS[1];
    const int K2 = OUTPUT0_DIMS[2];
    const int K = K2 / 2;
    const bool valid = (query_idx < N1);

    /* ── Tile buffer in local memory (256 × 3 floats = 3 KB) ── */
    __local float tile_x[TILE_SIZE];
    __local float tile_y[TILE_SIZE];
    __local float tile_z[TILE_SIZE];

    /* ── Load query point ── */
    float qx = 0.0f, qy = 0.0f, qz = 0.0f;
    if (valid) {
        const int p1_base = batch * INPUT0_PITCHES[0]
                          + query_idx * INPUT0_PITCHES[1];
        qx = (float)p1[p1_base + 0 * INPUT0_PITCHES[2]];
        qy = (float)p1[p1_base + 1 * INPUT0_PITCHES[2]];
        qz = (float)p1[p1_base + 2 * INPUT0_PITCHES[2]];
    }

    /* ── MinK with max-tracking (matches pytorch3d MinK::add) ── */
    float best_dists[MAX_K];
    int   best_idx[MAX_K];
    float max_dist = 1e10f;
    int   max_pos  = 0;

    for (int k = 0; k < K; k++) {
        best_dists[k] = 1e10f;
        best_idx[k]   = 0;
    }

    /* ── Tiled scan over source points ── */
    for (int tile_start = 0; tile_start < N2; tile_start += TILE_SIZE) {

        /* Cooperative load: each work-item loads one source point */
        const int load_idx = tile_start + local_id;
        if (load_idx < N2) {
            const int p2_base = batch * INPUT1_PITCHES[0]
                              + load_idx * INPUT1_PITCHES[1];
            tile_x[local_id] = (float)p2[p2_base + 0 * INPUT1_PITCHES[2]];
            tile_y[local_id] = (float)p2[p2_base + 1 * INPUT1_PITCHES[2]];
            tile_z[local_id] = (float)p2[p2_base + 2 * INPUT1_PITCHES[2]];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        /* Each work-item scans the tile against its query point */
        if (valid) {
            const int tile_end = min(TILE_SIZE, N2 - tile_start);
            for (int j = 0; j < tile_end; j++) {
                const float dx = qx - tile_x[j];
                const float dy = qy - tile_y[j];
                const float dz = qz - tile_z[j];
                const float dist_sq = dx*dx + dy*dy + dz*dz;

                if (dist_sq < max_dist) {
                    /* Replace worst entry, then rescan for new max */
                    best_dists[max_pos] = dist_sq;
                    best_idx[max_pos]   = tile_start + j;

                    max_dist = dist_sq;
                    max_pos  = 0;
                    for (int k = 0; k < K; k++) {
                        if (best_dists[k] > max_dist) {
                            max_dist = best_dists[k];
                            max_pos  = k;
                        }
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* ── Sort results (bubble sort, K≤8 so negligible) ── */
    if (valid) {
        for (int i = 0; i < K - 1; i++) {
            for (int j2 = 0; j2 < K - i - 1; j2++) {
                if (best_dists[j2 + 1] < best_dists[j2]) {
                    const float td = best_dists[j2];
                    const int   ti = best_idx[j2];
                    best_dists[j2]     = best_dists[j2 + 1];
                    best_idx[j2]       = best_idx[j2 + 1];
                    best_dists[j2 + 1] = td;
                    best_idx[j2 + 1]   = ti;
                }
            }
        }

        /* Write packed output: [dists_0..K-1, idx_0..K-1] */
        const int out_base = batch * OUTPUT0_PITCHES[0]
                           + query_idx * OUTPUT0_PITCHES[1];
        for (int k = 0; k < K; k++) {
            output[out_base + k * OUTPUT0_PITCHES[2]]
                = (OUTPUT0_TYPE)best_dists[k];
            output[out_base + (K + k) * OUTPUT0_PITCHES[2]]
                = (OUTPUT0_TYPE)best_idx[k];
        }
    }
}
