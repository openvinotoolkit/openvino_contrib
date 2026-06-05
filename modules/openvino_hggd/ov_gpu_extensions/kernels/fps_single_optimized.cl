/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Farthest Point Sampling - Optimized O(N×K) kernel
 *
 * Key optimization vs fps_single_v2 (O(N×K²)):
 *   Maintains persistent min_dist[N] in __local memory.
 *   Each iteration only computes distance to the LAST selected point,
 *   then updates min_dist[i] = min(min_dist[i], dist_to_last).
 *   This matches the pytorch3d CUDA approach exactly.
 *
 *   With N=3000, K=512:
 *     Old: each iteration scans all k previously selected → N×K×(K/2) = 393M ops
 *     New: each iteration scans 1 selected point        → N×K        = 1.5M ops
 *
 * Input:  points [B, N, 3]
 * Output: [B, K, 4] packed [x, y, z, idx]
 *
 * WorkSize: global=(256, B, 1) local=(256, 1, 1)
 *
 * Local memory: MAX_N floats (16KB) + reduction arrays (2KB) = ~18KB
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WORKGROUP_SIZE 256
#define MAX_N 8192

__kernel void fps_single_optimized_kernel(
    const __global INPUT0_TYPE* points,
          __global OUTPUT0_TYPE* output)
{
    const int local_id = get_local_id(0);
    const int batch = get_global_id(1);

    if (batch >= INPUT0_DIMS[0]) return;

    const int N = INPUT0_DIMS[1];
    const int K = OUTPUT0_DIMS[1];
    const int n_eff = (N < MAX_N) ? N : MAX_N;

    /* ── Shared state ── */
    __local float min_dist[MAX_N];          /* persistent per-point min distance */
    __local float local_max_dist[WORKGROUP_SIZE];
    __local int   local_max_idx[WORKGROUP_SIZE];
    __local int   selected;                 /* index of last selected point       */

    /* ── Initialize min_dist to infinity ── */
    for (int i = local_id; i < n_eff; i += WORKGROUP_SIZE)
        min_dist[i] = 1e10f;
    barrier(CLK_LOCAL_MEM_FENCE);

    /* ── Select point 0 as first sample ── */
    if (local_id == 0) {
        selected = 0;
        const int p_base = batch * INPUT0_PITCHES[0];
        const float px = (float)points[p_base + 0 * INPUT0_PITCHES[2]];
        const float py = (float)points[p_base + 1 * INPUT0_PITCHES[2]];
        const float pz = (float)points[p_base + 2 * INPUT0_PITCHES[2]];

        const int out_base = batch * OUTPUT0_PITCHES[0];
        output[out_base + 0 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)px;
        output[out_base + 1 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)py;
        output[out_base + 2 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)pz;
        output[out_base + 3 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* ── Main FPS loop ── */
    for (int k = 1; k < K; k++) {

        /* Load last selected point from global input (FP32 precision) */
        const int sel = selected;
        const int sel_base = batch * INPUT0_PITCHES[0] + sel * INPUT0_PITCHES[1];
        const float sx = (float)points[sel_base + 0 * INPUT0_PITCHES[2]];
        const float sy = (float)points[sel_base + 1 * INPUT0_PITCHES[2]];
        const float sz = (float)points[sel_base + 2 * INPUT0_PITCHES[2]];

        /* Update min_dist for assigned points & find local argmax */
        float my_max = -1.0f;
        int   my_idx = 0;

        for (int i = local_id; i < n_eff; i += WORKGROUP_SIZE) {
            const int pi_base = batch * INPUT0_PITCHES[0] + i * INPUT0_PITCHES[1];
            const float dx = (float)points[pi_base + 0 * INPUT0_PITCHES[2]] - sx;
            const float dy = (float)points[pi_base + 1 * INPUT0_PITCHES[2]] - sy;
            const float dz = (float)points[pi_base + 2 * INPUT0_PITCHES[2]] - sz;
            const float dist = dx*dx + dy*dy + dz*dz;

            /* Persistent update: min of old distance and distance to last selected */
            float md = min_dist[i];
            md = fmin(md, dist);
            min_dist[i] = md;

            if (md > my_max) {
                my_max = md;
                my_idx = i;
            }
        }

        /* ── Workgroup reduction: find global argmax ── */
        local_max_dist[local_id] = my_max;
        local_max_idx[local_id]  = my_idx;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int stride = WORKGROUP_SIZE / 2; stride > 0; stride >>= 1) {
            if (local_id < stride) {
                if (local_max_dist[local_id + stride] > local_max_dist[local_id]) {
                    local_max_dist[local_id] = local_max_dist[local_id + stride];
                    local_max_idx[local_id]  = local_max_idx[local_id + stride];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        /* Work item 0 writes selected point to output */
        if (local_id == 0) {
            const int best = local_max_idx[0];
            selected = best;

            const int pb = batch * INPUT0_PITCHES[0] + best * INPUT0_PITCHES[1];
            const int out_base = batch * OUTPUT0_PITCHES[0] + k * OUTPUT0_PITCHES[1];
            output[out_base + 0 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)points[pb + 0 * INPUT0_PITCHES[2]];
            output[out_base + 1 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)points[pb + 1 * INPUT0_PITCHES[2]];
            output[out_base + 2 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)points[pb + 2 * INPUT0_PITCHES[2]];
            output[out_base + 3 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)best;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
