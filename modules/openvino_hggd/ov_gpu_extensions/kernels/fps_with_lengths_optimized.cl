/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Farthest Point Sampling with Lengths - Optimized O(N×K) kernel
 *
 * Same persistent min_dist optimization as fps_single_optimized.cl,
 * with variable-length batch support (each batch has different valid_len).
 *
 * Input 0: points  [B, N, 3] (may be zero-padded)
 * Input 1: lengths [B]       (actual valid length per batch)
 * Output:  [B, K, 4] packed [x, y, z, idx]
 *
 * WorkSize: global=(256, B, 1) local=(256, 1, 1)
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WORKGROUP_SIZE 256
#define MAX_N 8192

__kernel void fps_with_lengths_optimized_kernel(
    const __global INPUT0_TYPE* points,
    const __global INPUT1_TYPE* lengths,
          __global OUTPUT0_TYPE* output)
{
    const int local_id = get_local_id(0);
    const int batch = get_global_id(1);

    if (batch >= INPUT0_DIMS[0]) return;

    const int N = INPUT0_DIMS[1];
    const int K = OUTPUT0_DIMS[1];

    /* Get actual valid length, clamped to N and MAX_N */
    int valid_len = (int)lengths[batch];
    if (valid_len > N) valid_len = N;
    if (valid_len > MAX_N) valid_len = MAX_N;

    const int k_actual = (K < valid_len) ? K : valid_len;

    /* ── Shared state ── */
    __local float min_dist[MAX_N];
    __local float local_max_dist[WORKGROUP_SIZE];
    __local int   local_max_idx[WORKGROUP_SIZE];
    __local int   selected;

    /* ── Handle empty batch ── */
    if (valid_len <= 0) {
        if (local_id == 0) {
            for (int k = 0; k < K; k++) {
                const int out_base = batch * OUTPUT0_PITCHES[0] + k * OUTPUT0_PITCHES[1];
                output[out_base + 0 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)0.0f;
                output[out_base + 1 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)0.0f;
                output[out_base + 2 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)0.0f;
                output[out_base + 3 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)0;
            }
        }
        return;
    }

    /* ── Initialize min_dist to infinity ── */
    for (int i = local_id; i < valid_len; i += WORKGROUP_SIZE)
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
    for (int k = 1; k < k_actual; k++) {

        const int sel = selected;
        const int sel_base = batch * INPUT0_PITCHES[0] + sel * INPUT0_PITCHES[1];
        const float sx = (float)points[sel_base + 0 * INPUT0_PITCHES[2]];
        const float sy = (float)points[sel_base + 1 * INPUT0_PITCHES[2]];
        const float sz = (float)points[sel_base + 2 * INPUT0_PITCHES[2]];

        float my_max = -1.0f;
        int   my_idx = 0;

        for (int i = local_id; i < valid_len; i += WORKGROUP_SIZE) {
            const int pi_base = batch * INPUT0_PITCHES[0] + i * INPUT0_PITCHES[1];
            const float dx = (float)points[pi_base + 0 * INPUT0_PITCHES[2]] - sx;
            const float dy = (float)points[pi_base + 1 * INPUT0_PITCHES[2]] - sy;
            const float dz = (float)points[pi_base + 2 * INPUT0_PITCHES[2]] - sz;
            const float dist = dx*dx + dy*dy + dz*dz;

            float md = min_dist[i];
            md = fmin(md, dist);
            min_dist[i] = md;

            if (md > my_max) {
                my_max = md;
                my_idx = i;
            }
        }

        /* ── Workgroup reduction ── */
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

    /* ── Pad remaining slots if K > k_actual ── */
    if (local_id == 0 && k_actual < K) {
        const int last_out = batch * OUTPUT0_PITCHES[0] + (k_actual - 1) * OUTPUT0_PITCHES[1];
        const float lx = (float)output[last_out + 0 * OUTPUT0_PITCHES[2]];
        const float ly = (float)output[last_out + 1 * OUTPUT0_PITCHES[2]];
        const float lz = (float)output[last_out + 2 * OUTPUT0_PITCHES[2]];
        const float li = (float)output[last_out + 3 * OUTPUT0_PITCHES[2]];

        for (int k = k_actual; k < K; k++) {
            const int out_base = batch * OUTPUT0_PITCHES[0] + k * OUTPUT0_PITCHES[1];
            output[out_base + 0 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)lx;
            output[out_base + 1 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)ly;
            output[out_base + 2 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)lz;
            output[out_base + 3 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)li;
        }
    }
}
