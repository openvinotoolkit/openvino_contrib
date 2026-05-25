// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

__kernel void ov_furthest_point_sampling(
    __global const INPUT0_TYPE* pts,  // (B, N, 3)
    __global const INPUT1_TYPE* npoint_ptr, // npoint scalar
    __global OUTPUT0_TYPE* output,    // (B, npoint)
    __global float* dist              // (B, N) internal buffer; keep FP32 under FP16 inference
) {
    int tid = get_global_id(0);
    int b   = get_global_id(1);  // batch index

    int B = INPUT0_DIMS[0];  // batch size
    int N = INPUT0_DIMS[1];  // number of points
    int C = INPUT0_DIMS[2];  // channels (should be 3)
    int npoint = INPUT1_DIMS[0];

    if (b >= B) return;

    // Offsets for batch memory
    int pts_offset  = b * N * C;
    int dist_offset = b * N;
    int out_offset  = b * npoint;

    int local_id = get_local_id(0);
    int local_size = get_local_size(0);

    // Keep squared distances in FP32. Quantizing these to FP16 changes the
    // argmax ordering and corrupts FPS point selection.
    for (int i = local_id; i < N; i += local_size)
        dist[dist_offset + i] = FLT_MAX;

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // Initialize first output point = 0
    if (local_id == 0) output[out_offset] = 0;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    __local float local_dist[256];
    __local int local_idx[256];

    for (int j = 1; j < npoint; ++j) {
        int last_idx = output[out_offset + j - 1];

        // Read coordinates and cast to float for computation
        float last_x = (float)pts[pts_offset + last_idx * 3 + 0];
        float last_y = (float)pts[pts_offset + last_idx * 3 + 1];
        float last_z = (float)pts[pts_offset + last_idx * 3 + 2];

        // update dist array in parallel
        for (int i = local_id; i < N; i += local_size) {
            float dx = (float)pts[pts_offset + i * 3 + 0] - last_x;
            float dy = (float)pts[pts_offset + i * 3 + 1] - last_y;
            float dz = (float)pts[pts_offset + i * 3 + 2] - last_z;
            float d = dx*dx + dy*dy + dz*dz;
            float cur_dist = dist[dist_offset + i];
            if (d < cur_dist)
                dist[dist_offset + i] = d;
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        // find max dist and argmax with local reduction
        float best_dist = -1.0f;
        int best_idx = -1;
        for (int i = local_id; i < N; i += local_size) {
            float v = dist[dist_offset + i];
            if (v > best_dist) {
                best_dist = v;
                best_idx = i;
            }
        }

        local_dist[local_id] = best_dist;
        local_idx[local_id] = best_idx;
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        // reduce within local memory to find global max
        for (int stride = local_size/2; stride > 0; stride /= 2) {
            if (local_id < stride) {
                if (local_dist[local_id] < local_dist[local_id + stride]) {
                    local_dist[local_id] = local_dist[local_id + stride];
                    local_idx[local_id] = local_idx[local_id + stride];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }

        if (local_id == 0) output[out_offset + j] = local_idx[0];
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
}
