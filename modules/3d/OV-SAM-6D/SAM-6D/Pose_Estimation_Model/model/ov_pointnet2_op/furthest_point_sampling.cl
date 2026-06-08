// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

__kernel void ov_furthest_point_sampling(
    __global const INPUT0_TYPE* pts,  // (B, N, 3)
    __global const INPUT1_TYPE* npoint_ptr, // npoint scalar
    __global OUTPUT0_TYPE* output,    // (B, npoint)
    __global float* dist,             // (B, N) internal buffer; keep FP32 under FP16 inference
    __global float* pts_f32           // (B, N, 3) internal FP32 pts cache
) {
    int b   = get_global_id(1);  // batch index

    int B = INPUT0_DIMS[0];  // batch size
    int N = INPUT0_DIMS[1];  // number of points
    int C = INPUT0_DIMS[2];  // channels (should be 3)
    int npoint = INPUT1_DIMS[0] * INPUT1_DIMS[1];

    if (b >= B) return;

    int pts_offset  = b * N * C;
    int dist_offset = b * N;
    int out_offset  = b * npoint;

    int local_id   = get_local_id(0);
    int local_size = get_local_size(0);

    __local float local_dist[256];
    __local int   local_idx[256];

    // Pre-convert pts from INPUT0_TYPE to float once; amortizes half->float
    // conversion cost across the npoint outer iterations (hot inner loop
    // below then always reads from the float cache).
    for (int k = local_id; k < N * C; k += local_size)
        pts_f32[pts_offset + k] = (float)pts[pts_offset + k];

    for (int i = local_id; i < N; i += local_size)
        dist[dist_offset + i] = FLT_MAX;

    if (local_id == 0) output[out_offset] = 0;
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int j = 1; j < npoint; ++j) {
        int last_idx = output[out_offset + j - 1];

        float last_x = pts_f32[pts_offset + last_idx * 3 + 0];
        float last_y = pts_f32[pts_offset + last_idx * 3 + 1];
        float last_z = pts_f32[pts_offset + last_idx * 3 + 2];

        float best_dist = -1.0f;
        int   best_idx  = -1;

        // 4x unrolled: issue all 4 loads before any arithmetic to hide memory
        // latency. Each WI owns a disjoint strided slice of dist[] so writes
        // in the same pass do not create cross-WI dependencies.
        int i = local_id;
        for (; i + 3 * local_size < N; i += 4 * local_size) {
            int i0 = i;
            int i1 = i +     local_size;
            int i2 = i + 2 * local_size;
            int i3 = i + 3 * local_size;

            float px0 = pts_f32[pts_offset + i0 * 3 + 0];
            float py0 = pts_f32[pts_offset + i0 * 3 + 1];
            float pz0 = pts_f32[pts_offset + i0 * 3 + 2];
            float px1 = pts_f32[pts_offset + i1 * 3 + 0];
            float py1 = pts_f32[pts_offset + i1 * 3 + 1];
            float pz1 = pts_f32[pts_offset + i1 * 3 + 2];
            float px2 = pts_f32[pts_offset + i2 * 3 + 0];
            float py2 = pts_f32[pts_offset + i2 * 3 + 1];
            float pz2 = pts_f32[pts_offset + i2 * 3 + 2];
            float px3 = pts_f32[pts_offset + i3 * 3 + 0];
            float py3 = pts_f32[pts_offset + i3 * 3 + 1];
            float pz3 = pts_f32[pts_offset + i3 * 3 + 2];

            // Load current distances (4 more reads)
            float cd0 = dist[dist_offset + i0];
            float cd1 = dist[dist_offset + i1];
            float cd2 = dist[dist_offset + i2];
            float cd3 = dist[dist_offset + i3];

            // Squared distances
            float dx0 = px0 - last_x, dy0 = py0 - last_y, dz0 = pz0 - last_z;
            float dx1 = px1 - last_x, dy1 = py1 - last_y, dz1 = pz1 - last_z;
            float dx2 = px2 - last_x, dy2 = py2 - last_y, dz2 = pz2 - last_z;
            float dx3 = px3 - last_x, dy3 = py3 - last_y, dz3 = pz3 - last_z;

            float nd0 = min(dx0*dx0 + dy0*dy0 + dz0*dz0, cd0);
            float nd1 = min(dx1*dx1 + dy1*dy1 + dz1*dz1, cd1);
            float nd2 = min(dx2*dx2 + dy2*dy2 + dz2*dz2, cd2);
            float nd3 = min(dx3*dx3 + dy3*dy3 + dz3*dz3, cd3);

            dist[dist_offset + i0] = nd0;
            dist[dist_offset + i1] = nd1;
            dist[dist_offset + i2] = nd2;
            dist[dist_offset + i3] = nd3;

            if (nd0 > best_dist) { best_dist = nd0; best_idx = i0; }
            if (nd1 > best_dist) { best_dist = nd1; best_idx = i1; }
            if (nd2 > best_dist) { best_dist = nd2; best_idx = i2; }
            if (nd3 > best_dist) { best_dist = nd3; best_idx = i3; }
        }
        // Scalar tail for remaining elements
        for (; i < N; i += local_size) {
            float dx = pts_f32[pts_offset + i * 3 + 0] - last_x;
            float dy = pts_f32[pts_offset + i * 3 + 1] - last_y;
            float dz = pts_f32[pts_offset + i * 3 + 2] - last_z;
            float new_d = min(dx*dx + dy*dy + dz*dz, dist[dist_offset + i]);
            dist[dist_offset + i] = new_d;
            if (new_d > best_dist) { best_dist = new_d; best_idx = i; }
        }

        local_dist[local_id] = best_dist;
        local_idx[local_id]  = best_idx;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int stride = local_size / 2; stride > 0; stride >>= 1) {
            if (local_id < stride) {
                if (local_dist[local_id] < local_dist[local_id + stride]) {
                    local_dist[local_id] = local_dist[local_id + stride];
                    local_idx[local_id]  = local_idx[local_id + stride];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (local_id == 0) output[out_offset + j] = local_idx[0];
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
