// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

__kernel void ov_furthest_point_sampling(
    __global const INPUT0_TYPE* pts,  // (B, N, 3)
    __global const INPUT1_TYPE* npoint_ptr, // npoint scalar
    __global OUTPUT0_TYPE* output,    // (B, npoint)
    __global float* dist,             // (B, N) internal buffer; keep FP32 under FP16 inference
    __global float* pts_f32           // (B, N, 3) internal FP32 pts cache
) {
    int b = get_global_id(1); // batch index

    int B = INPUT0_DIMS[0];  // batch size
    int N = INPUT0_DIMS[1];  // number of points
    int C = INPUT0_DIMS[2];  // channels (should be 3)
    int npoint = INPUT1_DIMS[0] * INPUT1_DIMS[1];

    if (b >= B) return;

    int pts_offset  = b * N * C;
    int dist_offset = b * N;
    int out_offset  = b * npoint;

    int lid   = get_local_id(0);
    int lsize = get_local_size(0);   // 1024

    // Work-group local memory - sized for max 1024 work-items
    __local float s_val[1024];
    __local int   s_idx[1024];

    // Convert pts to float once (amortises half/bfloat->float cost)
    for (int k = lid; k < N * C; k += lsize)
        pts_f32[pts_offset + k] = (float)pts[pts_offset + k];

    // Init distances to FLT_MAX
    for (int i = lid; i < N; i += lsize)
        dist[dist_offset + i] = FLT_MAX;

    // First sampled point = 0; cache in local memory
    if (lid == 0) {
        output[out_offset] = 0;
        s_idx[0] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int j = 1; j < npoint; ++j) {
        // Read last selected index from LOCAL memory
        int last_idx = s_idx[0];

        float lx = pts_f32[pts_offset + last_idx * 3 + 0];
        float ly = pts_f32[pts_offset + last_idx * 3 + 1];
        float lz = pts_f32[pts_offset + last_idx * 3 + 2];

        float best_dist = -1.0f;
        int   best_idx  = -1;

        // 8-way unrolled distance update + local argmax
        // Each WI owns a disjoint strided slice of dist[] - no cross-WI deps.
        int i = lid;
        for (; i + 7 * lsize < N; i += 8 * lsize) {
            int i0 = i;
            int i1 = i + lsize;
            int i2 = i + 2 * lsize;
            int i3 = i + 3 * lsize;
            int i4 = i + 4 * lsize;
            int i5 = i + 5 * lsize;
            int i6 = i + 6 * lsize;
            int i7 = i + 7 * lsize;

            float3 p0 = vload3(0, pts_f32 + pts_offset + i0 * 3);
            float3 p1 = vload3(0, pts_f32 + pts_offset + i1 * 3);
            float3 p2 = vload3(0, pts_f32 + pts_offset + i2 * 3);
            float3 p3 = vload3(0, pts_f32 + pts_offset + i3 * 3);
            float3 p4 = vload3(0, pts_f32 + pts_offset + i4 * 3);
            float3 p5 = vload3(0, pts_f32 + pts_offset + i5 * 3);
            float3 p6 = vload3(0, pts_f32 + pts_offset + i6 * 3);
            float3 p7 = vload3(0, pts_f32 + pts_offset + i7 * 3);

            float cd0 = dist[dist_offset + i0];
            float cd1 = dist[dist_offset + i1];
            float cd2 = dist[dist_offset + i2];
            float cd3 = dist[dist_offset + i3];
            float cd4 = dist[dist_offset + i4];
            float cd5 = dist[dist_offset + i5];
            float cd6 = dist[dist_offset + i6];
            float cd7 = dist[dist_offset + i7];

            float dx0 = p0.x - lx, dy0 = p0.y - ly, dz0 = p0.z - lz;
            float dx1 = p1.x - lx, dy1 = p1.y - ly, dz1 = p1.z - lz;
            float dx2 = p2.x - lx, dy2 = p2.y - ly, dz2 = p2.z - lz;
            float dx3 = p3.x - lx, dy3 = p3.y - ly, dz3 = p3.z - lz;
            float dx4 = p4.x - lx, dy4 = p4.y - ly, dz4 = p4.z - lz;
            float dx5 = p5.x - lx, dy5 = p5.y - ly, dz5 = p5.z - lz;
            float dx6 = p6.x - lx, dy6 = p6.y - ly, dz6 = p6.z - lz;
            float dx7 = p7.x - lx, dy7 = p7.y - ly, dz7 = p7.z - lz;

            float nd0 = fmin(dx0*dx0 + dy0*dy0 + dz0*dz0, cd0);
            float nd1 = fmin(dx1*dx1 + dy1*dy1 + dz1*dz1, cd1);
            float nd2 = fmin(dx2*dx2 + dy2*dy2 + dz2*dz2, cd2);
            float nd3 = fmin(dx3*dx3 + dy3*dy3 + dz3*dz3, cd3);
            float nd4 = fmin(dx4*dx4 + dy4*dy4 + dz4*dz4, cd4);
            float nd5 = fmin(dx5*dx5 + dy5*dy5 + dz5*dz5, cd5);
            float nd6 = fmin(dx6*dx6 + dy6*dy6 + dz6*dz6, cd6);
            float nd7 = fmin(dx7*dx7 + dy7*dy7 + dz7*dz7, cd7);

            dist[dist_offset + i0] = nd0;
            dist[dist_offset + i1] = nd1;
            dist[dist_offset + i2] = nd2;
            dist[dist_offset + i3] = nd3;
            dist[dist_offset + i4] = nd4;
            dist[dist_offset + i5] = nd5;
            dist[dist_offset + i6] = nd6;
            dist[dist_offset + i7] = nd7;

            if (nd0 > best_dist) { best_dist = nd0; best_idx = i0; }
            if (nd1 > best_dist) { best_dist = nd1; best_idx = i1; }
            if (nd2 > best_dist) { best_dist = nd2; best_idx = i2; }
            if (nd3 > best_dist) { best_dist = nd3; best_idx = i3; }
            if (nd4 > best_dist) { best_dist = nd4; best_idx = i4; }
            if (nd5 > best_dist) { best_dist = nd5; best_idx = i5; }
            if (nd6 > best_dist) { best_dist = nd6; best_idx = i6; }
            if (nd7 > best_dist) { best_dist = nd7; best_idx = i7; }
        }
        // Scalar tail
        for (; i < N; i += lsize) {
            float3 p = vload3(0, pts_f32 + pts_offset + i * 3);
            float dx = p.x - lx, dy = p.y - ly, dz = p.z - lz;
            float new_d = fmin(dx*dx + dy*dy + dz*dz, dist[dist_offset + i]);
            dist[dist_offset + i] = new_d;
            if (new_d > best_dist) { best_dist = new_d; best_idx = i; }
        }

        // Work-group reduction (standard tree reduction)
        s_val[lid] = best_dist;
        s_idx[lid] = best_idx;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int stride = lsize / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                if (s_val[lid] < s_val[lid + stride]) {
                    s_val[lid] = s_val[lid + stride];
                    s_idx[lid] = s_idx[lid + stride];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Write result; s_idx[0] is already valid for next iteration
        if (lid == 0)
            output[out_offset + j] = (OUTPUT0_TYPE)s_idx[0];
    }
}
