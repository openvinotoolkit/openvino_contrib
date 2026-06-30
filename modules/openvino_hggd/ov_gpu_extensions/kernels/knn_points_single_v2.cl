/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * KNN Points Single - GPU kernel with BFYX format
 * Supports FP16/FP32 via OpenVINO type macros
 *
 * Inputs:  p1 [B, N1, 3] - query points
 *          p2 [B, N2, 3] - source points
 * Output:  [B, N1, K*2] - packed [dists_0..K-1, idx_0..K-1]
 *
 * WorkSize: global=(N1, B, 1)
 * Uses PITCHES for memory access to handle GPU internal layouts
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define MAX_K 64

__kernel void knn_points_single_v2_kernel(
    const __global INPUT0_TYPE* p1,
    const __global INPUT1_TYPE* p2,
          __global OUTPUT0_TYPE* output)
{
    const int query_idx = get_global_id(0);
    const int batch = get_global_id(1);
    
    if (query_idx >= INPUT0_DIMS[1] || batch >= INPUT0_DIMS[0]) return;
    
    const int K2 = OUTPUT0_DIMS[2];
    const int K = K2 / 2;
    const int N2 = INPUT1_DIMS[1];
    
    // Read query point (cast to float for precision)
    const int p1_base = batch * INPUT0_PITCHES[0] + query_idx * INPUT0_PITCHES[1];
    const float qx = (float)p1[p1_base + 0 * INPUT0_PITCHES[2]];
    const float qy = (float)p1[p1_base + 1 * INPUT0_PITCHES[2]];
    const float qz = (float)p1[p1_base + 2 * INPUT0_PITCHES[2]];
    
    // Storage for K nearest neighbors
    float best_dists[MAX_K];
    int best_idx[MAX_K];
    
    for (int k = 0; k < K; k++) {
        best_dists[k] = 1e10f;
        best_idx[k] = 0;
    }
    
    // Find K nearest neighbors
    for (int j = 0; j < N2; j++) {
        const int p2_base = batch * INPUT1_PITCHES[0] + j * INPUT1_PITCHES[1];
        const float sx = (float)p2[p2_base + 0 * INPUT1_PITCHES[2]];
        const float sy = (float)p2[p2_base + 1 * INPUT1_PITCHES[2]];
        const float sz = (float)p2[p2_base + 2 * INPUT1_PITCHES[2]];
        
        const float dx = qx - sx;
        const float dy = qy - sy;
        const float dz = qz - sz;
        const float dist_sq = dx*dx + dy*dy + dz*dz;
        
        // Insert into sorted list if closer than K-th neighbor
        if (dist_sq < best_dists[K-1]) {
            int insert_pos = K - 1;
            while (insert_pos > 0 && dist_sq < best_dists[insert_pos - 1]) {
                best_dists[insert_pos] = best_dists[insert_pos - 1];
                best_idx[insert_pos] = best_idx[insert_pos - 1];
                insert_pos--;
            }
            best_dists[insert_pos] = dist_sq;
            best_idx[insert_pos] = j;
        }
    }
    
    // Write output
    const int out_base = batch * OUTPUT0_PITCHES[0] + query_idx * OUTPUT0_PITCHES[1];
    for (int k = 0; k < K; k++) {
        output[out_base + k * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)best_dists[k];
        output[out_base + (K + k) * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)best_idx[k];
    }
}
