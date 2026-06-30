/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Farthest Point Sampling - Parallel GPU kernel
 * 
 * Uses parallel reduction to find farthest point each iteration.
 * Each work item handles one candidate point, computes min_dist to selected points,
 * then parallel reduction finds the global maximum.
 *
 * Input points: [B, N, 3] - source points
 * Output: [B, K, 4] - packed [x, y, z, idx] for each sampled point
 *
 * WorkSize: global=(256, B, 1) local=(256, 1, 1) - 256 work items per batch for reduction
 * 
 * Algorithm: For each iteration k, each work item processes N/256 points,
 * finds local max, then workgroup reduction finds global max.
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WORKGROUP_SIZE 256

__kernel void fps_single_v2_kernel(
    const __global INPUT0_TYPE* points,
          __global OUTPUT0_TYPE* output)
{
    const int local_id = get_local_id(0);
    const int batch = get_global_id(1);
    
    if (batch >= INPUT0_DIMS[0]) return;
    
    const int N = INPUT0_DIMS[1];  // number of source points
    const int K = OUTPUT0_DIMS[1]; // number of points to sample
    
    // Local memory for parallel reduction
    __local float local_max_dist[WORKGROUP_SIZE];
    __local int local_max_idx[WORKGROUP_SIZE];
    
    // Global memory for tracking min distances (use output buffer's last K entries as scratch)
    // We'll use a single min_dist value per work item, recomputed each iteration
    
    // Initialize: select point 0 as first point
    if (local_id == 0) {
        const int p_base = batch * INPUT0_PITCHES[0] + 0 * INPUT0_PITCHES[1];
        const float px = (float)points[p_base + 0 * INPUT0_PITCHES[2]];
        const float py = (float)points[p_base + 1 * INPUT0_PITCHES[2]];
        const float pz = (float)points[p_base + 2 * INPUT0_PITCHES[2]];
        
        const int out_base = batch * OUTPUT0_PITCHES[0] + 0 * OUTPUT0_PITCHES[1];
        output[out_base + 0 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)px;
        output[out_base + 1 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)py;
        output[out_base + 2 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)pz;
        output[out_base + 3 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)0;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    // Main FPS loop
    for (int k = 1; k < K; k++) {
        // Each work item finds max among its assigned points
        float my_max_dist = -1.0f;
        int my_max_idx = 0;
        
        // Process points in strided fashion
        for (int i = local_id; i < N; i += WORKGROUP_SIZE) {
            const int pi_base = batch * INPUT0_PITCHES[0] + i * INPUT0_PITCHES[1];
            const float pix = (float)points[pi_base + 0 * INPUT0_PITCHES[2]];
            const float piy = (float)points[pi_base + 1 * INPUT0_PITCHES[2]];
            const float piz = (float)points[pi_base + 2 * INPUT0_PITCHES[2]];
            
            // Find minimum distance to all selected points (0..k-1)
            // Read index from output, then coords from INPUT to avoid FP16 precision loss
            float min_dist = 1e10f;
            for (int j = 0; j < k; j++) {
                const int out_base = batch * OUTPUT0_PITCHES[0] + j * OUTPUT0_PITCHES[1];
                const int sel_idx = (int)output[out_base + 3 * OUTPUT0_PITCHES[2]];
                
                // Read from original FP32 input buffer using stored index
                const int sel_base = batch * INPUT0_PITCHES[0] + sel_idx * INPUT0_PITCHES[1];
                const float sx = (float)points[sel_base + 0 * INPUT0_PITCHES[2]];
                const float sy = (float)points[sel_base + 1 * INPUT0_PITCHES[2]];
                const float sz = (float)points[sel_base + 2 * INPUT0_PITCHES[2]];
                
                const float dx = pix - sx;
                const float dy = piy - sy;
                const float dz = piz - sz;
                const float dist = dx*dx + dy*dy + dz*dz;
                
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
            
            // Track this work item's best candidate
            if (min_dist > my_max_dist) {
                my_max_dist = min_dist;
                my_max_idx = i;
            }
        }
        
        // Store to local memory for reduction
        local_max_dist[local_id] = my_max_dist;
        local_max_idx[local_id] = my_max_idx;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Parallel reduction to find global max
        for (int stride = WORKGROUP_SIZE / 2; stride > 0; stride >>= 1) {
            if (local_id < stride) {
                if (local_max_dist[local_id + stride] > local_max_dist[local_id]) {
                    local_max_dist[local_id] = local_max_dist[local_id + stride];
                    local_max_idx[local_id] = local_max_idx[local_id + stride];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        // Work item 0 writes the selected point
        if (local_id == 0) {
            int best_idx = local_max_idx[0];
            const int sel_base = batch * INPUT0_PITCHES[0] + best_idx * INPUT0_PITCHES[1];
            const float sx = (float)points[sel_base + 0 * INPUT0_PITCHES[2]];
            const float sy = (float)points[sel_base + 1 * INPUT0_PITCHES[2]];
            const float sz = (float)points[sel_base + 2 * INPUT0_PITCHES[2]];
            
            const int out_base = batch * OUTPUT0_PITCHES[0] + k * OUTPUT0_PITCHES[1];
            output[out_base + 0 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)sx;
            output[out_base + 1 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)sy;
            output[out_base + 2 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)sz;
            output[out_base + 3 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)best_idx;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
