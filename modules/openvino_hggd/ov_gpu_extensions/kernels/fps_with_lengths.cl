/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Farthest Point Sampling with Lengths Support - GPU Native
 * 
 * Handles variable-length padded batches efficiently on GPU.
 * Each batch can have different valid lengths, avoiding zero-padded regions.
 *
 * Input 0 (points):  [B, N, 3] - source points (may be zero-padded)
 * Input 1 (lengths): [B] - actual valid length for each batch element
 * Output: [B, K, 4] - packed [x, y, z, idx] for each sampled point
 *
 * WorkSize: global=(256, B, 1) local=(256, 1, 1)
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WORKGROUP_SIZE 256

__kernel void fps_with_lengths_kernel(
    const __global INPUT0_TYPE* points,    // [B, N, 3]
    const __global INPUT1_TYPE* lengths,   // [B]
          __global OUTPUT0_TYPE* output)   // [B, K, 4]
{
    const int local_id = get_local_id(0);
    const int batch = get_global_id(1);
    
    if (batch >= INPUT0_DIMS[0]) return;
    
    const int N = INPUT0_DIMS[1];  // max points (padded dimension)
    const int K = OUTPUT0_DIMS[1]; // number of points to sample
    
    // Get actual valid length for this batch, clamped to N
    int valid_len = (int)lengths[batch];
    if (valid_len > N) valid_len = N;  // Safety clamp
    
    // If valid_len is 0 or very small, output zeros
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
    
    // Clamp K to valid length
    const int k_actual = (K < valid_len) ? K : valid_len;
    
    // Local memory for parallel reduction
    __local float local_max_dist[WORKGROUP_SIZE];
    __local int local_max_idx[WORKGROUP_SIZE];
    
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
    
    // Main FPS loop - only iterate up to k_actual
    for (int k = 1; k < k_actual; k++) {
        float my_max_dist = -1.0f;
        int my_max_idx = 0;
        
        // Only iterate over valid points [0, valid_len)
        for (int i = local_id; i < valid_len; i += WORKGROUP_SIZE) {
            const int pi_base = batch * INPUT0_PITCHES[0] + i * INPUT0_PITCHES[1];
            const float pix = (float)points[pi_base + 0 * INPUT0_PITCHES[2]];
            const float piy = (float)points[pi_base + 1 * INPUT0_PITCHES[2]];
            const float piz = (float)points[pi_base + 2 * INPUT0_PITCHES[2]];
            
            // Find minimum distance to all selected points
            float min_dist = 1e10f;
            for (int j = 0; j < k; j++) {
                const int out_base = batch * OUTPUT0_PITCHES[0] + j * OUTPUT0_PITCHES[1];
                const int sel_idx = (int)output[out_base + 3 * OUTPUT0_PITCHES[2]];
                
                // Read from original input buffer
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
            
            if (min_dist > my_max_dist) {
                my_max_dist = min_dist;
                my_max_idx = i;
            }
        }
        
        local_max_dist[local_id] = my_max_dist;
        local_max_idx[local_id] = my_max_idx;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Parallel reduction
        for (int stride = WORKGROUP_SIZE / 2; stride > 0; stride >>= 1) {
            if (local_id < stride) {
                if (local_max_dist[local_id + stride] > local_max_dist[local_id]) {
                    local_max_dist[local_id] = local_max_dist[local_id + stride];
                    local_max_idx[local_id] = local_max_idx[local_id + stride];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
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
    
    // If K > k_actual, pad remaining with last valid point
    if (local_id == 0 && k_actual < K) {
        // Get last valid sampled point
        const int last_out = batch * OUTPUT0_PITCHES[0] + (k_actual - 1) * OUTPUT0_PITCHES[1];
        const float last_x = (float)output[last_out + 0 * OUTPUT0_PITCHES[2]];
        const float last_y = (float)output[last_out + 1 * OUTPUT0_PITCHES[2]];
        const float last_z = (float)output[last_out + 2 * OUTPUT0_PITCHES[2]];
        const float last_idx = (float)output[last_out + 3 * OUTPUT0_PITCHES[2]];
        
        for (int k = k_actual; k < K; k++) {
            const int out_base = batch * OUTPUT0_PITCHES[0] + k * OUTPUT0_PITCHES[1];
            output[out_base + 0 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)last_x;
            output[out_base + 1 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)last_y;
            output[out_base + 2 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)last_z;
            output[out_base + 3 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)last_idx;
        }
    }
}
