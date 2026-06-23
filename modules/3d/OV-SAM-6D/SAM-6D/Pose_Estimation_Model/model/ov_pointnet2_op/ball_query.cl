// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

__kernel void ov_ball_query(
    __global const INPUT0_TYPE* new_xyz,   // (B, npoint, 3)
    __global const INPUT1_TYPE* xyz,       // (B, N, 3)
    __global OUTPUT0_TYPE* idx)            // (B, npoint, nsample)
{
    uint batch_index = get_global_id(0);      
    uint point_index = get_global_id(1);      

    int b = INPUT0_DIMS[0];
    int n = INPUT1_DIMS[1];
    int npoint = INPUT0_DIMS[1];
    // nsample and radius are passed as #defines from XML

    if (batch_index >= b || point_index >= npoint) return;

    float radius2 = radius * radius;
    int cnt = 0;

    // Offsets
    uint new_xyz_offset = batch_index * npoint * 3 + point_index * 3;
    uint xyz_batch_offset = batch_index * n * 3;
    uint output_offset = batch_index * npoint * nsample + point_index * nsample;

    float new_x = (float)new_xyz[new_xyz_offset + 0];
    float new_y = (float)new_xyz[new_xyz_offset + 1];
    float new_z = (float)new_xyz[new_xyz_offset + 2];

    // Stream directly from global memory to avoid local memory overflow and barrier deadlocks
    for (int k = 0; k < n && cnt < nsample; ++k) {
        float x = (float)xyz[xyz_batch_offset + k * 3 + 0];
        float y = (float)xyz[xyz_batch_offset + k * 3 + 1];
        float z = (float)xyz[xyz_batch_offset + k * 3 + 2];

        float dx = new_x - x;
        float dy = new_y - y;
        float dz = new_z - z;
        float d2 = dx * dx + dy * dy + dz * dz;

        if (d2 < radius2) {
            if (cnt == 0) {
                // Initialize all slots with the first neighbor index
                for (int l = 0; l < nsample; ++l) {
                    idx[output_offset + l] = k;
                }
            }
            // Assign current neighbor
            idx[output_offset + cnt] = k;
            cnt++;
        }
    }
}
