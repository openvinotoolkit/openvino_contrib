// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

__kernel void grouping_operation(
    __global const INPUT0_TYPE* points,    // (B, C, N)
    __global const INPUT1_TYPE* idx,         // (B, NPOINT, NSAMPLE)
    __global OUTPUT0_TYPE* out              // (B, C, NPOINT, NSAMPLE)
) {
    //Get the ID of the current work-group, corresponding to CUDA blockIdx.x
    int global_id = get_global_id(0);

    //Get the ID of the current work item in the work group, corresponding to threadIdx.x
    int local_id_x = get_local_id(0);
    int local_id_y = get_local_id(1);

    //Get the workgroup size, corresponding to blockDim.x
    int local_size_x = get_local_size(0);
    int local_size_y = get_local_size(1);

    //Calculate the total_threads processed by the current work group (optional, used to loop to process more data)
    int total_threads = get_num_groups(0) * get_local_size(0);

    int B = INPUT0_DIMS[0];
    int C = INPUT0_DIMS[1];
    int N = INPUT0_DIMS[2];
    int NPOINT = INPUT1_DIMS[1];
    int NSAMPLE = INPUT1_DIMS[2];

    int total = B * C * NPOINT * NSAMPLE;

    if (global_id >= total) return;

    int batch = global_id / (C * NPOINT * NSAMPLE);
    int channel = (global_id / (NSAMPLE * NPOINT)) % C;
    int point = (global_id / NSAMPLE) % NPOINT;
    int sample = global_id % NSAMPLE;

    int idx_offset = batch * (NPOINT * NSAMPLE) + point * NSAMPLE + sample;
    int a = idx[idx_offset];

    float val = 0.0f;

    if (a >= 0 && a < N) {
        int input_offset = batch * (C * N) + channel * N + a;
        val = (float)points[input_offset];

        if (isnan(val) || isinf(val)) {
            val = 0.0f;
        }
    }

    int out_offset = batch * (C * NPOINT * NSAMPLE) + channel * (NPOINT * NSAMPLE) + point * NSAMPLE + sample;
    out[out_offset] = (OUTPUT0_TYPE)val;
}
