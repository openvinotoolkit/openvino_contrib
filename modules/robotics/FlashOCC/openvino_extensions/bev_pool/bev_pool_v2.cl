/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 * Adapted from OpenVINO BEVFusion module BEVPool sources
 * (https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/openvino_bevfusion), Apache-2.0.
 * BEV pooling algorithm adapted from BEVDet bev_pool_v2
 * (https://github.com/HuangJunJie2017/BEVDet), Apache-2.0, Copyright (c) Phigent Robotics.
 */

/*
 * BEVPool V2 GPU Kernel — Pre-sorted Interval-Based Scatter (No Atomics)
 *
 * Reads from a packed sort buffer produced by BEVPoolBinSort:
 *   [sorted_ranks | cell_scratch | interval_starts | interval_lengths]
 *   offsets: 0      TOTAL_PTS     2*TOTAL_PTS      2*TOTAL_PTS+NX*NY
 *
 * Work sizing:
 *   - One workgroup per BEV cell: NX * NY = 129,600 workgroups
 *   - One work-item per output channel: CHANNELS = 80 per workgroup
 *   - global = 129,600 * 80 = 10,368,000;  local = 80
 *
 * OpenVINO custom layer defines:
 *   NX, NY, CHANNELS, FEAT_HW, DEPTH_HW, TOTAL_PTS
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/* Offsets into packed sort buffer */
#define OFF_SORTED  0
#define OFF_STARTS  (2 * TOTAL_PTS)
#define OFF_LENGTHS (2 * TOTAL_PTS + NX * NY)

__kernel void bevpool_v2(
    __global const INPUT0_TYPE* depth_probs,    /* [N, D, H, W]  FP32 (softmax) */
    __global const INPUT1_TYPE* context_feats,  /* [N, C, H, W]  FP32           */
    __global const INPUT2_TYPE* packed_sort,    /* packed sort buffer I32        */
    __global OUTPUT0_TYPE* output               /* [1, C, NX, NY] FP32          */
) {
    const int cell = get_group_id(0);       /* 0 .. NX*NY-1                     */
    const int c    = get_local_id(0);       /* 0 .. C-1                         */
    const int C    = CHANNELS;
    const int n_cells = NX * NY;

    if (cell >= n_cells) return;

    __global const int* sorted_ranks     = packed_sort + OFF_SORTED;
    __global const int* interval_starts  = packed_sort + OFF_STARTS;
    __global const int* interval_lengths = packed_sort + OFF_LENGTHS;

    const int start = interval_starts[cell];
    const int len   = interval_lengths[cell];

    float sum = 0.0f;

    for (int i = 0; i < len; ++i) {
        const int point_idx = sorted_ranks[start + i];
        const float dw      = depth_probs[point_idx];

        /* Derive context feature location from point index */
        const int cam = point_idx / DEPTH_HW;
        const int hw  = point_idx % DEPTH_HW % FEAT_HW;
        const int feat_base = cam * C * FEAT_HW + hw;

        sum += dw * context_feats[feat_base + c * FEAT_HW];
    }

    /* Write in Y-major order [C, NY, NX] so the output tensor is C-contiguous.
     * BinSort encodes cell = ix * NY + iy, so decode to write as [c, iy, ix]. */
    const int ix = cell / NY;
    const int iy = cell % NY;
    output[c * n_cells + iy * NX + ix] = sum;
}
