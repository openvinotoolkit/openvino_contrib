// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// Postprocess stage 4: greedy suppression and detection gather.
// ---------------------------------------------------------------------------
// Work-item zero selects surviving candidates. The work-group copies their
// records to the output in parallel.

__kernel __attribute__((reqd_work_group_size(GATHER_WG, 1, 1)))
void pp_postproc_gather(__global const float* candidates,
                        __global const int* mask,
                        __global float* detections) {
    __local int keep[PP_MAX_DETECTIONS];
    __local uint removed[PP_NMS_COL_BLOCKS];
    __local int kept;

    const int lid = (int)get_local_id(0);

    for (int i = lid; i < PP_NMS_COL_BLOCKS; i += GATHER_WG) removed[i] = 0;
    // Clear unused detection rows.
    for (int i = lid; i < PP_MAX_DETECTIONS * PP_DET_CHANNELS; i += GATHER_WG) detections[i] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    const int count = min(max(((__global const int*)candidates)[0], 0), PP_MAX_CANDIDATES);
    const int col_blocks = (count + PP_NMS_BLOCK - 1) / PP_NMS_BLOCK;

    if (lid == 0) {
        int found = 0;
        for (int i = 0; i < count && found < PP_MAX_DETECTIONS; ++i) {
            const int block = i / PP_NMS_BLOCK;
            if (removed[block] & (1u << (i % PP_NMS_BLOCK))) continue;
            keep[found++] = i;
            __global const int* row = mask + (long)i * PP_NMS_COL_BLOCKS;
            for (int j = block; j < col_blocks; ++j) removed[j] |= as_uint(row[j]);
        }
        kept = found;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    __global const float* boxes = candidates + PP_DET_CHANNELS;  // skip the header row
    for (int i = lid; i < kept * PP_DET_CHANNELS; i += GATHER_WG) {
        const int det = i / PP_DET_CHANNELS;
        const int channel = i - det * PP_DET_CHANNELS;
        detections[i] = boxes[(long)keep[det] * PP_DET_CHANNELS + channel];
    }
}
