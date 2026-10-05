// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// Postprocess stage 3: rotated-BEV IoU bitmask.
// ---------------------------------------------------------------------------
// Tiles cover the upper block triangle. Each work-item evaluates one box pair,
// and each row is combined into a bitmask in local memory.

// x, y, z, w, l, h, yaw + circumradius.
#define NMS_CACHE_STRIDE (PP_NUM_BOX_VALUES + 1)

inline void pp_nms_cache(__local float* dst, __global const float* src) {
    for (int i = 0; i < PP_NUM_BOX_VALUES; ++i) dst[i] = src[i];
    dst[PP_NUM_BOX_VALUES] = pp_box_reach(src[3], src[4]);
}

__kernel __attribute__((reqd_work_group_size(PP_NMS_TILE_WG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(NMS_SIMD)))
void pp_postproc_nms(__global const float* candidates, __global int* mask) {
    __local float row_cache[PP_NMS_BLOCK * NMS_CACHE_STRIDE];
    __local float col_cache[PP_NMS_BLOCK * NMS_CACHE_STRIDE];
    __local uint row_bits[PP_NMS_BLOCK];

    const int lid = (int)get_local_id(0);
    const int row = lid / PP_NMS_BLOCK;
    const int col = lid % PP_NMS_BLOCK;

    const int count = min(max(((__global const int*)candidates)[0], 0), PP_MAX_CANDIDATES);
    const int col_blocks = (count + PP_NMS_BLOCK - 1) / PP_NMS_BLOCK;
    const int tiles = col_blocks * (col_blocks + 1) / 2;
    __global const float* boxes = candidates + PP_DET_CHANNELS;  // skip the header row

    for (int tile = (int)get_group_id(0); tile < tiles; tile += NMS_GROUPS) {
        // Convert the triangular tile index to row and column block indices.
        int col_start = (int)((sqrt(8.0f * (float)tile + 1.0f) - 1.0f) * 0.5f);
        while ((col_start + 1) * (col_start + 2) / 2 <= tile) col_start++;
        while (col_start > 0 && col_start * (col_start + 1) / 2 > tile) col_start--;
        const int row_start = tile - col_start * (col_start + 1) / 2;

        const int row_size = min(PP_NMS_BLOCK, count - row_start * PP_NMS_BLOCK);
        const int col_size = min(PP_NMS_BLOCK, count - col_start * PP_NMS_BLOCK);

        // Synchronize before reusing the local tile caches.
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < PP_NMS_BLOCK) {
            row_bits[lid] = 0;
            if (lid < row_size)
                pp_nms_cache(row_cache + lid * NMS_CACHE_STRIDE,
                             boxes + (long)(row_start * PP_NMS_BLOCK + lid) * PP_DET_CHANNELS);
        } else if (lid < 2 * PP_NMS_BLOCK) {
            const int slot = lid - PP_NMS_BLOCK;
            if (slot < col_size)
                pp_nms_cache(col_cache + slot * NMS_CACHE_STRIDE,
                             boxes + (long)(col_start * PP_NMS_BLOCK + slot) * PP_DET_CHANNELS);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Diagonal tiles compare only against later candidates.
        const int begin = (row_start == col_start) ? row + 1 : 0;
        if (row < row_size && col < col_size && col >= begin) {
            __local const float* a = row_cache + row * NMS_CACHE_STRIDE;
            __local const float* b = col_cache + col * NMS_CACHE_STRIDE;
            // Skip pairs whose centers are farther apart than their box reach.
            const float reach = a[PP_NUM_BOX_VALUES] + b[PP_NUM_BOX_VALUES] + 2e-2f;
            if (fabs(b[0] - a[0]) <= reach && fabs(b[1] - a[1]) <= reach) {
                float box_a[PP_NUM_BOX_VALUES];
                float box_b[PP_NUM_BOX_VALUES];
                for (int i = 0; i < PP_NUM_BOX_VALUES; ++i) {
                    box_a[i] = a[i];
                    box_b[i] = b[i];
                }
                if (pp_rotated_iou(box_a, box_b) >= PP_NMS_THRESH)
                    atomic_or(&row_bits[row], 1u << col);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid < row_size)
            mask[(long)(row_start * PP_NMS_BLOCK + lid) * PP_NMS_COL_BLOCKS + col_start] =
                as_int(row_bits[lid]);
    }
}
