// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// Postprocess stage 2: compact, decode, and sort.
// ---------------------------------------------------------------------------
// Row zero stores the candidate count. Remaining rows store detection records
// ordered by descending score.

__kernel __attribute__((reqd_work_group_size(CAND_WG, 1, 1)))
void pp_postproc_candidates(__global const int* select,
                            __global const float* cls,
                            __global const float* box,
                            __global const float* dir,
                            __global float* candidates) {
    __local int survivor[PP_MAX_CANDIDATES];
    __local float score[PP_MAX_CANDIDATES];
    __local int found;

    const int lid = (int)get_local_id(0);
    if (lid == 0) found = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 1: collect surviving anchor indices.
    for (int word = lid; word < PP_SELECT_WORDS; word += CAND_WG) {
        uint bits = as_uint(select[word]);
        while (bits) {
            const int bit = 31 - clz(bits & (0u - bits));  // lowest set bit
            bits &= bits - 1;
            const int slot = atomic_inc(&found);
            if (slot < PP_MAX_CANDIDATES) survivor[slot] = word * PP_SELECT_BITS + bit;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int count = min(found, PP_MAX_CANDIDATES);

    // Phase 2: compute scores for surviving anchors.
    for (int i = lid; i < count; i += CAND_WG) score[i] = pp_anchor_score(cls, survivor[i], 0);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 3: rank, decode, and write records.
    for (int i = lid; i < count; i += CAND_WG) {
        const float my_score = score[i];
        const int my_anchor = survivor[i];
        int rank = 0;
        for (int j = 0; j < count; ++j) {
            const float other = score[j];
            if (other > my_score || (other == my_score && survivor[j] < my_anchor)) rank++;
        }

        int class_id = 0;
        const float value = pp_anchor_score(cls, my_anchor, &class_id);
        float record[PP_DET_CHANNELS];
        pp_decode(box, dir, my_anchor, class_id, value, record);

        __global float* dst = candidates + (long)(rank + 1) * PP_DET_CHANNELS;
        for (int c = 0; c < PP_DET_CHANNELS; ++c) dst[c] = record[c];
    }

    // Store the candidate count in the header row.
    if (lid == 0) ((__global int*)candidates)[0] = count;
}
