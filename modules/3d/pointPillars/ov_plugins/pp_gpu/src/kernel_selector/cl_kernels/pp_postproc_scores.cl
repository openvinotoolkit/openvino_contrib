// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// Postprocess stage 1: score filter.
// ---------------------------------------------------------------------------
// Each work-item evaluates one anchor and sets its bit in the selection bitmap.

#define SCORES_WORDS_PER_WG (SCORES_WG / PP_SELECT_BITS)

__kernel __attribute__((reqd_work_group_size(SCORES_WG, 1, 1)))
void pp_postproc_scores(__global const float* cls, __global int* words) {
    __local uint block_words[SCORES_WORDS_PER_WG];

    const int lid = (int)get_local_id(0);
    if (lid < SCORES_WORDS_PER_WG) block_words[lid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    const int anchor = (int)get_global_id(0);
    if (pp_anchor_score(cls, anchor, 0) >= PP_SCORE_THRESH)
        atomic_or(&block_words[lid / PP_SELECT_BITS], 1u << (lid & (PP_SELECT_BITS - 1)));
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < SCORES_WORDS_PER_WG)
        words[(int)get_group_id(0) * SCORES_WORDS_PER_WG + lid] = as_int(block_words[lid]);
}
