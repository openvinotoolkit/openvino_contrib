// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// Postprocess stage 2: compact, decode, and sort.
// ---------------------------------------------------------------------------
// Row zero stores the candidate count. Remaining rows store detection records
// ordered by descending score, ties by ascending anchor index. On overflow,
// CAND_DETERMINISTIC keeps the PP_MAX_CANDIDATES best records in that order;
// otherwise the kernel keeps the first atomic arrivals.

#ifndef CAND_DETERMINISTIC
#error "CAND_DETERMINISTIC must come from pp_tuning_config_generated.cl"
#endif

#if CAND_DETERMINISTIC
#define CAND_KEY_BITS 64
#define CAND_KEY_SCORE_SHIFT 32
#define CAND_RADIX_BITS 8
#define CAND_RADIX_BINS (1 << CAND_RADIX_BITS)
#define CAND_RADIX_MASK (CAND_RADIX_BINS - 1)
#define CAND_RADIX_TOP_SHIFT (CAND_KEY_BITS - CAND_RADIX_BITS)

// Unique per anchor, ordered by (score desc, anchor asc). Positive float bits
// compare like the floats.
inline ulong pp_candidate_key(__global const float* class_logits, int anchor_index) {
    const uint score_bits = as_uint(pp_anchor_score(class_logits, anchor_index, 0));
    return ((ulong)score_bits << CAND_KEY_SCORE_SHIFT) | (uint)(PP_ANCHOR_COUNT - 1 - anchor_index);
}
#endif

__kernel __attribute__((reqd_work_group_size(CAND_WG, 1, 1)))
void pp_postproc_candidates(__global const int* select,
                            __global const float* cls,
                            __global const float* box,
                            __global const float* dir,
                            __global float* candidates) {
    __local int survivor[PP_MAX_CANDIDATES];
    __local float score[PP_MAX_CANDIDATES];
    __local int found;
#if CAND_DETERMINISTIC
    __local int digit_histogram[CAND_RADIX_BINS];
    __local ulong cut_key;
    __local int needed_count;
    __local int cut_done;
#endif

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

#if CAND_DETERMINISTIC
    // Overflow: radix-select the key of the PP_MAX_CANDIDATES-th best anchor,
    // one digit per pass, then collect the survivors again.
    if (found > PP_MAX_CANDIDATES) {
        if (lid == 0) {
            cut_key = 0;
            needed_count = PP_MAX_CANDIDATES;
            cut_done = 0;
        }
        for (int digit_shift = CAND_RADIX_TOP_SHIFT; digit_shift >= 0; digit_shift -= CAND_RADIX_BITS) {
            for (int bin_index = lid; bin_index < CAND_RADIX_BINS; bin_index += CAND_WG) {
                digit_histogram[bin_index] = 0;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            const ulong decided_mask =
                digit_shift == CAND_RADIX_TOP_SHIFT ? 0 : ~0UL << (digit_shift + CAND_RADIX_BITS);
            const ulong decided_prefix = cut_key;
            for (int word_index = lid; word_index < PP_SELECT_WORDS; word_index += CAND_WG) {
                uint word_bits = as_uint(select[word_index]);
                while (word_bits) {
                    const int bit_index = 31 - clz(word_bits & (0u - word_bits));
                    word_bits &= word_bits - 1;
                    const ulong candidate_key = pp_candidate_key(cls, word_index * PP_SELECT_BITS + bit_index);
                    if ((candidate_key & decided_mask) == decided_prefix) {
                        atomic_inc(&digit_histogram[(int)(candidate_key >> digit_shift) & CAND_RADIX_MASK]);
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if (lid == 0) {
                int bin_index = CAND_RADIX_MASK;
                while (digit_histogram[bin_index] < needed_count) {
                    needed_count -= digit_histogram[bin_index--];
                }
                cut_key |= (ulong)bin_index << digit_shift;
                cut_done = digit_histogram[bin_index] == needed_count;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (cut_done) {
                break;
            }
        }

        // Keys are unique, so exactly PP_MAX_CANDIDATES keys are >= cut_key.
        const ulong threshold_key = cut_key;
        if (lid == 0) {
            found = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int word_index = lid; word_index < PP_SELECT_WORDS; word_index += CAND_WG) {
            uint word_bits = as_uint(select[word_index]);
            while (word_bits) {
                const int bit_index = 31 - clz(word_bits & (0u - word_bits));
                word_bits &= word_bits - 1;
                const int anchor_index = word_index * PP_SELECT_BITS + bit_index;
                if (pp_candidate_key(cls, anchor_index) >= threshold_key) {
                    const int slot_index = atomic_inc(&found);
                    if (slot_index < PP_MAX_CANDIDATES) {
                        survivor[slot_index] = anchor_index;
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#endif

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
