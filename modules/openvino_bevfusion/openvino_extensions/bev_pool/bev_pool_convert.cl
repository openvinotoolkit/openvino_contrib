/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * BEVPoolConvert GPU Kernel for OpenVINO Custom Layer
 *
 * Simple element-wise conversion: int32 fixed-point → float32
 *   output[i] = (float)input[i] / FP_SCALE
 * then input[i] = 0, so the accumulator holds zeros for the next
 * bevpool_scatter, which only adds into it.
 *
 * Work sizing: one work-item per element
 */

#define FP_SCALE 8192  /* must match bevpool_scatter */

__kernel void bevpool_convert(
    __global const INPUT0_TYPE* token,   /* [1] int32 from bevpool_scatter; only orders the kernels */
    __global INPUT1_TYPE* input,         /* [1, C, NX, NY] int32 accumulator */
    __global OUTPUT0_TYPE* output        /* [1, C, NX, NY] float32 */
) {
    const int idx = get_global_id(0);
    output[idx] = (float)input[idx] / (float)FP_SCALE;
    input[idx] = 0;
}
