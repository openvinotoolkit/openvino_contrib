/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 * Adapted from OpenVINO BEVFusion module BEVPool sources
 * (https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/openvino_bevfusion), Apache-2.0.
 * BEV pooling algorithm adapted from BEVDet bev_pool_v2
 * (https://github.com/HuangJunJie2017/BEVDet), Apache-2.0, Copyright (c) Phigent Robotics.
 */

/*
 * BEVPoolConvert GPU Kernel for OpenVINO Custom Layer
 *
 * Simple element-wise conversion: int32 fixed-point → float32
 *   output[i] = (float)input[i] / FP_SCALE
 *
 * Work sizing: one work-item per element
 */

#ifndef FP_SCALE
#define FP_SCALE 8192  /* must match bevpool_scatter; overridden from the op attribute */
#endif

__kernel void bevpool_convert(
    __global const INPUT0_TYPE* input,   /* [1, C, NX, NY] int32 */
    __global OUTPUT0_TYPE* output        /* [1, C, NX, NY] float32 */
) {
    const int idx = get_global_id(0);
    output[idx] = (float)input[idx] / (float)FP_SCALE;
}
