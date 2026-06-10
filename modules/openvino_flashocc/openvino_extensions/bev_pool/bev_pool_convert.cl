/*
 * BEVPoolConvert GPU Kernel for OpenVINO Custom Layer
 *
 * Simple element-wise conversion: int32 fixed-point → float32
 *   output[i] = (float)input[i] / FP_SCALE
 *
 * Work sizing: one work-item per element
 */

#define FP_SCALE 8192  /* must match bevpool_scatter */

__kernel void bevpool_convert(
    __global const INPUT0_TYPE* input,   /* [1, C, NX, NY] int32 */
    __global OUTPUT0_TYPE* output        /* [1, C, NX, NY] float32 */
) {
    const int idx = get_global_id(0);
    output[idx] = (float)input[idx] / (float)FP_SCALE;
}
