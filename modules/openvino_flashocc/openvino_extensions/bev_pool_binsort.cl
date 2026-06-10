/*
 * BEVPoolBinSort GPU Kernel — Single-Workgroup Counting Sort
 *
 * Produces a single packed output buffer containing:
 *   [sorted_ranks | cell_scratch | interval_starts | interval_lengths]
 *   offsets: 0      TOTAL_PTS     2*TOTAL_PTS      2*TOTAL_PTS+NX*NY
 *
 * Packed into one buffer because SimpleGPU custom layers only support
 * single-output operations.
 *
 * WorkSizes: global=1024, local=1024 (single workgroup)
 *
 * OpenVINO custom layer defines: NX, NY, TOTAL_PTS
 */

#ifndef X_MIN
#define X_MIN  -54.0f
#endif

#ifndef Y_MIN
#define Y_MIN  -54.0f
#endif

#ifndef X_STEP
#define X_STEP   0.3f
#endif

#ifndef Y_STEP
#define Y_STEP   0.3f
#endif

/* Offsets into packed output buffer */
#define OFF_SORTED  0
#define OFF_CELL    TOTAL_PTS
#define OFF_STARTS  (2 * TOTAL_PTS)
#define OFF_LENGTHS (2 * TOTAL_PTS + NX * NY)

__kernel __attribute__((reqd_work_group_size(1024, 1, 1)))
void bevpool_binsort(
    __global const INPUT0_TYPE* geom,       /* [TOTAL_PTS, 3] FP32 */
    __global OUTPUT0_TYPE* packed            /* [TOTAL_PTS*2 + NX*NY*2] I32 */
) {
    __global int* sorted   = packed + OFF_SORTED;
    __global int* cells    = packed + OFF_CELL;
    __global int* starts   = packed + OFF_STARTS;
    __global int* lengths  = packed + OFF_LENGTHS;

    const int lid = get_local_id(0);
    const int WG  = 1024;
    const int total_pts = TOTAL_PTS;
    const int n_cells   = NX * NY;
    const int nx = NX;
    const int ny = NY;

    /* Phase 0: Zero lengths */
    for (int i = lid; i < n_cells; i += WG) {
        lengths[i] = 0;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    /* Phase 1: Compute cell per point + count */
    for (int p = lid; p < total_pts; p += WG) {
        float gx = geom[p * 3 + 0];
        float gy = geom[p * 3 + 1];
        int ix = (int)floor((gx - X_MIN) / X_STEP);
        int iy = (int)floor((gy - Y_MIN) / Y_STEP);

        int cell = -1;
        if (ix >= 0 && ix < nx && iy >= 0 && iy < ny) {
            cell = ix * ny + iy;
            atomic_add(&lengths[cell], 1);
        }
        cells[p] = cell;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    /* Phase 2: Exclusive prefix sum -> interval_starts (single thread) */
    if (lid == 0) {
        int acc = 0;
        for (int i = 0; i < n_cells; ++i) {
            starts[i] = acc;
            acc += lengths[i];
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    /* Phase 3: Scatter to sorted positions */
    for (int p = lid; p < total_pts; p += WG) {
        int cell = cells[p];
        if (cell >= 0) {
            int pos = atomic_add(&starts[cell], 1);
            sorted[pos] = p;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    /* Phase 4: Restore starts */
    for (int i = lid; i < n_cells; i += WG) {
        starts[i] -= lengths[i];
    }
}
