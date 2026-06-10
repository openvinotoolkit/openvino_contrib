/*
 * VoxelizeMean GPU Kernel for OpenVINO Custom Layer
 *
 * Single work-group approach (GWS=LWS=1024) to avoid stale-buffer
 * issue with the result output buffer.
 *
 * Phase 1: Zero metadata row + compact counter.
 * Barrier.
 * Phase 2: Each WI scans ~128 hash slots, compacts valid entries.
 *
 * Input:  workspace [WORKSPACE_SIZE] int32  (from scatter)
 * Output: result    [MAX_VOXELS + 1, 9] float32
 */

#define MAX_VOXELS          60000
#define MAX_POINTS_PER_VOX  10
#define NUM_FEATURES        5
#define FP_SCALE            4096
#define HASH_SIZE           131072
#define HT_OFFSET           1
#define ENTRY_STRIDE        11
#define WG_SIZE             1024


__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void voxelize_mean(
    __global const INPUT0_TYPE* workspace,   /* [WORKSPACE_SIZE] int  */
    __global OUTPUT0_TYPE* result            /* [MAX_VOXELS + 1, 9] float */
) {
    const int lid = get_local_id(0);

    /* ══════════ Phase 1: Zero metadata row and compact counter ══════════ */
    if (lid == 0) {
        int nv = workspace[0];
        if (nv > MAX_VOXELS) nv = MAX_VOXELS;
        result[0] = (float)nv;
        for (int c = 1; c < 9; c++) result[c] = 0.0f;
        /* Zero the compact counter stored at result[8] as int32 */
        ((__global int*)result)[8] = 0;
    }

    /* ── Full work-group barrier (valid: single WG) ── */
    barrier(CLK_GLOBAL_MEM_FENCE);

    /* ══════════ Phase 2: Scan hash slots and compact ══════════ */
    const int slots_chunk = (HASH_SIZE + WG_SIZE - 1) / WG_SIZE;
    const int s_start     = lid * slots_chunk;
    const int s_end       = min(s_start + slots_chunk, HASH_SIZE);

    for (int slot = s_start; slot < s_end; slot++) {
        const int entry_base = HT_OFFSET + slot * ENTRY_STRIDE;
        const int key_stored = workspace[entry_base + 0];

        if (key_stored <= 0) continue;   /* empty or poisoned */

        int count = workspace[entry_base + 6];
        if (count <= 0) continue;

        /* ── Atomically grab an output row ── */
        int row = atomic_inc((__global int*)(result + 8)) + 1;
        if (row > MAX_VOXELS) continue;

        /* Mean: divide by total count (matches reference behavior) */
        const float inv = 1.0f / ((float)FP_SCALE * (float)count);
        const int out_base = row * 9;

        /* Mean features */
        for (int f = 0; f < NUM_FEATURES; f++) {
            result[out_base + f] = (float)workspace[entry_base + 1 + f] * inv;
        }

        /* Coordinates */
        result[out_base + 5] = (float)workspace[entry_base + 7];   /* batch */
        result[out_base + 6] = (float)workspace[entry_base + 8];   /* coord_x */
        result[out_base + 7] = (float)workspace[entry_base + 9];   /* coord_y */
        result[out_base + 8] = (float)workspace[entry_base + 10];  /* coord_z */
    }
}
