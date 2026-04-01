/*
 * VoxelizeScatter GPU Kernel for OpenVINO Custom Layer
 *
 * Single work-group approach to avoid stale-buffer issue.
 * GWS = LWS = 1024 (all WIs in one work-group).
 *
 * Phase 1: Zero entire workspace (each WI zeros ~1409 ints).
 * Barrier (CLK_GLOBAL_MEM_FENCE) — valid because single WG.
 * Phase 2: Hash-insert all points (each WI processes ~68 points).
 *
 * Workspace layout (int32 flat buffer):
 *   [0]:                        num_voxels counter
 *   [1 .. HASH_SIZE*ENTRY_STRIDE]: hash table entries, each ENTRY_STRIDE ints:
 *       [key_stored, accum0..4, count, batch, coord_x, coord_y, coord_z]
 */

/* ── Grid parameters (BEVFusion defaults) ── */
#define GRID_X      1440
#define GRID_Y      1440
#define GRID_Z      41
#define VOXEL_X     0.075f
#define VOXEL_Y     0.075f
#define VOXEL_Z     0.2f
#define RANGE_X_MIN -54.0f
#define RANGE_Y_MIN -54.0f
#define RANGE_Z_MIN -5.0f

/* ── Workspace layout ── */
#define MAX_VOXELS          60000
#define MAX_POINTS_PER_VOX  10
#define NUM_FEATURES        5
#define FP_SCALE            4096
#define HASH_SIZE           131072
#define HASH_MASK           131071
#define HT_OFFSET           1
#define ENTRY_STRIDE        11  /* key, feat*5, count, batch, x, y, z */
#define WORKSPACE_SIZE      1441793
#define WG_SIZE             1024


__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void voxelize_scatter(
    __global const INPUT0_TYPE* points,      /* [MAX_POINTS, 5] float   */
    __global const INPUT1_TYPE* num_pts_in,  /* [1]             int     */
    __global OUTPUT0_TYPE* workspace         /* [WORKSPACE_SIZE] int    */
) {
    const int lid = get_local_id(0);
    const int num_pts = num_pts_in[0];

    /* ══════════ Phase 1: Zero entire workspace ══════════ */
    const int chunk = (WORKSPACE_SIZE + WG_SIZE - 1) / WG_SIZE;
    const int start = lid * chunk;
    const int end   = min(start + chunk, WORKSPACE_SIZE);
    for (int i = start; i < end; i++) {
        workspace[i] = 0;
    }

    /* ── Full work-group barrier (valid: single WG) ── */
    barrier(CLK_GLOBAL_MEM_FENCE);

    /* ══════════ Phase 2: Hash-insert all points ══════════ */
    const int pts_chunk = (num_pts + WG_SIZE - 1) / WG_SIZE;
    const int p_start   = lid * pts_chunk;
    const int p_end     = min(p_start + pts_chunk, num_pts);

    for (int pid = p_start; pid < p_end; pid++) {
        const int p_base = pid * NUM_FEATURES;
        const float px = (float)points[p_base + 0];
        const float py = (float)points[p_base + 1];
        const float pz = (float)points[p_base + 2];

        /* ── Compute grid coordinates ── */
        const int ix = (int)floor((px - RANGE_X_MIN) / VOXEL_X);
        const int iy = (int)floor((py - RANGE_Y_MIN) / VOXEL_Y);
        const int iz = (int)floor((pz - RANGE_Z_MIN) / VOXEL_Z);

        if (ix < 0 || ix >= GRID_X || iy < 0 || iy >= GRID_Y ||
            iz < 0 || iz >= GRID_Z)
            continue;

        /* ── Hash key (max ~85M < 2^31) ── */
        const int key = ix * (GRID_Y * GRID_Z) + iy * GRID_Z + iz;
        const int stored_key = key + 1;  /* 0 = empty sentinel */

        /* ── Open-address hash table insert ── */
        int slot = key & HASH_MASK;

        for (int probe = 0; probe < 256; probe++) {
            const int entry_base = HT_OFFSET + slot * ENTRY_STRIDE;

            int old = atomic_cmpxchg(&workspace[entry_base], 0, stored_key);

            if (old == 0) {
                /* ── Claimed empty slot: new voxel ── */
                int nv = atomic_inc(&workspace[0]);
                if (nv >= MAX_VOXELS) {
                    workspace[entry_base] = -1;
                    break;
                }
                workspace[entry_base + 7]  = 0;    /* batch */
                workspace[entry_base + 8]  = ix;
                workspace[entry_base + 9]  = iy;
                workspace[entry_base + 10] = iz;
                /* fall through to accumulate */
            } else if (old == stored_key) {
                /* ── Found our key — accumulate ── */
            } else if (old == -1) {
                break;  /* poisoned */
            } else {
                slot = (slot + 1) & HASH_MASK;
                continue;
            }

            /* ── Accumulate features (capped at MAX_POINTS_PER_VOX) ── */
            const int ticket = atomic_inc(&workspace[entry_base + 6]);
            if (ticket < MAX_POINTS_PER_VOX) {
                for (int f = 0; f < NUM_FEATURES; f++) {
                    int ival = (int)((float)points[p_base + f] * (float)FP_SCALE);
                    atomic_add(&workspace[entry_base + 1 + f], ival);
                }
            }
            break;
        }
    }
}
