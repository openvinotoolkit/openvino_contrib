/*
 * BEVPoolScatter GPU Kernel for OpenVINO Custom Layer — Per-Pixel Optimized
 *
 * Scatter pre-computed depth softmax weights × context features into BEV grid
 * using native int32 atomic_add with fixed-point scaling.
 *
 * Per-pixel approach: one work-group per pixel (cam, h, w), each WI handles
 * one output channel. All depth bins for a pixel are processed together.
 *
 * Advantages over per-point approach:
 *   1. ~120× fewer work-groups (16,896 vs ~2M), reduces dispatch overhead
 *   2. Geometry loaded once via local memory (shared across all channels)
 *   3. Context features read once per pixel (not per depth bin)
 *   4. Local accumulation for same-cell depth bins reduces atomics
 *   5. Total memory reads reduced from ~108MB to ~17MB
 *
 * Work sizing: one work-group per pixel
 *   total_pixels = NUM_CAMS * FEAT_H * FEAT_W = 6 * 32 * 88 = 16,896
 *   global = total_pixels * CHANNELS
 *   local  = CHANNELS (80)
 *
 * OpenVINO custom layer defines (via XML):
 *   NX, NY, NZ, NUM_CAMS, DEPTH_BINS, CHANNELS, FEAT_H, FEAT_W
 */

/* Grid bounds — constant for BEVFusion */
#define X_MIN  -54.0f
#define Y_MIN  -54.0f
#define Z_MIN  -10.0f
#define X_STEP   0.3f
#define Y_STEP   0.3f
#define Z_STEP  20.0f

#define FP_SCALE 8192  /* 2^13 fixed-point */

/* Depth bins with probability < threshold are skipped */
#define DEPTH_THRESHOLD 0.005f

__kernel void bevpool_scatter(
    __global const INPUT0_TYPE* depth_probs,    /* [N, D, H, W] (softmax) */
    __global const INPUT1_TYPE* context_feats,  /* [N, C, H, W] */
    __global const INPUT2_TYPE* geom,           /* [N*D*H*W, 3] */
    __global OUTPUT0_TYPE* output               /* [1, C*NZ, NX, NY] int32 */
) {
    const int D  = DEPTH_BINS;   /* 118 */
    const int H  = FEAT_H;      /* 32  */
    const int W  = FEAT_W;      /* 88  */
    const int C  = CHANNELS;    /* 80  */
    const int nx = NX;          /* 360 */
    const int ny = NY;          /* 360 */
    const int nz = NZ;          /* 1   */
    const int HW = H * W;       /* 2816 */

    const int pixel_idx = get_group_id(0);  /* 0 .. NUM_CAMS*HW-1 */
    const int c         = get_local_id(0);  /* 0 .. C-1 (channel idx) */

    const int total_pixels = NUM_CAMS * HW;
    if (pixel_idx >= total_pixels) return;

    const int cam = pixel_idx / HW;
    const int hw  = pixel_idx % HW;

    /* ── Phase 1: Collaborative loading of depth weights + geometry ── */
    /* D=118 bins, C=80 workers → each WI loads ceil(118/80) ≈ 2 bins */
    __local float loc_depth_w[DEPTH_BINS];
    __local int   loc_grid[DEPTH_BINS];  /* flattened grid index or -1 */

    for (int d = c; d < D; d += C) {
        int pid = cam * D * HW + d * HW + hw;
        float dw = depth_probs[pid];

        if (dw < DEPTH_THRESHOLD) {
            loc_grid[d] = -1;
        } else {
            float gx = geom[pid * 3 + 0];
            float gy = geom[pid * 3 + 1];

            int ix = (int)((gx - X_MIN) / X_STEP);
            int iy = (int)((gy - Y_MIN) / Y_STEP);

            if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) {
                loc_grid[d] = -1;
            } else {
                /* NZ=1 : grid_idx = ix * ny + iy
                 * For NZ>1: would need iz computation and (c*nz+iz)*nx*ny layout */
                loc_grid[d] = ix * ny + iy;
                loc_depth_w[d] = dw;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* ── Phase 2: Read context feature for this channel ── */
    float ctx_val = context_feats[cam * C * HW + c * HW + hw];

    /* ── Phase 3: Scatter with local accumulation ──
     * Adjacent depth bins often map to the same BEV cell (along a ray).
     * Accumulate locally and flush once per unique cell → fewer atomics. */
    int   prev_grid = -2;  /* impossible sentinel */
    float accum = 0.0f;
    const int nx_ny = nx * ny;

    for (int d = 0; d < D; ++d) {
        int g = loc_grid[d];
        if (g < 0) continue;

        float val = loc_depth_w[d] * ctx_val;

        if (g == prev_grid) {
            accum += val;  /* same cell → accumulate locally */
        } else {
            /* Flush previous accumulation */
            if (prev_grid >= 0) {
                int oidx = (c * nz) * nx_ny + prev_grid;
                atomic_add(&output[oidx], (int)(accum * FP_SCALE));
            }
            prev_grid = g;
            accum = val;
        }
    }
    /* Flush final accumulation */
    if (prev_grid >= 0) {
        int oidx = (c * nz) * nx_ny + prev_grid;
        atomic_add(&output[oidx], (int)(accum * FP_SCALE));
    }
}
