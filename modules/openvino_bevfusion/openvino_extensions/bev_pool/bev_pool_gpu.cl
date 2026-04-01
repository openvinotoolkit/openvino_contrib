/*
 * Optimized BEV Pool GPU Kernel for OpenVINO Custom Layer
 *
 * Single fused kernel: depth softmax + weighted scatter via float CAS atomics.
 *
 * Key optimizations:
 *   - Per-point parallelism (one work-item per point in N*D*H*W)
 *   - Float CAS atomic add for BEV grid scatter
 *   - Geometry-based grid lookup (no pre-sorting / interval tables)
 *   - All grid bounds hardcoded (constants for BEVFusion)
 *
 * OpenVINO custom layer defines (via XML):
 *   NX, NY, NZ, NUM_CAMS, DEPTH_BINS, CHANNELS, FEAT_H, FEAT_W
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/* Grid bounds — constant for BEVFusion */
#define X_MIN  -54.0f
#define Y_MIN  -54.0f
#define Z_MIN  -10.0f
#define X_STEP   0.3f
#define Y_STEP   0.3f
#define Z_STEP  20.0f

/* ─── Float atomic add via compare-and-swap ──────────────────────── */
inline void atomic_add_f(__global float* addr, float val) {
    union { unsigned int u; float f; } next, expected, current;
    current.f = *addr;
    do {
        expected.f = current.f;
        next.f     = expected.f + val;
        current.u  = atomic_cmpxchg(
            (__global unsigned int*)addr, expected.u, next.u);
    } while (current.u != expected.u);
}

/*
 * Fused BEV pool kernel — per-point parallelism.
 *
 * One work-item per point in (N * D * H * W).
 * Each work-item:
 *   1. Checks geometry → BEV grid cell (skip if out of bounds)
 *   2. Computes depth softmax for its (cam, h, w) pixel
 *   3. Scatter-adds weighted features to output BEV grid
 */
__kernel void bevpool_fused(
    __global const INPUT0_TYPE* depth_logits,   /* [N, D, H, W] */
    __global const INPUT1_TYPE* context_feats,  /* [N, C, H, W] */
    __global const INPUT2_TYPE* geom,           /* [N*D*H*W, 3] */
    __global OUTPUT0_TYPE* output               /* [1, C*NZ, NX, NY] */
) {
    const int D     = DEPTH_BINS;
    const int H     = FEAT_H;
    const int W     = FEAT_W;
    const int C     = CHANNELS;
    const int nx    = NX;
    const int ny    = NY;
    const int nz    = NZ;
    const int HW    = H * W;

    const int point_idx    = get_global_id(0);
    const int total_points = NUM_CAMS * D * H * W;
    if (point_idx >= total_points) return;

    /* ── geometry → grid index (early exit if out of bounds) ── */
    const float gx = geom[point_idx * 3 + 0];
    const float gy = geom[point_idx * 3 + 1];
    const float gz = geom[point_idx * 3 + 2];

    const int ix = (int)((gx - X_MIN) / X_STEP);
    const int iy = (int)((gy - Y_MIN) / Y_STEP);
    const int iz = (int)((gz - Z_MIN) / Z_STEP);

    if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) return;

    /* ── decode point → (cam, d, hw) ── */
    const int D_HW = D * HW;
    const int cam  = point_idx / D_HW;
    const int rem  = point_idx % D_HW;
    const int d    = rem / HW;
    const int hw   = rem % HW;

    /* ── depth softmax for this pixel ── */
    const int depth_base   = cam * D * HW + hw;
    const int depth_stride = HW;

    float max_val = -1e30f;
    for (int dd = 0; dd < D; ++dd) {
        float v = depth_logits[depth_base + dd * depth_stride];
        max_val = fmax(max_val, v);
    }

    float sum_exp = 0.0f;
    float my_exp  = 0.0f;
    for (int dd = 0; dd < D; ++dd) {
        float e = exp(depth_logits[depth_base + dd * depth_stride] - max_val);
        sum_exp += e;
        if (dd == d) my_exp = e;
    }

    float depth_w = my_exp / sum_exp;
    if (depth_w < 1e-6f) return;

    /* ── scatter weighted features to BEV grid ── */
    const int ctx_base   = cam * C * HW + hw;
    const int ctx_stride = HW;
    const int nx_ny      = nx * ny;

    for (int c = 0; c < C; ++c) {
        float ctx_val = context_feats[ctx_base + c * ctx_stride];
        float val     = depth_w * ctx_val;
        int   oidx    = (c * nz + iz) * nx_ny + ix * ny + iy;
        atomic_add_f(&output[oidx], val);
    }
}
