// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// CdpnPnpSolve GPU kernel - EPnP + RANSAC PnP solver.
//
// Self-contained single-work-item OpenCL kernel that implements the EPnP
// algorithm:
//   1. Build 2D-3D correspondences from denorm_coords + confidence
//   2. RANSAC: sample 5 -> EPnP -> project -> count inliers -> keep best
//   3. Refit EPnP on all inliers
//   4. Output R, T, num_corres, pnp_success
//
// Tensors (4D BFYX):
//   input0  denorm_coords: [N, 3, 64, 64]  f32
//   input1  confidence:    [N, 1, 64, 64]  f32
//   input2  obj_extents:   [N, 1, 1, 3]    f32
//   input3  crop_meta:     [N, 1, 1, 5]    f32 (c_w, c_h, s, w_begin, h_begin)
//   input4  cam_K:         [N, 1, 1, 4]    f32 (fx, fy, cx, cy)
//   output0 R:             [N, 1, 3, 3]    f32
//   output1 T_pnp:         [N, 1, 1, 3]    f32
//   output2 num_corres:    [N, 1, 1, 1]    f32
//   output3 pnp_success:   [N, 1, 1, 1]    f32
//
// WorkSizes: global = "B", local = "1" from output0 [N,1,3,3].
//
// JIT defines from XML:
//   MASK_THR     : float (0.5)
//   OUT_RES      : int   (64)
//   MAX_ITERS    : int   (100)
//   REPROJ_THR   : float (8.0)

#ifndef OUT_RES
#define OUT_RES 64
#endif
#ifndef MASK_THR
#define MASK_THR 0.5f
#endif
#ifndef MAX_ITERS
#define MAX_ITERS 100
#endif
#ifndef REPROJ_THR
#define REPROJ_THR 8.0f
#endif

#define SPATIAL (OUT_RES * OUT_RES)
#define MAX_CORRES SPATIAL
#define MODEL_POINTS 5

// --- Tiny inline helpers ---

inline float dot3f(float a0, float a1, float a2, float b0, float b1, float b2) {
    return a0*b0 + a1*b1 + a2*b2;
}

inline float dist2f(float a0, float a1, float a2, float b0, float b1, float b2) {
    float d0 = a0-b0, d1 = a1-b1, d2 = a2-b2;
    return d0*d0 + d1*d1 + d2*d2;
}

// Simple LCG pseudo-random number generator
inline uint lcg_rand(uint* state) {
    *state = (*state) * 1103515245u + 12345u;
    return (*state >> 16) & 0x7FFF;
}

// --- 3x3 matrix inverse ---
inline void inv3x3(float m00, float m01, float m02,
                   float m10, float m11, float m12,
                   float m20, float m21, float m22,
                   float* inv, float* det_out) {
    float det = m00*(m11*m22 - m12*m21)
              - m01*(m10*m22 - m12*m20)
              + m02*(m10*m21 - m11*m20);
    *det_out = det;
    if (fabs(det) < 1e-15f) return;
    float inv_det = 1.0f / det;
    inv[0] = (m11*m22 - m12*m21) * inv_det;
    inv[1] = (m02*m21 - m01*m22) * inv_det;
    inv[2] = (m01*m12 - m02*m11) * inv_det;
    inv[3] = (m12*m20 - m10*m22) * inv_det;
    inv[4] = (m00*m22 - m02*m20) * inv_det;
    inv[5] = (m02*m10 - m00*m12) * inv_det;
    inv[6] = (m10*m21 - m11*m20) * inv_det;
    inv[7] = (m01*m20 - m00*m21) * inv_det;
    inv[8] = (m00*m11 - m01*m10) * inv_det;
}

// --- Jacobi eigenvalue decomposition for symmetric 3x3 ---
inline void sym3x3_eigen(float A[9], float eigenvalues[3], float V[9]) {
    // A stored row-major [3][3], V stored row-major [3][3]
    float M[9]; for (int i = 0; i < 9; i++) M[i] = A[i];
    for (int i = 0; i < 9; i++) V[i] = 0;
    V[0] = V[4] = V[8] = 1.0f;

    for (int sweep = 0; sweep < 30; sweep++) {
        float off = fabs(M[1]) + fabs(M[2]) + fabs(M[5]);
        if (off < 1e-10f) break;

        for (int p = 0; p < 2; p++) {
            for (int q = p+1; q < 3; q++) {
                if (fabs(M[p*3+q]) < 1e-14f) continue;
                float tau = (M[q*3+q] - M[p*3+p]) / (2.0f * M[p*3+q]);
                float t;
                if (fabs(tau) > 1e10f)
                    t = 1.0f / (2.0f * tau);
                else
                    t = ((tau >= 0) ? 1.0f : -1.0f) /
                        (fabs(tau) + sqrt(1.0f + tau*tau));
                float c = 1.0f / sqrt(1.0f + t*t);
                float s = t * c;

                float Mpp = M[p*3+p], Mqq = M[q*3+q], Mpq = M[p*3+q];
                M[p*3+p] = Mpp - t * Mpq;
                M[q*3+q] = Mqq + t * Mpq;
                M[p*3+q] = M[q*3+p] = 0.0f;

                for (int r = 0; r < 3; r++) {
                    if (r == p || r == q) continue;
                    float Mrp = M[r*3+p], Mrq = M[r*3+q];
                    M[r*3+p] = M[p*3+r] = c * Mrp - s * Mrq;
                    M[r*3+q] = M[q*3+r] = s * Mrp + c * Mrq;
                }
                for (int r = 0; r < 3; r++) {
                    float Vrp = V[r*3+p], Vrq = V[r*3+q];
                    V[r*3+p] = c * Vrp - s * Vrq;
                    V[r*3+q] = s * Vrp + c * Vrq;
                }
            }
        }
    }
    eigenvalues[0] = M[0]; eigenvalues[1] = M[4]; eigenvalues[2] = M[8];

    // Sort descending
    for (int i = 0; i < 2; i++) {
        for (int j = i+1; j < 3; j++) {
            if (eigenvalues[j] > eigenvalues[i]) {
                float tmp = eigenvalues[i]; eigenvalues[i] = eigenvalues[j]; eigenvalues[j] = tmp;
                for (int k = 0; k < 3; k++) {
                    float tv = V[k*3+i]; V[k*3+i] = V[k*3+j]; V[k*3+j] = tv;
                }
            }
        }
    }
}

// --- Jacobi eigenvalue for 12x12 symmetric (in private memory) ---
// Uses flat arrays. This is expensive but needed for EPnP.
// Note: We store the 12x12 in local memory to avoid register pressure.
inline void sym12x12_eigen_smallest4(
    __local float* M,      // [144] in/out
    __local float* V,      // [144] eigenvectors columns
    float evals[12],       // eigenvalues out (sorted ascending)
    int sorted_idx[12])    // sorted indices
{
    // Initialize V = I
    for (int i = 0; i < 144; i++) V[i] = 0;
    for (int i = 0; i < 12; i++) V[i*12+i] = 1.0f;

    // Jacobi sweeps
    for (int sweep = 0; sweep < 60; sweep++) {
        float off = 0;
        for (int p = 0; p < 11; p++)
            for (int q = p+1; q < 12; q++)
                off += M[p*12+q] * M[p*12+q];
        if (off < 1e-16f) break;

        for (int p = 0; p < 11; p++) {
            for (int q = p+1; q < 12; q++) {
                if (fabs(M[p*12+q]) < 1e-16f) continue;
                float tau = (M[q*12+q] - M[p*12+p]) / (2.0f * M[p*12+q]);
                float t;
                if (fabs(tau) > 1e10f)
                    t = 1.0f / (2.0f * tau);
                else
                    t = ((tau >= 0) ? 1.0f : -1.0f) /
                        (fabs(tau) + sqrt(1.0f + tau*tau));
                float c = 1.0f / sqrt(1.0f + t*t);
                float s = t * c;

                float Mpp = M[p*12+p], Mqq = M[q*12+q], Mpq = M[p*12+q];
                M[p*12+p] = Mpp - t * Mpq;
                M[q*12+q] = Mqq + t * Mpq;
                M[p*12+q] = M[q*12+p] = 0.0f;

                for (int r = 0; r < 12; r++) {
                    if (r == p || r == q) continue;
                    float Mrp = M[r*12+p], Mrq = M[r*12+q];
                    M[r*12+p] = M[p*12+r] = c * Mrp - s * Mrq;
                    M[r*12+q] = M[q*12+r] = s * Mrp + c * Mrq;
                }
                for (int r = 0; r < 12; r++) {
                    float Vrp = V[p*12+r], Vrq = V[q*12+r];
                    V[p*12+r] = c * Vrp - s * Vrq;
                    V[q*12+r] = s * Vrp + c * Vrq;
                }
            }
        }
    }

    // Extract eigenvalues
    for (int i = 0; i < 12; i++) {
        evals[i] = M[i*12+i];
        sorted_idx[i] = i;
    }
    // Sort ascending by eigenvalue (bubble sort, N=12)
    for (int i = 0; i < 11; i++)
        for (int j = i+1; j < 12; j++)
            if (evals[sorted_idx[j]] < evals[sorted_idx[i]]) {
                int tmp = sorted_idx[i];
                sorted_idx[i] = sorted_idx[j];
                sorted_idx[j] = tmp;
            }
}

// --- Least-squares solve for small systems via normal equations ---
// Solves L^T L x = L^T rhs  for rowsxcols system
inline void ls_solve(float* L, int rows, int cols, float* rhs, float* x) {
    // AtA and Atb
    float AtA[25], Atb[5]; // max cols = 5
    for (int i = 0; i < cols; i++) {
        Atb[i] = 0;
        for (int j = 0; j < cols; j++) AtA[i*cols+j] = 0;
    }
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < cols; j++) {
            float s = 0;
            for (int k = 0; k < rows; k++) s += L[k*cols+i] * L[k*cols+j];
            AtA[i*cols+j] = s;
        }
        float s = 0;
        for (int k = 0; k < rows; k++) s += L[k*cols+i] * rhs[k];
        Atb[i] = s;
    }
    // Gaussian elimination
    float aug[30]; // max cols=5, augmented cols=6
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < cols; j++) aug[i*(cols+1)+j] = AtA[i*cols+j];
        aug[i*(cols+1)+cols] = Atb[i];
    }
    for (int k = 0; k < cols; k++) {
        int pivot = k;
        for (int i = k+1; i < cols; i++)
            if (fabs(aug[i*(cols+1)+k]) > fabs(aug[pivot*(cols+1)+k])) pivot = i;
        if (pivot != k)
            for (int j = k; j <= cols; j++) {
                float tmp = aug[k*(cols+1)+j];
                aug[k*(cols+1)+j] = aug[pivot*(cols+1)+j];
                aug[pivot*(cols+1)+j] = tmp;
            }
        if (fabs(aug[k*(cols+1)+k]) < 1e-12f) { x[k] = 0; continue; }
        for (int i = k+1; i < cols; i++) {
            float f = aug[i*(cols+1)+k] / aug[k*(cols+1)+k];
            for (int j = k; j <= cols; j++)
                aug[i*(cols+1)+j] -= f * aug[k*(cols+1)+j];
        }
    }
    for (int i = cols-1; i >= 0; i--) {
        x[i] = aug[i*(cols+1)+cols];
        for (int j = i+1; j < cols; j++) x[i] -= aug[i*(cols+1)+j] * x[j];
        x[i] = (fabs(aug[i*(cols+1)+i]) > 1e-12f) ? x[i] / aug[i*(cols+1)+i] : 0.0f;
    }
}

// --- QR solve 6x4 (Householder) ---
inline void qr_solve_6x4(float A[24], float b[6], float x[4]) {
    for (int k = 0; k < 4; k++) {
        float norm = 0;
        for (int i = k; i < 6; i++) norm += A[i*4+k] * A[i*4+k];
        norm = sqrt(norm);
        if (norm < 1e-12f) { x[k] = 0; continue; }
        float sign = (A[k*4+k] >= 0) ? 1.0f : -1.0f;
        float alpha = sign * norm;
        A[k*4+k] += alpha;
        float scale = 1.0f / (alpha * A[k*4+k]);
        for (int j = k+1; j < 4; j++) {
            float d = 0;
            for (int i = k; i < 6; i++) d += A[i*4+k] * A[i*4+j];
            d *= scale;
            for (int i = k; i < 6; i++) A[i*4+j] -= d * A[i*4+k];
        }
        {
            float d = 0;
            for (int i = k; i < 6; i++) d += A[i*4+k] * b[i];
            d *= scale;
            for (int i = k; i < 6; i++) b[i] -= d * A[i*4+k];
        }
        A[k*4+k] = -alpha;
    }
    for (int i = 3; i >= 0; i--) {
        float s = b[i];
        for (int j = i+1; j < 4; j++) s -= A[i*4+j] * x[j];
        x[i] = (fabs(A[i*4+i]) > 1e-12f) ? s / A[i*4+i] : 0.0f;
    }
}

// --- SVD of 3x3 for R estimation ---
inline void svd3x3(float A[9], float U[9], float S[3], float Vt[9]) {
    float AtA[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float s = 0;
            for (int k = 0; k < 3; k++) s += A[k*3+i] * A[k*3+j];
            AtA[i*3+j] = s;
        }
    float evals[3], V[9];
    sym3x3_eigen(AtA, evals, V);
    for (int i = 0; i < 3; i++) S[i] = sqrt(max(evals[i], 0.0f));
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) Vt[i*3+j] = V[j*3+i];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float s = 0;
            for (int k = 0; k < 3; k++) s += A[i*3+k] * V[k*3+j];
            U[i*3+j] = (S[j] > 1e-8f) ? s / S[j] : 0.0f;
        }
}


// ===============================================================
// Main kernel
// ===============================================================

__kernel void cdpn_pnp_solve_gpu(
    const __global float* restrict denorm_coords,   // [N,3,64,64]
    const __global float* restrict confidence,      // [N,1,64,64]
    const __global float* restrict obj_extents,     // [N,1,1,3]
    const __global float* restrict crop_meta,       // [N,1,1,5]
    const __global float* restrict cam_K,           // [N,1,1,4]
    __global float* restrict R_out,                 // [N,1,3,3]
    __global float* restrict T_out,                 // [N,1,1,3]
    __global float* restrict num_corres_out,        // [N,1,1,1]
    __global float* restrict pnp_success_out)       // [N,1,1,1]
{
    const int n = get_global_id(0);  // batch index
    if (n >= OUTPUT0_DIMS[0]) return;

    // Use GPU plugin pitch macros for correct buffer addressing.
    // INPUT0 = denorm_coords [N,3,64,64],
    // INPUT1 = confidence [N,1,64,64]
    // INPUT2 = obj_extents [N,1,1,3],
    // INPUT3 = crop_meta [N,1,1,5]
    // INPUT4 = cam_K [N,1,1,4]
    // OUTPUT0 = R [N,1,3,3],
    // OUTPUT1 = T_pnp [N,1,1,3]
    // OUTPUT2 = num_corres [N,1,1,1],
    // OUTPUT3 = pnp_success [N,1,1,1]
    const int denorm_base = n * INPUT0_PITCHES[0] + INPUT0_OFFSET;
    const int conf_base   = n * INPUT1_PITCHES[0] + INPUT1_OFFSET;
    const int ext_base    = n * INPUT2_PITCHES[0] + INPUT2_OFFSET;
    const int meta_base   = n * INPUT3_PITCHES[0] + INPUT3_OFFSET;
    const int camK_base   = n * INPUT4_PITCHES[0] + INPUT4_OFFSET;
    const int R_base      = n * OUTPUT0_PITCHES[0] + OUTPUT0_OFFSET;
    const int T_base      = n * OUTPUT1_PITCHES[0] + OUTPUT1_OFFSET;
    const int nc_base     = n * OUTPUT2_PITCHES[0] + OUTPUT2_OFFSET;
    const int ok_base     = n * OUTPUT3_PITCHES[0] + OUTPUT3_OFFSET;

    // --- Default outputs ---
    R_out[R_base + 0*OUTPUT0_PITCHES[2] + 0] = 1;
    R_out[R_base + 0*OUTPUT0_PITCHES[2] + 1] = 0;
    R_out[R_base + 0*OUTPUT0_PITCHES[2] + 2] = 0;
    R_out[R_base + 1*OUTPUT0_PITCHES[2] + 0] = 0;
    R_out[R_base + 1*OUTPUT0_PITCHES[2] + 1] = 1;
    R_out[R_base + 1*OUTPUT0_PITCHES[2] + 2] = 0;
    R_out[R_base + 2*OUTPUT0_PITCHES[2] + 0] = 0;
    R_out[R_base + 2*OUTPUT0_PITCHES[2] + 1] = 0;
    R_out[R_base + 2*OUTPUT0_PITCHES[2] + 2] = 1;
    T_out[T_base + 0] = T_out[T_base + 1] = T_out[T_base + 2] = 0;
    num_corres_out[nc_base] = 0;
    pnp_success_out[ok_base] = 0;

    const float fx = cam_K[camK_base + 0], fy = cam_K[camK_base + 1];
    const float cx = cam_K[camK_base + 2], cy = cam_K[camK_base + 3];
    const float s = crop_meta[meta_base + 2];
    const float w_begin = crop_meta[meta_base + 3], h_begin = crop_meta[meta_base + 4];
    const float w_unit = s / (float)OUT_RES, h_unit = s / (float)OUT_RES;
    const float ext_x = obj_extents[ext_base + 0];
    const float ext_y = obj_extents[ext_base + 1];
    const float ext_z = obj_extents[ext_base + 2];
    const float min_x = 0.001f * ext_x, min_y = 0.001f * ext_y, min_z = 0.001f * ext_z;

    // --- Build correspondences into local memory ---
    // Max correspondences = SPATIAL = 4096
    __local float pts3d[MAX_CORRES * 3];
    __local float pts2d[MAX_CORRES * 2];
    __local float mtm_buf[144];  // for MtM [12x12]
    __local float eig_buf[144];  // for eigenvectors [12x12]

    int n_pts = 0;
    for (int row = 0; row < OUT_RES; row++) {
        for (int col = 0; col < OUT_RES; col++) {
            // Use pitch macros for confidence [N,1,64,64]
            float c = confidence[conf_base + row * INPUT1_PITCHES[2] + col * INPUT1_PITCHES[3]];
            if (c < MASK_THR) continue;
            // Use pitch macros for denorm_coords [N,3,64,64]
            float x3 = denorm_coords[denorm_base + 0 * INPUT0_PITCHES[1]
                                    + row * INPUT0_PITCHES[2] + col * INPUT0_PITCHES[3]];
            float y3 = denorm_coords[denorm_base + 1 * INPUT0_PITCHES[1]
                                    + row * INPUT0_PITCHES[2] + col * INPUT0_PITCHES[3]];
            float z3 = denorm_coords[denorm_base + 2 * INPUT0_PITCHES[1]
                                    + row * INPUT0_PITCHES[2] + col * INPUT0_PITCHES[3]];
            if (fabs(x3) < min_x && fabs(y3) < min_y && fabs(z3) < min_z)
                continue;
            pts3d[n_pts*3]   = x3;
            pts3d[n_pts*3+1] = y3;
            pts3d[n_pts*3+2] = z3;
            pts2d[n_pts*2]   = w_begin + col * w_unit;
            pts2d[n_pts*2+1] = h_begin + row * h_unit;
            n_pts++;
        }
    }
    num_corres_out[nc_base] = (float)n_pts;
    if (n_pts < 5) return;

    // --- EPnP function (operates on a subset in private mem) ---
    // A macro-like inline approach: extract subset, run EPnP, return R,t

    const float REPROJ_THR_SQ = REPROJ_THR * REPROJ_THR;
    int niters = MAX_ITERS;
    if (niters < 1) niters = 1;

    int max_good_count = 0;
    float best_R[9] = {1,0,0, 0,1,0, 0,0,1};
    float best_t[3] = {0,0,0};

    // RNG state
    uint rng_state = 12345u;

    for (int iter = 0; iter < niters; iter++) {
        // Sample 5 distinct indices
        int sample[5];
        for (int i = 0; i < 5; i++) {
            bool ok;
            do {
                sample[i] = (int)(lcg_rand(&rng_state) % (uint)n_pts);
                ok = true;
                for (int j = 0; j < i; j++)
                    if (sample[j] == sample[i]) { ok = false; break; }
            } while (!ok);
        }

        // Extract sample into private arrays
        float s3d[15], s2d[10]; // 5 points x 3, 5 points x 2
        for (int i = 0; i < 5; i++) {
            s3d[3*i]   = pts3d[3*sample[i]];
            s3d[3*i+1] = pts3d[3*sample[i]+1];
            s3d[3*i+2] = pts3d[3*sample[i]+2];
            s2d[2*i]   = pts2d[2*sample[i]];
            s2d[2*i+1] = pts2d[2*sample[i]+1];
        }

        // --- EPnP on 5-point sample ---
        int np = 5;

        // 1. Choose control points
        float cws[4][3];
        cws[0][0] = cws[0][1] = cws[0][2] = 0;
        for (int i = 0; i < np; i++) {
            cws[0][0] += s3d[3*i]; cws[0][1] += s3d[3*i+1]; cws[0][2] += s3d[3*i+2];
        }
        cws[0][0] /= np; cws[0][1] /= np; cws[0][2] /= np;

        // PCA
        float cov[9] = {};
        for (int i = 0; i < np; i++) {
            float dx = s3d[3*i]-cws[0][0], dy = s3d[3*i+1]-cws[0][1], dz = s3d[3*i+2]-cws[0][2];
            cov[0] += dx*dx; cov[1] += dx*dy; cov[2] += dx*dz;
            cov[3] += dy*dx; cov[4] += dy*dy; cov[5] += dy*dz;
            cov[6] += dz*dx; cov[7] += dz*dy; cov[8] += dz*dz;
        }
        float evals3[3], evecs3[9];
        sym3x3_eigen(cov, evals3, evecs3);
        for (int i = 1; i < 4; i++) {
            float k = sqrt(max(evals3[i-1], 0.0f) / (float)np);
            for (int j = 0; j < 3; j++)
                cws[i][j] = cws[0][j] + k * evecs3[j*3+(i-1)];
        }

        // 2. Barycentric coordinates
        float cc_inv[9], det_val;
        inv3x3(cws[1][0]-cws[0][0], cws[2][0]-cws[0][0], cws[3][0]-cws[0][0],
               cws[1][1]-cws[0][1], cws[2][1]-cws[0][1], cws[3][1]-cws[0][1],
               cws[1][2]-cws[0][2], cws[2][2]-cws[0][2], cws[3][2]-cws[0][2],
               cc_inv, &det_val);
        if (fabs(det_val) < 1e-12f) continue;

        float alphas[20]; // 5x4
        for (int i = 0; i < np; i++) {
            float dx = s3d[3*i]-cws[0][0], dy = s3d[3*i+1]-cws[0][1], dz = s3d[3*i+2]-cws[0][2];
            alphas[4*i+1] = cc_inv[0]*dx + cc_inv[1]*dy + cc_inv[2]*dz;
            alphas[4*i+2] = cc_inv[3]*dx + cc_inv[4]*dy + cc_inv[5]*dz;
            alphas[4*i+3] = cc_inv[6]*dx + cc_inv[7]*dy + cc_inv[8]*dz;
            alphas[4*i]   = 1.0f - alphas[4*i+1] - alphas[4*i+2] - alphas[4*i+3];
        }

        // 3. Build M [10 x 12] and MtM [12 x 12]
        for (int i = 0; i < 144; i++) mtm_buf[i] = 0;
        for (int i = 0; i < np; i++) {
            float* a = &alphas[4*i];
            float u = s2d[2*i], v = s2d[2*i+1];
            float M1[12], M2[12];
            for (int j = 0; j < 4; j++) {
                M1[3*j]   = a[j] * fx;
                M1[3*j+1] = 0;
                M1[3*j+2] = a[j] * (cx - u);
                M2[3*j]   = 0;
                M2[3*j+1] = a[j] * fy;
                M2[3*j+2] = a[j] * (cy - v);
            }
            for (int r = 0; r < 12; r++)
                for (int c = r; c < 12; c++) {
                    float val = M1[r]*M1[c] + M2[r]*M2[c];
                    mtm_buf[r*12+c] += val;
                    if (r != c) mtm_buf[c*12+r] += val;
                }
        }

        // 4. Eigendecomposition of MtM
        float evals12[12];
        int sorted12[12];
        sym12x12_eigen_smallest4(mtm_buf, eig_buf, evals12, sorted12);

        // Build ut: row i of ut = eigenvectors[sorted12[i]]  (ascending order)
        // Need rows 8,9,10,11 (mapped from smallest 4 eigenvalues)
        // OpenCV convention: ut[11] = smallest, ut[8] = 4th smallest
        // This is sorted ascending, so sorted12[0]=smallest, etc.
        // Map: ut row (11-k) for k=0..11 ← eigenvector sorted12[k]
        float ut[144];
        for (int k = 0; k < 12; k++) {
            int src = sorted12[k];
            for (int j = 0; j < 12; j++)
                ut[(11-k)*12+j] = eig_buf[src*12+j];
        }

        // 5. L_6x10 and rho
        const float* vp[4];
        float v_buf[48]; // 4 x 12
        vp[0] = &ut[11*12]; vp[1] = &ut[10*12]; vp[2] = &ut[9*12]; vp[3] = &ut[8*12];
        for (int i = 0; i < 48; i++) v_buf[i] = (i < 12) ? vp[0][i] : (i < 24) ? vp[1][i-12] : (i < 36) ? vp[2][i-24] : vp[3][i-36];

        float dv[4][6][3];
        for (int i = 0; i < 4; i++) {
            const float* vi = &v_buf[i*12];
            int a = 0, b = 1;
            for (int j = 0; j < 6; j++) {
                dv[i][j][0] = vi[3*a]   - vi[3*b];
                dv[i][j][1] = vi[3*a+1] - vi[3*b+1];
                dv[i][j][2] = vi[3*a+2] - vi[3*b+2];
                b++; if (b > 3) { a++; b = a+1; }
            }
        }

        float l_6x10[60]; // [6][10]
        for (int i = 0; i < 6; i++) {
            l_6x10[i*10+0] =       dot3f(dv[0][i][0],dv[0][i][1],dv[0][i][2], dv[0][i][0],dv[0][i][1],dv[0][i][2]);
            l_6x10[i*10+1] = 2.0f* dot3f(dv[0][i][0],dv[0][i][1],dv[0][i][2], dv[1][i][0],dv[1][i][1],dv[1][i][2]);
            l_6x10[i*10+2] =       dot3f(dv[1][i][0],dv[1][i][1],dv[1][i][2], dv[1][i][0],dv[1][i][1],dv[1][i][2]);
            l_6x10[i*10+3] = 2.0f* dot3f(dv[0][i][0],dv[0][i][1],dv[0][i][2], dv[2][i][0],dv[2][i][1],dv[2][i][2]);
            l_6x10[i*10+4] = 2.0f* dot3f(dv[1][i][0],dv[1][i][1],dv[1][i][2], dv[2][i][0],dv[2][i][1],dv[2][i][2]);
            l_6x10[i*10+5] =       dot3f(dv[2][i][0],dv[2][i][1],dv[2][i][2], dv[2][i][0],dv[2][i][1],dv[2][i][2]);
            l_6x10[i*10+6] = 2.0f* dot3f(dv[0][i][0],dv[0][i][1],dv[0][i][2], dv[3][i][0],dv[3][i][1],dv[3][i][2]);
            l_6x10[i*10+7] = 2.0f* dot3f(dv[1][i][0],dv[1][i][1],dv[1][i][2], dv[3][i][0],dv[3][i][1],dv[3][i][2]);
            l_6x10[i*10+8] = 2.0f* dot3f(dv[2][i][0],dv[2][i][1],dv[2][i][2], dv[3][i][0],dv[3][i][1],dv[3][i][2]);
            l_6x10[i*10+9] =       dot3f(dv[3][i][0],dv[3][i][1],dv[3][i][2], dv[3][i][0],dv[3][i][1],dv[3][i][2]);
        }

        float rho[6];
        rho[0] = dist2f(cws[0][0],cws[0][1],cws[0][2], cws[1][0],cws[1][1],cws[1][2]);
        rho[1] = dist2f(cws[0][0],cws[0][1],cws[0][2], cws[2][0],cws[2][1],cws[2][2]);
        rho[2] = dist2f(cws[0][0],cws[0][1],cws[0][2], cws[3][0],cws[3][1],cws[3][2]);
        rho[3] = dist2f(cws[1][0],cws[1][1],cws[1][2], cws[2][0],cws[2][1],cws[2][2]);
        rho[4] = dist2f(cws[1][0],cws[1][1],cws[1][2], cws[3][0],cws[3][1],cws[3][2]);
        rho[5] = dist2f(cws[2][0],cws[2][1],cws[2][2], cws[3][0],cws[3][1],cws[3][2]);

        // 6. Three beta approximations + Gauss-Newton + compute R,t
        float best_iter_R[9], best_iter_t[3];
        float best_iter_err = 1e20f;

        for (int approx = 1; approx <= 3; approx++) {
            float betas[4] = {0,0,0,0};

            if (approx == 1) {
                float L64[24], rho6[6], b4[4];
                for (int i = 0; i < 6; i++) {
                    L64[i*4+0] = l_6x10[i*10+0];
                    L64[i*4+1] = l_6x10[i*10+1];
                    L64[i*4+2] = l_6x10[i*10+3];
                    L64[i*4+3] = l_6x10[i*10+6];
                    rho6[i] = rho[i];
                }
                ls_solve(L64, 6, 4, rho6, b4);
                if (b4[0] < 0) {
                    betas[0] = sqrt(-b4[0]);
                    betas[1] = (fabs(betas[0]) > 1e-10f) ? -b4[1]/betas[0] : 0;
                    betas[2] = (fabs(betas[0]) > 1e-10f) ? -b4[2]/betas[0] : 0;
                    betas[3] = (fabs(betas[0]) > 1e-10f) ? -b4[3]/betas[0] : 0;
                } else {
                    betas[0] = sqrt(max(b4[0], 0.0f));
                    betas[1] = (fabs(betas[0]) > 1e-10f) ? b4[1]/betas[0] : 0;
                    betas[2] = (fabs(betas[0]) > 1e-10f) ? b4[2]/betas[0] : 0;
                    betas[3] = (fabs(betas[0]) > 1e-10f) ? b4[3]/betas[0] : 0;
                }
            } else if (approx == 2) {
                float L63[18], rho6[6], b3[3];
                for (int i = 0; i < 6; i++) {
                    L63[i*3+0] = l_6x10[i*10+0];
                    L63[i*3+1] = l_6x10[i*10+1];
                    L63[i*3+2] = l_6x10[i*10+2];
                    rho6[i] = rho[i];
                }
                ls_solve(L63, 6, 3, rho6, b3);
                if (b3[0] < 0) {
                    betas[0] = sqrt(-b3[0]);
                    betas[1] = (b3[2] < 0) ? sqrt(-b3[2]) : 0;
                } else {
                    betas[0] = sqrt(max(b3[0], 0.0f));
                    betas[1] = (b3[2] > 0) ? sqrt(b3[2]) : 0;
                }
                if (b3[1] < 0) betas[0] = -betas[0];
                betas[2] = 0; betas[3] = 0;
            } else {
                float L65[30], rho6[6], b5[5];
                for (int i = 0; i < 6; i++) {
                    L65[i*5+0] = l_6x10[i*10+0];
                    L65[i*5+1] = l_6x10[i*10+1];
                    L65[i*5+2] = l_6x10[i*10+2];
                    L65[i*5+3] = l_6x10[i*10+3];
                    L65[i*5+4] = l_6x10[i*10+4];
                    rho6[i] = rho[i];
                }
                ls_solve(L65, 6, 5, rho6, b5);
                if (b5[0] < 0) {
                    betas[0] = sqrt(-b5[0]);
                    betas[1] = (b5[2] < 0) ? sqrt(-b5[2]) : 0;
                } else {
                    betas[0] = sqrt(max(b5[0], 0.0f));
                    betas[1] = (b5[2] > 0) ? sqrt(b5[2]) : 0;
                }
                if (b5[1] < 0) betas[0] = -betas[0];
                betas[2] = (fabs(betas[0]) > 1e-10f) ? b5[3]/betas[0] : 0;
                betas[3] = 0;
            }

            // Gauss-Newton (5 iterations)
            for (int gn = 0; gn < 5; gn++) {
                float A[24], b[6], dx[4];
                for (int i = 0; i < 6; i++) {
                    const float* L = &l_6x10[i*10];
                    A[i*4+0] = 2*L[0]*betas[0]+L[1]*betas[1]+L[3]*betas[2]+L[6]*betas[3];
                    A[i*4+1] = L[1]*betas[0]+2*L[2]*betas[1]+L[4]*betas[2]+L[7]*betas[3];
                    A[i*4+2] = L[3]*betas[0]+L[4]*betas[1]+2*L[5]*betas[2]+L[8]*betas[3];
                    A[i*4+3] = L[6]*betas[0]+L[7]*betas[1]+L[8]*betas[2]+2*L[9]*betas[3];
                    b[i] = rho[i] - (L[0]*betas[0]*betas[0]+L[1]*betas[0]*betas[1]+
                        L[2]*betas[1]*betas[1]+L[3]*betas[0]*betas[2]+L[4]*betas[1]*betas[2]+
                        L[5]*betas[2]*betas[2]+L[6]*betas[0]*betas[3]+L[7]*betas[1]*betas[3]+
                        L[8]*betas[2]*betas[3]+L[9]*betas[3]*betas[3]);
                }
                qr_solve_6x4(A, b, dx);
                for (int i = 0; i < 4; i++) betas[i] += dx[i];
            }

            // Compute camera-frame control points
            float ccs[4][3];
            for (int i = 0; i < 4; i++) ccs[i][0] = ccs[i][1] = ccs[i][2] = 0;
            for (int i = 0; i < 4; i++) {
                const float* vi = &ut[(11-i)*12];
                for (int j = 0; j < 4; j++)
                    for (int k = 0; k < 3; k++)
                        ccs[j][k] += betas[i] * vi[3*j+k];
            }

            // Camera-frame 3D points
            float pcs[15]; // 5x3
            for (int i = 0; i < np; i++)
                for (int j = 0; j < 3; j++)
                    pcs[3*i+j] = alphas[4*i]*ccs[0][j]+alphas[4*i+1]*ccs[1][j]+
                                 alphas[4*i+2]*ccs[2][j]+alphas[4*i+3]*ccs[3][j];

            // Solve for sign
            if (pcs[2] < 0) {
                for (int i = 0; i < 4; i++) for (int j = 0; j < 3; j++) ccs[i][j] = -ccs[i][j];
                for (int i = 0; i < np*3; i++) pcs[i] = -pcs[i];
            }

            // Estimate R,t via 3D-3D alignment
            float pc0[3] = {0,0,0}, pw0[3] = {0,0,0};
            for (int i = 0; i < np; i++) {
                for (int j = 0; j < 3; j++) { pc0[j] += pcs[3*i+j]; pw0[j] += s3d[3*i+j]; }
            }
            for (int j = 0; j < 3; j++) { pc0[j] /= np; pw0[j] /= np; }

            float ABt[9] = {};
            for (int i = 0; i < np; i++)
                for (int j = 0; j < 3; j++) {
                    float pc_d = pcs[3*i+j] - pc0[j];
                    ABt[j*3+0] += pc_d * (s3d[3*i]-pw0[0]);
                    ABt[j*3+1] += pc_d * (s3d[3*i+1]-pw0[1]);
                    ABt[j*3+2] += pc_d * (s3d[3*i+2]-pw0[2]);
                }

            float U[9], Sv[3], VtM[9];
            svd3x3(ABt, U, Sv, VtM);
            float tmpR[9];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++) {
                    tmpR[i*3+j] = 0;
                    for (int k = 0; k < 3; k++) tmpR[i*3+j] += U[i*3+k]*VtM[k*3+j];
                }
            float det = tmpR[0]*(tmpR[4]*tmpR[8]-tmpR[5]*tmpR[7])
                       -tmpR[1]*(tmpR[3]*tmpR[8]-tmpR[5]*tmpR[6])
                       +tmpR[2]*(tmpR[3]*tmpR[7]-tmpR[4]*tmpR[6]);
            if (det < 0) { tmpR[6]=-tmpR[6]; tmpR[7]=-tmpR[7]; tmpR[8]=-tmpR[8]; }

            float tmpt[3];
            for (int i = 0; i < 3; i++)
                tmpt[i] = pc0[i] - (tmpR[i*3]*pw0[0]+tmpR[i*3+1]*pw0[1]+tmpR[i*3+2]*pw0[2]);

            // Reprojection error
            float err = 0;
            for (int i = 0; i < np; i++) {
                float Xc = tmpR[0]*s3d[3*i]+tmpR[1]*s3d[3*i+1]+tmpR[2]*s3d[3*i+2]+tmpt[0];
                float Yc = tmpR[3]*s3d[3*i]+tmpR[4]*s3d[3*i+1]+tmpR[5]*s3d[3*i+2]+tmpt[1];
                float Zc = tmpR[6]*s3d[3*i]+tmpR[7]*s3d[3*i+1]+tmpR[8]*s3d[3*i+2]+tmpt[2];
                if (fabs(Zc) < 1e-8f) { err += 1e6f; continue; }
                float ue = cx + fx*Xc/Zc;
                float ve = cy + fy*Yc/Zc;
                float du = s2d[2*i]-ue, dv2 = s2d[2*i+1]-ve;
                err += sqrt(du*du + dv2*dv2);
            }
            err /= np;

            if (err < best_iter_err) {
                best_iter_err = err;
                for (int i = 0; i < 9; i++) best_iter_R[i] = tmpR[i];
                for (int i = 0; i < 3; i++) best_iter_t[i] = tmpt[i];
            }
        } // end approx loop

        // Count inliers with this EPnP result
        int good_count = 0;
        for (int i = 0; i < n_pts; i++) {
            float Xc=best_iter_R[0]*pts3d[3*i]+best_iter_R[1]*pts3d[3*i+1]+best_iter_R[2]*pts3d[3*i+2]+best_iter_t[0];
            float Yc=best_iter_R[3]*pts3d[3*i]+best_iter_R[4]*pts3d[3*i+1]+best_iter_R[5]*pts3d[3*i+2]+best_iter_t[1];
            float Zc=best_iter_R[6]*pts3d[3*i]+best_iter_R[7]*pts3d[3*i+1]+best_iter_R[8]*pts3d[3*i+2]+best_iter_t[2];
            if (fabs(Zc) < 1e-8f) continue;
            float px = cx + fx*Xc/Zc;
            float py = cy + fy*Yc/Zc;
            float dx = pts2d[2*i]-px, dy = pts2d[2*i+1]-py;
            if (dx*dx+dy*dy <= REPROJ_THR_SQ) good_count++;
        }

        if (good_count > max(max_good_count, MODEL_POINTS-1)) {
            for (int i = 0; i < 9; i++) best_R[i] = best_iter_R[i];
            for (int i = 0; i < 3; i++) best_t[i] = best_iter_t[i];
            max_good_count = good_count;
            // Adaptive iteration count
            float out_ratio = (float)(n_pts - good_count) / (float)n_pts;
            float num = log(max(1.0f - 0.99f, 1e-10f));
            float denom = log(1.0f - pow(1.0f - out_ratio, (float)MODEL_POINTS));
            if (denom < 0) {
                int new_iters = (int)ceil(num / denom);
                if (new_iters < niters) niters = new_iters;
            }
        }
    } // end RANSAC loop

    if (max_good_count <= 0) return;

    // --- Refit on inliers ---
    // Collect inliers into a temporary buffer in local memory
    // Reuse pts3d/pts2d for inlier extraction
    int n_inliers = 0;
    // Need temporary storage for inlier points
    // Reuse mtm_buf area (144 floats) is not enough for large inlier sets
    // Instead, directly run EPnP on the full correspondence set
    // but only include inliers. Build ephemeral arrays.
    // Given GPU local memory constraints, run the refit only if inliers fit.
    // Max inliers = MAX_CORRES = 4096; need 5*4096 floats = 20K.
    // This exceeds typical local memory. Instead, store inlier indices.
    __local int inlier_idx[MAX_CORRES];
    for (int i = 0; i < n_pts; i++) {
        float Xc=best_R[0]*pts3d[3*i]+best_R[1]*pts3d[3*i+1]+best_R[2]*pts3d[3*i+2]+best_t[0];
        float Yc=best_R[3]*pts3d[3*i]+best_R[4]*pts3d[3*i+1]+best_R[5]*pts3d[3*i+2]+best_t[1];
        float Zc=best_R[6]*pts3d[3*i]+best_R[7]*pts3d[3*i+1]+best_R[8]*pts3d[3*i+2]+best_t[2];
        if (fabs(Zc) < 1e-8f) continue;
        float px = cx + fx*Xc/Zc;
        float py = cy + fy*Yc/Zc;
        float dx = pts2d[2*i]-px, dy = pts2d[2*i+1]-py;
        if (dx*dx+dy*dy <= REPROJ_THR_SQ)
            inlier_idx[n_inliers++] = i;
    }

    // Refit EPnP if enough inliers (same algorithm, using pts3d/pts2d with indirect indexing)
    // For simplicity and to avoid excessive local memory, perform a simplified refit:
    // Use all inlier points for choose_control_points, bary coords, matrix method (M), etc.
    // This is done in-place using inlier_idx for indirect access.
    if (n_inliers >= 5) {
        int np = n_inliers;

        // Control points
        float cws_r[4][3];
        cws_r[0][0] = cws_r[0][1] = cws_r[0][2] = 0;
        for (int ii = 0; ii < np; ii++) {
            int i = inlier_idx[ii];
            cws_r[0][0] += pts3d[3*i]; cws_r[0][1] += pts3d[3*i+1]; cws_r[0][2] += pts3d[3*i+2];
        }
        cws_r[0][0] /= np; cws_r[0][1] /= np; cws_r[0][2] /= np;

        float cov_r[9] = {};
        for (int ii = 0; ii < np; ii++) {
            int i = inlier_idx[ii];
            float dx = pts3d[3*i]-cws_r[0][0], dy = pts3d[3*i+1]-cws_r[0][1], dz = pts3d[3*i+2]-cws_r[0][2];
            cov_r[0]+=dx*dx; cov_r[1]+=dx*dy; cov_r[2]+=dx*dz;
            cov_r[3]+=dy*dx; cov_r[4]+=dy*dy; cov_r[5]+=dy*dz;
            cov_r[6]+=dz*dx; cov_r[7]+=dz*dy; cov_r[8]+=dz*dz;
        }
        float evals_r[3], evecs_r[9];
        sym3x3_eigen(cov_r, evals_r, evecs_r);
        for (int i = 1; i < 4; i++) {
            float k = sqrt(max(evals_r[i-1], 0.0f) / (float)np);
            for (int j = 0; j < 3; j++) cws_r[i][j] = cws_r[0][j] + k * evecs_r[j*3+(i-1)];
        }

        float cc_inv_r[9], det_r;
        inv3x3(cws_r[1][0]-cws_r[0][0], cws_r[2][0]-cws_r[0][0], cws_r[3][0]-cws_r[0][0],
               cws_r[1][1]-cws_r[0][1], cws_r[2][1]-cws_r[0][1], cws_r[3][1]-cws_r[0][1],
               cws_r[1][2]-cws_r[0][2], cws_r[2][2]-cws_r[0][2], cws_r[3][2]-cws_r[0][2],
               cc_inv_r, &det_r);

        if (fabs(det_r) > 1e-12f) {
            // Build MtM for refitting
            for (int i = 0; i < 144; i++) mtm_buf[i] = 0;
            for (int ii = 0; ii < np; ii++) {
                int idx = inlier_idx[ii];
                float dx = pts3d[3*idx]-cws_r[0][0], dy = pts3d[3*idx+1]-cws_r[0][1], dz = pts3d[3*idx+2]-cws_r[0][2];
                float a[4];
                a[1] = cc_inv_r[0]*dx + cc_inv_r[1]*dy + cc_inv_r[2]*dz;
                a[2] = cc_inv_r[3]*dx + cc_inv_r[4]*dy + cc_inv_r[5]*dz;
                a[3] = cc_inv_r[6]*dx + cc_inv_r[7]*dy + cc_inv_r[8]*dz;
                a[0] = 1.0f - a[1] - a[2] - a[3];

                float u = pts2d[2*idx], v = pts2d[2*idx+1];
                float M1[12], M2[12];
                for (int j = 0; j < 4; j++) {
                    M1[3*j]   = a[j]*fx;   M1[3*j+1] = 0;          M1[3*j+2] = a[j]*(cx-u);
                    M2[3*j]   = 0;         M2[3*j+1] = a[j]*fy;    M2[3*j+2] = a[j]*(cy-v);
                }
                for (int r = 0; r < 12; r++)
                    for (int c = r; c < 12; c++) {
                        float val = M1[r]*M1[c] + M2[r]*M2[c];
                        mtm_buf[r*12+c] += val;
                        if (r != c) mtm_buf[c*12+r] += val;
                    }
            }

            float evals12_r[12]; int sorted12_r[12];
            sym12x12_eigen_smallest4(mtm_buf, eig_buf, evals12_r, sorted12_r);

            float ut_r[144];
            for (int k = 0; k < 12; k++) {
                int src = sorted12_r[k];
                for (int j = 0; j < 12; j++) ut_r[(11-k)*12+j] = eig_buf[src*12+j];
            }

            // L_6x10 and rho for refit
            float vr_buf[48];
            for (int i = 0; i < 12; i++) {
                vr_buf[i]    = ut_r[11*12+i];
                vr_buf[12+i] = ut_r[10*12+i];
                vr_buf[24+i] = ut_r[9*12+i];
                vr_buf[36+i] = ut_r[8*12+i];
            }

            float dv_r[4][6][3];
            for (int i = 0; i < 4; i++) {
                const float* vi = &vr_buf[i*12];
                int a = 0, b = 1;
                for (int j = 0; j < 6; j++) {
                    dv_r[i][j][0] = vi[3*a]-vi[3*b]; dv_r[i][j][1] = vi[3*a+1]-vi[3*b+1]; dv_r[i][j][2] = vi[3*a+2]-vi[3*b+2];
                    b++; if (b>3){a++;b=a+1;}
                }
            }

            float l_r[60];
            for (int i = 0; i < 6; i++) {
                l_r[i*10+0]=      dot3f(dv_r[0][i][0],dv_r[0][i][1],dv_r[0][i][2],dv_r[0][i][0],dv_r[0][i][1],dv_r[0][i][2]);
                l_r[i*10+1]=2.0f* dot3f(dv_r[0][i][0],dv_r[0][i][1],dv_r[0][i][2],dv_r[1][i][0],dv_r[1][i][1],dv_r[1][i][2]);
                l_r[i*10+2]=      dot3f(dv_r[1][i][0],dv_r[1][i][1],dv_r[1][i][2],dv_r[1][i][0],dv_r[1][i][1],dv_r[1][i][2]);
                l_r[i*10+3]=2.0f* dot3f(dv_r[0][i][0],dv_r[0][i][1],dv_r[0][i][2],dv_r[2][i][0],dv_r[2][i][1],dv_r[2][i][2]);
                l_r[i*10+4]=2.0f* dot3f(dv_r[1][i][0],dv_r[1][i][1],dv_r[1][i][2],dv_r[2][i][0],dv_r[2][i][1],dv_r[2][i][2]);
                l_r[i*10+5]=      dot3f(dv_r[2][i][0],dv_r[2][i][1],dv_r[2][i][2],dv_r[2][i][0],dv_r[2][i][1],dv_r[2][i][2]);
                l_r[i*10+6]=2.0f* dot3f(dv_r[0][i][0],dv_r[0][i][1],dv_r[0][i][2],dv_r[3][i][0],dv_r[3][i][1],dv_r[3][i][2]);
                l_r[i*10+7]=2.0f* dot3f(dv_r[1][i][0],dv_r[1][i][1],dv_r[1][i][2],dv_r[3][i][0],dv_r[3][i][1],dv_r[3][i][2]);
                l_r[i*10+8]=2.0f* dot3f(dv_r[2][i][0],dv_r[2][i][1],dv_r[2][i][2],dv_r[3][i][0],dv_r[3][i][1],dv_r[3][i][2]);
                l_r[i*10+9]=      dot3f(dv_r[3][i][0],dv_r[3][i][1],dv_r[3][i][2],dv_r[3][i][0],dv_r[3][i][1],dv_r[3][i][2]);
            }
            float rho_r[6];
            rho_r[0]=dist2f(cws_r[0][0],cws_r[0][1],cws_r[0][2],cws_r[1][0],cws_r[1][1],cws_r[1][2]);
            rho_r[1]=dist2f(cws_r[0][0],cws_r[0][1],cws_r[0][2],cws_r[2][0],cws_r[2][1],cws_r[2][2]);
            rho_r[2]=dist2f(cws_r[0][0],cws_r[0][1],cws_r[0][2],cws_r[3][0],cws_r[3][1],cws_r[3][2]);
            rho_r[3]=dist2f(cws_r[1][0],cws_r[1][1],cws_r[1][2],cws_r[2][0],cws_r[2][1],cws_r[2][2]);
            rho_r[4]=dist2f(cws_r[1][0],cws_r[1][1],cws_r[1][2],cws_r[3][0],cws_r[3][1],cws_r[3][2]);
            rho_r[5]=dist2f(cws_r[2][0],cws_r[2][1],cws_r[2][2],cws_r[3][0],cws_r[3][1],cws_r[3][2]);

            float best_refit_err = 1e20f;
            float refit_R[9], refit_t[3];

            for (int approx = 1; approx <= 3; approx++) {
                float betas[4] = {0,0,0,0};
                if (approx == 1) {
                    float Lt[24], rh[6], b4[4];
                    for (int i=0;i<6;i++){Lt[i*4]=l_r[i*10];Lt[i*4+1]=l_r[i*10+1];Lt[i*4+2]=l_r[i*10+3];Lt[i*4+3]=l_r[i*10+6];rh[i]=rho_r[i];}
                    ls_solve(Lt,6,4,rh,b4);
                    if(b4[0]<0){betas[0]=sqrt(-b4[0]);betas[1]=(fabs(betas[0])>1e-10f)?-b4[1]/betas[0]:0;betas[2]=(fabs(betas[0])>1e-10f)?-b4[2]/betas[0]:0;betas[3]=(fabs(betas[0])>1e-10f)?-b4[3]/betas[0]:0;}
                    else{betas[0]=sqrt(max(b4[0],0.0f));betas[1]=(fabs(betas[0])>1e-10f)?b4[1]/betas[0]:0;betas[2]=(fabs(betas[0])>1e-10f)?b4[2]/betas[0]:0;betas[3]=(fabs(betas[0])>1e-10f)?b4[3]/betas[0]:0;}
                } else if (approx == 2) {
                    float Lt[18],rh[6],b3[3];
                    for(int i=0;i<6;i++){Lt[i*3]=l_r[i*10];Lt[i*3+1]=l_r[i*10+1];Lt[i*3+2]=l_r[i*10+2];rh[i]=rho_r[i];}
                    ls_solve(Lt,6,3,rh,b3);
                    if(b3[0]<0){betas[0]=sqrt(-b3[0]);betas[1]=(b3[2]<0)?sqrt(-b3[2]):0;}
                    else{betas[0]=sqrt(max(b3[0],0.0f));betas[1]=(b3[2]>0)?sqrt(b3[2]):0;}
                    if(b3[1]<0)betas[0]=-betas[0]; betas[2]=0;betas[3]=0;
                } else {
                    float Lt[30],rh[6],bb[5];
                    for(int i=0;i<6;i++){Lt[i*5]=l_r[i*10];Lt[i*5+1]=l_r[i*10+1];Lt[i*5+2]=l_r[i*10+2];Lt[i*5+3]=l_r[i*10+3];Lt[i*5+4]=l_r[i*10+4];rh[i]=rho_r[i];}
                    ls_solve(Lt,6,5,rh,bb);
                    if(bb[0]<0){betas[0]=sqrt(-bb[0]);betas[1]=(bb[2]<0)?sqrt(-bb[2]):0;}
                    else{betas[0]=sqrt(max(bb[0],0.0f));betas[1]=(bb[2]>0)?sqrt(bb[2]):0;}
                    if(bb[1]<0)betas[0]=-betas[0]; betas[2]=(fabs(betas[0])>1e-10f)?bb[3]/betas[0]:0;betas[3]=0;
                }

                for (int gn=0;gn<5;gn++){
                    float A[24],bb[6],dx[4];
                    for(int i=0;i<6;i++){
                        const float*L=&l_r[i*10];
                        A[i*4+0]=2*L[0]*betas[0]+L[1]*betas[1]+L[3]*betas[2]+L[6]*betas[3];
                        A[i*4+1]=L[1]*betas[0]+2*L[2]*betas[1]+L[4]*betas[2]+L[7]*betas[3];
                        A[i*4+2]=L[3]*betas[0]+L[4]*betas[1]+2*L[5]*betas[2]+L[8]*betas[3];
                        A[i*4+3]=L[6]*betas[0]+L[7]*betas[1]+L[8]*betas[2]+2*L[9]*betas[3];
                        bb[i]=rho_r[i]-(L[0]*betas[0]*betas[0]+L[1]*betas[0]*betas[1]+L[2]*betas[1]*betas[1]+L[3]*betas[0]*betas[2]+L[4]*betas[1]*betas[2]+L[5]*betas[2]*betas[2]+L[6]*betas[0]*betas[3]+L[7]*betas[1]*betas[3]+L[8]*betas[2]*betas[3]+L[9]*betas[3]*betas[3]);
                    }
                    qr_solve_6x4(A,bb,dx);
                    for(int i=0;i<4;i++)betas[i]+=dx[i];
                }

                // Compute camera-frame control & 3D points for inliers
                float ccs_r[4][3];
                for(int i=0;i<4;i++)ccs_r[i][0]=ccs_r[i][1]=ccs_r[i][2]=0;
                for(int i=0;i<4;i++){const float*vi=&ut_r[(11-i)*12];for(int j=0;j<4;j++)for(int k=0;k<3;k++)ccs_r[j][k]+=betas[i]*vi[3*j+k];}

                // Centroids for alignment using inlier points
                float pc0[3]={0,0,0},pw0[3]={0,0,0};
                // Check sign using first inlier's camera-frame point
                {
                    int fi = inlier_idx[0];
                    float dxi=pts3d[3*fi]-cws_r[0][0],dyi=pts3d[3*fi+1]-cws_r[0][1],dzi=pts3d[3*fi+2]-cws_r[0][2];
                    float a1_=cc_inv_r[0]*dxi+cc_inv_r[1]*dyi+cc_inv_r[2]*dzi;
                    float a2_=cc_inv_r[3]*dxi+cc_inv_r[4]*dyi+cc_inv_r[5]*dzi;
                    float a3_=cc_inv_r[6]*dxi+cc_inv_r[7]*dyi+cc_inv_r[8]*dzi;
                    float a0_=1.0f-a1_-a2_-a3_;
                    float pz=a0_*ccs_r[0][2]+a1_*ccs_r[1][2]+a2_*ccs_r[2][2]+a3_*ccs_r[3][2];
                    if(pz<0){for(int i=0;i<4;i++)for(int j=0;j<3;j++)ccs_r[i][j]=-ccs_r[i][j];}
                }

                for(int ii=0;ii<np;ii++){
                    int idx=inlier_idx[ii];
                    float dxi=pts3d[3*idx]-cws_r[0][0],dyi=pts3d[3*idx+1]-cws_r[0][1],dzi=pts3d[3*idx+2]-cws_r[0][2];
                    float a1_=cc_inv_r[0]*dxi+cc_inv_r[1]*dyi+cc_inv_r[2]*dzi;
                    float a2_=cc_inv_r[3]*dxi+cc_inv_r[4]*dyi+cc_inv_r[5]*dzi;
                    float a3_=cc_inv_r[6]*dxi+cc_inv_r[7]*dyi+cc_inv_r[8]*dzi;
                    float a0_=1.0f-a1_-a2_-a3_;
                    float pcx=a0_*ccs_r[0][0]+a1_*ccs_r[1][0]+a2_*ccs_r[2][0]+a3_*ccs_r[3][0];
                    float pcy=a0_*ccs_r[0][1]+a1_*ccs_r[1][1]+a2_*ccs_r[2][1]+a3_*ccs_r[3][1];
                    float pcz=a0_*ccs_r[0][2]+a1_*ccs_r[1][2]+a2_*ccs_r[2][2]+a3_*ccs_r[3][2];
                    pc0[0]+=pcx; pc0[1]+=pcy; pc0[2]+=pcz;
                    pw0[0]+=pts3d[3*idx]; pw0[1]+=pts3d[3*idx+1]; pw0[2]+=pts3d[3*idx+2];
                }
                for(int j=0;j<3;j++){pc0[j]/=np;pw0[j]/=np;}

                float ABt_r[9]={};
                for(int ii=0;ii<np;ii++){
                    int idx=inlier_idx[ii];
                    float dxi=pts3d[3*idx]-cws_r[0][0],dyi=pts3d[3*idx+1]-cws_r[0][1],dzi=pts3d[3*idx+2]-cws_r[0][2];
                    float a1_=cc_inv_r[0]*dxi+cc_inv_r[1]*dyi+cc_inv_r[2]*dzi;
                    float a2_=cc_inv_r[3]*dxi+cc_inv_r[4]*dyi+cc_inv_r[5]*dzi;
                    float a3_=cc_inv_r[6]*dxi+cc_inv_r[7]*dyi+cc_inv_r[8]*dzi;
                    float a0_=1.0f-a1_-a2_-a3_;
                    float pcx=a0_*ccs_r[0][0]+a1_*ccs_r[1][0]+a2_*ccs_r[2][0]+a3_*ccs_r[3][0];
                    float pcy=a0_*ccs_r[0][1]+a1_*ccs_r[1][1]+a2_*ccs_r[2][1]+a3_*ccs_r[3][1];
                    float pcz=a0_*ccs_r[0][2]+a1_*ccs_r[1][2]+a2_*ccs_r[2][2]+a3_*ccs_r[3][2];
                    for(int j=0;j<3;j++){
                        float pcd=(j==0?pcx:j==1?pcy:pcz)-pc0[j];
                        ABt_r[j*3+0]+=pcd*(pts3d[3*idx]-pw0[0]);
                        ABt_r[j*3+1]+=pcd*(pts3d[3*idx+1]-pw0[1]);
                        ABt_r[j*3+2]+=pcd*(pts3d[3*idx+2]-pw0[2]);
                    }
                }

                float Ur[9],Sr[3],VtR[9];
                svd3x3(ABt_r,Ur,Sr,VtR);
                float tmpR2[9];
                for(int i=0;i<3;i++)for(int j=0;j<3;j++){tmpR2[i*3+j]=0;for(int k=0;k<3;k++)tmpR2[i*3+j]+=Ur[i*3+k]*VtR[k*3+j];}
                float det2=tmpR2[0]*(tmpR2[4]*tmpR2[8]-tmpR2[5]*tmpR2[7])-tmpR2[1]*(tmpR2[3]*tmpR2[8]-tmpR2[5]*tmpR2[6])+tmpR2[2]*(tmpR2[3]*tmpR2[7]-tmpR2[4]*tmpR2[6]);
                if(det2<0){tmpR2[6]=-tmpR2[6];tmpR2[7]=-tmpR2[7];tmpR2[8]=-tmpR2[8];}
                float tmpt2[3];
                for(int i=0;i<3;i++)tmpt2[i]=pc0[i]-(tmpR2[i*3]*pw0[0]+tmpR2[i*3+1]*pw0[1]+tmpR2[i*3+2]*pw0[2]);

                // Check reprojection error on inliers
                float err=0;
                for(int ii=0;ii<np;ii++){
                    int idx=inlier_idx[ii];
                    float Xc=tmpR2[0]*pts3d[3*idx]+tmpR2[1]*pts3d[3*idx+1]+tmpR2[2]*pts3d[3*idx+2]+tmpt2[0];
                    float Yc=tmpR2[3]*pts3d[3*idx]+tmpR2[4]*pts3d[3*idx+1]+tmpR2[5]*pts3d[3*idx+2]+tmpt2[1];
                    float Zc=tmpR2[6]*pts3d[3*idx]+tmpR2[7]*pts3d[3*idx+1]+tmpR2[8]*pts3d[3*idx+2]+tmpt2[2];
                    if(fabs(Zc)<1e-8f){err+=1e6f;continue;}
                    float ue=cx+fx*Xc/Zc,ve=cy+fy*Yc/Zc;
                    float du=pts2d[2*idx]-ue,dv2=pts2d[2*idx+1]-ve;
                    err+=sqrt(du*du+dv2*dv2);
                }
                err/=np;
                if(err<best_refit_err){best_refit_err=err;for(int i=0;i<9;i++)refit_R[i]=tmpR2[i];for(int i=0;i<3;i++)refit_t[i]=tmpt2[i];}
            }

            // Use refit if it produced a valid result
            if (best_refit_err < 1e19f) {
                for (int i = 0; i < 9; i++) best_R[i] = refit_R[i];
                for (int i = 0; i < 3; i++) best_t[i] = refit_t[i];
            }
        }
    }

    // --- Write final outputs ---
    // R [N,1,3,3]
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            R_out[R_base + r * OUTPUT0_PITCHES[2] + c] = best_R[r * 3 + c];
    // T [N,1,1,3]
    for (int i = 0; i < 3; i++) T_out[T_base + i] = best_t[i];
    pnp_success_out[ok_base] = 1.0f;
}
