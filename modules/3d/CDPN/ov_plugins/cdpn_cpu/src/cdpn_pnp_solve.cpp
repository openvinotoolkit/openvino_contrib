// CdpnPnpSolve - Self-contained EPnP + RANSAC implementation for
// the CDPN E2E OpenVINO pipeline.
//
// Matches OpenCV's cv2.solvePnPRansac(pts3d, pts2d, K, dist_coeffs,
//                                      flags=cv2.SOLVEPNP_EPNP)
// with default parameters:
//   iterationsCount = 100
//   reprojectionError = 8.0
//   confidence = 0.99
//   model_points (minimal sample) = 5
//
// Algorithm outline (solvepnp + epnp + ptsetreg):
//
// RANSAC outer loop (ptsetreg RANSACPointSetRegistrator::run):
//   for each iteration:
//     1. Sample 5 random correspondences (getSubset)
//     2. Run EPnP on the 5-point sample (PnPRansacCallback::runKernel)
//     3. Project ALL 3D points with candidate (R,T) -> 2D
//        (PnPRansacCallback::computeError)
//     4. Count inliers where squared_reproj_error <= threshold^2
//     5. If best so far, update best model and adapt iteration count
//        using RANSACUpdateNumIters(confidence, outlier_ratio, 5, niters)
//   After loop:
//     6. Collect inlier correspondences from best model
//     7. Refit EPnP on ALL inliers to get final (R, T)
//
// EPnP solver (epnp):
//   1. Choose 4 control points (centroid + 3 PCA directions)
//   2. Express each 3D point as barycentric combination of control points
//   3. Build matrix M (2n x 12) from barycentric weights + image coords
//   4. Compute MtM and its SVD (12x12)
//   5. Form L_6x10 linearization and rho (pairwise control-point distances)
//   6. Solve for betas (3 approximations: case N=1,2,3)
//   7. Refine betas with 5 Gauss-Newton iterations
//   8. Recover camera-frame control points, then camera-frame 3D points
//   9. Estimate R,T via 3D-3D alignment (SVD of cross-covariance)
//  10. Pick solution (of 3 beta-approximations) with lowest reprojection error

#include "cdpn_pnp_solve.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace CdpnExtension {

// ===========================================================================
// Tiny linear-algebra helpers (no external dependencies)
// ===========================================================================

static inline double dot3(const double* a, const double* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static inline double dist2(const double* a, const double* b) {
    double dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
    return dx*dx + dy*dy + dz*dz;
}

// --- Jacobi eigenvalue decomposition for symmetric 3x3 ---
// (used in control-point PCA)
static void symmetric_3x3_eigenvalues(const double A[3][3],
                                      double eigenvalues[3],
                                      double eigenvectors[3][3]) {
    // Full Jacobi rotation method for 3x3 symmetric matrix
    double V[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    double M[3][3];
    std::memcpy(M, A, sizeof(M));

    for (int sweep = 0; sweep < 50; ++sweep) {
        // Off-diagonal sum
        double off = std::fabs(M[0][1]) + std::fabs(M[0][2]) + std::fabs(M[1][2]);
        if (off < 1e-15) break;

        for (int p = 0; p < 2; ++p) {
            for (int q = p+1; q < 3; ++q) {
                if (std::fabs(M[p][q]) < 1e-20) continue;
                double tau = (M[q][q] - M[p][p]) / (2.0 * M[p][q]);
                double t;
                if (std::fabs(tau) > 1e15)
                    t = 1.0 / (2.0 * tau);
                else
                    t = ((tau >= 0) ? 1.0 : -1.0) /
                        (std::fabs(tau) + std::sqrt(1.0 + tau*tau));
                double c = 1.0 / std::sqrt(1.0 + t*t);
                double s = t * c;

                double Mpp = M[p][p], Mqq = M[q][q], Mpq = M[p][q];
                M[p][p] = Mpp - t * Mpq;
                M[q][q] = Mqq + t * Mpq;
                M[p][q] = M[q][p] = 0.0;

                for (int r = 0; r < 3; ++r) {
                    if (r == p || r == q) continue;
                    double Mrp = M[r][p], Mrq = M[r][q];
                    M[r][p] = M[p][r] = c * Mrp - s * Mrq;
                    M[r][q] = M[q][r] = s * Mrp + c * Mrq;
                }
                for (int r = 0; r < 3; ++r) {
                    double Vrp = V[r][p], Vrq = V[r][q];
                    V[r][p] = c * Vrp - s * Vrq;
                    V[r][q] = s * Vrp + c * Vrq;
                }
            }
        }
    }

    // Sort by decreasing eigenvalue
    int idx[3] = {0, 1, 2};
    if (M[idx[0]][idx[0]] < M[idx[1]][idx[1]]) std::swap(idx[0], idx[1]);
    if (M[idx[0]][idx[0]] < M[idx[2]][idx[2]]) std::swap(idx[0], idx[2]);
    if (M[idx[1]][idx[1]] < M[idx[2]][idx[2]]) std::swap(idx[1], idx[2]);

    for (int i = 0; i < 3; ++i) {
        eigenvalues[i] = M[idx[i]][idx[i]];
        for (int j = 0; j < 3; ++j)
            eigenvectors[j][i] = V[j][idx[i]];
    }
}

// --- SVD of 3x3 for R estimation (thin SVD using cross-covariance) ---
// Returns U, S, Vt such that A ≈ U * diag(S) * Vt
static void svd_3x3(const double A[3][3],
                     double U[3][3], double S[3], double Vt[3][3]) {
    // Compute AtA
    double AtA[3][3] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                AtA[i][j] += A[k][i] * A[k][j];

    double eigenvalues[3];
    double V[3][3];
    symmetric_3x3_eigenvalues(AtA, eigenvalues, V);

    for (int i = 0; i < 3; ++i)
        S[i] = std::sqrt(std::max(eigenvalues[i], 0.0));

    // Vt = V transposed
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Vt[i][j] = V[j][i];

    // U = A * V * diag(1/S)
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            U[i][j] = 0;
            for (int k = 0; k < 3; ++k)
                U[i][j] += A[i][k] * V[k][j];
            if (S[j] > 1e-12)
                U[i][j] /= S[j];
            else
                U[i][j] = 0;
        }
    }
}

// --- MtM SVD for EPnP (12x12 symmetric matrix) ---
// The last 4 right singular vectors (nullspace).
// Uses Jacobi eigendecomposition on MtM (12x12 symmetric).
static void symmetric_NxN_eigen(const double* A, int N,
                                double* eigenvalues, double* eigenvectors) {
    // eigenvectors stored column-major: eigenvectors[col*N + row]
    // Work on a copy
    std::vector<double> M(N * N);
    std::memcpy(M.data(), A, N * N * sizeof(double));

    // Initialize eigenvectors to identity
    std::memset(eigenvectors, 0, N * N * sizeof(double));
    for (int i = 0; i < N; ++i) eigenvectors[i * N + i] = 1.0;

    auto idx = [N](int r, int c) { return r * N + c; };

    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0;
        for (int p = 0; p < N - 1; ++p)
            for (int q = p + 1; q < N; ++q)
                off += M[idx(p, q)] * M[idx(p, q)];
        if (off < 1e-24) break;

        for (int p = 0; p < N - 1; ++p) {
            for (int q = p + 1; q < N; ++q) {
                if (std::fabs(M[idx(p, q)]) < 1e-20) continue;

                double denom = 2.0 * M[idx(p, q)];
                double tau = (M[idx(q, q)] - M[idx(p, p)]) / denom;
                double t;
                if (std::fabs(tau) > 1e15)
                    t = 1.0 / (2.0 * tau);
                else
                    t = ((tau >= 0) ? 1.0 : -1.0) /
                        (std::fabs(tau) + std::sqrt(1.0 + tau * tau));
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = t * c;

                double Mpp = M[idx(p, p)], Mqq = M[idx(q, q)];
                M[idx(p, p)] = Mpp - t * M[idx(p, q)];
                M[idx(q, q)] = Mqq + t * M[idx(p, q)];
                M[idx(p, q)] = M[idx(q, p)] = 0.0;

                for (int r = 0; r < N; ++r) {
                    if (r == p || r == q) continue;
                    double Mrp = M[idx(r, p)], Mrq = M[idx(r, q)];
                    M[idx(r, p)] = M[idx(p, r)] = c * Mrp - s * Mrq;
                    M[idx(r, q)] = M[idx(q, r)] = s * Mrp + c * Mrq;
                }
                for (int r = 0; r < N; ++r) {
                    double Vrp = eigenvectors[p * N + r];
                    double Vrq = eigenvectors[q * N + r];
                    eigenvectors[p * N + r] = c * Vrp - s * Vrq;
                    eigenvectors[q * N + r] = s * Vrp + c * Vrq;
                }
            }
        }
    }

    for (int i = 0; i < N; ++i)
        eigenvalues[i] = M[idx(i, i)];
}

// --- QR solve (Householder) for 6x4 systems ---
static void qr_solve_6x4(double A[6][4], double b[6], double x[4]) {
    static constexpr int NR = 6, NC = 4;
    for (int k = 0; k < NC; ++k) {
        // Compute norm of column k below diagonal
        double norm = 0;
        for (int i = k; i < NR; ++i) norm += A[i][k] * A[i][k];
        norm = std::sqrt(norm);
        if (norm < 1e-15) { x[k] = 0; continue; }

        double sign = (A[k][k] >= 0) ? 1.0 : -1.0;
        double alpha = sign * norm;
        A[k][k] += alpha;
        double scale = 1.0 / (alpha * A[k][k]);

        // Apply Householder to remaining columns
        for (int j = k + 1; j < NC; ++j) {
            double dot = 0;
            for (int i = k; i < NR; ++i) dot += A[i][k] * A[i][j];
            dot *= scale;
            for (int i = k; i < NR; ++i) A[i][j] -= dot * A[i][k];
        }
        // Apply to b
        {
            double dot = 0;
            for (int i = k; i < NR; ++i) dot += A[i][k] * b[i];
            dot *= scale;
            for (int i = k; i < NR; ++i) b[i] -= dot * A[i][k];
        }
        A[k][k] = -alpha;
    }
    // Back-substitution
    for (int i = NC - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < NC; ++j) s -= A[i][j] * x[j];
        x[i] = (std::fabs(A[i][i]) > 1e-15) ? s / A[i][i] : 0.0;
    }
}

// --- Least-squares solve using normal equations (for 6xN with N<=10) ---
static void least_squares_solve(const double* L, int rows, int cols,
                                const double* rhs, double* x) {
    // AtA x = At b
    std::vector<double> AtA(cols * cols, 0.0);
    std::vector<double> Atb(cols, 0.0);
    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < rows; ++k)
                AtA[i * cols + j] += L[k * cols + i] * L[k * cols + j];
        }
        for (int k = 0; k < rows; ++k)
            Atb[i] += L[k * cols + i] * rhs[k];
    }
    // Simple Cholesky or just invert small matrix via Gauss elimination
    // For small systems, use Gaussian elimination with partial pivoting
    int N = cols;
    std::vector<double> aug(N * (N + 1));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) aug[i * (N+1) + j] = AtA[i * N + j];
        aug[i * (N+1) + N] = Atb[i];
    }
    for (int k = 0; k < N; ++k) {
        int pivot = k;
        for (int i = k + 1; i < N; ++i)
            if (std::fabs(aug[i*(N+1)+k]) > std::fabs(aug[pivot*(N+1)+k]))
                pivot = i;
        if (pivot != k)
            for (int j = k; j <= N; ++j)
                std::swap(aug[k*(N+1)+j], aug[pivot*(N+1)+j]);
        if (std::fabs(aug[k*(N+1)+k]) < 1e-15) { x[k] = 0; continue; }
        for (int i = k + 1; i < N; ++i) {
            double f = aug[i*(N+1)+k] / aug[k*(N+1)+k];
            for (int j = k; j <= N; ++j)
                aug[i*(N+1)+j] -= f * aug[k*(N+1)+j];
        }
    }
    for (int i = N - 1; i >= 0; --i) {
        x[i] = aug[i*(N+1)+N];
        for (int j = i + 1; j < N; ++j)
            x[i] -= aug[i*(N+1)+j] * x[j];
        if (std::fabs(aug[i*(N+1)+i]) > 1e-15)
            x[i] /= aug[i*(N+1)+i];
        else
            x[i] = 0;
    }
}


// ===========================================================================
// EPnP solver
// ===========================================================================

struct EPnPSolver {
    int n_pts;
    double fu, fv, uc, vc;

    std::vector<double> pws;  // 3D points (world)  [n*3]
    std::vector<double> us;   // 2D points (image)  [n*2]
    std::vector<double> alphas; // barycentric coords [n*4]
    std::vector<double> pcs;  // 3D points (camera) [n*3]

    double cws[4][3];  // control points (world)
    double ccs[4][3];  // control points (camera)

    EPnPSolver(const double* pts3d, const double* pts2d, int npts,
               double fx, double fy, double cx, double cy)
        : n_pts(npts), fu(fx), fv(fy), uc(cx), vc(cy),
          pws(3 * npts), us(2 * npts), alphas(4 * npts), pcs(3 * npts)
    {
        for (int i = 0; i < npts; ++i) {
            pws[3*i]   = pts3d[3*i];
            pws[3*i+1] = pts3d[3*i+1];
            pws[3*i+2] = pts3d[3*i+2];

            us[2*i]   = pts2d[2*i];
            us[2*i+1] = pts2d[2*i+1];
        }
    }

    void choose_control_points() {
        // C0 = centroid
        cws[0][0] = cws[0][1] = cws[0][2] = 0;
        for (int i = 0; i < n_pts; ++i) {
            cws[0][0] += pws[3*i];
            cws[0][1] += pws[3*i+1];
            cws[0][2] += pws[3*i+2];
        }
        cws[0][0] /= n_pts;
        cws[0][1] /= n_pts;
        cws[0][2] /= n_pts;

        // PCA on centred points
        // Compute 3x3 covariance PW0t * PW0
        double cov[3][3] = {};
        for (int i = 0; i < n_pts; ++i) {
            double dx = pws[3*i]   - cws[0][0];
            double dy = pws[3*i+1] - cws[0][1];
            double dz = pws[3*i+2] - cws[0][2];
            cov[0][0] += dx*dx; cov[0][1] += dx*dy; cov[0][2] += dx*dz;
            cov[1][0] += dy*dx; cov[1][1] += dy*dy; cov[1][2] += dy*dz;
            cov[2][0] += dz*dx; cov[2][1] += dz*dy; cov[2][2] += dz*dz;
        }

        double eigenvalues[3];
        double eigenvectors[3][3];
        symmetric_3x3_eigenvalues(cov, eigenvalues, eigenvectors);

        // C1, C2, C3 at centroid ± scaled eigenvectors
        for (int i = 1; i < 4; ++i) {
            double k = std::sqrt(std::max(eigenvalues[i-1], 0.0) / n_pts);
            for (int j = 0; j < 3; ++j)
                cws[i][j] = cws[0][j] + k * eigenvectors[j][i-1];
        }
    }

    void compute_barycentric_coordinates() {
        // CC = [C1-C0 | C2-C0 | C3-C0]  (3x3)
        double cc[3][3], cc_inv[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                cc[i][j] = cws[j+1][i] - cws[0][i];

        // Invert 3x3 using adjugate
        double det =
            cc[0][0]*(cc[1][1]*cc[2][2] - cc[1][2]*cc[2][1]) -
            cc[0][1]*(cc[1][0]*cc[2][2] - cc[1][2]*cc[2][0]) +
            cc[0][2]*(cc[1][0]*cc[2][1] - cc[1][1]*cc[2][0]);

        if (std::fabs(det) < 1e-15) {
            // Degenerate - fill with equal weights
            for (int i = 0; i < n_pts; ++i) {
                alphas[4*i]   = 0.25;
                alphas[4*i+1] = 0.25;
                alphas[4*i+2] = 0.25;
                alphas[4*i+3] = 0.25;
            }
            return;
        }
        double inv_det = 1.0 / det;
        cc_inv[0][0] = (cc[1][1]*cc[2][2] - cc[1][2]*cc[2][1]) * inv_det;
        cc_inv[0][1] = (cc[0][2]*cc[2][1] - cc[0][1]*cc[2][2]) * inv_det;
        cc_inv[0][2] = (cc[0][1]*cc[1][2] - cc[0][2]*cc[1][1]) * inv_det;
        cc_inv[1][0] = (cc[1][2]*cc[2][0] - cc[1][0]*cc[2][2]) * inv_det;
        cc_inv[1][1] = (cc[0][0]*cc[2][2] - cc[0][2]*cc[2][0]) * inv_det;
        cc_inv[1][2] = (cc[0][2]*cc[1][0] - cc[0][0]*cc[1][2]) * inv_det;
        cc_inv[2][0] = (cc[1][0]*cc[2][1] - cc[1][1]*cc[2][0]) * inv_det;
        cc_inv[2][1] = (cc[0][1]*cc[2][0] - cc[0][0]*cc[2][1]) * inv_det;
        cc_inv[2][2] = (cc[0][0]*cc[1][1] - cc[0][1]*cc[1][0]) * inv_det;

        for (int i = 0; i < n_pts; ++i) {
            double dx = pws[3*i]   - cws[0][0];
            double dy = pws[3*i+1] - cws[0][1];
            double dz = pws[3*i+2] - cws[0][2];
            alphas[4*i+1] = cc_inv[0][0]*dx + cc_inv[0][1]*dy + cc_inv[0][2]*dz;
            alphas[4*i+2] = cc_inv[1][0]*dx + cc_inv[1][1]*dy + cc_inv[1][2]*dz;
            alphas[4*i+3] = cc_inv[2][0]*dx + cc_inv[2][1]*dy + cc_inv[2][2]*dz;
            alphas[4*i]   = 1.0 - alphas[4*i+1] - alphas[4*i+2] - alphas[4*i+3];
        }
    }

    void fill_M_row(double* M_row, const double* a, double u, double v) {
        // Two rows of M for one correspondence
        double* M1 = M_row;
        double* M2 = M_row + 12;
        for (int i = 0; i < 4; ++i) {
            M1[3*i]     = a[i] * fu;
            M1[3*i + 1] = 0.0;
            M1[3*i + 2] = a[i] * (uc - u);
            M2[3*i]     = 0.0;
            M2[3*i + 1] = a[i] * fv;
            M2[3*i + 2] = a[i] * (vc - v);
        }
    }

    void compute_ccs(const double* betas, const double* ut) {
        for (int i = 0; i < 4; ++i)
            ccs[i][0] = ccs[i][1] = ccs[i][2] = 0;
        for (int i = 0; i < 4; ++i) {
            const double* v = ut + 12 * (11 - i);
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 3; ++k)
                    ccs[j][k] += betas[i] * v[3*j + k];
        }
    }

    void compute_pcs() {
        for (int i = 0; i < n_pts; ++i) {
            for (int j = 0; j < 3; ++j)
                pcs[3*i + j] = alphas[4*i]*ccs[0][j] + alphas[4*i+1]*ccs[1][j]
                              + alphas[4*i+2]*ccs[2][j] + alphas[4*i+3]*ccs[3][j];
        }
    }

    void solve_for_sign() {
        if (pcs[2] < 0.0) {
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 3; ++j) ccs[i][j] = -ccs[i][j];
            for (int i = 0; i < n_pts; ++i)
                for (int j = 0; j < 3; ++j) pcs[3*i+j] = -pcs[3*i+j];
        }
    }

    void estimate_R_and_t(double R[3][3], double t[3]) {
        // Centroids
        double pc0[3] = {}, pw0[3] = {};
        for (int i = 0; i < n_pts; ++i) {
            for (int j = 0; j < 3; ++j) {
                pc0[j] += pcs[3*i+j];
                pw0[j] += pws[3*i+j];
            }
        }
        for (int j = 0; j < 3; ++j) {
            pc0[j] /= n_pts;
            pw0[j] /= n_pts;
        }

        // Cross-covariance ABt
        double ABt[3][3] = {};
        for (int i = 0; i < n_pts; ++i) {
            for (int j = 0; j < 3; ++j) {
                double pc_d = pcs[3*i+j] - pc0[j];
                ABt[j][0] += pc_d * (pws[3*i]   - pw0[0]);
                ABt[j][1] += pc_d * (pws[3*i+1] - pw0[1]);
                ABt[j][2] += pc_d * (pws[3*i+2] - pw0[2]);
            }
        }

        // SVD of ABt
        double U[3][3], S[3], Vt[3][3];
        svd_3x3(ABt, U, S, Vt);

        // R = U * Vt
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                R[i][j] = 0;
                for (int k = 0; k < 3; ++k)
                    R[i][j] += U[i][k] * Vt[k][j];
            }

        // Ensure proper rotation (det = +1)
        double det =
            R[0][0]*(R[1][1]*R[2][2] - R[1][2]*R[2][1]) -
            R[0][1]*(R[1][0]*R[2][2] - R[1][2]*R[2][0]) +
            R[0][2]*(R[1][0]*R[2][1] - R[1][1]*R[2][0]);
        if (det < 0) {
            R[2][0] = -R[2][0];
            R[2][1] = -R[2][1];
            R[2][2] = -R[2][2];
        }

        t[0] = pc0[0] - (R[0][0]*pw0[0] + R[0][1]*pw0[1] + R[0][2]*pw0[2]);
        t[1] = pc0[1] - (R[1][0]*pw0[0] + R[1][1]*pw0[1] + R[1][2]*pw0[2]);
        t[2] = pc0[2] - (R[2][0]*pw0[0] + R[2][1]*pw0[1] + R[2][2]*pw0[2]);
    }

    double compute_R_and_t(const double* ut, const double* betas,
                           double R[3][3], double t[3]) {
        compute_ccs(betas, ut);
        compute_pcs();
        solve_for_sign();
        estimate_R_and_t(R, t);
        return reprojection_error(R, t);
    }

    double reprojection_error(const double R[3][3], const double t[3]) {
        double sum = 0;
        for (int i = 0; i < n_pts; ++i) {
            double Xc = R[0][0]*pws[3*i] + R[0][1]*pws[3*i+1] + R[0][2]*pws[3*i+2] + t[0];
            double Yc = R[1][0]*pws[3*i] + R[1][1]*pws[3*i+1] + R[1][2]*pws[3*i+2] + t[1];
            double Zc = R[2][0]*pws[3*i] + R[2][1]*pws[3*i+1] + R[2][2]*pws[3*i+2] + t[2];
            if (std::fabs(Zc) < 1e-10) { sum += 1e6; continue; }
            double inv_Zc = 1.0 / Zc;
            double ue = uc + fu * Xc * inv_Zc;
            double ve = vc + fv * Yc * inv_Zc;
            double du = us[2*i] - ue;
            double dv = us[2*i+1] - ve;
            sum += std::sqrt(du*du + dv*dv);
        }
        return sum / n_pts;
    }

    void compute_L_6x10(const double* ut, double l_6x10[6][10]) {
        const double* v[4];
        v[0] = ut + 12 * 11;
        v[1] = ut + 12 * 10;
        v[2] = ut + 12 * 9;
        v[3] = ut + 12 * 8;

        double dv[4][6][3];
        for (int i = 0; i < 4; ++i) {
            int a = 0, b = 1;
            for (int j = 0; j < 6; ++j) {
                dv[i][j][0] = v[i][3*a]   - v[i][3*b];
                dv[i][j][1] = v[i][3*a+1] - v[i][3*b+1];
                dv[i][j][2] = v[i][3*a+2] - v[i][3*b+2];
                b++;
                if (b > 3) { a++; b = a + 1; }
            }
        }

        for (int i = 0; i < 6; ++i) {
            l_6x10[i][0] =        dot3(dv[0][i], dv[0][i]);
            l_6x10[i][1] = 2.0 *  dot3(dv[0][i], dv[1][i]);
            l_6x10[i][2] =        dot3(dv[1][i], dv[1][i]);
            l_6x10[i][3] = 2.0 *  dot3(dv[0][i], dv[2][i]);
            l_6x10[i][4] = 2.0 *  dot3(dv[1][i], dv[2][i]);
            l_6x10[i][5] =        dot3(dv[2][i], dv[2][i]);
            l_6x10[i][6] = 2.0 *  dot3(dv[0][i], dv[3][i]);
            l_6x10[i][7] = 2.0 *  dot3(dv[1][i], dv[3][i]);
            l_6x10[i][8] = 2.0 *  dot3(dv[2][i], dv[3][i]);
            l_6x10[i][9] =        dot3(dv[3][i], dv[3][i]);
        }
    }

    void compute_rho(double rho[6]) {
        rho[0] = dist2(cws[0], cws[1]);
        rho[1] = dist2(cws[0], cws[2]);
        rho[2] = dist2(cws[0], cws[3]);
        rho[3] = dist2(cws[1], cws[2]);
        rho[4] = dist2(cws[1], cws[3]);
        rho[5] = dist2(cws[2], cws[3]);
    }

    void find_betas_approx_1(const double l_6x10[6][10], const double rho[6],
                             double betas[4]) {
        // betas10 = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
        // approx_1 = [B11 B12 B13 B14]
        double L64[6][4], b4[4];
        for (int i = 0; i < 6; ++i) {
            L64[i][0] = l_6x10[i][0];
            L64[i][1] = l_6x10[i][1];
            L64[i][2] = l_6x10[i][3];
            L64[i][3] = l_6x10[i][6];
        }
        double L_flat[24], rho_copy[6];
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 4; ++j) L_flat[i*4+j] = L64[i][j];
            rho_copy[i] = rho[i];
        }
        least_squares_solve(L_flat, 6, 4, rho_copy, b4);

        if (b4[0] < 0) {
            betas[0] = std::sqrt(-b4[0]);
            betas[1] = -b4[1] / betas[0];
            betas[2] = -b4[2] / betas[0];
            betas[3] = -b4[3] / betas[0];
        } else {
            betas[0] = std::sqrt(std::max(b4[0], 0.0));
            if (betas[0] > 1e-15) {
                betas[1] = b4[1] / betas[0];
                betas[2] = b4[2] / betas[0];
                betas[3] = b4[3] / betas[0];
            } else {
                betas[0] = betas[1] = betas[2] = betas[3] = 0;
            }
        }
    }

    void find_betas_approx_2(const double l_6x10[6][10], const double rho[6],
                             double betas[4]) {
        // approx_2 = [B11 B12 B22]
        double L63[6][3], b3[3];
        for (int i = 0; i < 6; ++i) {
            L63[i][0] = l_6x10[i][0];
            L63[i][1] = l_6x10[i][1];
            L63[i][2] = l_6x10[i][2];
        }
        double L_flat[18], rho_copy[6];
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 3; ++j) L_flat[i*3+j] = L63[i][j];
            rho_copy[i] = rho[i];
        }
        least_squares_solve(L_flat, 6, 3, rho_copy, b3);

        if (b3[0] < 0) {
            betas[0] = std::sqrt(-b3[0]);
            betas[1] = (b3[2] < 0) ? std::sqrt(-b3[2]) : 0.0;
        } else {
            betas[0] = std::sqrt(std::max(b3[0], 0.0));
            betas[1] = (b3[2] > 0) ? std::sqrt(b3[2]) : 0.0;
        }
        if (b3[1] < 0) betas[0] = -betas[0];
        betas[2] = 0;
        betas[3] = 0;
    }

    void find_betas_approx_3(const double l_6x10[6][10], const double rho[6],
                             double betas[4]) {
        // approx_3 = [B11 B12 B22 B13 B23]
        double L65[6][5], b5[5];
        for (int i = 0; i < 6; ++i) {
            L65[i][0] = l_6x10[i][0];
            L65[i][1] = l_6x10[i][1];
            L65[i][2] = l_6x10[i][2];
            L65[i][3] = l_6x10[i][3];
            L65[i][4] = l_6x10[i][4];
        }
        double L_flat[30], rho_copy[6];
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 5; ++j) L_flat[i*5+j] = L65[i][j];
            rho_copy[i] = rho[i];
        }
        least_squares_solve(L_flat, 6, 5, rho_copy, b5);

        if (b5[0] < 0) {
            betas[0] = std::sqrt(-b5[0]);
            betas[1] = (b5[2] < 0) ? std::sqrt(-b5[2]) : 0.0;
        } else {
            betas[0] = std::sqrt(std::max(b5[0], 0.0));
            betas[1] = (b5[2] > 0) ? std::sqrt(b5[2]) : 0.0;
        }
        if (b5[1] < 0) betas[0] = -betas[0];
        betas[2] = (std::fabs(betas[0]) > 1e-15) ? b5[3] / betas[0] : 0.0;
        betas[3] = 0;
    }

    void gauss_newton(const double l_6x10[6][10], const double rho[6],
                      double betas[4]) {
        for (int iter = 0; iter < 5; ++iter) {
            double A[6][4], b[6], x[4];
            for (int i = 0; i < 6; ++i) {
                const double* L = l_6x10[i];
                A[i][0] = 2*L[0]*betas[0] + L[1]*betas[1] + L[3]*betas[2] + L[6]*betas[3];
                A[i][1] = L[1]*betas[0] + 2*L[2]*betas[1] + L[4]*betas[2] + L[7]*betas[3];
                A[i][2] = L[3]*betas[0] + L[4]*betas[1] + 2*L[5]*betas[2] + L[8]*betas[3];
                A[i][3] = L[6]*betas[0] + L[7]*betas[1] + L[8]*betas[2] + 2*L[9]*betas[3];

                b[i] = rho[i] - (
                    L[0]*betas[0]*betas[0] +
                    L[1]*betas[0]*betas[1] +
                    L[2]*betas[1]*betas[1] +
                    L[3]*betas[0]*betas[2] +
                    L[4]*betas[1]*betas[2] +
                    L[5]*betas[2]*betas[2] +
                    L[6]*betas[0]*betas[3] +
                    L[7]*betas[1]*betas[3] +
                    L[8]*betas[2]*betas[3] +
                    L[9]*betas[3]*betas[3]);
            }
            qr_solve_6x4(A, b, x);
            for (int i = 0; i < 4; ++i) betas[i] += x[i];
        }
    }

    // Main EPnP compute_pose
    bool compute_pose(double R_out[3][3], double t_out[3]) {
        if (n_pts < 4) return false;

        choose_control_points();
        compute_barycentric_coordinates();

        // Build M matrix [2n x 12]
        int M_rows = 2 * n_pts;
        std::vector<double> M(M_rows * 12, 0.0);
        for (int i = 0; i < n_pts; ++i)
            fill_M_row(&M[2*i*12], &alphas[4*i], us[2*i], us[2*i+1]);

        // MtM [12 x 12]
        double mtm[144] = {};
        for (int i = 0; i < 12; ++i)
            for (int j = i; j < 12; ++j) {
                double s = 0;
                for (int k = 0; k < M_rows; ++k)
                    s += M[k*12+i] * M[k*12+j];
                mtm[i*12+j] = s;
                mtm[j*12+i] = s;
            }

        // Eigendecomposition of MtM
        double eigenvalues[12];
        double eigenvectors[144];  // column major: col[i] = eigenvectors[i*12 .. i*12+11]
        symmetric_NxN_eigen(mtm, 12, eigenvalues, eigenvectors);

        // Sort eigenvalues ascending (we need smallest 4 - the nullspace)
        int idx[12];
        std::iota(idx, idx + 12, 0);
        std::sort(idx, idx + 12, [&](int a, int b) {
            return eigenvalues[a] < eigenvalues[b];
        });

        // ut: eigenvectors as rows, sorted by ascending eigenvalue
        // ut[i*12 + j] = eigenvectors[idx[i]*12 + j] (row i, col j)
        // The OpenCV convention: ut rows ordered so that
        // indices 11,10,9,8 correspond to the 4 smallest eigenvalues.
        // In OpenCV's SVD decomposition: ut is ordered descending,
        // so ut[11*12..] is the smallest, ut[10*12..] is next, etc.
        double ut[144];
        for (int i = 0; i < 12; ++i) {
            // Map: row (11-i) of ut <- eigenvector with i-th smallest eigenvalue
            int src = idx[i];
            for (int j = 0; j < 12; ++j)
                ut[(11-i)*12 + j] = eigenvectors[src*12 + j];
        }

        double l_6x10[6][10], rho[6];
        compute_L_6x10(ut, l_6x10);
        compute_rho(rho);

        double Betas[4][4], rep_errors[4];
        double Rs[4][3][3], ts[4][3];

        find_betas_approx_1(l_6x10, rho, Betas[1]);
        gauss_newton(l_6x10, rho, Betas[1]);
        rep_errors[1] = compute_R_and_t(ut, Betas[1], Rs[1], ts[1]);

        find_betas_approx_2(l_6x10, rho, Betas[2]);
        gauss_newton(l_6x10, rho, Betas[2]);
        rep_errors[2] = compute_R_and_t(ut, Betas[2], Rs[2], ts[2]);

        find_betas_approx_3(l_6x10, rho, Betas[3]);
        gauss_newton(l_6x10, rho, Betas[3]);
        rep_errors[3] = compute_R_and_t(ut, Betas[3], Rs[3], ts[3]);

        int best = 1;
        if (rep_errors[2] < rep_errors[best]) best = 2;
        if (rep_errors[3] < rep_errors[best]) best = 3;

        std::memcpy(R_out, Rs[best], 9 * sizeof(double));
        std::memcpy(t_out, ts[best], 3 * sizeof(double));
        return true;
    }
};


// ===========================================================================
// RANSAC iteration count update
// ===========================================================================

static int ransac_update_num_iters(double confidence, double outlier_ratio,
                                   int model_points, int max_iters) {
    double p = std::max(std::min(confidence, 1.0), 0.0);
    double ep = std::max(std::min(outlier_ratio, 1.0), 0.0);

    double num = std::log(std::max(1.0 - p, 1e-15));
    double denom = std::log(1.0 - std::pow(1.0 - ep, model_points));
    if (denom >= 0) return max_iters;
    int result = static_cast<int>(std::ceil(num / denom));
    return std::min(result, max_iters);
}


// ===========================================================================
// CdpnPnpSolve OpenVINO Op implementation
// ===========================================================================

CdpnPnpSolve::CdpnPnpSolve(const ov::Output<ov::Node>& denorm_coords,
                             const ov::Output<ov::Node>& confidence,
                             const ov::Output<ov::Node>& obj_extents,
                             const ov::Output<ov::Node>& crop_meta,
                             const ov::Output<ov::Node>& cam_K,
                             float mask_threshold,
                             int out_res,
                             int max_iterations,
                             float reproj_threshold)
    : Op({denorm_coords, confidence, obj_extents, crop_meta, cam_K}),
      m_mask_threshold(mask_threshold),
      m_out_res(out_res),
      m_max_iterations(max_iterations),
      m_reproj_threshold(reproj_threshold) {
    constructor_validate_and_infer_types();
}

void CdpnPnpSolve::validate_and_infer_types() {
    set_output_type(0, ov::element::f32, ov::PartialShape({1, 1, 3, 3}));  // R
    set_output_type(1, ov::element::f32, ov::PartialShape({1, 1, 1, 3}));  // T
    set_output_type(2, ov::element::f32, ov::PartialShape({1, 1, 1, 1}));  // num_corres
    set_output_type(3, ov::element::f32, ov::PartialShape({1, 1, 1, 1}));  // pnp_success
}

std::shared_ptr<ov::Node> CdpnPnpSolve::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 5, "CdpnPnpSolve expects 5 inputs");
    return std::make_shared<CdpnPnpSolve>(
        new_args[0], new_args[1], new_args[2], new_args[3], new_args[4],
        m_mask_threshold, m_out_res, m_max_iterations, m_reproj_threshold);
}

bool CdpnPnpSolve::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("mask_threshold", m_mask_threshold);
    visitor.on_attribute("out_res", m_out_res);
    visitor.on_attribute("max_iterations", m_max_iterations);
    visitor.on_attribute("reproj_threshold", m_reproj_threshold);
    return true;
}

bool CdpnPnpSolve::evaluate(ov::TensorVector& outputs,
                             const ov::TensorVector& inputs) const {
    // --- Read inputs ---
    const float* denorm = inputs[0].data<float>();   // [1, 3, 64, 64]
    const float* conf   = inputs[1].data<float>();   // [1, 1, 64, 64]
    const float* ext    = inputs[2].data<float>();   // [1, 1, 1, 3]
    const float* meta   = inputs[3].data<float>();   // [1, 1, 1, 5]
    const float* camK   = inputs[4].data<float>();   // [1, 1, 1, 4]

    // --- Set output shapes ---
    outputs[0].set_shape({1, 1, 3, 3});
    outputs[1].set_shape({1, 1, 1, 3});
    outputs[2].set_shape({1, 1, 1, 1});
    outputs[3].set_shape({1, 1, 1, 1});

    float* R_out = outputs[0].data<float>();
    float* T_out = outputs[1].data<float>();
    float* nc_out = outputs[2].data<float>();
    float* ok_out = outputs[3].data<float>();

    // Default output: identity R, zero T, failure
    for (int i = 0; i < 9; ++i) R_out[i] = (i % 4 == 0) ? 1.0f : 0.0f;
    std::memset(T_out, 0, 3 * sizeof(float));
    nc_out[0] = 0.0f;
    ok_out[0] = 0.0f;

    const int RES = m_out_res;
    const int spatial = RES * RES;
    const float mask_thr = m_mask_threshold;

    // Camera intrinsics
    const double fx = camK[0], fy = camK[1], cx = camK[2], cy = camK[3];

    // Crop metadata
    const double s       = meta[2];
    const double w_begin = meta[3];
    const double h_begin = meta[4];
    const double w_unit  = s / RES;
    const double h_unit  = s / RES;

    // Object extents
    const double ext_x = ext[0], ext_y = ext[1], ext_z = ext[2];
    const double min_x = 0.001 * ext_x;
    const double min_y = 0.001 * ext_y;
    const double min_z = 0.001 * ext_z;

    // --- Step 1: Build 2D-3D correspondences ---
    // denorm layout: [1, 3, 64, 64] -> channel c at offset c*spatial + row*RES + col
    std::vector<double> pts3d, pts2d;
    pts3d.reserve(spatial * 3);
    pts2d.reserve(spatial * 2);

    for (int row = 0; row < RES; ++row) {
        for (int col = 0; col < RES; ++col) {
            int idx = row * RES + col;
            if (conf[idx] < mask_thr) continue;

            double x3 = denorm[0 * spatial + idx];
            double y3 = denorm[1 * spatial + idx];
            double z3 = denorm[2 * spatial + idx];

            if (std::fabs(x3) < min_x && std::fabs(y3) < min_y &&
                std::fabs(z3) < min_z)
                continue;

            pts3d.push_back(x3);
            pts3d.push_back(y3);
            pts3d.push_back(z3);
            // 2D: col -> x (horizontal), row -> y (vertical)
            pts2d.push_back(w_begin + col * w_unit);
            pts2d.push_back(h_begin + row * h_unit);
        }
    }

    int n_pts = static_cast<int>(pts3d.size() / 3);
    nc_out[0] = static_cast<float>(n_pts);

    if (n_pts < 5) return true;  // Not enough points

    // --- Step 2: RANSAC + EPnP ---
    // Parameters matching OpenCV defaults
    const int MODEL_POINTS = 5;
    const double REPROJ_THR_SQ = static_cast<double>(m_reproj_threshold) *
                                  static_cast<double>(m_reproj_threshold);
    const double CONFIDENCE = 0.99;
    int niters = std::max(m_max_iterations, 1);

    // Best model tracking
    int max_good_count = 0;
    std::vector<uint8_t> best_mask(n_pts, 0);
    double best_R[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    double best_t[3] = {};

    // RNG (fixed seed for reproducibility within one call)
    std::mt19937 rng(12345);

    for (int iter = 0; iter < niters; ++iter) {
        // Sample MODEL_POINTS distinct indices
        int sample[MODEL_POINTS];
        for (int i = 0; i < MODEL_POINTS; ++i) {
            bool unique;
            do {
                sample[i] = static_cast<int>(
                    rng() % static_cast<uint32_t>(n_pts));
                unique = true;
                for (int j = 0; j < i; ++j)
                    if (sample[j] == sample[i]) { unique = false; break; }
            } while (!unique);
        }

        // Extract sample points
        double sample_3d[MODEL_POINTS * 3], sample_2d[MODEL_POINTS * 2];
        for (int i = 0; i < MODEL_POINTS; ++i) {
            sample_3d[3*i]   = pts3d[3*sample[i]];
            sample_3d[3*i+1] = pts3d[3*sample[i]+1];
            sample_3d[3*i+2] = pts3d[3*sample[i]+2];
            sample_2d[2*i]   = pts2d[2*sample[i]];
            sample_2d[2*i+1] = pts2d[2*sample[i]+1];
        }

        // Run EPnP on sample
        EPnPSolver solver(sample_3d, sample_2d, MODEL_POINTS, fx, fy, cx, cy);
        double cand_R[3][3], cand_t[3];
        if (!solver.compute_pose(cand_R, cand_t)) continue;

        // Count inliers: project ALL points, compute squared reprojection error
        int good_count = 0;
        std::vector<uint8_t> mask(n_pts, 0);

        for (int i = 0; i < n_pts; ++i) {
            double Xc = cand_R[0][0]*pts3d[3*i] + cand_R[0][1]*pts3d[3*i+1] +
                        cand_R[0][2]*pts3d[3*i+2] + cand_t[0];
            double Yc = cand_R[1][0]*pts3d[3*i] + cand_R[1][1]*pts3d[3*i+1] +
                        cand_R[1][2]*pts3d[3*i+2] + cand_t[1];
            double Zc = cand_R[2][0]*pts3d[3*i] + cand_R[2][1]*pts3d[3*i+1] +
                        cand_R[2][2]*pts3d[3*i+2] + cand_t[2];
            if (std::fabs(Zc) < 1e-10) continue;
            double inv_Zc = 1.0 / Zc;
            double proj_x = cx + fx * Xc * inv_Zc;
            double proj_y = cy + fy * Yc * inv_Zc;
            double dx = pts2d[2*i]   - proj_x;
            double dy = pts2d[2*i+1] - proj_y;
            double err_sq = dx*dx + dy*dy;

            if (err_sq <= REPROJ_THR_SQ) {
                mask[i] = 1;
                good_count++;
            }
        }

        if (good_count > std::max(max_good_count, MODEL_POINTS - 1)) {
            best_mask = mask;
            std::memcpy(best_R, cand_R, sizeof(best_R));
            std::memcpy(best_t, cand_t, sizeof(best_t));
            max_good_count = good_count;

            // Adaptive iteration count
            double outlier_ratio = static_cast<double>(n_pts - good_count) / n_pts;
            niters = ransac_update_num_iters(
                CONFIDENCE, outlier_ratio, MODEL_POINTS, niters);
        }
    }

    if (max_good_count <= 0) return true;  // RANSAC failed

    // --- Step 3: Refit EPnP on all inliers ---
    std::vector<double> inlier_3d, inlier_2d;
    inlier_3d.reserve(max_good_count * 3);
    inlier_2d.reserve(max_good_count * 2);
    for (int i = 0; i < n_pts; ++i) {
        if (!best_mask[i]) continue;
        inlier_3d.push_back(pts3d[3*i]);
        inlier_3d.push_back(pts3d[3*i+1]);
        inlier_3d.push_back(pts3d[3*i+2]);
        inlier_2d.push_back(pts2d[2*i]);
        inlier_2d.push_back(pts2d[2*i+1]);
    }

    int n_inliers = static_cast<int>(inlier_3d.size() / 3);
    if (n_inliers >= 5) {
        EPnPSolver refit(inlier_3d.data(), inlier_2d.data(), n_inliers,
                         fx, fy, cx, cy);
        double final_R[3][3], final_t[3];
        if (refit.compute_pose(final_R, final_t)) {
            std::memcpy(best_R, final_R, sizeof(best_R));
            std::memcpy(best_t, final_t, sizeof(best_t));
        }
    }

    // --- Step 4: Write outputs ---
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R_out[i*3+j] = static_cast<float>(best_R[i][j]);

    T_out[0] = static_cast<float>(best_t[0]);
    T_out[1] = static_cast<float>(best_t[1]);
    T_out[2] = static_cast<float>(best_t[2]);

    ok_out[0] = 1.0f;

    return true;
}

bool CdpnPnpSolve::has_evaluate() const {
    return true;
}

}  // namespace CdpnExtension
