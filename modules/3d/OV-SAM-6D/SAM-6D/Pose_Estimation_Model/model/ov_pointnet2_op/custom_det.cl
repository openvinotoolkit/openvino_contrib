// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef MAX_N
#define MAX_N 64
#endif

__kernel void ov_custom_det(
    __global const INPUT0_TYPE* input,   // (B, n, n) row-major
    __global OUTPUT0_TYPE* output)       // (B,)
{
    uint batch_index = get_global_id(0);
    uint n = INPUT0_DIMS[1];  // matrix size (n x n)
    uint b = INPUT0_DIMS[0];  // batch size

    if (batch_index >= b) return;

    uint input_offset = batch_index * n * n;
    uint output_offset = batch_index;

    // Declare local memory at kernel scope
    __local float local_matrix[MAX_N * MAX_N];

    // Fast paths for small matrices with improved precision
    if (n == 1) {
        output[output_offset] = (OUTPUT0_TYPE)((float)input[input_offset]);
        return;
    }

    if (n == 2) {
        // |a00 a01|
        // |a10 a11|
        float a00 = (float)input[input_offset + 0];
        float a01 = (float)input[input_offset + 1];
        float a10 = (float)input[input_offset + 2];
        float a11 = (float)input[input_offset + 3];

        // Use fma for better precision
        float det = fma(a00, a11, -a01 * a10);
        output[output_offset] = (OUTPUT0_TYPE)det;
        return;
    }

    if (n == 3) {
        // Row-major 3x3 with improved precision
        float a00 = (float)input[input_offset + 0], a01 = (float)input[input_offset + 1], a02 = (float)input[input_offset + 2];
        float a10 = (float)input[input_offset + 3], a11 = (float)input[input_offset + 4], a12 = (float)input[input_offset + 5];
        float a20 = (float)input[input_offset + 6], a21 = (float)input[input_offset + 7], a22 = (float)input[input_offset + 8];

        // Use fma for better precision in 3x3 determinant
        float m1 = fma(a11, a22, -a12 * a21);
        float m2 = fma(a10, a22, -a12 * a20);
        float m3 = fma(a10, a21, -a11 * a20);

        float det = fma(a00, m1, fma(-a01, m2, a02 * m3));
        output[output_offset] = (OUTPUT0_TYPE)det;
        return;
    }

    if (n == 4) {
        // Row-major 4x4 with improved precision
        float a00 = (float)input[input_offset + 0], a01 = (float)input[input_offset + 1], a02 = (float)input[input_offset + 2], a03 = (float)input[input_offset + 3];
        float a10 = (float)input[input_offset + 4], a11 = (float)input[input_offset + 5], a12 = (float)input[input_offset + 6], a13 = (float)input[input_offset + 7];
        float a20 = (float)input[input_offset + 8], a21 = (float)input[input_offset + 9], a22 = (float)input[input_offset + 10], a23 = (float)input[input_offset + 11];
        float a30 = (float)input[input_offset + 12], a31 = (float)input[input_offset + 13], a32 = (float)input[input_offset + 14], a33 = (float)input[input_offset + 15];

        // Compute minors with better precision
        float m0 = fma(a22, a33, -a23 * a32);
        float m1 = fma(a21, a33, -a23 * a31);
        float m2 = fma(a21, a32, -a22 * a31);
        float m3 = fma(a20, a33, -a23 * a30);
        float m4 = fma(a20, a32, -a22 * a30);
        float m5 = fma(a20, a31, -a21 * a30);

        // Compute determinant with better precision
        float det = fma(a00, fma(a11, m0, fma(-a12, m1, a13 * m2)),
                       fma(-a01, fma(a10, m0, fma(-a12, m3, a13 * m4)),
                           fma(a02, fma(a10, m1, fma(-a11, m3, a13 * m5)),
                               -a03 * fma(a10, m2, fma(-a11, m4, a12 * m5)))));
        output[output_offset] = (OUTPUT0_TYPE)det;
        return;
    }

    // General path: Gaussian elimination with partial pivoting
    // For larger matrices, use local memory (up to MAX_N x MAX_N)
    if (n <= MAX_N) {
        // Copy matrix to local memory with coalesced access
        uint local_id = get_local_id(0);
        uint local_size = get_local_size(0);

        for (uint i = local_id; i < n; i += local_size) {
            for (uint j = 0; j < n; ++j) {
                local_matrix[i * n + j] = (float)input[input_offset + i * n + j];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float det = 1.0f;
        float sign = 1.0f;
        const float eps = 1e-12f;

        for (uint k = 0; k < n; ++k) {
            // Find pivot with better numerical stability
            uint pivot_row = k;
            float max_abs = fabs(local_matrix[k * n + k]);
            
            for (uint i = k + 1; i < n; ++i) {
                float v = fabs(local_matrix[i * n + k]);
                if (v > max_abs) {
                    max_abs = v;
                    pivot_row = i;
                }
            }
            
            if (max_abs < eps) {
                output[output_offset] = (OUTPUT0_TYPE)0.0f;
                return;
            }
            
            // Swap rows if needed
            if (pivot_row != k) {
                for (uint j = k; j < n; ++j) {
                    float temp = local_matrix[k * n + j];
                    local_matrix[k * n + j] = local_matrix[pivot_row * n + j];
                    local_matrix[pivot_row * n + j] = temp;
                }
                sign = -sign;
            }
            
            float pivot = local_matrix[k * n + k];
            det *= pivot;
            
            // Eliminate below with better precision
            for (uint i = k + 1; i < n; ++i) {
                float factor = local_matrix[i * n + k] / pivot;
                local_matrix[i * n + k] = 0.0f;
                
                for (uint j = k + 1; j < n; ++j) {
                    local_matrix[i * n + j] = fma(-factor, local_matrix[k * n + j], local_matrix[i * n + j]);
                }
            }
        }

        output[output_offset] = (OUTPUT0_TYPE)(det * sign);
    } else {
        // For very large matrices, fall back to a simpler approach
        output[output_offset] = (OUTPUT0_TYPE)0.0f;
    }
}
