// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

typedef struct {
    float c;
    float s;
} JacobiRotation;

inline void makeJacobi(float x, float y, float z, JacobiRotation* jr)
{
    // Use double for intermediate math
    float dx = x;
    float dy = y;
    float dz = z;

    float deno = 2.0 * fabs(dy);
    if(deno < 1e-30) { // Slightly safer epsilon
        jr->c = 1.0f;
        jr->s = 0.0f;
    } else {
        float tau = (dx - dz) / deno;
        float w = sqrt(tau*tau + 1.0);
        float t;
        if(tau > 0.0) {
            t = 1.0 / (tau + w);
        } else {
            t = 1.0 / (tau - w);
        }
        float sign_t = (t >= 0.0) ? 1.0 : -1.0;
        float n = 1.0 / sqrt(t*t + 1.0);

        // Cast back to float at the very end
        jr->s = (-sign_t * (dy / fabs(dy)) * fabs(t) * n);
        jr->c = n;
    }
}

inline void real_2x2_jacobi_svd(
        float a, float b, float c, float d,
        JacobiRotation* jl, JacobiRotation* jr)
{
    // Step 1: temporary left rotation
    float t = a + d;
    float delta = c - b;

    JacobiRotation rot1;
    if(fabs(delta) < 1e-38f) {
        rot1.c = 1.0f;
        rot1.s = 0.0f;
    } else {
        float u = t / delta;
        float tmp = sqrt(1.0f + u*u);
        rot1.s = 1.0f / tmp;
        rot1.c = u / tmp;
    }

    // Step 2: Apply rot1 to 2x2 matrix
    float m00 = rot1.c*a + rot1.s*c;
    float m01 = rot1.c*b + rot1.s*d;
    float m10 = -rot1.s*a + rot1.c*c;
    float m11 = -rot1.s*b + rot1.c*d;

    // Step 3: Compute right rotation using makeJacobi
    makeJacobi(m00, m01, m11, jr);

    // Step 4: Compute final left rotation jl = rot1 * jr^T
    jl->c = rot1.c * jr->c + rot1.s * jr->s;
    jl->s = -rot1.c * jr->s + rot1.s * jr->c;
}


inline void apply_left_rotation(float W[3][3], int p, int q, JacobiRotation jl)
{
    const float EPSILON = 1e-9;

    for(int k = 0; k < 3; k++) {
        float v_p = W[p][k];
        float v_q = W[q][k];

        // Standard rotation: [ c  s] [v_p]
        //                    [-s  c] [v_q]
        float next_p = jl.c * v_p + (float)jl.s * v_q;
        float next_q = -jl.s * v_p + (float)jl.c * v_q;

        // Force tiny values to zero to prevent divergence
        if (fabs(next_p) < EPSILON) next_p = 0.0;
        if (fabs(next_q) < EPSILON) next_q = 0.0;

        W[p][k] = next_p;
        W[q][k] = next_q;
    }
}

inline void apply_right_rotation(float W[3][3], int p, int q, JacobiRotation jr)
{
    const float EPSILON = 1e-9;

    for(int k = 0; k < 3; k++) {
        float v_p = W[k][p];
        float v_q = W[k][q];

        // Right rotation is the Transpose: [v_p v_q] [ c -s]
        //                                            [ s  c]
        float next_p = jr.c * v_p - jr.s * v_q;
        float next_q = jr.s * v_p + jr.c * v_q;

        if (fabs(next_p) < EPSILON) next_p = 0.0;
        if (fabs(next_q) < EPSILON) next_q = 0.0;

        W[k][p] = next_p;
        W[k][q] = next_q;
    }
}

inline void accumulate_V(float V[3][3], int p, int q, JacobiRotation jr)
{
    float c = jr.c;
    float s = jr.s;

    for (int k = 0; k < 3; k++)
    {
        float vp = V[k][p];
        float vq = V[k][q];

        V[k][p] =  c * vp - s * vq;
        V[k][q] =  s * vp + c * vq;
    }
}

inline void accumulate_U(float U[3][3], int p, int q, JacobiRotation jl)
{
    // jl is j_left (NOT transposed)
    // We explicitly apply jl.transpose() here
    float c = jl.c;
    float s = jl.s;

    for (int k = 0; k < 3; k++)
    {
        float up = U[k][p];
        float uq = U[k][q];

        U[k][p] =  c * up - s * uq;
        U[k][q] =  s * up + c * uq;
    }
}

__kernel void ov_custom_svd_v(
        __global const INPUT0_TYPE* input,   // (B,3,3)
        __global OUTPUT0_TYPE* V)             // (B,3,3)
{
    int b = get_global_id(0);
    int base = b*9;

    //float Ut[3][3];
    float W[3][3], Vt[3][3];
    for(int i=0;i<3;i++) for(int j=0;j<3;j++){
        W[i][j] = (float)input[base+i*3+j];
        Vt[i][j] = (i==j)?1.0f:0.0f;
        //Ut[i][j] = (i==j)?1.0f:0.0f;
    }

    // STEP 1: scale
    float scale=0.0f;
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) scale=fmax(scale,fabs(W[i][j]));

    if(scale==0.0f) scale=1.0f;
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) W[i][j]/=scale;
    //STEP 2
    // Compute max diagonal entry
    float maxDiagEntry = fmax(fabs(W[0][0]), fmax(fabs(W[1][1]), fabs(W[2][2])));

    // Compute threshold like Eigen
    float considerAsZero = 1.17549435e-7f;        // FLT_MIN
    float precision      = 2.0f * 1.1920929e-07f; // 2*FLT_EPSILON
    float threshold      = fmax(considerAsZero, precision * maxDiagEntry);

    int pairs[3][2] = { {1,0}, {2,0}, {2,1} };
    int max_sweeps = 10;

    int finished = 0;
    int cnt = 0;
    while (!finished)
    {
        finished = 1;

        for (int p = 1; p < 3; ++p)
        {
            for (int q = 0; q < p; ++q)
            {
                // threshold recomputed PER (p,q)
                float threshold = fmax(1.17549435e-38f, 2.0f * 1.1920929e-07f * maxDiagEntry);
                //printf("\n threashhold:%.8f", threshold);

                if (fabs(W[p][q]) > threshold || fabs(W[q][p]) > threshold)
                {
                    finished = 0;

                    JacobiRotation jl, jr;
                    jl.c = 1; jl.s=0;
                    jr.c = 1; jr.s=0;
                    real_2x2_jacobi_svd( W[p][p], W[p][q], W[q][p], W[q][q], &jl, &jr);
                    apply_left_rotation(W, p, q, jl);
                    apply_right_rotation(W, p, q, jr);
                    accumulate_V(Vt, p, q, jr);
                    // update maxDiagEntry INSIDE loop (Eigen does this!)
                    float ap = fabs(W[p][p]);
                    float aq = fabs(W[q][q]);
                    maxDiagEntry = fmax(maxDiagEntry, fmax(ap, aq));
                }
            }
        }
    }

    //Step -3 (calculating S and fliping sign)
    const float eps = 1e-6f;   // for FP32 (use 1e-12 for FP64)
    float S[3];
    for (int i = 0; i < 3; i++)
    {
        float a = W[i][i];
        // singular value is always positive
        S[i] = fabs(a);
    }

    //Step 4
    for (int i = 0; i < 3; i++)
        S[i] *= scale;

    int nonzeroSingularValues = 3;

    for (int i = 0; i < 3; i++)
    {
        // find index of max singular value in [i..2]
        int pos = i;
        float maxVal = S[i];

        for (int j = i + 1; j < 3; j++)
        {
            if (S[j] > maxVal)
            {
                maxVal = S[j];
                pos = j;
            }
        }

        // if remaining singular values are zero → stop
        if (maxVal == 0.0f)
        {
            nonzeroSingularValues = i;
            break;
        }

        // swap singular values
        if (pos != i)
        {
            float tmp = S[i];
            S[i] = S[pos];
            S[pos] = tmp;
            // swap columns of V if you compute it
            
               for (int k = 0; k < 3; k++)
               {
               float t = Vt[k][i];
               Vt[k][i] = Vt[k][pos];
               Vt[k][pos] = t;
               }
        }
    }

    // Store U
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) V[base+i*3+j]=(OUTPUT0_TYPE)Vt[i][j];
}
