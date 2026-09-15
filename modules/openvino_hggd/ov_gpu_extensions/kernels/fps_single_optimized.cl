/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Portions derived from PyTorch3D (https://github.com/facebookresearch/pytorch3d).
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither the name Meta nor the names of its contributors may be used to
 *    endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/*
 * Farthest Point Sampling - Recomputing O(N*K*K) kernel
 *
 * Removes the hard MAX_N=8192 cap from the previous persistent-min-dist
 * optimistic kernel.  Instead of keeping an O(N) min_dist array in local
 * memory, each work-item recomputes the minimum distance to already-selected
 * points from the output buffer on every iteration.  Selected-point indices
 * are stored in the output port, so no extra scratch buffer is required.
 *
 * Input:  points  [B, N, 3]
 * Output: [B, K, 4] packed [x, y, z, idx]
 *
 * WorkSize: global=(256, B, 1) ******=(256, 1, 1)
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WORKGROUP_SIZE 256
#define MAX_K 64

__kernel void fps_single_optimized_kernel(
    const __global INPUT0_TYPE* points,
          __global OUTPUT0_TYPE* output)
{
    const int local_id = get_local_id(0);
    const int batch = get_global_id(1);

    if (batch >= INPUT0_DIMS[0]) return;

    const int N = INPUT0_DIMS[1];
    const int K = OUTPUT0_DIMS[1];
    const int K_eff = (K < N) ? K : N;

    /* ── Shared state ── */
    __local float local_max_dist[WORKGROUP_SIZE];
    __local int   local_max_idx[WORKGROUP_SIZE];

    /* ── Handle empty batch ── */
    if (N <= 0) {
        if (local_id == 0) {
            for (int k = 0; k < K; k++) {
                const int out_base = batch * OUTPUT0_PITCHES[0] + k * OUTPUT0_PITCHES[1];
                output[out_base + 0 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)0.0f;
                output[out_base + 1 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)0.0f;
                output[out_base + 2 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)0.0f;
                output[out_base + 3 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)0;
            }
        }
        return;
    }

    /* ── Initialize: select point 0 as first sample ── */
    if (local_id == 0) {
        const int p_base = batch * INPUT0_PITCHES[0];
        const float px = (float)points[p_base + 0 * INPUT0_PITCHES[2]];
        const float py = (float)points[p_base + 1 * INPUT0_PITCHES[2]];
        const float pz = (float)points[p_base + 2 * INPUT0_PITCHES[2]];

        const int out_base = batch * OUTPUT0_PITCHES[0];
        output[out_base + 0 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)px;
        output[out_base + 1 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)py;
        output[out_base + 2 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)pz;
        output[out_base + 3 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* ── Main FPS loop ── */
    for (int k = 1; k < K_eff; k++) {

        float my_max = -1.0f;
        int   my_idx = 0;

        for (int i = local_id; i < N; i += WORKGROUP_SIZE) {
            const int pi_base = batch * INPUT0_PITCHES[0] + i * INPUT0_PITCHES[1];
            const float px = (float)points[pi_base + 0 * INPUT0_PITCHES[2]];
            const float py = (float)points[pi_base + 1 * INPUT0_PITCHES[2]];
            const float pz = (float)points[pi_base + 2 * INPUT0_PITCHES[2]];

            /* Recompute min distance to all already-selected points */
            float md = 1e10f;
            for (int j = 0; j < k; j++) {
                const int out_base = batch * OUTPUT0_PITCHES[0] + j * OUTPUT0_PITCHES[1];
                const int sel_idx = (int)output[out_base + 3 * OUTPUT0_PITCHES[2]];

                const int sel_base = batch * INPUT0_PITCHES[0] + sel_idx * INPUT0_PITCHES[1];
                const float sx = (float)points[sel_base + 0 * INPUT0_PITCHES[2]];
                const float sy = (float)points[sel_base + 1 * INPUT0_PITCHES[2]];
                const float sz = (float)points[sel_base + 2 * INPUT0_PITCHES[2]];

                const float dx = px - sx;
                const float dy = py - sy;
                const float dz = pz - sz;
                const float d2 = dx*dx + dy*dy + dz*dz;
                if (d2 < md) md = d2;
            }

            /* Skip already-selected points: md == 0 means this point is
             * coincident with (or is) an already-selected point; selecting it
             * again would emit a duplicate index (diverges from CPU reference
             * which marks selected points with min_dist = -1). */
            if (md > 0.0f && md > my_max) {
                my_max = md;
                my_idx = i;
            }
        }

        /* ── Workgroup reduction: find global argmax ── */
        local_max_dist[local_id] = my_max;
        local_max_idx[local_id]  = my_idx;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int stride = WORKGROUP_SIZE / 2; stride > 0; stride >>= 1) {
            if (local_id < stride) {
                if (local_max_dist[local_id + stride] > local_max_dist[local_id]) {
                    local_max_dist[local_id] = local_max_dist[local_id + stride];
                    local_max_idx[local_id]  = local_max_idx[local_id + stride];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (local_id == 0) {
            const int best = local_max_idx[0];
            const int pb = batch * INPUT0_PITCHES[0] + best * INPUT0_PITCHES[1];
            const int out_base = batch * OUTPUT0_PITCHES[0] + k * OUTPUT0_PITCHES[1];

            output[out_base + 0 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)points[pb + 0 * INPUT0_PITCHES[2]];
            output[out_base + 1 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)points[pb + 1 * INPUT0_PITCHES[2]];
            output[out_base + 2 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)points[pb + 2 * INPUT0_PITCHES[2]];
            output[out_base + 3 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)best;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* ── Pad remaining slots if K > K_eff ── */
    if (local_id == 0 && K_eff < K) {
        const int last_out = batch * OUTPUT0_PITCHES[0] + (K_eff - 1) * OUTPUT0_PITCHES[1];
        const float lx = (float)output[last_out + 0 * OUTPUT0_PITCHES[2]];
        const float ly = (float)output[last_out + 1 * OUTPUT0_PITCHES[2]];
        const float lz = (float)output[last_out + 2 * OUTPUT0_PITCHES[2]];
        const float li = (float)output[last_out + 3 * OUTPUT0_PITCHES[2]];

        for (int k = K_eff; k < K; k++) {
            const int out_base = batch * OUTPUT0_PITCHES[0] + k * OUTPUT0_PITCHES[1];
            output[out_base + 0 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)lx;
            output[out_base + 1 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)ly;
            output[out_base + 2 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)lz;
            output[out_base + 3 * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)li;
        }
    }
}
