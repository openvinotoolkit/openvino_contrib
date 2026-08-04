// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// CdpnPreprocess GPU kernel - image crop + resize + normalise + HWC->CHW.
//
// Steps:
//   1. Int-truncate centre and scale
//   2. Compute integer crop edges with rounding
//   3. Zero-padded crop -> bilinear resize -> [0,1] -> CHW
//
// Tensors (padded to 4D BFYX):
//   input0  image:   [N, IM_H, IM_W, 3]  u8   (HWC pixel data)
//   input1  bbox:    [N, 1, 1, 4]         f32  (x, y, w, h)
//   output0 tensor:  [N, 3, INP_RES, INP_RES] f32 (CHW normalised)
//   output1 meta:    [N, 1, 1, 5]         f32  (c_w, c_h, s, w_begin, h_begin)
//
// WorkSizes: global = "X,Y,B" from output0 [N,3,INP_RES,INP_RES]
//   get_global_id(0) = output column
//   get_global_id(1) = output row
//   get_global_id(2) = batch index
//
// JIT defines from XML:
//   INP_RES   : int   (256)
//   PAD_RATIO : float (1.5)
//   IM_W      : int   (640)
//   IM_H      : int   (480)
//
// GPU plugin auto-defines (pitches account for internal buffer padding):
//   OUTPUT0_DIMS[4] — tensor dimensions in BFYX order
//   INPUT0_PITCHES[4], INPUT0_OFFSET  — image buffer pitches
//   INPUT1_PITCHES[4], INPUT1_OFFSET  — bbox buffer pitches
//   OUTPUT0_PITCHES[4], OUTPUT0_OFFSET — tensor output pitches
//   OUTPUT1_PITCHES[4], OUTPUT1_OFFSET — meta output pitches

#ifndef INP_RES
#define INP_RES 256
#endif
#ifndef PAD_RATIO
#define PAD_RATIO 1.5f
#endif
#ifndef IM_W
#define IM_W 640
#endif
#ifndef IM_H
#define IM_H 480
#endif

// Read a pixel value from the zero-padded canvas.
// Canvas [s x s] maps to image at (u,l). Valid region is
// [local_u..local_b) x [local_l..local_r). Outside -> 0.
// n = batch index, needed for pitch-based addressing.
inline float read_canvas(const __global uchar* image,
                         int n,
                         int row, int col, int ch,
                         int u, int l,
                         int local_u, int local_l,
                         int local_b, int local_r)
{
    if (row >= local_u && row < local_b &&
        col >= local_l && col < local_r) {
        int src_row = u + (row - local_u);
        int src_col = l + (col - local_l);
        // BFYX: B=N, F=IM_H, Y=IM_W, X=3
        return (float)image[n * INPUT0_PITCHES[0]
                          + src_row * INPUT0_PITCHES[1]
                          + src_col * INPUT0_PITCHES[2]
                          + ch * INPUT0_PITCHES[3]
                          + INPUT0_OFFSET];
    }
    return 0.0f;
}

__kernel void cdpn_preprocess_gpu(
    const __global uchar* restrict image,
    const __global float* restrict bbox,
    __global float* restrict out_tensor,
    __global float* restrict out_meta)
{
    const int ox = get_global_id(0);  // column in output
    const int oy = get_global_id(1);  // row    in output
    const int n  = get_global_id(2);  // batch  index

    if (ox >= INP_RES || oy >= INP_RES || n >= OUTPUT0_DIMS[0])
        return;

    // --- Read bbox ---
    // bbox [N,1,1,4] BFYX: B=N, F=1, Y=1, X=4
    // Pitches are in elements (not bytes).
    const int bbox_base = n * INPUT1_PITCHES[0] + INPUT1_OFFSET;
    const float bx = bbox[bbox_base + 0];
    const float by = bbox[bbox_base + 1];
    const float bw = bbox[bbox_base + 2];
    const float bh = bbox[bbox_base + 3];

    // --- Step 1: centre & scale (float) ---
    float c_w_f = bx + 0.5f * bw;
    float c_h_f = by + 0.5f * bh;
    float s_f   = max(bw, bh) * PAD_RATIO;
    s_f = min(s_f, (float)max(IM_W, IM_H));

    // --- Step 2: int-truncate ---
    const int c_w = (int)c_w_f;
    const int c_h = (int)c_h_f;
    const int s   = (int)s_f;

    if (s <= 0) {
        for (int ch = 0; ch < 3; ++ch)
            out_tensor[n * OUTPUT0_PITCHES[0] + ch * OUTPUT0_PITCHES[1]
                     + oy * OUTPUT0_PITCHES[2] + ox * OUTPUT0_PITCHES[3]
                     + OUTPUT0_OFFSET] = 0.0f;
        if (ox == 0 && oy == 0) {
            int meta_base = n * OUTPUT1_PITCHES[0] + OUTPUT1_OFFSET;
            out_meta[meta_base + 0] = 0.0f;
            out_meta[meta_base + 1] = 0.0f;
            out_meta[meta_base + 2] = 0.0f;
            out_meta[meta_base + 3] = 0.0f;
            out_meta[meta_base + 4] = 0.0f;
        }
        return;
    }

    // --- Step 3: crop edges ---
    int u = (int)((float)c_h - 0.5f * (float)s + 0.5f);
    int l = (int)((float)c_w - 0.5f * (float)s + 0.5f);
    int b = u + s;
    int r = l + s;

    // --- Step 4: clamp to image bounds ---
    int local_u = 0, local_l = 0, local_b = s, local_r = s;

    if (u < 0) { local_u = -u; u = 0; }
    if (l < 0) { local_l = -l; l = 0; }
    if (b > IM_H) { local_b = s - (b - IM_H); b = IM_H; }
    if (r > IM_W) { local_r = s - (r - IM_W); r = IM_W; }

    // bilinear resize from [s,s] canvas -> [INP_RES, INP_RES]
    // cv2 INTER_LINEAR: src = (dst + 0.5) * (s / INP_RES) - 0.5
    const float scale = (float)s / (float)INP_RES;
    const float src_y = ((float)oy + 0.5f) * scale - 0.5f;
    const float src_x = ((float)ox + 0.5f) * scale - 0.5f;

    const int sy0 = (int)floor(src_y);
    const int sy1 = sy0 + 1;
    const int sx0 = (int)floor(src_x);
    const int sx1 = sx0 + 1;
    const float fy = src_y - (float)sy0;
    const float fx = src_x - (float)sx0;

    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w01 = fx          * (1.0f - fy);
    const float w10 = (1.0f - fx) * fy;
    const float w11 = fx          * fy;

    // Clamp to canvas [0, s-1]
    const int cy0 = clamp(sy0, 0, s - 1);
    const int cy1 = clamp(sy1, 0, s - 1);
    const int cx0 = clamp(sx0, 0, s - 1);
    const int cx1 = clamp(sx1, 0, s - 1);

    for (int ch = 0; ch < 3; ++ch) {
        float v00 = read_canvas(image, n, cy0, cx0, ch, u, l, local_u, local_l, local_b, local_r);
        float v01 = read_canvas(image, n, cy0, cx1, ch, u, l, local_u, local_l, local_b, local_r);
        float v10 = read_canvas(image, n, cy1, cx0, ch, u, l, local_u, local_l, local_b, local_r);
        float v11 = read_canvas(image, n, cy1, cx1, ch, u, l, local_u, local_l, local_b, local_r);

        float val = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
        // Use pitch macros for output tensor [N,3,INP_RES,INP_RES]
        out_tensor[n * OUTPUT0_PITCHES[0] + ch * OUTPUT0_PITCHES[1]
                 + oy * OUTPUT0_PITCHES[2] + ox * OUTPUT0_PITCHES[3]
                 + OUTPUT0_OFFSET] = val / 255.0f;
    }

    // --- Step 7: crop metadata (only work-item (0,0)) ---
    if (ox == 0 && oy == 0) {
        const float c_h_out = 0.5f * (float)(u + b);
        const float c_w_out = 0.5f * (float)(l + r);
        const float s_out   = (float)s;
        const float w_begin = (float)((int)c_w_out) - s_out / 2.0f;
        const float h_begin = (float)((int)c_h_out) - s_out / 2.0f;

        // Use pitch macros for meta output [N,1,1,5]
        int meta_base = n * OUTPUT1_PITCHES[0] + OUTPUT1_OFFSET;
        out_meta[meta_base + 0] = c_w_out;
        out_meta[meta_base + 1] = c_h_out;
        out_meta[meta_base + 2] = s_out;
        out_meta[meta_base + 3] = w_begin;
        out_meta[meta_base + 4] = h_begin;
    }
}
