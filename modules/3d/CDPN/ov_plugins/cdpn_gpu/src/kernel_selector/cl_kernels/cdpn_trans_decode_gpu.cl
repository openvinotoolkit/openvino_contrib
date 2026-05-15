// CdpnTransDecode GPU kernel - translation decode (CDPN paper Eq. 5).
//
// Tensors (4D BFYX):
//   input0  pred_trans: [1, 1, 1, 3]  f32  (ratio_delta_cx, ratio_delta_cy, ratio_depth)
//   input1  bbox_wh:    [1, 1, 1, 2]  f32  (bw, bh)
//   input2  crop_meta:  [1, 1, 1, 5]  f32  (c_w, c_h, s, w_begin, h_begin)
//   input3  cam_K:      [1, 1, 1, 4]  f32  (fx, fy, cx, cy)
//   output0 translation:[1, 1, 1, 3]  f32  (tx, ty, tz)
//
// Single work-item kernel (global=1).
//
// JIT defines from XML:
//   OUT_RES : int (64)

#ifndef OUT_RES
#define OUT_RES 64
#endif

__kernel void cdpn_trans_decode_gpu(
    const __global float* restrict pred_trans,
    const __global float* restrict bbox_wh,
    const __global float* restrict crop_meta,
    const __global float* restrict cam_K,
    __global float* restrict translation)
{
    const float ratio_delta_cx = pred_trans[0];
    const float ratio_delta_cy = pred_trans[1];
    const float ratio_depth    = pred_trans[2];

    const float bw = bbox_wh[0];
    const float bh = bbox_wh[1];

    const float c_w = crop_meta[0];
    const float c_h = crop_meta[1];
    const float s   = crop_meta[2];

    const float fx = cam_K[0];
    const float fy = cam_K[1];
    const float cx = cam_K[2];
    const float cy = cam_K[3];

    // CDPN paper Eq. 5
    const float pred_depth = ratio_depth * ((float)OUT_RES / s);
    const float pred_cx = ratio_delta_cx * bw + c_w;
    const float pred_cy = ratio_delta_cy * bh + c_h;

    translation[0] = (pred_cx - cx) * pred_depth / fx;  // tx
    translation[1] = (pred_cy - cy) * pred_depth / fy;  // ty
    translation[2] = pred_depth;                        // tz
}
