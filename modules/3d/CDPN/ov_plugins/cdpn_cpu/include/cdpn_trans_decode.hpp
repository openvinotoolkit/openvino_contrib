// CdpnTransDecode - translation head decoding.
//
//   pred_depth = ratio_depth * (out_res / s)
//   pred_c     = ratio_delta_c * [bbox_w, bbox_h] + [c_w, c_h]
//   tx = (pred_cx - cx) * pred_depth / fx
//   ty = (pred_cy - cy) * pred_depth / fy
//   tz = pred_depth
//
// Uses int-truncated c_w/c_h from crop_meta (matches zoom_in).
//
// Inputs:
//   0: pred_trans [1, 1, 1, 3] f32 - ratio_delta_cx, ratio_delta_cy, ratio_depth
//   1: bbox_wh    [1, 1, 1, 2] f32 - bbox width, height
//   2: crop_meta  [1, 1, 1, 5] f32 - c_w, c_h, s, w_begin, h_begin
//   3: cam_K      [1, 1, 1, 4] f32 - fx, fy, cx, cy
//
// Outputs:
//   0: translation [1, 1, 1, 3] f32 - tx, ty, tz
//
// Attributes:
//   out_res : int (default 64)

#pragma once

#include <openvino/op/op.hpp>

namespace CdpnExtension {

class CdpnTransDecode : public ov::op::Op {
public:
    OPENVINO_OP("CdpnTransDecode");

    CdpnTransDecode() = default;

    CdpnTransDecode(const ov::Output<ov::Node>& pred_trans,
                    const ov::Output<ov::Node>& bbox_wh,
                    const ov::Output<ov::Node>& crop_meta,
                    const ov::Output<ov::Node>& cam_K,
                    int out_res = 64);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;

private:
    int m_out_res = 64;
};

}  // namespace CdpnExtension
