#include "cdpn_trans_decode.hpp"

namespace CdpnExtension {

CdpnTransDecode::CdpnTransDecode(const ov::Output<ov::Node>& pred_trans,
                                 const ov::Output<ov::Node>& bbox_wh,
                                 const ov::Output<ov::Node>& crop_meta,
                                 const ov::Output<ov::Node>& cam_K,
                                 int out_res)
    : Op({pred_trans, bbox_wh, crop_meta, cam_K}),
      m_out_res(out_res) {
    constructor_validate_and_infer_types();
}

void CdpnTransDecode::validate_and_infer_types() {
    // Output: [1, 1, 1, 3] - 4D BFYX for GPU compatibility
    set_output_type(0, ov::element::f32, ov::PartialShape({1, 1, 1, 3}));
}

std::shared_ptr<ov::Node> CdpnTransDecode::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 4, "CdpnTransDecode expects 4 inputs");
    return std::make_shared<CdpnTransDecode>(
        new_args[0], new_args[1], new_args[2], new_args[3], m_out_res);
}

bool CdpnTransDecode::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("out_res", m_out_res);
    return true;
}

bool CdpnTransDecode::evaluate(ov::TensorVector& outputs,
                               const ov::TensorVector& inputs) const {
    const auto* pred_trans = inputs[0].data<float>();  // [1,1,1,3]
    const auto* bbox_wh = inputs[1].data<float>();     // [1,1,1,2]
    const auto* crop_meta = inputs[2].data<float>();   // [1,1,1,5]
    const auto* cam_K = inputs[3].data<float>();       // [1,1,1,4]

    outputs[0].set_shape({1, 1, 1, 3});
    auto* out = outputs[0].data<float>();

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

    // Translation decode (CDPN paper Eq. 5)
    const float pred_depth = ratio_depth *
        (static_cast<float>(m_out_res) / s);
    const float pred_cx = ratio_delta_cx * bw + c_w;
    const float pred_cy = ratio_delta_cy * bh + c_h;

    out[0] = (pred_cx - cx) * pred_depth / fx;  // tx
    out[1] = (pred_cy - cy) * pred_depth / fy;  // ty
    out[2] = pred_depth;                         // tz

    return true;
}

bool CdpnTransDecode::has_evaluate() const {
    return true;
}

}  // namespace CdpnExtension
