// CdpnPnpSolve - EPnP + RANSAC PnP solver for CDPN E2E pipeline.
//
// Self-contained implementation of the EPnP algorithm
// wrapped in a standard RANSAC loop
// with default parameters (iterationsCount=100, reprojectionError=8.0,
// confidence=0.99, model_points=5).
//
// Steps:
//   1. Build 2D-3D correspondences from denorm_coords + confidence
//      (confidence threshold + near-zero object-extent filter)
//   2. RANSAC loop:
//      a. Sample 5 random correspondences
//      b. Run EPnP on the sample to get candidate R, T
//      c. Project all 3D points using candidate R, T, K
//      d. Count inliers (squared reprojection error <= threshold^2)
//      e. Keep model with most inliers; adaptively update iteration count
//   3. Refit EPnP on all inliers of the best model
//   4. Output R[3,3], T[3], num_correspondences, pnp_success
//
// Inputs (all 4D BFYX for GPU compatibility):
//   0: denorm_coords [1, 3, 64, 64] f32 -- denormalised 3D coordinates
//   1: confidence    [1, 1, 64, 64] f32 -- normalised confidence map
//   2: obj_extents   [1, 1, 1, 3]   f32 -- |min_x|, |min_y|, |min_z|
//   3: crop_meta     [1, 1, 1, 5]   f32 -- c_w, c_h, s, w_begin, h_begin
//   4: cam_K         [1, 1, 1, 4]   f32 -- fx, fy, cx, cy
//
// Outputs:
//   0: R            [1, 1, 3, 3] f32 -- rotation matrix
//   1: T_pnp        [1, 1, 1, 3] f32 -- translation vector
//   2: num_corres   [1, 1, 1, 1] f32 -- number of correspondences
//   3: pnp_success  [1, 1, 1, 1] f32 -- 1.0 if PnP succeeded, else 0.0
//
// Attributes:
//   mask_threshold   : float (default 0.5)
//   out_res          : int   (default 64)
//   max_iterations   : int   (default 100)
//   reproj_threshold : float (default 8.0)

#pragma once

#include <openvino/op/op.hpp>

namespace CdpnExtension {

class CdpnPnpSolve : public ov::op::Op {
public:
    OPENVINO_OP("CdpnPnpSolve");

    CdpnPnpSolve() = default;

    CdpnPnpSolve(const ov::Output<ov::Node>& denorm_coords,
                  const ov::Output<ov::Node>& confidence,
                  const ov::Output<ov::Node>& obj_extents,
                  const ov::Output<ov::Node>& crop_meta,
                  const ov::Output<ov::Node>& cam_K,
                  float mask_threshold = 0.5f,
                  int out_res = 64,
                  int max_iterations = 100,
                  float reproj_threshold = 8.0f);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;

private:
    float m_mask_threshold = 0.5f;
    int m_out_res = 64;
    int m_max_iterations = 100;
    float m_reproj_threshold = 8.0f;
};

}  // namespace CdpnExtension
