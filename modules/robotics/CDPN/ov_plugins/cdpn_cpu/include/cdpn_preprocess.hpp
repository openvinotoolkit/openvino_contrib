// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// CdpnPreprocess - image crop + resize + normalise + HWC->CHW.
//
// Steps:
//   1. Int-truncate centre and scale
//   2. Compute integer crop edges (u, l, b, r) with +0.5 rounding
//   3. Copy pixel-aligned rectangle into zero-padded canvas
//   4. cv2.resize with bilinear interpolation to inp_res x inp_res
//   5. Normalise [0,255] -> [0,1], transpose HWC -> CHW
//
// Inputs:
//   0: image  [H, W, 3] u8  - BGR HWC
//   1: bbox   [4] f32       - (x, y, w, h)
//
// Outputs:
//   0: tensor    [1, 3, inp_res, inp_res] f32
//   1: crop_meta [1, 1, 1, 5] f32 - (c_w, c_h, s, w_begin, h_begin)
//
// Attributes:
//   inp_res   : int   (default 256)
//   pad_ratio : float (default 1.5)
//   im_w      : int   (default 640)
//   im_h      : int   (default 480)

#pragma once

#include <openvino/op/op.hpp>

namespace CdpnExtension {

class CdpnPreprocess : public ov::op::Op {
public:
  OPENVINO_OP("CdpnPreprocess");

  CdpnPreprocess() = default;

  CdpnPreprocess(const ov::Output<ov::Node> &image,
                 const ov::Output<ov::Node> &bbox, int inp_res = 256,
                 float pad_ratio = 1.5f, int im_w = 640, int im_h = 480);

  void validate_and_infer_types() override;
  std::shared_ptr<ov::Node>
  clone_with_new_inputs(const ov::OutputVector &new_args) const override;
  bool visit_attributes(ov::AttributeVisitor &visitor) override;
  bool evaluate(ov::TensorVector &outputs,
                const ov::TensorVector &inputs) const override;
  bool has_evaluate() const override;

private:
  int m_inp_res = 256;
  float m_pad_ratio = 1.5f;
  int m_im_w = 640;
  int m_im_h = 480;
};

} // namespace CdpnExtension
