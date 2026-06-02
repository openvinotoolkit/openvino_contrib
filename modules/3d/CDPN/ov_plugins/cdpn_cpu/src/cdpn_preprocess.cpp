// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cdpn_preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace CdpnExtension {

CdpnPreprocess::CdpnPreprocess(const ov::Output<ov::Node>& image,
                               const ov::Output<ov::Node>& bbox,
                               int inp_res,
                               float pad_ratio,
                               int im_w,
                               int im_h)
    : Op({image, bbox}),
      m_inp_res(inp_res),
      m_pad_ratio(pad_ratio),
      m_im_w(im_w),
      m_im_h(im_h) {
    constructor_validate_and_infer_types();
}

void CdpnPreprocess::validate_and_infer_types() {
  // Propagate batch dim from image input (dynamic or static)
  auto batch = get_input_partial_shape(0)[0];
  set_output_type(0, ov::element::f32,
                  ov::PartialShape({batch, 3, m_inp_res, m_inp_res}));
  // crop_meta padded to 4D BFYX
  set_output_type(1, ov::element::f32, ov::PartialShape({batch, 1, 1, 5}));
}

std::shared_ptr<ov::Node> CdpnPreprocess::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 2, "CdpnPreprocess expects 2 inputs");
    return std::make_shared<CdpnPreprocess>(
        new_args[0], new_args[1],
        m_inp_res, m_pad_ratio, m_im_w, m_im_h);
}

bool CdpnPreprocess::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("inp_res", m_inp_res);
    visitor.on_attribute("pad_ratio", m_pad_ratio);
    visitor.on_attribute("im_w", m_im_w);
    visitor.on_attribute("im_h", m_im_h);
    return true;
}

bool CdpnPreprocess::evaluate(ov::TensorVector& outputs,
                              const ov::TensorVector& inputs) const {
    // --- Read inputs ---
    // Image may be 3D [H,W,3] or 4D [1,H,W,3]
    const auto img_shape = inputs[0].get_shape();
    int N, H, W;
    if (img_shape.size() == 4) {
      N = static_cast<int>(img_shape[0]);
      H = static_cast<int>(img_shape[1]);
      W = static_cast<int>(img_shape[2]);
    } else {
      N = 1;
      H = static_cast<int>(img_shape[0]);
      W = static_cast<int>(img_shape[1]);
    }
    const auto *img_base = inputs[0].data<uint8_t>();
    const auto *bbox_base = inputs[1].data<float>();

    const int res = m_inp_res;
    const int R2 = res * res;
    const int img_stride = H * W * 3;
    const int tensor_stride = 3 * R2;
    const int bbox_stride = 4;
    const int meta_stride = 5;

    // --- Set output shapes ---
    outputs[0].set_shape({static_cast<size_t>(N), 3, static_cast<size_t>(res),
                          static_cast<size_t>(res)});
    outputs[1].set_shape({static_cast<size_t>(N), 1, 1, 5});
    auto *out_tensor_base = outputs[0].data<float>();
    auto *out_meta_base = outputs[1].data<float>();
    for (int n = 0; n < N; ++n) {
      const auto *img = img_base + n * img_stride;
      const auto *bbox = bbox_base + n * bbox_stride;
      auto *out_tensor = out_tensor_base + n * tensor_stride;
      auto *out_meta = out_meta_base + n * meta_stride;

      // --- zoom_in() ---
      const float bx = bbox[0], by = bbox[1], bw = bbox[2], bh = bbox[3];
      // Guard: zero-padded batch entries have bbox=[0,0,0,0] -> s=0.
      // Skip crop and write zeros to avoid out-of-bounds canvas access.
      if (bw <= 0.0f && bh <= 0.0f) {
        std::memset(out_tensor, 0, tensor_stride * sizeof(float));
        std::memset(out_meta, 0, meta_stride * sizeof(float));
        continue;
      }

      // Step 1: compute centre and scale (float)
      float c_w_f = bx + 0.5f * bw;
      float c_h_f = by + 0.5f * bh;
      float s_f = std::max(bw, bh) * m_pad_ratio;
      s_f = std::min(s_f, static_cast<float>(std::max(m_im_w, m_im_h)));

      // Step 2: int-truncate
      const int c_w = static_cast<int>(c_w_f);
      const int c_h = static_cast<int>(c_h_f);
      const int s = static_cast<int>(s_f);

      // Step 3: compute crop edges
      int u = static_cast<int>(c_h - 0.5 * s + 0.5); // top
      int l = static_cast<int>(c_w - 0.5 * s + 0.5); // left
      int b = u + s;                                 // bottom
      int r = l + s;                                 // right

      // Step 4: clamp to image bounds
      int local_u = 0, local_l = 0, local_b = s, local_r = s;

      if (u < 0) {
        local_u = -u;
        u = 0;
      }
      if (l < 0) {
        local_l = -l;
        l = 0;
      }
      if (b > H) {
        local_b = s - (b - H);
        b = H;
      }
      if (r > W) {
        local_r = s - (r - W);
        r = W;
      }

      // Step 5: create zero-padded canvas [s, s, 3] and copy valid region
      std::vector<float> canvas(s * s * 3, 0.0f);
      for (int row = local_u; row < local_b; ++row) {
        const int src_row = u + (row - local_u);
        for (int col = local_l; col < local_r; ++col) {
            const int src_col = l + (col - local_l);
            for (int ch = 0; ch < 3; ++ch) {
                canvas[(row * s + col) * 3 + ch] =
                    static_cast<float>(img[(src_row * W + src_col) * 3 + ch]);
            }
        }
      }

    // Step 6: bilinear resize from [s, s] to [res, res]
    const float scale_y = static_cast<float>(s) / static_cast<float>(res);
    const float scale_x = static_cast<float>(s) / static_cast<float>(res);

    for (int oy = 0; oy < res; ++oy) {
        const float src_y = (static_cast<float>(oy) + 0.5f) * scale_y - 0.5f;
        const int y0 = static_cast<int>(std::floor(src_y));
        const int y1 = y0 + 1;
        const float fy = src_y - static_cast<float>(y0);

        const int y0c = std::max(0, std::min(y0, s - 1));
        const int y1c = std::max(0, std::min(y1, s - 1));

        for (int ox = 0; ox < res; ++ox) {
            const float src_x = (static_cast<float>(ox) + 0.5f) * scale_x - 0.5f;
            const int x0 = static_cast<int>(std::floor(src_x));
            const int x1 = x0 + 1;
            const float fx = src_x - static_cast<float>(x0);

            const int x0c = std::max(0, std::min(x0, s - 1));
            const int x1c = std::max(0, std::min(x1, s - 1));

            const float w00 = (1.0f - fx) * (1.0f - fy);
            const float w01 = fx * (1.0f - fy);
            const float w10 = (1.0f - fx) * fy;
            const float w11 = fx * fy;

            for (int ch = 0; ch < 3; ++ch) {
              const float v = canvas[(y0c * s + x0c) * 3 + ch] * w00 +
                              canvas[(y0c * s + x1c) * 3 + ch] * w01 +
                              canvas[(y1c * s + x0c) * 3 + ch] * w10 +
                              canvas[(y1c * s + x1c) * 3 + ch] * w11;

              // Normalise [0,255] -> [0,1] and write CHW layout
              out_tensor[ch * R2 + oy * res + ox] = v / 255.0f;
            }
        }
    }

    // Step 7: crop metadata
    // c_h_out = 0.5*(u_orig + b_orig), c_w_out = 0.5*(l_orig + r_orig)
    // But after clamping, u and b have changed.
    // zoom_in: c_h = 0.5*(u+b), c_w = 0.5*(l+r) after clamping
    const float c_h_out = 0.5f * static_cast<float>(u + b);
    const float c_w_out = 0.5f * static_cast<float>(l + r);
    const float s_out = static_cast<float>(s);

    out_meta[0] = c_w_out;
    out_meta[1] = c_h_out;
    out_meta[2] = s_out;
    // w_begin / h_begin used by PnP correspondence builder
    // c_w_int = int(c_w_out), w_begin = c_w_int - s/2.0
    const float w_begin = static_cast<float>(static_cast<int>(c_w_out))
                          - s_out / 2.0f;
    const float h_begin = static_cast<float>(static_cast<int>(c_h_out))
                          - s_out / 2.0f;
    out_meta[3] = w_begin;
    out_meta[4] = h_begin;

    } // end batch loop

    return true;
}

bool CdpnPreprocess::has_evaluate() const {
    return true;
}

}  // namespace CdpnExtension
