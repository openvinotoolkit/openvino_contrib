// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace gfx_plugin {

std::string generate_msl_for_interpolate(const InterpolateCodegenDesc& d, mlir::ModuleOp module) {
    std::string scalar_t = "float";
    if (auto func = get_entry_func(module)) {
        auto ft = func.getFunctionType();
        if (ft.getNumInputs() >= 1) {
            scalar_t = msl_type_from_mlir(ft.getInput(0));
        }
    } else {
        switch (d.element_type) {
            case ov::element::f16: scalar_t = "half"; break;
            case ov::element::f32: scalar_t = "float"; break;
            default: break;
        }
    }
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "using scalar_t = " << scalar_t << ";\n";
    ss << "struct InterpolateParams {\n"
          "  uint N;\n"
          "  uint C;\n"
          "  uint H_in;\n"
          "  uint W_in;\n"
          "  uint H_out;\n"
          "  uint W_out;\n"
          "  float scale_h;\n"
          "  float scale_w;\n"
          "  uint align_corners;\n"
          "  uint use_half_pixel;\n"
          "  uint nearest_mode;\n"
          "};\n";
    ss << "kernel void interpolate_kernel(\n"
          "  device const scalar_t* src [[buffer(0)]],\n"
          "  device scalar_t* dst [[buffer(1)]],\n"
          "  constant InterpolateParams& p [[buffer(2)]],\n"
          "  uint gid [[thread_position_in_grid]]) {\n"
          "  uint total = p.N * p.C * p.H_out * p.W_out;\n"
          "  if (gid >= total) return;\n"
          "  uint tmp = gid;\n"
          "  uint w = tmp % p.W_out; tmp /= p.W_out;\n"
          "  uint h = tmp % p.H_out; tmp /= p.H_out;\n"
          "  uint c = tmp % p.C;     tmp /= p.C;\n"
          "  uint n = tmp;\n"
          "  float fh, fw;\n"
          "  if (p.align_corners && p.H_out > 1) fh = (float)h * (float)(p.H_in - 1) / (float)(p.H_out - 1);\n"
          "  else if (p.use_half_pixel) fh = ((float)h + 0.5f) * p.scale_h - 0.5f;\n"
          "  else fh = (float)h * p.scale_h;\n"
          "  if (p.align_corners && p.W_out > 1) fw = (float)w * (float)(p.W_in - 1) / (float)(p.W_out - 1);\n"
          "  else if (p.use_half_pixel) fw = ((float)w + 0.5f) * p.scale_w - 0.5f;\n"
          "  else fw = (float)w * p.scale_w;\n";
    if (d.nearest) {
        ss << "  int ih;\n"
              "  int iw;\n"
              "  if (p.nearest_mode == 1) {\n"
              "    ih = clamp((int)floor(fh), 0, (int)p.H_in - 1);\n"
              "    iw = clamp((int)floor(fw), 0, (int)p.W_in - 1);\n"
              "  } else if (p.nearest_mode == 2) {\n"
              "    ih = clamp((int)ceil(fh), 0, (int)p.H_in - 1);\n"
              "    iw = clamp((int)ceil(fw), 0, (int)p.W_in - 1);\n"
              "  } else {\n"
              "    ih = clamp((int)round(fh), 0, (int)p.H_in - 1);\n"
              "    iw = clamp((int)round(fw), 0, (int)p.W_in - 1);\n"
              "  }\n"
              "  uint src_idx = (((n * p.C + c) * p.H_in) + (uint)ih) * p.W_in + (uint)iw;\n"
              "  dst[gid] = src[src_idx];\n";
    } else {
        ss << "  float fh0 = floor(fh);\n"
              "  float fw0 = floor(fw);\n"
              "  int h0 = clamp((int)fh0, 0, (int)p.H_in - 1);\n"
              "  int w0 = clamp((int)fw0, 0, (int)p.W_in - 1);\n"
              "  int h1 = min(h0 + 1, (int)p.H_in - 1);\n"
              "  int w1 = min(w0 + 1, (int)p.W_in - 1);\n"
              "  float dh = fh - fh0;\n"
              "  float dw = fw - fw0;\n"
              "  uint base = ((n * p.C + c) * p.H_in) * p.W_in;\n"
              "  uint idx00 = base + (uint)h0 * p.W_in + (uint)w0;\n"
              "  uint idx01 = base + (uint)h0 * p.W_in + (uint)w1;\n"
              "  uint idx10 = base + (uint)h1 * p.W_in + (uint)w0;\n"
              "  uint idx11 = base + (uint)h1 * p.W_in + (uint)w1;\n"
              "  float v00 = (float)src[idx00];\n"
              "  float v01 = (float)src[idx01];\n"
              "  float v10 = (float)src[idx10];\n"
              "  float v11 = (float)src[idx11];\n"
              "  float v0 = mix(v00, v01, dw);\n"
              "  float v1 = mix(v10, v11, dw);\n"
              "  float v = mix(v0, v1, dh);\n"
              "  dst[gid] = (scalar_t)v;\n";
    }
    ss << "}\n";
    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
