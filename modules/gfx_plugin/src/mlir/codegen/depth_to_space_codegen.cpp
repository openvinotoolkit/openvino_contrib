// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace gfx_plugin {

std::string generate_msl_for_depth_to_space(const DepthToSpaceCodegenDesc& d, mlir::ModuleOp module) {
    std::ostringstream ss;
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
            case ov::element::i32: scalar_t = "int"; break;
            case ov::element::i64: scalar_t = "long"; break;
            default: break;
        }
    }

    ss << "#include <metal_stdlib>\nusing namespace metal;\n";
    ss << "using scalar_t = " << scalar_t << ";\n";
    ss << "struct DepthToSpaceParams {\n";
    ss << "  uint N; uint C; uint H; uint W;\n";
    ss << "  uint C_out; uint H_out; uint W_out;\n";
    ss << "  uint block; uint mode; uint total;\n";
    ss << "};\n";
    ss << "kernel void depth_to_space_kernel(\n";
    ss << "  device const scalar_t* data [[buffer(0)]],\n";
    ss << "  device scalar_t* out [[buffer(1)]],\n";
    ss << "  constant DepthToSpaceParams& p [[buffer(2)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "  if (gid >= p.total) return;\n";
    ss << "  uint w_out = gid % p.W_out;\n";
    ss << "  uint tmp = gid / p.W_out;\n";
    ss << "  uint h_out = tmp % p.H_out;\n";
    ss << "  tmp = tmp / p.H_out;\n";
    ss << "  uint c_out = tmp % p.C_out;\n";
    ss << "  uint n = tmp / p.C_out;\n";
    ss << "  uint h_in = h_out / p.block;\n";
    ss << "  uint w_in = w_out / p.block;\n";
    ss << "  uint off_h = h_out - h_in * p.block;\n";
    ss << "  uint off_w = w_out - w_in * p.block;\n";
    ss << "  uint c_in = 0;\n";
    ss << "  if (p.mode == 0) {\n";
    ss << "    c_in = (off_h * p.block + off_w) * p.C_out + c_out;\n";
    ss << "  } else {\n";
    ss << "    c_in = c_out * p.block * p.block + off_h * p.block + off_w;\n";
    ss << "  }\n";
    ss << "  uint in_idx = ((n * p.C + c_in) * p.H + h_in) * p.W + w_in;\n";
    ss << "  out[gid] = data[in_idx];\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
