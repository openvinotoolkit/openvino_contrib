// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_codegen/codegen_common.hpp"

#include <sstream>

#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

std::string generate_msl_for_batchnorm2d(const BatchNorm2DCodegenDesc& d, mlir::ModuleOp module) {
    OPENVINO_ASSERT(d.N && d.C && d.H && d.W, "BatchNorm2D desc missing dims");
    std::string scalar = "float";
    if (auto func = get_entry_func(module)) {
        auto ft = func.getFunctionType();
        if (ft.getNumInputs() >= 1) {
            scalar = msl_type_from_mlir(ft.getInput(0));
        }
    } else {
        scalar = (d.element_type == ov::element::f16) ? "half" : "float";
    }
    const bool use_half = (scalar == "half");

    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "struct BNParams { uint N, C, H, W; };\n";
    ss << "kernel void batchnorm2d_kernel(\n";
    ss << "  device const " << scalar << "* in0    [[buffer(0)]],\n";
    ss << "  device const float* params [[buffer(1)]],\n";  // params kept in f32
    ss << "  device " << scalar << "* out          [[buffer(2)]],\n";
    ss << "  constant BNParams& p       [[buffer(3)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    uint total = p.N * p.C * p.H * p.W;\n";
    ss << "    if (gid >= total) return;\n";
    ss << "    uint tmp = gid;\n";
    ss << "    uint w = tmp % p.W; tmp /= p.W;\n";
    ss << "    uint h = tmp % p.H; tmp /= p.H;\n";
    ss << "    uint c = tmp % p.C; tmp /= p.C;\n";
    ss << "    uint n = tmp;\n";
    ss << "    float g = params[c];\n";
    ss << "    float b = params[p.C + c];\n";
    ss << "    float m = params[2 * p.C + c];\n";
    ss << "    float v = params[3 * p.C + c];\n";
    ss << "    float eps = params[4 * p.C];\n";
    ss << "    float inv_std = rsqrt(v + eps);\n";
    ss << "    uint idx = ((n * p.C + c) * p.H + h) * p.W + w;\n";
    ss << "    float x = static_cast<float>(in0[idx]);\n";
    ss << "    float y = g * (x - m) * inv_std + b;\n";
    if (use_half) {
        ss << "    out[idx] = static_cast<" << scalar << ">(y);\n";
    } else {
        ss << "    out[idx] = y;\n";
    }
    ss << "}\n";
    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
