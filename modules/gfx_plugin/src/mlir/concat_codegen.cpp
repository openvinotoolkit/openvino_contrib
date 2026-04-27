// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace gfx_plugin {

namespace {
}  // namespace

std::string generate_msl_for_concat(const ConcatCodegenDesc& d, mlir::ModuleOp module) {
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
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "using scalar_t = " << scalar_t << ";\n";
    ss << "struct ConcatParams { uint outer; uint inner; uint axis_offset; uint axis_len; uint axis_total; };\n";
    ss << "kernel void concat_kernel(\n";
    ss << "  device const scalar_t* src [[buffer(0)]],\n";
    ss << "  device scalar_t* dst [[buffer(1)]],\n";
    ss << "  constant ConcatParams& p [[buffer(2)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "  uint total = p.outer * p.axis_len * p.inner;\n";
    ss << "  if (gid >= total) return;\n";
    ss << "  uint tmp = gid;\n";
    ss << "  uint outer = tmp / (p.axis_len * p.inner);\n";
    ss << "  tmp -= outer * p.axis_len * p.inner;\n";
    ss << "  uint axis = tmp / p.inner;\n";
    ss << "  uint inner = tmp - axis * p.inner;\n";
    ss << "  uint dst_idx = ((outer * p.axis_total + (p.axis_offset + axis)) * p.inner) + inner;\n";
    ss << "  uint src_idx = ((outer * p.axis_len + axis) * p.inner) + inner;\n";
    ss << "  dst[dst_idx] = src[src_idx];\n";
    ss << "}\n";
    return ss.str();
}

std::string generate_msl_for_concat_binary(const ConcatCodegenDesc& d, mlir::ModuleOp module) {
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
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "using scalar_t = " << scalar_t << ";\n";
    ss << "struct ConcatBinaryParams { uint outer; uint inner; uint axis0; uint axis1; uint axis_total; };\n";
    ss << "kernel void concat_binary_kernel(\n";
    ss << "  device const scalar_t* src0 [[buffer(0)]],\n";
    ss << "  device const scalar_t* src1 [[buffer(1)]],\n";
    ss << "  device scalar_t* dst [[buffer(2)]],\n";
    ss << "  constant ConcatBinaryParams& p [[buffer(3)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "  uint active_axis = p.axis0 + p.axis1;\n";
    ss << "  uint total = p.outer * active_axis * p.inner;\n";
    ss << "  if (gid >= total) return;\n";
    ss << "  uint inner = gid % p.inner;\n";
    ss << "  uint tmp = gid / p.inner;\n";
    ss << "  uint axis = tmp % active_axis;\n";
    ss << "  uint outer = tmp / active_axis;\n";
    ss << "  uint dst_idx = ((outer * p.axis_total + axis) * p.inner) + inner;\n";
    ss << "  if (axis < p.axis0) {\n";
    ss << "    uint src_idx = ((outer * p.axis0 + axis) * p.inner) + inner;\n";
    ss << "    dst[dst_idx] = src0[src_idx];\n";
    ss << "  } else {\n";
    ss << "    uint axis1 = axis - p.axis0;\n";
    ss << "    uint src_idx = ((outer * p.axis1 + axis1) * p.inner) + inner;\n";
    ss << "    dst[dst_idx] = src1[src_idx];\n";
    ss << "  }\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
