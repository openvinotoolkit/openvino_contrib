// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_codegen/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace metal_plugin {

std::string generate_msl_for_gather(const GatherCodegenDesc& d, mlir::ModuleOp module) {
    std::ostringstream ss;
    std::string scalar_t = "float";
    std::string index_t = "int";
    if (auto func = get_entry_func(module)) {
        auto ft = func.getFunctionType();
        if (ft.getNumInputs() >= 1) {
            scalar_t = msl_type_from_mlir(ft.getInput(0));
        }
        if (ft.getNumInputs() >= 2) {
            index_t = msl_type_from_mlir(ft.getInput(1));
        }
    } else {
        switch (d.element_type) {
            case ov::element::f16: scalar_t = "half"; break;
            case ov::element::f32: scalar_t = "float"; break;
            case ov::element::i32: scalar_t = "int"; break;
            case ov::element::i64: scalar_t = "long"; break;
            default: break;
        }
        switch (d.index_type) {
            case ov::element::i32: index_t = "int"; break;
            case ov::element::i64: index_t = "long"; break;
            default: break;
        }
    }
    ss << "#include <metal_stdlib>\nusing namespace metal;\n";
    ss << "using scalar_t = " << scalar_t << ";\n";
    ss << "using index_t = " << index_t << ";\n";
    ss << "struct GatherParams { uint outer; uint inner; uint axis_dim; uint indices_count; };\n";
    ss << "kernel void gather_kernel(\n";
    ss << "  device const scalar_t* data [[buffer(0)]],\n";
    ss << "  device const index_t* indices [[buffer(1)]],\n";
    ss << "  device scalar_t* out [[buffer(2)]],\n";
    ss << "  constant GatherParams& p [[buffer(3)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "  uint total = p.outer * p.indices_count * p.inner;\n";
    ss << "  if (gid >= total) return;\n";
    ss << "  uint tmp = gid;\n";
    ss << "  uint inner_idx = tmp % p.inner; tmp /= p.inner;\n";
    ss << "  uint idx_idx = tmp % p.indices_count; tmp /= p.indices_count;\n";
    ss << "  uint outer_idx = tmp;\n";
    ss << "  long ix = (long)indices[idx_idx];\n";
    ss << "  if (ix < 0) ix += (long)p.axis_dim;\n";
    ss << "  if (ix < 0) ix = 0;\n";
    ss << "  if (ix >= (long)p.axis_dim) ix = (long)p.axis_dim - 1;\n";
    ss << "  uint in_idx = ((outer_idx * p.axis_dim + (uint)ix) * p.inner) + inner_idx;\n";
    ss << "  out[gid] = data[in_idx];\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov
