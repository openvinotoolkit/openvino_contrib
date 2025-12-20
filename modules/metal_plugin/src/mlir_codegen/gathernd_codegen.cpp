// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_codegen/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace metal_plugin {

std::string generate_msl_for_gathernd(const GatherNDCodegenDesc& d, mlir::ModuleOp module) {
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
    ss << "struct GatherNDParams {\n";
    ss << "  uint inner;\n";
    ss << "  uint num_indices;\n";
    ss << "  uint k;\n";
    ss << "  uint total;\n";
    ss << "  uint strides[" << GatherNDCodegenDesc::kMaxDims << "];\n";
    ss << "  uint dims[" << GatherNDCodegenDesc::kMaxDims << "];\n";
    ss << "};\n";
    ss << "kernel void gathernd_kernel(\n";
    ss << "  device const scalar_t* data [[buffer(0)]],\n";
    ss << "  device const index_t* indices [[buffer(1)]],\n";
    ss << "  device scalar_t* out [[buffer(2)]],\n";
    ss << "  constant GatherNDParams& p [[buffer(3)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "  if (gid >= p.total) return;\n";
    ss << "  uint inner_idx = gid % p.inner;\n";
    ss << "  uint index_pos = gid / p.inner;\n";
    ss << "  uint base = 0;\n";
    ss << "  for (uint i = 0; i < p.k; ++i) {\n";
    ss << "    long dim = (long)p.dims[i];\n";
    ss << "    if (dim <= 0) return;\n";
    ss << "    long idx = (long)indices[index_pos * p.k + i];\n";
    ss << "    if (idx < 0) idx += dim;\n";
    ss << "    if (idx < 0) idx = 0;\n";
    ss << "    if (idx >= dim) idx = dim - 1;\n";
    ss << "    base += (uint)idx * p.strides[i];\n";
    ss << "  }\n";
    ss << "  uint in_idx = base + inner_idx;\n";
    ss << "  out[gid] = data[in_idx];\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov
