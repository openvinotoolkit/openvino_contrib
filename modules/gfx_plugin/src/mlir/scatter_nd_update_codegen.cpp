// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace gfx_plugin {

std::string generate_msl_for_scatter_nd_update(const ScatterNDUpdateCodegenDesc& d, mlir::ModuleOp module) {
    std::ostringstream ss;
    std::string scalar_t = "float";
    std::string index_t = "int";
    bool types_from_module = false;
    if (module) {
        if (auto func = get_entry_func(module)) {
            auto ft = func.getFunctionType();
            if (ft.getNumInputs() >= 1) {
                scalar_t = msl_type_from_mlir(ft.getInput(0));
            }
            if (ft.getNumInputs() >= 2) {
                index_t = msl_type_from_mlir(ft.getInput(1));
            }
            types_from_module = ft.getNumInputs() >= 2;
        }
    }
    if (!types_from_module) {
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
    ss << "struct ScatterNDParams {\n";
    ss << "  uint inner;\n";
    ss << "  uint num_indices;\n";
    ss << "  uint k;\n";
    ss << "  uint total_updates;\n";
    ss << "  uint total_data;\n";
    ss << "  uint strides[" << ScatterNDUpdateCodegenDesc::kMaxDims << "];\n";
    ss << "  uint dims[" << ScatterNDUpdateCodegenDesc::kMaxDims << "];\n";
    ss << "};\n";

    ss << "kernel void scatter_nd_update(\n";
    ss << "  device const scalar_t* data [[buffer(0)]],\n";
    ss << "  device const index_t* indices [[buffer(1)]],\n";
    ss << "  device const scalar_t* updates [[buffer(2)]],\n";
    ss << "  device scalar_t* out [[buffer(3)]],\n";
    ss << "  constant ScatterNDParams& p [[buffer(4)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "  if (gid >= p.total_data) return;\n";
    ss << "  scalar_t value = data[gid];\n";
    ss << "  if (p.inner == 0) { out[gid] = value; return; }\n";
    ss << "  for (uint linear = 0; linear < p.total_updates; ++linear) {\n";
    ss << "    uint inner_idx = linear % p.inner;\n";
    ss << "    uint index_pos = linear / p.inner;\n";
    ss << "    if (index_pos >= p.num_indices) continue;\n";
    ss << "    bool valid = true;\n";
    ss << "    uint base = 0;\n";
    ss << "    for (uint i = 0; i < p.k; ++i) {\n";
    ss << "      long dim = (long)p.dims[i];\n";
    ss << "      if (dim <= 0) { valid = false; break; }\n";
    ss << "      long idx = (long)indices[index_pos * p.k + i];\n";
    ss << "      if (idx < 0) idx += dim;\n";
    ss << "      if (idx < 0 || idx >= dim) { valid = false; break; }\n";
    ss << "      base += (uint)idx * p.strides[i];\n";
    ss << "    }\n";
    ss << "    uint out_idx = base + inner_idx;\n";
    ss << "    if (valid && out_idx == gid) value = updates[linear];\n";
    ss << "  }\n";
    ss << "  out[gid] = value;\n";
    ss << "}\n";

    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
