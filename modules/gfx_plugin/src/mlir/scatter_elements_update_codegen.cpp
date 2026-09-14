// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace gfx_plugin {

std::string generate_msl_for_scatter_elements_update(const ScatterElementsUpdateCodegenDesc& d, mlir::ModuleOp module) {
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
    ss << "struct ScatterElementsParams {\n";
    ss << "  uint rank;\n";
    ss << "  uint axis;\n";
    ss << "  uint total_updates;\n";
    ss << "  uint total_data;\n";
    ss << "  uint update_dims[" << ScatterElementsUpdateCodegenDesc::kMaxDims << "];\n";
    ss << "  uint update_strides[" << ScatterElementsUpdateCodegenDesc::kMaxDims << "];\n";
    ss << "  uint data_dims[" << ScatterElementsUpdateCodegenDesc::kMaxDims << "];\n";
    ss << "  uint data_strides[" << ScatterElementsUpdateCodegenDesc::kMaxDims << "];\n";
    ss << "};\n";

    ss << "kernel void scatter_elements_update(\n";
    ss << "  device const scalar_t* data [[buffer(0)]],\n";
    ss << "  device const index_t* indices [[buffer(1)]],\n";
    ss << "  device const scalar_t* updates [[buffer(2)]],\n";
    ss << "  device scalar_t* out [[buffer(3)]],\n";
    ss << "  constant ScatterElementsParams& p [[buffer(4)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "  if (gid >= p.total_data) return;\n";
    ss << "  scalar_t value = data[gid];\n";
    ss << "  for (uint linear = 0; linear < p.total_updates; ++linear) {\n";
    ss << "    bool valid = true;\n";
    ss << "    uint out_idx = 0;\n";
    ss << "    for (uint i = 0; i < p.rank; ++i) {\n";
    ss << "      uint stride = p.update_strides[i];\n";
    ss << "      uint dim = p.update_dims[i];\n";
    ss << "      uint c = stride ? (linear / stride) % dim : 0;\n";
    ss << "      if (i == p.axis) {\n";
    ss << "        long ix = (long)indices[linear];\n";
    ss << "        long axis_dim = (long)p.data_dims[p.axis];\n";
    ss << "        if (axis_dim <= 0) { valid = false; break; }\n";
    ss << "        if (ix < 0) ix += axis_dim;\n";
    ss << "        if (ix < 0 || ix >= axis_dim) { valid = false; break; }\n";
    ss << "        c = (uint)ix;\n";
    ss << "      }\n";
    ss << "      out_idx += c * p.data_strides[i];\n";
    ss << "    }\n";
    ss << "    if (valid && out_idx == gid) value = updates[linear];\n";
    ss << "  }\n";
    ss << "  out[gid] = value;\n";
    ss << "}\n";

    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
