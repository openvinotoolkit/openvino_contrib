// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_codegen/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace metal_plugin {

std::string generate_msl_for_gather_elements(const GatherElementsCodegenDesc& d, mlir::ModuleOp module) {
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
    ss << "struct GatherElementsParams {\n";
    ss << "  uint rank;\n";
    ss << "  uint axis;\n";
    ss << "  uint total;\n";
    ss << "  uint out_dims[" << GatherElementsCodegenDesc::kMaxDims << "];\n";
    ss << "  uint out_strides[" << GatherElementsCodegenDesc::kMaxDims << "];\n";
    ss << "  uint data_dims[" << GatherElementsCodegenDesc::kMaxDims << "];\n";
    ss << "  uint data_strides[" << GatherElementsCodegenDesc::kMaxDims << "];\n";
    ss << "};\n";
    ss << "kernel void gather_elements_kernel(\n";
    ss << "  device const scalar_t* data [[buffer(0)]],\n";
    ss << "  device const index_t* indices [[buffer(1)]],\n";
    ss << "  device scalar_t* out [[buffer(2)]],\n";
    ss << "  constant GatherElementsParams& p [[buffer(3)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "  if (gid >= p.total) return;\n";
    ss << "  uint coord[" << GatherElementsCodegenDesc::kMaxDims << "];\n";
    ss << "  for (uint i = 0; i < p.rank; ++i) {\n";
    ss << "    uint stride = p.out_strides[i];\n";
    ss << "    uint dim = p.out_dims[i];\n";
    ss << "    uint c = stride ? (gid / stride) % dim : 0;\n";
    ss << "    coord[i] = c;\n";
    ss << "  }\n";
    ss << "  long ix = (long)indices[gid];\n";
    ss << "  uint axis = p.axis;\n";
    ss << "  long axis_dim = (long)p.data_dims[axis];\n";
    ss << "  if (axis_dim <= 0) return;\n";
    ss << "  if (ix < 0) ix += axis_dim;\n";
    ss << "  if (ix < 0) ix = 0;\n";
    ss << "  if (ix >= axis_dim) ix = axis_dim - 1;\n";
    ss << "  uint in_idx = 0;\n";
    ss << "  for (uint i = 0; i < p.rank; ++i) {\n";
    ss << "    uint c = (i == axis) ? (uint)ix : coord[i];\n";
    ss << "    in_idx += c * p.data_strides[i];\n";
    ss << "  }\n";
    ss << "  out[gid] = data[in_idx];\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov
