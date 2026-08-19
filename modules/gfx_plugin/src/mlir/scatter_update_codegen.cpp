// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace gfx_plugin {

std::string generate_msl_for_scatter_update(const ScatterUpdateCodegenDesc& d, mlir::ModuleOp module) {
    std::string scalar_t = msl_type_from_element(d.element_type == ov::element::dynamic ? ov::element::f32
                                                                                       : d.element_type);
    std::string index_t = msl_type_from_element(d.index_type == ov::element::i64 ? ov::element::i64
                                                                                 : ov::element::i32);
    if (auto func = get_entry_func(module)) {
        auto ft = func.getFunctionType();
        if (ft.getNumInputs() >= 1) {
            scalar_t = msl_type_from_mlir(ft.getInput(0));
        }
        if (ft.getNumInputs() >= 2) {
            index_t = msl_type_from_mlir(ft.getInput(1));
        }
    }

    std::ostringstream ss;
    ss << "#include <metal_stdlib>\nusing namespace metal;\n";
    ss << "using scalar_t = " << scalar_t << ";\n";
    ss << "using index_t = " << index_t << ";\n";
    ss << "struct ScatterUpdateParams {\n";
    ss << "  uint data_rank;\n";
    ss << "  uint idx_rank;\n";
    ss << "  uint update_rank;\n";
    ss << "  uint axis;\n";
    ss << "  uint total_data;\n";
    ss << "  uint idx_total;\n";
    ss << "  uint data_dims[8];\n";
    ss << "  uint data_strides[8];\n";
    ss << "  uint idx_dims[8];\n";
    ss << "  uint idx_strides[8];\n";
    ss << "  uint update_strides[16];\n";
    ss << "};\n";
    ss << "kernel void scatter_update_kernel(\n";
    ss << "  device const scalar_t* data [[buffer(0)]],\n";
    ss << "  device const index_t* indices [[buffer(1)]],\n";
    ss << "  device const scalar_t* updates [[buffer(2)]],\n";
    ss << "  device scalar_t* out [[buffer(3)]],\n";
    ss << "  constant ScatterUpdateParams& p [[buffer(4)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "  if (gid >= p.total_data) return;\n";
    ss << "  uint coord[8];\n";
    ss << "  uint rem = gid;\n";
    ss << "  for (uint d = 0; d < p.data_rank; ++d) {\n";
    ss << "    uint stride = p.data_strides[d];\n";
    ss << "    coord[d] = stride == 0 ? 0 : rem / stride;\n";
    ss << "    rem = stride == 0 ? 0 : rem - coord[d] * stride;\n";
    ss << "  }\n";
    ss << "  scalar_t value = data[gid];\n";
    ss << "  for (uint linear = 0; linear < p.idx_total; ++linear) {\n";
    ss << "    uint idx_coord[8];\n";
    ss << "    uint idx_rem = linear;\n";
    ss << "    for (uint d = 0; d < p.idx_rank; ++d) {\n";
    ss << "      uint stride = p.idx_strides[d];\n";
    ss << "      idx_coord[d] = stride == 0 ? 0 : idx_rem / stride;\n";
    ss << "      idx_rem = stride == 0 ? 0 : idx_rem - idx_coord[d] * stride;\n";
    ss << "    }\n";
    ss << "    long raw = static_cast<long>(indices[linear]);\n";
    ss << "    long axis_dim = static_cast<long>(p.data_dims[p.axis]);\n";
    ss << "    long normalized = raw < 0 ? raw + axis_dim : raw;\n";
    ss << "    if (normalized != static_cast<long>(coord[p.axis])) continue;\n";
    ss << "    uint upd_off = 0;\n";
    ss << "    uint upd_dim = 0;\n";
    ss << "    for (uint d = 0; d < p.axis; ++d) upd_off += coord[d] * p.update_strides[upd_dim++];\n";
    ss << "    for (uint d = 0; d < p.idx_rank; ++d) upd_off += idx_coord[d] * p.update_strides[upd_dim++];\n";
    ss << "    for (uint d = p.axis + 1; d < p.data_rank; ++d) upd_off += coord[d] * p.update_strides[upd_dim++];\n";
    ss << "    value = updates[upd_off];\n";
    ss << "  }\n";
    ss << "  out[gid] = value;\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
