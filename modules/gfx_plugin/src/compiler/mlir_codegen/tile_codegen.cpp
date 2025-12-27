// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_codegen/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace gfx_plugin {

std::string generate_msl_for_tile(const TileCodegenDesc& d, mlir::ModuleOp module) {
    std::string scalar = (d.element_type == ov::element::f16) ? "half" : "float";
    if (auto func = get_entry_func(module)) {
        auto ft = func.getFunctionType();
        if (ft.getNumInputs() >= 1) {
            scalar = msl_type_from_mlir(ft.getInput(0));
        }
    }

    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "kernel void tile_kernel(\n";
    ss << "  device const " << scalar << "* in0 [[buffer(0)]],\n";
    ss << "  device " << scalar << "* out [[buffer(1)]],\n";
    ss << "  constant uint& NUM_ELEMS [[buffer(2)]],\n";
    ss << "  constant uint& RANK [[buffer(3)]],\n";
    ss << "  constant int* out_dims [[buffer(4)]],\n";
    ss << "  constant int* in_dims [[buffer(5)]],\n";
    ss << "  constant int* out_strides [[buffer(6)]],\n";
    ss << "  constant int* in_strides [[buffer(7)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    if (gid >= NUM_ELEMS) return;\n";
    ss << "    int in_index = 0;\n";
    ss << "    for (uint i = 0; i < RANK; ++i) {\n";
    ss << "      int coord = int(gid / out_strides[i]) % out_dims[i];\n";
    ss << "      int in_coord = (in_dims[i] == 0) ? 0 : (coord % in_dims[i]);\n";
    ss << "      in_index += in_coord * in_strides[i];\n";
    ss << "    }\n";
    ss << "    out[gid] = in0[in_index];\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
