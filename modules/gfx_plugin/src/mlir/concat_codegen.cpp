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
    ss << "kernel void concat_kernel(\n";
    for (size_t i = 0; i < d.input_axis_lengths.size(); ++i) {
        ss << "  device const scalar_t* src" << i << " [[buffer(" << i << ")]],\n";
    }
    ss << "  device scalar_t* dst [[buffer(" << d.input_axis_lengths.size() << ")]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "  uint total = " << static_cast<uint32_t>(d.outer * d.axis_total * d.inner) << ";\n";
    ss << "  if (gid >= total) return;\n";
    ss << "  uint tmp = gid;\n";
    ss << "  uint outer = tmp / " << static_cast<uint32_t>(d.axis_total * d.inner) << ";\n";
    ss << "  tmp -= outer * " << static_cast<uint32_t>(d.axis_total * d.inner) << ";\n";
    ss << "  uint axis = tmp / " << static_cast<uint32_t>(d.inner) << ";\n";
    ss << "  uint inner = tmp - axis * " << static_cast<uint32_t>(d.inner) << ";\n";
    uint64_t axis_offset = 0;
    for (size_t i = 0; i < d.input_axis_lengths.size(); ++i) {
        const uint64_t axis_len = d.input_axis_lengths[i];
        ss << "  if (axis >= " << static_cast<uint32_t>(axis_offset)
           << " && axis < " << static_cast<uint32_t>(axis_offset + axis_len) << ") {\n";
        ss << "    uint local_axis = axis - " << static_cast<uint32_t>(axis_offset) << ";\n";
        ss << "    uint src_idx = ((outer * " << static_cast<uint32_t>(axis_len)
           << " + local_axis) * " << static_cast<uint32_t>(d.inner) << ") + inner;\n";
        ss << "    dst[gid] = src" << i << "[src_idx];\n";
        ss << "    return;\n";
        ss << "  }\n";
        axis_offset += axis_len;
    }
    ss << "}\n";
    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
