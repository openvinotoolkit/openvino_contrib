// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace gfx_plugin {

std::string generate_msl_for_reduce(const ReduceCodegenDesc& d, mlir::ModuleOp module) {
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
    const bool is_int = (scalar_t == "int" || scalar_t == "long");
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "using scalar_t = " << scalar_t << ";\n";
    ss << "kernel void reduce_kernel(device const scalar_t* A [[buffer(0)]],\n";
    ss << "                           device scalar_t* O [[buffer(1)]],\n";
    ss << "                           constant uint& NUM_ELEMS [[buffer(2)]],\n";
    ss << "                           constant uint& RANK [[buffer(3)]],\n";
    ss << "                           constant int* OUT_DIMS [[buffer(4)]],\n";
    ss << "                           constant int* IN_DIMS [[buffer(5)]],\n";
    ss << "                           constant int* IN_STRIDES [[buffer(6)]],\n";
    ss << "                           constant int* AXIS_MASK [[buffer(7)]],\n";
    ss << "                           constant int* REDUCE_DIMS [[buffer(8)]],\n";
    ss << "                           uint gid [[thread_position_in_grid]]) {\n";
    ss << "    if (gid >= NUM_ELEMS) return;\n";
    ss << "    uint idx = gid;\n";
    ss << "    int coords[8]; // support up to rank 8 for now\n";
    ss << "    for (uint d = RANK; d-- > 0;) {\n";
    ss << "        coords[d] = idx % OUT_DIMS[d];\n";
    ss << "        idx /= OUT_DIMS[d];\n";
    ss << "    }\n";
    ss << "    uint reduce_size = 1;\n";
    ss << "    for (uint d = 0; d < RANK; ++d) reduce_size *= (AXIS_MASK[d] ? (uint)REDUCE_DIMS[d] : 1u);\n";
    if (d.kind == ReduceKind::Prod) {
        ss << "    scalar_t acc = static_cast<scalar_t>(1);\n";
    } else if (d.kind == ReduceKind::Max || d.kind == ReduceKind::Min) {
        ss << "    scalar_t acc = 0;\n";
        ss << "    bool first = true;\n";
    } else {
        ss << "    scalar_t acc = 0;\n";
    }
    ss << "    for (uint r = 0; r < reduce_size; ++r) {\n";
    ss << "        uint tmp = r;\n";
    ss << "        int in_idx = 0;\n";
    ss << "        for (uint d = RANK; d-- > 0;) {\n";
    ss << "            int coord = coords[d];\n";
    ss << "            if (AXIS_MASK[d]) {\n";
    ss << "                int rd = tmp % REDUCE_DIMS[d];\n";
    ss << "                tmp /= REDUCE_DIMS[d];\n";
    ss << "                coord = rd;\n";
    ss << "            }\n";
    ss << "            in_idx += coord * IN_STRIDES[d];\n";
    ss << "        }\n";
    ss << "        scalar_t v = A[in_idx];\n";
    if (d.kind == ReduceKind::Max) {
        ss << "        if (first) { acc = v; first = false; } else { acc = (v > acc ? v : acc); }\n";
    } else if (d.kind == ReduceKind::Min) {
        ss << "        if (first) { acc = v; first = false; } else { acc = (v < acc ? v : acc); }\n";
    } else if (d.kind == ReduceKind::Prod) {
        ss << "        acc *= v;\n";
    } else if (d.kind == ReduceKind::L1) {
        if (is_int) {
            ss << "        acc += abs(v);\n";
        } else {
            ss << "        acc += fabs(v);\n";
        }
    } else if (d.kind == ReduceKind::L2) {
        ss << "        acc += v * v;\n";
    } else {
        ss << "        acc += v;\n";
    }
    ss << "    }\n";
    if (d.kind == ReduceKind::Mean) {
        ss << "    acc = acc / static_cast<scalar_t>(reduce_size);\n";
    } else if (d.kind == ReduceKind::L2) {
        ss << "    acc = sqrt(acc);\n";
    }
    ss << "    O[gid] = acc;\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
