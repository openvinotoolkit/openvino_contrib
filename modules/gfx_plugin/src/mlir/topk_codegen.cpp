// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace gfx_plugin {

std::string generate_msl_for_topk(const TopKCodegenDesc& d, mlir::ModuleOp module) {
    std::string val_scalar = "float";
    std::string idx_scalar = d.index_type == ov::element::i64 ? "long" : "int";
    if (auto func = get_entry_func(module)) {
        auto ft = func.getFunctionType();
        if (ft.getNumResults() >= 1) {
            val_scalar = msl_type_from_mlir(ft.getResult(0));
        }
        if (ft.getNumResults() >= 2) {
            idx_scalar = msl_type_from_mlir(ft.getResult(1));
        }
    }

    const bool sort_indices = d.sort_type == TopKSortType::SortIndices;
    const std::string init_val = d.mode_max ? "-INFINITY" : "INFINITY";
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "constexpr uint kTopK = " << d.k << ";\n";
    ss << "constexpr uint kAxis = " << d.axis_len << ";\n";
    ss << "constexpr uint kInner = " << (d.inner == 0 ? 1u : d.inner) << ";\n";
    ss << "constexpr uint kOuter = " << (d.outer == 0 ? 1u : d.outer) << ";\n";
    ss << "kernel void topk_kernel(\n";
    ss << "  device const " << val_scalar << "* in0 [[buffer(0)]],\n";
    ss << "  device " << val_scalar << "* out_vals [[buffer(1)]],\n";
    ss << "  device " << idx_scalar << "* out_idx [[buffer(2)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    const uint rows = kOuter * kInner;\n";
    ss << "    if (gid >= rows) return;\n";
    ss << "    const uint outer = gid / kInner;\n";
    ss << "    const uint inner = gid - outer * kInner;\n";
    ss << "    float top_vals[kTopK];\n";
    ss << "    int top_ids[kTopK];\n";
    ss << "    for (uint i = 0; i < kTopK; ++i) { top_vals[i] = " << init_val << "; top_ids[i] = 0; }\n";
    ss << "    for (uint a = 0; a < kAxis; ++a) {\n";
    ss << "      const uint idx = (outer * kAxis + a) * kInner + inner;\n";
    ss << "      const float v = (float)in0[idx];\n";
    ss << "      uint insert = kTopK;\n";
    if (d.mode_max) {
        ss << "      for (uint i = 0; i < kTopK; ++i) { if (v > top_vals[i]) { insert = i; break; } }\n";
    } else {
        ss << "      for (uint i = 0; i < kTopK; ++i) { if (v < top_vals[i]) { insert = i; break; } }\n";
    }
    ss << "      if (insert < kTopK) {\n";
    ss << "        for (uint j = kTopK - 1; j > insert; --j) { top_vals[j] = top_vals[j - 1]; top_ids[j] = top_ids[j - 1]; }\n";
    ss << "        top_vals[insert] = v;\n";
    ss << "        top_ids[insert] = (int)a;\n";
    ss << "      }\n";
    ss << "    }\n";
    if (sort_indices) {
        ss << "    for (uint i = 0; i < kTopK; ++i) {\n";
        ss << "      for (uint j = i + 1; j < kTopK; ++j) {\n";
        ss << "        if (top_ids[j] < top_ids[i]) {\n";
        ss << "          int ti = top_ids[i]; float tv = top_vals[i];\n";
        ss << "          top_ids[i] = top_ids[j]; top_vals[i] = top_vals[j];\n";
        ss << "          top_ids[j] = ti; top_vals[j] = tv;\n";
        ss << "        }\n";
        ss << "      }\n";
        ss << "    }\n";
    }
    ss << "    for (uint k = 0; k < kTopK; ++k) {\n";
    ss << "      const uint out_idx_flat = (outer * kTopK + k) * kInner + inner;\n";
    ss << "      out_vals[out_idx_flat] = (" << val_scalar << ")top_vals[k];\n";
    ss << "      out_idx[out_idx_flat] = (" << idx_scalar << ")top_ids[k];\n";
    ss << "    }\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
