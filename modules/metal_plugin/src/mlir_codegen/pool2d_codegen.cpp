// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_codegen/codegen_common.hpp"
#include "mlir_codegen/index_expr_utils.hpp"

#include <sstream>
#include <unordered_set>
#include <vector>

#include "openvino/core/except.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace ov {
namespace metal_plugin {
namespace {

mlir::func::FuncOp find_func(mlir::ModuleOp module) {
    for (auto f : module.getOps<mlir::func::FuncOp>())
        return f;
    return nullptr;
}

std::vector<mlir::scf::ForOp> collect_loop_chain(mlir::func::FuncOp func) {
    std::vector<mlir::scf::ForOp> loops;
    mlir::scf::ForOp outer = nullptr;
    func.walk([&](mlir::scf::ForOp op) {
        if (!outer)
            outer = op;
    });
    if (!outer)
        return loops;
    auto cur = outer;
    while (cur) {
        loops.push_back(cur);
        auto inner = cur.getBody()->getOps<mlir::scf::ForOp>();
        if (inner.empty())
            break;
        cur = *inner.begin();
    }
    return loops;
}

bool shape_matches(mlir::MemRefType ty, std::initializer_list<uint32_t> dims) {
    if (!ty || ty.getRank() != static_cast<int>(dims.size()))
        return false;
    auto shape = ty.getShape();
    size_t idx = 0;
    for (auto d : dims) {
        const auto v = shape[idx++];
        if (v == mlir::ShapedType::kDynamic)
            continue;
        if (static_cast<uint32_t>(v) != d)
            return false;
    }
    return true;
}

std::vector<std::string> render_indices(mlir::Operation::operand_range range,
                                        const llvm::DenseMap<mlir::Value, std::string>& names) {
    std::vector<std::string> out;
    out.reserve(range.size());
    for (auto v : range)
        out.push_back(render_index_expr(v, names));
    return out;
}

std::string emit_pool2d_msl(const Pool2DCodegenDesc& d,
                            const std::vector<std::string>& input_idx,
                            const std::vector<std::string>& output_idx) {
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "struct Pool2DParams {\n";
    ss << "  uint N, C, H, W;\n";
    ss << "  uint kH, kW;\n";
    ss << "  uint strideH, strideW;\n";
    ss << "  uint padTop, padLeft, padBottom, padRight;\n";
    ss << "  uint outH, outW;\n";
    ss << "  bool is_avg;\n";
    ss << "  bool exclude_pad;\n";
    ss << "};\n";
    ss << "kernel void pool2d_kernel(\n";
    ss << "  device const float* input  [[buffer(0)]],\n";
    ss << "  device float*       output [[buffer(1)]],\n";
    ss << "  constant Pool2DParams& p   [[buffer(2)]],\n";
    ss << "  uint gid_x [[thread_position_in_grid.x]],\n";
    ss << "  uint gid_y [[thread_position_in_grid.y]]) {\n";
    ss << "  uint n = gid_y;\n";
    ss << "  uint c = gid_x;\n";
    ss << "  if (n >= p.N || c >= p.C) return;\n";
    ss << "  int n_i = int(n);\n";
    ss << "  int c_i = int(c);\n";
    ss << "  for (uint oh = 0; oh < p.outH; ++oh) {\n";
    ss << "    int oh_i = int(oh);\n";
    ss << "    for (uint ow = 0; ow < p.outW; ++ow) {\n";
    ss << "      int ow_i = int(ow);\n";
    ss << "      float acc = p.is_avg ? 0.0f : -INFINITY;\n";
    ss << "      uint count = 0;\n";
    ss << "      for (uint kh = 0; kh < p.kH; ++kh) {\n";
    ss << "        int kh_i = int(kh);\n";
    ss << "        for (uint kw = 0; kw < p.kW; ++kw) {\n";
    ss << "          int kw_i = int(kw);\n";
    ss << "          int ih = " << input_idx[2] << ";\n";
    ss << "          int iw = " << input_idx[3] << ";\n";
    ss << "          if (ih < 0 || iw < 0 || ih >= int(p.H) || iw >= int(p.W)) {\n";
    ss << "            if (p.is_avg && !p.exclude_pad) { count++; }\n";
    ss << "            continue;\n";
    ss << "          }\n";
    ss << "          uint idx = " << flatten_indices(input_idx, {"p.C", "p.H", "p.W"}) << ";\n";
    ss << "          float v = input[idx];\n";
    ss << "          if (p.is_avg) {\n";
    ss << "            acc += v;\n";
    ss << "            count++;\n";
    ss << "          } else {\n";
    ss << "            acc = acc > v ? acc : v;\n";
    ss << "          }\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "      if (p.is_avg) {\n";
    ss << "        if (count == 0) count = 1;\n";
    ss << "        acc = acc / float(count);\n";
    ss << "      }\n";
    ss << "      uint out_idx = " << flatten_indices(output_idx, {"p.C", "p.outH", "p.outW"}) << ";\n";
    ss << "      output[out_idx] = acc;\n";
    ss << "    }\n";
    ss << "  }\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace

std::string generate_msl_for_pool2d(const Pool2DCodegenDesc& d, mlir::ModuleOp module) {
    OPENVINO_ASSERT(d.N && d.C && d.H && d.W && d.kH && d.kW && d.outH && d.outW, "Pool2D desc incomplete");
    if (!module)
        return emit_pool2d_msl(d, {"n_i", "c_i", "oh_i", "ow_i"}, {"n_i", "c_i", "oh_i", "ow_i"});

    auto func = find_func(module);
    OPENVINO_ASSERT(func, "Pool2D MLIR: function not found");

    auto loops = collect_loop_chain(func);
    OPENVINO_ASSERT(loops.size() >= 6, "Pool2D MLIR: expected 6-level loop nest n-c-oh-ow-kh-kw");

    const std::vector<std::string> loop_names = {"n_i", "c_i", "oh_i", "ow_i", "kh_i", "kw_i"};
    llvm::DenseMap<mlir::Value, std::string> names;
    for (size_t i = 0; i < loop_names.size(); ++i)
        names[loops[i].getInductionVar()] = loop_names[i];

    mlir::memref::LoadOp input_load = nullptr;
    mlir::memref::StoreOp output_store = nullptr;

    func.walk([&](mlir::memref::LoadOp op) {
        if (input_load)
            return;
        if (shape_matches(op.getMemRefType(), {d.N, d.C, d.H, d.W}))
            input_load = op;
    });
    func.walk([&](mlir::memref::StoreOp op) {
        if (output_store)
            return;
        if (shape_matches(op.getMemRefType(), {d.N, d.C, d.outH, d.outW}))
            output_store = op;
    });

    OPENVINO_ASSERT(input_load && output_store, "Pool2D MLIR: failed to find load/store");

    auto input_idx = render_indices(input_load.getIndices(), names);
    auto output_idx = render_indices(output_store.getIndices(), names);
    OPENVINO_ASSERT(input_idx.size() == 4 && output_idx.size() == 4, "Pool2D MLIR: unexpected index rank");

#if METAL_MLIR_DEBUG
    auto join = [](const std::vector<std::string>& v) {
        std::string s;
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) s += ", ";
            s += v[i];
        }
        return s;
    };
    mlir_codegen_log("[METAL MLIR] Pool2D func=" + func.getName().str());
    mlir_codegen_log("  input idx:  [" + join(input_idx) + "]");
    mlir_codegen_log("  output idx: [" + join(output_idx) + "]");
#endif

    return emit_pool2d_msl(d, input_idx, output_idx);
}

}  // namespace metal_plugin
}  // namespace ov
