// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_codegen/codegen_common.hpp"
#include "mlir_codegen/index_expr_utils.hpp"

#include <sstream>
#include <unordered_map>
#include <vector>

#include "openvino/core/except.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
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

std::string emit_conv2d_msl(const Conv2DCodegenDesc& d,
                            const std::vector<std::string>& input_idx,
                            const std::vector<std::string>& weight_idx,
                            const std::vector<std::string>& output_idx) {
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "struct Conv2DParams {\n";
    ss << "  uint N, C_in, H, W;\n";
    ss << "  uint C_out;\n";
    ss << "  uint kH, kW;\n";
    ss << "  uint strideH, strideW;\n";
    ss << "  uint dilationH, dilationW;\n";
    ss << "  uint padTop, padLeft, padBottom, padRight;\n";
    ss << "  uint outH, outW;\n";
    ss << "  uint groups;\n";
    ss << "};\n";
    ss << "kernel void conv2d_kernel(\n";
    ss << "  device const float* input  [[buffer(0)]],\n";
    ss << "  device const float* weight [[buffer(1)]],\n";
    ss << "  device float*       output [[buffer(2)]],\n";
    ss << "  constant Conv2DParams& p   [[buffer(3)]],\n";
    ss << "  uint2 gid [[thread_position_in_grid]]) {\n";
    ss << "  uint n = gid.y;\n";
    ss << "  uint oc = gid.x;\n";
    ss << "  if (n >= p.N || oc >= p.C_out) return;\n";
    ss << "  int n_i = int(n);\n";
    ss << "  int oc_i = int(oc);\n";
    ss << "  for (uint oh = 0; oh < p.outH; ++oh) {\n";
    ss << "    int oh_i = int(oh);\n";
    ss << "    for (uint ow = 0; ow < p.outW; ++ow) {\n";
    ss << "      int ow_i = int(ow);\n";
    ss << "      float acc = 0.0f;\n";
    ss << "      for (uint ic = 0; ic < p.C_in; ++ic) {\n";
    ss << "        int ic_i = int(ic);\n";
    ss << "        for (uint kh = 0; kh < p.kH; ++kh) {\n";
    ss << "          int kh_i = int(kh);\n";
    ss << "          for (uint kw = 0; kw < p.kW; ++kw) {\n";
    ss << "            int kw_i = int(kw);\n";
    ss << "            int ih = (oh_i * int(p.strideH) - int(p.padTop)  + kh_i * int(p.dilationH));\n";
    ss << "            int iw = (ow_i * int(p.strideW) - int(p.padLeft) + kw_i * int(p.dilationW));\n";
    ss << "            if (ih < 0 || iw < 0 || ih >= int(p.H) || iw >= int(p.W)) continue;\n";
    ss << "            uint in_idx = (((uint(n_i) * p.C_in + uint(ic_i)) * p.H + uint(ih)) * p.W + uint(iw));\n";
    ss << "            uint w_idx  = (((uint(oc_i) * p.C_in + uint(ic_i)) * p.kH + uint(kh_i)) * p.kW + uint(kw_i));\n";
    ss << "            acc += input[in_idx] * weight[w_idx];\n";
    ss << "          }\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "      uint out_idx = " << flatten_indices(output_idx, {"p.C_out", "p.outH", "p.outW"}) << ";\n";
    ss << "      output[out_idx] = acc;\n";
    ss << "    }\n";
    ss << "  }\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace

std::string generate_msl_for_conv2d(const Conv2DCodegenDesc& d, mlir::ModuleOp module) {
    OPENVINO_ASSERT(d.N && d.C_in && d.H && d.W && d.C_out && d.kH && d.kW, "Conv2D desc missing dims");
    OPENVINO_ASSERT(d.outH && d.outW, "Conv2D desc missing output dims");
    if (!module)
        return emit_conv2d_msl(d, {"n_i", "ic_i", "0", "0"}, {"oc_i", "ic_i", "kh_i", "kw_i"}, {"n_i", "oc_i", "oh_i", "ow_i"});

    auto func = find_func(module);
    OPENVINO_ASSERT(func, "Conv2D MLIR module does not contain a function");

    // Collect loop nest from the output store upward to skip padding/copy loops.
    mlir::memref::StoreOp output_store = nullptr;
    func.walk([&](mlir::memref::StoreOp op) {
        if (output_store)
            return;
        if (shape_matches(op.getMemRefType(), {d.N, d.C_out, d.outH, d.outW}))
            output_store = op;
    });
    OPENVINO_ASSERT(output_store, "Conv2D MLIR: failed to find output store");

    std::vector<mlir::scf::ForOp> loops;
    if (auto for_op = output_store->getParentOfType<mlir::scf::ForOp>()) {
        while (for_op) {
            loops.push_back(for_op);
            for_op = for_op->getParentOfType<mlir::scf::ForOp>();
        }
        std::reverse(loops.begin(), loops.end());
    }

    const std::vector<std::pair<std::string, uint32_t>> expected_loops = {
        {"n_i",    d.N},
        {"oc_i",   d.C_out},
        {"oh_i",   d.outH},
        {"ow_i",   d.outW},
        {"ic_i",   d.C_in},
        {"kh_i",   d.kH},
        {"kw_i",   d.kW},
    };

    auto get_trip = [](mlir::scf::ForOp op) -> std::optional<uint32_t> {
        auto lb = op.getLowerBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
        auto ub = op.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
        auto st = op.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>();
        if (!lb || !ub || !st) return std::nullopt;
        auto diff = ub.value() - lb.value();
        if (diff < 0 || st.value() <= 0) return std::nullopt;
        if (diff % st.value() != 0) return std::nullopt;
        return static_cast<uint32_t>(diff / st.value());
    };

    llvm::DenseMap<mlir::Value, std::string> names;
    std::vector<bool> used(expected_loops.size(), false);
    for (auto l : loops) {
        auto trip = get_trip(l);
        size_t chosen = expected_loops.size();
        if (trip) {
            for (size_t i = 0; i < expected_loops.size(); ++i) {
                if (!used[i] && expected_loops[i].second == *trip) {
                    chosen = i; break;
                }
            }
        }
        if (chosen == expected_loops.size()) {
            for (size_t i = 0; i < expected_loops.size(); ++i)
                if (!used[i]) { chosen = i; break; }
        }
        if (chosen < expected_loops.size()) {
            names[l.getInductionVar()] = expected_loops[chosen].first;
            used[chosen] = true;
        }
    }

    std::vector<std::string> output_idx = {"n_i", "oc_i", "oh_i", "ow_i"};

#if METAL_MLIR_DEBUG
    mlir_codegen_log("[METAL MLIR] Conv2D func=" + func.getName().str());
#endif

    return emit_conv2d_msl(d,
                           {"n_i", "ic_i", "ih_i", "iw_i"},
                           {"oc_i", "ic_i", "kh_i", "kw_i"},
                           output_idx);
}

}  // namespace metal_plugin
}  // namespace ov
