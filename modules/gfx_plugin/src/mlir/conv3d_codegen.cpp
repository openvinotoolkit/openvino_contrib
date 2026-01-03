// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen_common.hpp"
#include "mlir/index_expr_utils.hpp"

#include <sstream>
#include <vector>

#include "openvino/core/except.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace ov {
namespace gfx_plugin {
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

std::string emit_conv3d_msl(const Conv3DCodegenDesc& d,
                            const std::string& scalar,
                            const std::vector<std::string>& input_idx,
                            const std::vector<std::string>& weight_idx,
                            const std::vector<std::string>& output_idx) {
    const bool use_half = (scalar == "half");
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "struct Conv3DParams {\n";
    ss << "  uint N, C_in, D, H, W;\n";
    ss << "  uint C_out;\n";
    ss << "  uint kD, kH, kW;\n";
    ss << "  uint strideD, strideH, strideW;\n";
    ss << "  uint dilationD, dilationH, dilationW;\n";
    ss << "  uint padFront, padTop, padLeft, padBack, padBottom, padRight;\n";
    ss << "  uint outD, outH, outW;\n";
    ss << "};\n";
    ss << "kernel void conv3d_kernel(\n";
    ss << "  device const " << scalar << "* input  [[buffer(0)]],\n";
    ss << "  device const " << scalar << "* weight [[buffer(1)]],\n";
    ss << "  device " << scalar << "*       output [[buffer(2)]],\n";
    ss << "  constant Conv3DParams& p   [[buffer(3)]],\n";
    ss << "  uint2 gid [[thread_position_in_grid]]) {\n";
    ss << "  uint n = gid.y;\n";
    ss << "  uint oc = gid.x;\n";
    ss << "  if (n >= p.N || oc >= p.C_out) return;\n";
    ss << "  int n_i = int(n);\n";
    ss << "  int oc_i = int(oc);\n";
    ss << "  for (uint od = 0; od < p.outD; ++od) {\n";
    ss << "    int od_i = int(od);\n";
    ss << "    for (uint oh = 0; oh < p.outH; ++oh) {\n";
    ss << "      int oh_i = int(oh);\n";
    ss << "      for (uint ow = 0; ow < p.outW; ++ow) {\n";
    ss << "        int ow_i = int(ow);\n";
    ss << "        float acc = 0.0f;\n";
    ss << "        for (uint ic = 0; ic < p.C_in; ++ic) {\n";
    ss << "          int ic_i = int(ic);\n";
    ss << "          for (uint kd = 0; kd < p.kD; ++kd) {\n";
    ss << "            int kd_i = int(kd);\n";
    ss << "            for (uint kh = 0; kh < p.kH; ++kh) {\n";
    ss << "              int kh_i = int(kh);\n";
    ss << "              for (uint kw = 0; kw < p.kW; ++kw) {\n";
    ss << "                int kw_i = int(kw);\n";
    ss << "                int id = (od_i * int(p.strideD) - int(p.padFront) + kd_i * int(p.dilationD));\n";
    ss << "                int ih = (oh_i * int(p.strideH) - int(p.padTop)  + kh_i * int(p.dilationH));\n";
    ss << "                int iw = (ow_i * int(p.strideW) - int(p.padLeft) + kw_i * int(p.dilationW));\n";
    ss << "                if (id < 0 || ih < 0 || iw < 0 || id >= int(p.D) || ih >= int(p.H) || iw >= int(p.W)) continue;\n";
    ss << "                uint in_idx = (((uint(n_i) * p.C_in + uint(ic_i)) * p.D + uint(id)) * p.H + uint(ih)) * p.W + uint(iw);\n";
    ss << "                uint w_idx  = (((uint(oc_i) * p.C_in + uint(ic_i)) * p.kD + uint(kd_i)) * p.kH + uint(kh_i)) * p.kW + uint(kw_i);\n";
    ss << "                acc += static_cast<float>(input[in_idx]) * static_cast<float>(weight[w_idx]);\n";
    ss << "              }\n";
    ss << "            }\n";
    ss << "          }\n";
    ss << "        }\n";
    ss << "        uint out_idx = " << flatten_indices(output_idx, {"p.C_out", "p.outD", "p.outH", "p.outW"}) << ";\n";
    if (use_half) {
        ss << "        output[out_idx] = static_cast<" << scalar << ">(acc);\n";
    } else {
        ss << "        output[out_idx] = acc;\n";
    }
    ss << "      }\n";
    ss << "    }\n";
    ss << "  }\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace

std::string generate_msl_for_conv3d(const Conv3DCodegenDesc& d, mlir::ModuleOp module) {
    OPENVINO_ASSERT(d.N && d.C_in && d.D && d.H && d.W && d.C_out && d.kD && d.kH && d.kW, "Conv3D desc missing dims");
    OPENVINO_ASSERT(d.outD && d.outH && d.outW, "Conv3D desc missing output dims");
    std::string scalar = msl_type_from_element(d.element_type);
    if (scalar.empty())
        scalar = "float";
    auto fallback = [&]() {
        return emit_conv3d_msl(d, scalar,
                               {"n_i", "ic_i", "0", "0", "0"},
                               {"oc_i", "ic_i", "kd_i", "kh_i", "kw_i"},
                               {"n_i", "oc_i", "od_i", "oh_i", "ow_i"});
    };
    if (!module) {
        OPENVINO_THROW("Conv3D MLIR module is null");
    }

    auto func = find_func(module);
    OPENVINO_ASSERT(func, "Conv3D MLIR module does not contain a function");

    mlir::memref::LoadOp input_load = nullptr, weight_load = nullptr;
    mlir::memref::StoreOp output_store = nullptr;

    const std::initializer_list<uint32_t> input_shape = {d.N, d.C_in, d.D, d.H, d.W};
    const std::initializer_list<uint32_t> padded_input_shape = {
        d.N,
        d.C_in,
        d.D + d.padFront + d.padBack,
        d.H + d.padTop + d.padBottom,
        d.W + d.padLeft + d.padRight,
    };

    func.walk([&](mlir::memref::LoadOp op) {
        if (input_load && weight_load)
            return;
        auto ty = op.getMemRefType();
        if (!input_load && (shape_matches(ty, input_shape) || shape_matches(ty, padded_input_shape)))
            input_load = op;
        else if (!weight_load && shape_matches(ty, {d.C_out, d.C_in, d.kD, d.kH, d.kW}))
            weight_load = op;
    });

    func.walk([&](mlir::memref::StoreOp op) {
        if (output_store)
            return;
        if (shape_matches(op.getMemRefType(), {d.N, d.C_out, d.outD, d.outH, d.outW}))
            output_store = op;
    });

    if (!input_load || !weight_load || !output_store) {
        return fallback();
    }

    // Reconstruct loop chain based on the output store location to skip padding/copy loops.
    std::vector<mlir::scf::ForOp> loops;
    if (auto for_op = output_store->getParentOfType<mlir::scf::ForOp>()) {
        while (for_op) {
            loops.push_back(for_op);
            for_op = for_op->getParentOfType<mlir::scf::ForOp>();
        }
        std::reverse(loops.begin(), loops.end());
    }
    if (loops.size() < 9) {
        mlir_codegen_log("[GFX MLIR] Conv3D: loop depth " + std::to_string(loops.size()) + " (expected 9) — proceeding with best-effort parse");
    }

    const std::vector<std::pair<std::string, uint32_t>> expected_loops = {
        {"n_i",    d.N},
        {"oc_i",   d.C_out},
        {"od_i",   d.outD},
        {"oh_i",   d.outH},
        {"ow_i",   d.outW},
        {"ic_i",   d.C_in},
        {"kd_i",   d.kD},
        {"kh_i",   d.kH},
        {"kw_i",   d.kW},
    };

    auto get_trip_count = [](mlir::scf::ForOp for_op) -> std::optional<uint32_t> {
        auto lb = for_op.getLowerBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
        auto ub = for_op.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
        auto step = for_op.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>();
        if (!lb || !ub || !step)
            return std::nullopt;
        int64_t diff = ub.value() - lb.value();
        if (diff < 0 || step.value() <= 0)
            return std::nullopt;
        if (diff % step.value() != 0)
            return std::nullopt;
        return static_cast<uint32_t>(diff / step.value());
    };

    llvm::DenseMap<mlir::Value, std::string> names;
    std::vector<bool> used(expected_loops.size(), false);
    for (auto for_op : loops) {
        auto trip = get_trip_count(for_op);
        size_t chosen = expected_loops.size();
        if (trip) {
            for (size_t i = 0; i < expected_loops.size(); ++i) {
                if (used[i])
                    continue;
                if (expected_loops[i].second == *trip) {
                    chosen = i;
                    break;
                }
            }
        }
        if (chosen == expected_loops.size()) {
            // Fallback to first unused slot in order.
            for (size_t i = 0; i < expected_loops.size(); ++i) {
                if (!used[i]) {
                    chosen = i;
                    break;
                }
            }
        }
        if (chosen < expected_loops.size()) {
            names[for_op.getInductionVar()] = expected_loops[chosen].first;
            used[chosen] = true;
        }
    }

    auto input_idx = render_indices(input_load.getIndices(), names);
    auto weight_idx = render_indices(weight_load.getIndices(), names);
    auto output_idx = render_indices(output_store.getIndices(), names);

    if (input_idx.size() != 5 || weight_idx.size() != 5 || output_idx.size() != 5) {
        return fallback();
    }

    auto has_unknown = [](const std::vector<std::string>& idx) {
        for (const auto& s : idx) {
            if (s.find("<expr?>") != std::string::npos) {
                return true;
            }
        }
        return false;
    };
    if (has_unknown(input_idx) || has_unknown(weight_idx) || has_unknown(output_idx)) {
        return fallback();
    }

    if (GFX_MLIR_DEBUG) {
        auto join = [](const std::vector<std::string>& v) {
            std::string s;
            for (size_t i = 0; i < v.size(); ++i) {
                if (i) s += ", ";
                s += v[i];
            }
            return s;
        };
        mlir_codegen_log("[GFX MLIR] Conv3D func=" + func.getName().str());
        mlir_codegen_log("  input idx:  [" + join(input_idx) + "]");
        mlir_codegen_log("  weight idx: [" + join(weight_idx) + "]");
        mlir_codegen_log("  output idx: [" + join(output_idx) + "]");
    }

    return emit_conv3d_msl(d, scalar, input_idx, weight_idx, output_idx);
}

}  // namespace gfx_plugin
}  // namespace ov
