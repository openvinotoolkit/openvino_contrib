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

std::vector<std::string> render_indices(mlir::Operation::operand_range range,
                                        const llvm::DenseMap<mlir::Value, std::string>& names) {
    std::vector<std::string> out;
    out.reserve(range.size());
    for (auto v : range)
        out.push_back(render_index_expr(v, names));
    return out;
}

std::string emit_softmax_msl(const SoftmaxCodegenDesc& d,
                             const std::vector<std::string>& input_idx,
                             const std::vector<std::string>& output_idx,
                             uint32_t rank) {
    std::vector<std::string> dims;
    if (rank == 3)
        dims = {"p.cols", "p.inner"};
    else
        dims = {"p.cols"};
    const std::string in_flat = flatten_indices(input_idx, dims);
    const std::string out_flat = flatten_indices(output_idx, dims);

    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "struct SoftmaxParams { uint rows; uint cols; uint inner; };\n";
    ss << "kernel void softmax_kernel(\n";
    ss << "  device const float* input [[buffer(0)]],\n";
    ss << "  device float* output [[buffer(1)]],\n";
    ss << "  constant SoftmaxParams& p [[buffer(2)]],\n";
    ss << "  uint gid [[thread_position_in_grid.x]]) {\n";
    ss << "    uint row = gid / p.inner;\n";
    ss << "    uint inner_idx = gid - row * p.inner;\n";
    ss << "    if (row >= p.rows) return;\n";
    ss << "    int row_i = int(row);\n";
    ss << "    int inner_i = int(inner_idx);\n";
    ss << "    float m = -INFINITY;\n";
    ss << "    for (uint c = 0; c < p.cols; ++c) {\n";
    ss << "        int col_i = int(c);\n";
    ss << "        uint idx = " << in_flat << ";\n";
    ss << "        m = m > input[idx] ? m : input[idx];\n";
    ss << "    }\n";
    ss << "    float sum = 0.0f;\n";
    ss << "    for (uint c = 0; c < p.cols; ++c) {\n";
    ss << "        int col_i = int(c);\n";
    ss << "        uint idx = " << in_flat << ";\n";
    ss << "        sum += exp(input[idx] - m);\n";
    ss << "    }\n";
    ss << "    float inv = 1.0f / sum;\n";
    ss << "    for (uint c = 0; c < p.cols; ++c) {\n";
    ss << "        int col_i = int(c);\n";
    ss << "        uint out_idx = " << out_flat << ";\n";
    ss << "        output[out_idx] = exp(input[out_idx] - m) * inv;\n";
    ss << "    }\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace

std::string generate_msl_for_softmax(const SoftmaxCodegenDesc& d, mlir::ModuleOp module) {
    OPENVINO_ASSERT(d.rows > 0 && d.cols > 0, "Softmax: rows/cols must be positive");
    if (!module)
        return emit_softmax_msl(d, {"row_i", "col_i"}, {"row_i", "col_i"}, d.inner > 1 ? 3 : 2);

    auto func = find_func(module);
    OPENVINO_ASSERT(func, "Softmax MLIR: function not found");

    auto loops = collect_loop_chain(func);
    OPENVINO_ASSERT(!loops.empty(), "Softmax MLIR: expected loop nest");

    // Typical lowering: outer loops for row / inner (optional), inner for col
    uint32_t rank = 2;
    std::vector<std::string> loop_names;
    if (loops.size() == 2) {
        loop_names = {"row_i", "col_i"};
        rank = 2;
    } else if (loops.size() >= 3) {
        loop_names = {"row_i", "inner_i", "col_i"};
        rank = 3;
    } else {
        OPENVINO_THROW("Softmax MLIR: unexpected loop depth ", loops.size());
    }

    llvm::DenseMap<mlir::Value, std::string> names;
    for (size_t i = 0; i < loop_names.size(); ++i)
        names[loops[i].getInductionVar()] = loop_names[i];

    mlir::memref::LoadOp input_load = nullptr;
    mlir::memref::StoreOp output_store = nullptr;

    func.walk([&](mlir::memref::LoadOp op) {
        if (!input_load)
            input_load = op;
    });
    func.walk([&](mlir::memref::StoreOp op) {
        if (!output_store)
            output_store = op;
    });
    OPENVINO_ASSERT(input_load && output_store, "Softmax MLIR: failed to find load/store");

    auto input_idx = render_indices(input_load.getIndices(), names);
    auto output_idx = render_indices(output_store.getIndices(), names);

#if METAL_MLIR_DEBUG
    auto join = [](const std::vector<std::string>& v) {
        std::string s;
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) s += ", ";
            s += v[i];
        }
        return s;
    };
    mlir_codegen_log("[METAL MLIR] Softmax func=" + func.getName().str());
    mlir_codegen_log("  input idx:  [" + join(input_idx) + "]");
    mlir_codegen_log("  output idx: [" + join(output_idx) + "]");
#endif

    return emit_softmax_msl(d, input_idx, output_idx, rank);
}

}  // namespace metal_plugin
}  // namespace ov
