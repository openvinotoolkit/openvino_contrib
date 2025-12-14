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

std::string emit_pool2d_msl(const Pool2DCodegenDesc& d) {
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "struct Pool2DParams {\n";
    ss << "  uint N, C, H, W;\n";
    ss << "  uint kH, kW;\n";
    ss << "  uint strideH, strideW;\n";
    ss << "  uint dilationH, dilationW;\n";
    ss << "  uint padTop, padLeft, padBottom, padRight;\n";
    ss << "  uint outH, outW;\n";
    ss << "  bool is_avg;\n";
    ss << "  bool exclude_pad;\n";
    ss << "};\n";
    ss << "kernel void pool2d_kernel(\n";
    ss << "  device const float* input  [[buffer(0)]],\n";
    ss << "  device float*       output [[buffer(1)]],\n";
    ss << "  constant Pool2DParams& p   [[buffer(2)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "  uint total = p.N * p.C;\n";
    ss << "  if (gid >= total) return;\n";
    ss << "  uint n = gid / p.C;\n";
    ss << "  uint c = gid - n * p.C;\n";
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
    ss << "          int ih = oh_i * int(p.strideH) - int(p.padTop) + kh_i * int(p.dilationH);\n";
    ss << "          int iw = ow_i * int(p.strideW) - int(p.padLeft) + kw_i * int(p.dilationW);\n";
    ss << "          if (ih < 0 || iw < 0 || ih >= int(p.H) || iw >= int(p.W)) {\n";
    ss << "            if (p.is_avg && !p.exclude_pad) { count++; }\n";
    ss << "            continue;\n";
    ss << "          }\n";
    ss << "          uint idx = ((n * p.C + c) * p.H + uint(ih)) * p.W + uint(iw);\n";
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
    ss << "      uint out_idx = ((n * p.C + c) * p.outH + oh) * p.outW + ow;\n";
    ss << "      output[out_idx] = acc;\n";
    ss << "    }\n";
    ss << "  }\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace

std::string generate_msl_for_avgpool2d(const Pool2DCodegenDesc& d, mlir::ModuleOp module) {
    OPENVINO_ASSERT(d.N && d.C && d.H && d.W && d.kH && d.kW && d.outH && d.outW, "Pool2D desc incomplete");
    OPENVINO_ASSERT(d.is_avg, "AvgPool2D codegen expects is_avg=true");
    if (!module)
        return emit_pool2d_msl(d);

    (void)module;
    return emit_pool2d_msl(d);
}

}  // namespace metal_plugin
}  // namespace ov
