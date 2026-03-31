// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen_common.hpp"

#include <sstream>
#include <vector>

#include "openvino/core/except.hpp"
#include "llvm/Support/Casting.h"

namespace ov {
namespace gfx_plugin {

namespace {

ov::element::Type resolve_conv_buffer_type(const ov::element::Type& type,
                                           const ov::element::Type& fallback) {
    if (type != ov::element::dynamic) {
        return type;
    }
    if (fallback != ov::element::dynamic) {
        return fallback;
    }
    return ov::element::f32;
}

std::vector<int64_t> read_entry_argument_shape(mlir::ModuleOp module, size_t arg_idx) {
    std::vector<int64_t> shape;
    auto func = get_entry_func(module);
    if (!func) {
        return shape;
    }
    if (arg_idx >= func.getNumArguments()) {
        return shape;
    }

    auto type = func.getArgument(arg_idx).getType();
    if (auto ranked = llvm::dyn_cast<mlir::RankedTensorType>(type)) {
        if (!ranked.hasStaticShape()) {
            return {};
        }
        shape.assign(ranked.getShape().begin(), ranked.getShape().end());
        return shape;
    }
    if (auto memref = llvm::dyn_cast<mlir::MemRefType>(type)) {
        if (!memref.hasStaticShape()) {
            return {};
        }
        shape.assign(memref.getShape().begin(), memref.getShape().end());
        return shape;
    }
    return shape;
}

std::vector<int64_t> read_absorbed_input_permutation(mlir::ModuleOp module, size_t input_idx) {
    std::vector<int64_t> permutation;
    if (!module) {
        return permutation;
    }
    auto attr = module->getAttrOfType<mlir::ArrayAttr>("gfx.absorbed_input" + std::to_string(input_idx) + "_perm");
    if (!attr) {
        return permutation;
    }
    permutation.reserve(attr.size());
    for (auto value : attr) {
        auto int_attr = llvm::dyn_cast<mlir::IntegerAttr>(value);
        OPENVINO_ASSERT(int_attr, "Conv2D codegen: absorbed input permutation attr must be integer");
        permutation.push_back(int_attr.getInt());
    }
    return permutation;
}

std::vector<int64_t> invert_permutation(const std::vector<int64_t>& permutation) {
    std::vector<int64_t> inverse(permutation.size(), -1);
    for (size_t logical_axis = 0; logical_axis < permutation.size(); ++logical_axis) {
        const int64_t source_axis = permutation[logical_axis];
        OPENVINO_ASSERT(source_axis >= 0 && source_axis < static_cast<int64_t>(permutation.size()),
                        "Conv2D codegen: permutation axis out of range");
        OPENVINO_ASSERT(inverse[static_cast<size_t>(source_axis)] < 0,
                        "Conv2D codegen: permutation axis repeated");
        inverse[static_cast<size_t>(source_axis)] = static_cast<int64_t>(logical_axis);
    }
    return inverse;
}

}  // namespace

// Simple text generator: emits a single-kernel Conv2D with optional bias, batchnorm and activation.
// Generation is parameterized by Conv2DCodegenDesc; MLIR module is not required (we don't pattern-match it).
std::string generate_msl_for_conv2d(const Conv2DCodegenDesc& d, mlir::ModuleOp module) {
    OPENVINO_ASSERT(d.N && d.C_in && d.H && d.W && d.C_out && d.kH && d.kW, "Conv2D desc missing dims");
    uint32_t outH = d.outH;
    uint32_t outW = d.outW;
    const ov::element::Type output_type = resolve_conv_buffer_type(d.output_type, d.element_type);
    const ov::element::Type input_type = resolve_conv_buffer_type(d.input_type, output_type);
    const ov::element::Type weight_type = resolve_conv_buffer_type(d.weight_type, output_type);
    const ov::element::Type bias_type = resolve_conv_buffer_type(d.bias_type, output_type);
    const ov::element::Type bn_type = resolve_conv_buffer_type(d.bn_type, output_type);
    std::string input_scalar = msl_type_from_element(input_type);
    std::string weight_scalar = msl_type_from_element(weight_type);
    std::string bias_scalar = msl_type_from_element(bias_type);
    std::string bn_scalar = msl_type_from_element(bn_type);
    std::string output_scalar = msl_type_from_element(output_type);
    std::string accum = msl_accumulator_type_from_element(output_type);
    if (input_scalar.empty()) {
        input_scalar = "float";
    }
    if (weight_scalar.empty()) {
        weight_scalar = "float";
    }
    if (bias_scalar.empty()) {
        bias_scalar = output_scalar.empty() ? "float" : output_scalar;
    }
    if (bn_scalar.empty()) {
        bn_scalar = output_scalar.empty() ? "float" : output_scalar;
    }
    if (output_scalar.empty()) {
        output_scalar = "float";
    }
    const std::string input_compute = input_scalar;
    const std::string weight_compute = weight_scalar;
    const std::string bias_compute = bias_scalar;
    const std::string bn_compute = bn_scalar;
    const bool use_half = (output_scalar == "half");
    if (outH == 0) {
        int64_t eff_kh = static_cast<int64_t>(d.dilationH) * (static_cast<int64_t>(d.kH) - 1) + 1;
        outH = static_cast<uint32_t>((static_cast<int64_t>(d.H) + d.padTop + d.padBottom - eff_kh) / d.strideH + 1);
    }
    if (outW == 0) {
        int64_t eff_kw = static_cast<int64_t>(d.dilationW) * (static_cast<int64_t>(d.kW) - 1) + 1;
        outW = static_cast<uint32_t>((static_cast<int64_t>(d.W) + d.padLeft + d.padRight - eff_kw) / d.strideW + 1);
    }

    const auto input_permutation = read_absorbed_input_permutation(module, 0);
    const auto source_input_shape = read_entry_argument_shape(module, 0);
    const bool has_absorbed_input_transform = !input_permutation.empty();
    std::vector<int64_t> inverse_input_permutation;
    std::vector<uint64_t> source_input_strides;
    if (has_absorbed_input_transform) {
        OPENVINO_ASSERT(source_input_shape.size() == 4,
                        "Conv2D codegen: absorbed transpose expects rank-4 source input shape");
        OPENVINO_ASSERT(input_permutation.size() == 4,
                        "Conv2D codegen: absorbed transpose expects rank-4 permutation");
        inverse_input_permutation = invert_permutation(input_permutation);
        source_input_strides.assign(source_input_shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(source_input_shape.size()) - 2; i >= 0; --i) {
            source_input_strides[static_cast<size_t>(i)] =
                source_input_strides[static_cast<size_t>(i + 1)] *
                static_cast<uint64_t>(source_input_shape[static_cast<size_t>(i + 1)]);
        }
    }

    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "enum Activation : uint {\n";
    ss << "  ActIdentity = 0,\n";
    ss << "  ActRelu = 1,\n";
    ss << "  ActSigmoid = 2,\n";
    ss << "  ActTanh = 3,\n";
    ss << "  ActElu = 4,\n";
    ss << "  ActPrelu = 5,\n";
    ss << "  ActGelu = 6,\n";
    ss << "  ActSwish = 7,\n";
    ss << "  ActHSwish = 8,\n";
    ss << "  ActHSigmoid = 9,\n";
    ss << "  ActAbs = 10,\n";
    ss << "  ActSign = 11,\n";
    ss << "  ActClamp = 12,\n";
    ss << "};\n";
    ss << "struct ConvParams {\n";
    ss << "  uint N, C_in, H, W;\n";
    ss << "  uint C_out;\n";
    ss << "  uint groups;\n";
    ss << "  uint C_in_pg;\n";
    ss << "  uint C_out_pg;\n";
    ss << "  uint kH, kW;\n";
    ss << "  uint strideH, strideW;\n";
    ss << "  uint dilationH, dilationW;\n";
    ss << "  uint padTop, padLeft;\n";
    ss << "  uint padBottom, padRight;\n";
    ss << "  uint outH, outW;\n";
    ss << "  uint has_bias;\n";
    ss << "  uint has_bn;\n";
    ss << "  uint activation;\n";
    ss << "  float alpha;\n";
    ss << "  float epsilon;\n";
    ss << "  float clamp_min;\n";
    ss << "  float clamp_max;\n";
    ss << "};\n";

    ss << "kernel void conv2d_kernel(\n";
    ss << "  device const " << input_scalar << "* in0   [[buffer(0)]],\n";
    ss << "  device const " << weight_scalar << "* w     [[buffer(1)]],\n";
    ss << "  device const " << bias_scalar << "* bias  [[buffer(2)]],\n";
    ss << "  device const " << bn_scalar << "* gamma [[buffer(3)]],\n";
    ss << "  device const " << bn_scalar << "* beta  [[buffer(4)]],\n";
    ss << "  device const " << bn_scalar << "* mean  [[buffer(5)]],\n";
    ss << "  device const " << bn_scalar << "* var   [[buffer(6)]],\n";
    ss << "  constant ConvParams& p    [[buffer(7)]],\n";
    ss << "  device " << output_scalar << "* out         [[buffer(8)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    uint total = p.N * p.outH * p.outW * p.C_out;\n";
    ss << "    if (gid >= total) return;\n";
    ss << "    uint tmp = gid;\n";
    ss << "    uint ow = tmp % p.outW; tmp /= p.outW;\n";
    ss << "    uint oh = tmp % p.outH; tmp /= p.outH;\n";
    ss << "    uint co = tmp % p.C_out; tmp /= p.C_out;\n";
    ss << "    uint n = tmp;\n";
    ss << "    uint g = (p.groups == 0 || p.groups == 1) ? 0 : co / p.C_out_pg;\n";
    ss << "    uint co_g = (p.groups == 0 || p.groups == 1) ? co : co - g * p.C_out_pg;\n";
    ss << "    int in_h0 = int(oh) * int(p.strideH) - int(p.padTop);\n";
    ss << "    int in_w0 = int(ow) * int(p.strideW) - int(p.padLeft);\n";
    ss << "    " << accum << " acc = static_cast<" << accum << ">(0.0f);\n";
    ss << "    uint cin_pg = (p.groups == 0 || p.groups == 1) ? p.C_in : p.C_in_pg;\n";
    ss << "    for (uint ci = 0; ci < cin_pg; ++ci) {\n";
    ss << "        for (uint kh = 0; kh < p.kH; ++kh) {\n";
    ss << "            int ih = in_h0 + int(kh) * int(p.dilationH);\n";
    ss << "            if (ih < 0 || ih >= int(p.H)) continue;\n";
    ss << "            for (uint kw = 0; kw < p.kW; ++kw) {\n";
    ss << "                int iw = in_w0 + int(kw) * int(p.dilationW);\n";
    ss << "                if (iw < 0 || iw >= int(p.W)) continue;\n";
    ss << "                uint ci_global = (p.groups == 0 || p.groups == 1) ? ci : g * p.C_in_pg + ci;\n";
    if (!has_absorbed_input_transform) {
        ss << "                uint in_idx = ((n * p.C_in + ci_global) * p.H + uint(ih)) * p.W + uint(iw);\n";
    } else {
        ss << "                uint logical_idx[4] = {n, ci_global, uint(ih), uint(iw)};\n";
        ss << "                uint in_idx = 0u;\n";
        for (size_t src_dim = 0; src_dim < source_input_shape.size(); ++src_dim) {
            ss << "                in_idx += logical_idx[" << inverse_input_permutation[src_dim] << "] * "
               << static_cast<uint32_t>(source_input_strides[src_dim]) << "u;\n";
        }
    }
    ss << "                uint w_idx = (((g * p.C_out_pg + co_g) * p.C_in_pg + ci) * p.kH + kh) * p.kW + kw;\n";
    ss << "                acc += static_cast<" << accum << ">(static_cast<" << input_compute << ">(in0[in_idx])) * "
          "static_cast<" << accum << ">(static_cast<" << weight_compute << ">(w[w_idx]));\n";
    ss << "            }\n";
    ss << "        }\n";
    ss << "    }\n";
    ss << "    if (p.has_bias) {\n";
    ss << "        acc += static_cast<" << accum << ">(static_cast<" << bias_compute << ">(bias[co]));\n";
    ss << "    }\n";
    ss << "    if (p.has_bn) {\n";
    ss << "        " << accum << " g_scale = static_cast<" << accum << ">(static_cast<" << bn_compute << ">(gamma[co]));\n";
    ss << "        " << accum << " b_shift = static_cast<" << accum << ">(static_cast<" << bn_compute << ">(beta[co]));\n";
    ss << "        " << accum << " m = static_cast<" << accum << ">(static_cast<" << bn_compute << ">(mean[co]));\n";
    ss << "        " << accum << " v = static_cast<" << accum << ">(static_cast<" << bn_compute << ">(var[co]));\n";
    ss << "        " << accum << " inv_std = rsqrt(v + static_cast<" << accum << ">(p.epsilon));\n";
    ss << "        acc = g_scale * (acc - m) * inv_std + b_shift;\n";
    ss << "    }\n";
    ss << "    switch (p.activation) {\n";
    ss << "      case ActRelu: acc = max(acc, static_cast<" << accum << ">(0.0f)); break;\n";
    ss << "      case ActSigmoid: acc = static_cast<" << accum << ">(1.0f) / (static_cast<" << accum << ">(1.0f) + exp(-acc)); break;\n";
    ss << "      case ActTanh: acc = tanh(acc); break;\n";
    ss << "      case ActElu: acc = (acc > static_cast<" << accum << ">(0.0f)) ? acc : (exp(acc) - static_cast<" << accum << ">(1.0f)) * static_cast<" << accum << ">(p.alpha); break;\n";
    ss << "      case ActPrelu: acc = (acc >= static_cast<" << accum << ">(0.0f)) ? acc : acc * static_cast<" << accum << ">(p.alpha); break;\n";
    ss << "      case ActGelu: acc = static_cast<" << accum << ">(0.5f) * acc * (static_cast<" << accum << ">(1.0f) + tanh(static_cast<" << accum << ">(0.79788456f) * (acc + static_cast<" << accum << ">(0.044715f) * acc * acc * acc))); break;\n";
    ss << "      case ActSwish: acc = (acc >= static_cast<" << accum << ">(0.0f)) ? "
          "(acc / (static_cast<" << accum << ">(1.0f) + exp(-acc))) : "
          "(acc * exp(acc) / (static_cast<" << accum << ">(1.0f) + exp(acc))); break;\n";
    ss << "      case ActHSwish: acc = acc * clamp(acc + static_cast<" << accum << ">(3.0f), static_cast<" << accum << ">(0.0f), static_cast<" << accum << ">(6.0f)) / static_cast<" << accum << ">(6.0f); break;\n";
    ss << "      case ActHSigmoid: acc = clamp(acc + static_cast<" << accum << ">(3.0f), static_cast<" << accum << ">(0.0f), static_cast<" << accum << ">(6.0f)) / static_cast<" << accum << ">(6.0f); break;\n";
    ss << "      case ActAbs: acc = fabs(acc); break;\n";
    ss << "      case ActSign: acc = (acc > static_cast<" << accum << ">(0.0f)) ? static_cast<" << accum << ">(1.0f) : (acc < static_cast<" << accum << ">(0.0f) ? static_cast<" << accum << ">(-1.0f) : static_cast<" << accum << ">(0.0f)); break;\n";
    ss << "      case ActClamp: acc = clamp(acc, static_cast<" << accum << ">(p.clamp_min), static_cast<" << accum << ">(p.clamp_max)); break;\n";
    ss << "      case ActIdentity:\n";
    ss << "      default: break;\n";
    ss << "    }\n";
    ss << "    uint out_idx = ((n * p.C_out + co) * p.outH + oh) * p.outW + ow;\n";
    if (use_half || accum != output_scalar) {
        ss << "    out[out_idx] = static_cast<" << output_scalar << ">(acc);\n";
    } else {
        ss << "    out[out_idx] = acc;\n";
    }
    ss << "}\n";
    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
