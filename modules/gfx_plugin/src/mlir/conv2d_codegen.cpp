// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen_common.hpp"

#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "mlir/msl_codegen_apple_msl_common.hpp"
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
    if (!module) {
        return shape;
    }
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

bool read_bool_attr(mlir::ModuleOp module, const char* name) {
    if (!module) {
        return false;
    }
    auto attr = module->getAttrOfType<mlir::BoolAttr>(name);
    return attr && attr.getValue();
}

std::optional<std::string> read_first_string_attr(mlir::ModuleOp module, llvm::StringRef name) {
    std::optional<std::string> result;
    if (!module) {
        return result;
    }
    module.walk([&](mlir::Operation* op) {
        if (result) {
            return;
        }
        if (auto attr = op->getAttrOfType<mlir::StringAttr>(name)) {
            result = attr.getValue().str();
        }
    });
    return result;
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

std::string static_activation_expr(const std::string& kind, const char* value) {
    const std::string x(value);
    if (kind == "Relu") {
        return "max(" + x + ", 0.0f)";
    }
    if (kind == "Sigmoid") {
        return "1.0f / (1.0f + precise::exp(-" + x + "))";
    }
    if (kind == "Tanh") {
        return msl_stable_tanh_expr(x);
    }
    if (kind == "Elu") {
        return "(" + x + " > 0.0f) ? " + x + " : (exp(" + x + ") - 1.0f) * p.alpha";
    }
    if (kind == "Prelu") {
        return "(" + x + " >= 0.0f) ? " + x + " : " + x + " * p.alpha";
    }
    if (kind == "Gelu") {
        return msl_stable_gelu_tanh_expr(x);
    }
    if (kind == "Swish") {
        return x + " / (1.0f + precise::exp(-" + x + "))";
    }
    if (kind == "HSwish") {
        return x + " * clamp(" + x + " + 3.0f, 0.0f, 6.0f) / 6.0f";
    }
    if (kind == "HSigmoid") {
        return "clamp(" + x + " + 3.0f, 0.0f, 6.0f) / 6.0f";
    }
    if (kind == "Abs") {
        return "fabs(" + x + ")";
    }
    if (kind == "Sign") {
        return "(" + x + " > 0.0f) ? 1.0f : ((" + x + " < 0.0f) ? -1.0f : 0.0f)";
    }
    if (kind == "Clamp") {
        return "clamp(" + x + ", p.clamp_min, p.clamp_max)";
    }
    return x;
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
    const ov::element::Type weight_type =
        read_bool_attr(module, "gfx.conv2d_weight_storage_f16")
            ? ov::element::f16
            : resolve_conv_buffer_type(d.weight_type, output_type);
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
    Conv2DCodegenDesc resolved_desc = d;
    resolved_desc.outH = outH;
    resolved_desc.outW = outW;
    const uint32_t channel_block =
        std::max<uint32_t>(1u, resolved_desc.output_channels_per_thread
                                    ? resolved_desc.output_channels_per_thread
                                    : gfx_conv2d_output_channel_block(resolved_desc));
    const uint32_t width_block =
        std::max<uint32_t>(1u, resolved_desc.output_width_per_thread
                                    ? resolved_desc.output_width_per_thread
                                    : gfx_conv2d_output_width_block(resolved_desc));
    const bool weights_packed_oc4 =
        channel_block == 4 && read_bool_attr(module, "gfx.conv2d_weights_packed_oc4");
    const auto static_activation = read_first_string_attr(module, "gfx.activation_kind");
    const bool pointwise_1x1_fast_path =
        resolved_desc.groups <= 1 &&
        resolved_desc.kH == 1 &&
        resolved_desc.kW == 1 &&
        resolved_desc.strideH == 1 &&
        resolved_desc.strideW == 1 &&
        resolved_desc.dilationH == 1 &&
        resolved_desc.dilationW == 1 &&
        resolved_desc.padTop == 0 &&
        resolved_desc.padLeft == 0 &&
        resolved_desc.padBottom == 0 &&
        resolved_desc.padRight == 0;

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
    ss << "constant uint GFX_CONV_N = " << resolved_desc.N << "u;\n";
    ss << "constant uint GFX_CONV_C_IN = " << resolved_desc.C_in << "u;\n";
    ss << "constant uint GFX_CONV_H = " << resolved_desc.H << "u;\n";
    ss << "constant uint GFX_CONV_W = " << resolved_desc.W << "u;\n";
    ss << "constant uint GFX_CONV_C_OUT = " << resolved_desc.C_out << "u;\n";
    ss << "constant uint GFX_CONV_C_IN_PG = " << resolved_desc.C_in_pg << "u;\n";
    ss << "constant uint GFX_CONV_KH = " << resolved_desc.kH << "u;\n";
    ss << "constant uint GFX_CONV_KW = " << resolved_desc.kW << "u;\n";
    ss << "constant uint GFX_CONV_STRIDE_H = " << resolved_desc.strideH << "u;\n";
    ss << "constant uint GFX_CONV_STRIDE_W = " << resolved_desc.strideW << "u;\n";
    ss << "constant uint GFX_CONV_DILATION_H = " << resolved_desc.dilationH << "u;\n";
    ss << "constant uint GFX_CONV_DILATION_W = " << resolved_desc.dilationW << "u;\n";
    ss << "constant uint GFX_CONV_PAD_TOP = " << resolved_desc.padTop << "u;\n";
    ss << "constant uint GFX_CONV_PAD_LEFT = " << resolved_desc.padLeft << "u;\n";
    ss << "constant uint GFX_CONV_OUT_H = " << resolved_desc.outH << "u;\n";
    ss << "constant uint GFX_CONV_OUT_W = " << resolved_desc.outW << "u;\n";

    if (channel_block > 1) {
        OPENVINO_ASSERT(channel_block == 4, "Conv2D codegen: only 4-channel output blocking is supported");
        OPENVINO_ASSERT(width_block <= 2, "Conv2D codegen: only 2-pixel output width blocking is supported");
        OPENVINO_ASSERT(d.groups <= 1, "Conv2D codegen: channel-blocked kernel expects non-group convolution");
        if (static_activation) {
            ss << "inline float gfx_conv_apply_activation(float acc, constant ConvParams& p) {\n";
            ss << "    return " << static_activation_expr(*static_activation, "acc") << ";\n";
            ss << "}\n";
        } else {
            ss << "inline float gfx_conv_apply_activation(float acc, constant ConvParams& p) {\n";
            ss << "    switch (p.activation) {\n";
            ss << "      case ActRelu: return max(acc, 0.0f);\n";
            ss << "      case ActSigmoid: return 1.0f / (1.0f + precise::exp(-acc));\n";
            ss << "      case ActTanh: return " << msl_stable_tanh_expr("acc") << ";\n";
            ss << "      case ActElu: return (acc > 0.0f) ? acc : (exp(acc) - 1.0f) * p.alpha;\n";
            ss << "      case ActPrelu: return (acc >= 0.0f) ? acc : acc * p.alpha;\n";
            ss << "      case ActGelu: return " << msl_stable_gelu_tanh_expr("acc") << ";\n";
            ss << "      case ActSwish: return acc / (1.0f + precise::exp(-acc));\n";
            ss << "      case ActHSwish: return acc * clamp(acc + 3.0f, 0.0f, 6.0f) / 6.0f;\n";
            ss << "      case ActHSigmoid: return clamp(acc + 3.0f, 0.0f, 6.0f) / 6.0f;\n";
            ss << "      case ActAbs: return fabs(acc);\n";
            ss << "      case ActSign: return (acc > 0.0f) ? 1.0f : ((acc < 0.0f) ? -1.0f : 0.0f);\n";
            ss << "      case ActClamp: return clamp(acc, p.clamp_min, p.clamp_max);\n";
            ss << "      case ActIdentity:\n";
            ss << "      default: return acc;\n";
            ss << "    }\n";
            ss << "}\n";
        }
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
        ss << "    constexpr uint width_blocks = (GFX_CONV_OUT_W + " << (width_block - 1) << "u) / " << width_block << "u;\n";
        ss << "    constexpr uint spatial_total = GFX_CONV_N * GFX_CONV_OUT_H * width_blocks;\n";
        ss << "    constexpr uint channel_blocks = (GFX_CONV_C_OUT + " << (channel_block - 1) << "u) / " << channel_block << "u;\n";
        ss << "    uint total = spatial_total * channel_blocks;\n";
        ss << "    if (gid >= total) return;\n";
        ss << "    uint block = gid / spatial_total;\n";
        ss << "    uint spatial = gid - block * spatial_total;\n";
        ss << "    uint co_base = block * " << channel_block << "u;\n";
        ss << "    uint tmp = spatial;\n";
        ss << "    uint ow_base = (tmp % width_blocks) * " << width_block << "u; tmp /= width_blocks;\n";
        ss << "    uint oh = tmp % GFX_CONV_OUT_H; tmp /= GFX_CONV_OUT_H;\n";
        ss << "    uint n = tmp;\n";
        ss << "    int in_h0 = int(oh) * int(GFX_CONV_STRIDE_H) - int(GFX_CONV_PAD_TOP);\n";
        ss << "    int in_w0_base = int(ow_base) * int(GFX_CONV_STRIDE_W) - int(GFX_CONV_PAD_LEFT);\n";
        for (uint32_t px = 0; px < width_block; ++px) {
        for (uint32_t vec = 0; vec < channel_block / 4; ++vec) {
                ss << "    float4 acc" << px << "_" << vec << " = float4(0.0f);\n";
            }
        }
        ss << "    constexpr uint weight_channel_stride = GFX_CONV_C_IN_PG * GFX_CONV_KH * GFX_CONV_KW;\n";
        if (pointwise_1x1_fast_path) {
            ss << "    for (uint ci = 0; ci < GFX_CONV_C_IN; ++ci) {\n";
            for (uint32_t vec = 0; vec < channel_block / 4; ++vec) {
                const uint32_t co_offset = vec * 4;
                ss << "        if (co_base + " << co_offset << "u < GFX_CONV_C_OUT) {\n";
                if (weights_packed_oc4) {
                    ss << "            uint w_base" << vec
                       << " = ((block + " << vec << "u) * GFX_CONV_C_IN + ci) * 4u;\n";
                    ss << "            float4 ww" << vec << " = float4(static_cast<float>(static_cast<"
                       << weight_compute << ">(w[w_base" << vec << "])),\n";
                    ss << "                                        static_cast<float>(static_cast<"
                       << weight_compute << ">(w[w_base" << vec << " + 1u])),\n";
                    ss << "                                        static_cast<float>(static_cast<"
                       << weight_compute << ">(w[w_base" << vec << " + 2u])),\n";
                    ss << "                                        static_cast<float>(static_cast<"
                       << weight_compute << ">(w[w_base" << vec << " + 3u])));\n";
                } else {
                    ss << "            uint w_base" << vec << " = (co_base + " << co_offset
                       << "u) * weight_channel_stride + ci;\n";
                    ss << "            float4 ww" << vec << " = float4(0.0f);\n";
                    ss << "            if (co_base + " << (co_offset + 3) << "u < GFX_CONV_C_OUT) {\n";
                    ss << "                ww" << vec << " = float4(static_cast<float>(static_cast<"
                       << weight_compute << ">(w[w_base" << vec << "])),\n";
                    ss << "                            static_cast<float>(static_cast<"
                       << weight_compute << ">(w[w_base" << vec << " + weight_channel_stride])),\n";
                    ss << "                            static_cast<float>(static_cast<"
                       << weight_compute << ">(w[w_base" << vec << " + 2u * weight_channel_stride])),\n";
                    ss << "                            static_cast<float>(static_cast<"
                       << weight_compute << ">(w[w_base" << vec << " + 3u * weight_channel_stride])));\n";
                    ss << "            } else {\n";
                    for (uint32_t lane = 0; lane < 4; ++lane) {
                        ss << "                if (co_base + " << (co_offset + lane)
                           << "u < GFX_CONV_C_OUT) ww" << vec << "[" << lane
                           << "] = static_cast<float>(static_cast<" << weight_compute
                           << ">(w[w_base" << vec << " + " << lane << "u * weight_channel_stride]));\n";
                    }
                    ss << "            }\n";
                }
                for (uint32_t px = 0; px < width_block; ++px) {
                    ss << "            if (ow_base + " << px << "u < GFX_CONV_OUT_W) {\n";
                    if (!has_absorbed_input_transform) {
                        ss << "                uint in_idx" << px
                           << " = ((n * GFX_CONV_C_IN + ci) * GFX_CONV_H + oh) * GFX_CONV_W + (ow_base + "
                           << px << "u);\n";
                    } else {
                        ss << "                uint logical_idx" << px
                           << "[4] = {n, ci, oh, ow_base + " << px << "u};\n";
                        ss << "                uint in_idx" << px << " = 0u;\n";
                        for (size_t src_dim = 0; src_dim < source_input_shape.size(); ++src_dim) {
                            ss << "                in_idx" << px << " += logical_idx" << px
                               << "[" << inverse_input_permutation[src_dim] << "] * "
                               << static_cast<uint32_t>(source_input_strides[src_dim]) << "u;\n";
                        }
                    }
                    ss << "                float x" << px << " = static_cast<float>(static_cast<"
                       << input_compute << ">(in0[in_idx" << px << "]));\n";
                    ss << "                acc" << px << "_" << vec << " += x" << px
                       << " * ww" << vec << ";\n";
                    ss << "            }\n";
                }
                ss << "        }\n";
            }
            ss << "    }\n";
        } else {
            ss << "    for (uint ci = 0; ci < GFX_CONV_C_IN; ++ci) {\n";
            ss << "        for (uint kh = 0; kh < GFX_CONV_KH; ++kh) {\n";
            ss << "            int ih = in_h0 + int(kh) * int(GFX_CONV_DILATION_H);\n";
            ss << "            if (ih < 0 || ih >= int(GFX_CONV_H)) continue;\n";
            ss << "            for (uint kw = 0; kw < GFX_CONV_KW; ++kw) {\n";
        for (uint32_t vec = 0; vec < channel_block / 4; ++vec) {
            const uint32_t co_offset = vec * 4;
            ss << "                if (co_base + " << co_offset << "u < GFX_CONV_C_OUT) {\n";
            if (weights_packed_oc4) {
                ss << "                    uint w_base" << vec
                   << " = (((block + " << vec << "u) * GFX_CONV_C_IN + ci) * GFX_CONV_KH * GFX_CONV_KW + kh * GFX_CONV_KW + kw) * 4u;\n";
                ss << "                    float4 ww" << vec << " = float4(static_cast<float>(static_cast<"
                   << weight_compute << ">(w[w_base" << vec << "])),\n";
                ss << "                                                static_cast<float>(static_cast<"
                   << weight_compute << ">(w[w_base" << vec << " + 1u])),\n";
                ss << "                                                static_cast<float>(static_cast<"
                   << weight_compute << ">(w[w_base" << vec << " + 2u])),\n";
                ss << "                                                static_cast<float>(static_cast<"
                   << weight_compute << ">(w[w_base" << vec << " + 3u])));\n";
            } else {
                ss << "                    uint w_base" << vec << " = (co_base + " << co_offset
                   << "u) * weight_channel_stride + ci * GFX_CONV_KH * GFX_CONV_KW + kh * GFX_CONV_KW + kw;\n";
                ss << "                    float4 ww" << vec << " = float4(0.0f);\n";
                ss << "                    if (co_base + " << (co_offset + 3) << "u < GFX_CONV_C_OUT) {\n";
                ss << "                        ww" << vec << " = float4(static_cast<float>(static_cast<"
                   << weight_compute << ">(w[w_base" << vec << "])),\n";
                ss << "                                    static_cast<float>(static_cast<"
                   << weight_compute << ">(w[w_base" << vec << " + weight_channel_stride])),\n";
                ss << "                                    static_cast<float>(static_cast<"
                   << weight_compute << ">(w[w_base" << vec << " + 2u * weight_channel_stride])),\n";
                ss << "                                    static_cast<float>(static_cast<"
                   << weight_compute << ">(w[w_base" << vec << " + 3u * weight_channel_stride])));\n";
                ss << "                    } else {\n";
                for (uint32_t lane = 0; lane < 4; ++lane) {
                    ss << "                        if (co_base + " << (co_offset + lane)
                       << "u < GFX_CONV_C_OUT) ww" << vec << "[" << lane
                       << "] = static_cast<float>(static_cast<" << weight_compute
                       << ">(w[w_base" << vec << " + " << lane << "u * weight_channel_stride]));\n";
                }
                ss << "                    }\n";
            }
            for (uint32_t px = 0; px < width_block; ++px) {
                ss << "                    if (ow_base + " << px << "u < GFX_CONV_OUT_W) {\n";
                ss << "                        int iw" << px << " = in_w0_base + int("
                   << px << "u * GFX_CONV_STRIDE_W) + int(kw) * int(GFX_CONV_DILATION_W);\n";
                ss << "                        if (iw" << px << " >= 0 && iw" << px << " < int(GFX_CONV_W)) {\n";
                if (!has_absorbed_input_transform) {
                    ss << "                            uint in_idx" << px
                       << " = ((n * GFX_CONV_C_IN + ci) * GFX_CONV_H + uint(ih)) * GFX_CONV_W + uint(iw" << px << ");\n";
                } else {
                    ss << "                            uint logical_idx" << px
                       << "[4] = {n, ci, uint(ih), uint(iw" << px << ")};\n";
                    ss << "                            uint in_idx" << px << " = 0u;\n";
                    for (size_t src_dim = 0; src_dim < source_input_shape.size(); ++src_dim) {
                        ss << "                            in_idx" << px << " += logical_idx" << px
                           << "[" << inverse_input_permutation[src_dim] << "] * "
                           << static_cast<uint32_t>(source_input_strides[src_dim]) << "u;\n";
                    }
                }
                ss << "                            float x" << px << " = static_cast<float>(static_cast<"
                   << input_compute << ">(in0[in_idx" << px << "]));\n";
                ss << "                            acc" << px << "_" << vec << " += x" << px
                   << " * ww" << vec << ";\n";
                ss << "                        }\n";
                ss << "                    }\n";
            }
            ss << "                }\n";
        }
            ss << "            }\n";
            ss << "        }\n";
            ss << "    }\n";
        }
        for (uint32_t px = 0; px < width_block; ++px) {
            ss << "    if (ow_base + " << px << "u < GFX_CONV_OUT_W) {\n";
            for (uint32_t vec = 0; vec < channel_block / 4; ++vec) {
                for (uint32_t lane = 0; lane < 4; ++lane) {
                    const uint32_t co_offset = vec * 4 + lane;
                    ss << "        if (co_base + " << co_offset << "u < GFX_CONV_C_OUT) {\n";
                    ss << "            uint co = co_base + " << co_offset << "u;\n";
                    ss << "            float v = acc" << px << "_" << vec << "[" << lane << "];\n";
                    ss << "            if (p.has_bias) v += static_cast<float>(static_cast<"
                       << bias_compute << ">(bias[co]));\n";
                    ss << "            if (p.has_bn) {\n";
                    ss << "                float g_scale = static_cast<float>(static_cast<"
                       << bn_compute << ">(gamma[co]));\n";
                    ss << "                float b_shift = static_cast<float>(static_cast<"
                       << bn_compute << ">(beta[co]));\n";
                    ss << "                float m = static_cast<float>(static_cast<"
                       << bn_compute << ">(mean[co]));\n";
                    ss << "                float vv = static_cast<float>(static_cast<"
                       << bn_compute << ">(var[co]));\n";
                    ss << "                v = g_scale * (v - m) * rsqrt(vv + p.epsilon) + b_shift;\n";
                    ss << "            }\n";
                    ss << "            v = gfx_conv_apply_activation(v, p);\n";
                    ss << "            uint out_idx = ((n * GFX_CONV_C_OUT + co) * GFX_CONV_OUT_H + oh) * GFX_CONV_OUT_W + (ow_base + "
                       << px << "u);\n";
                    ss << "            out[out_idx] = static_cast<" << output_scalar << ">(v);\n";
                    ss << "        }\n";
                }
            }
            ss << "    }\n";
        }
        ss << "}\n";
        return ss.str();
    }

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
    ss << "      case ActSigmoid: acc = static_cast<" << accum << ">(1.0f) / (static_cast<" << accum
       << ">(1.0f) + precise::exp(-acc)); break;\n";
    ss << "      case ActTanh: acc = " << msl_stable_tanh_expr("acc") << "; break;\n";
    ss << "      case ActElu: acc = (acc > static_cast<" << accum << ">(0.0f)) ? acc : (exp(acc) - static_cast<" << accum << ">(1.0f)) * static_cast<" << accum << ">(p.alpha); break;\n";
    ss << "      case ActPrelu: acc = (acc >= static_cast<" << accum << ">(0.0f)) ? acc : acc * static_cast<" << accum << ">(p.alpha); break;\n";
    ss << "      case ActGelu: acc = " << msl_stable_gelu_tanh_expr("acc") << "; break;\n";
    ss << "      case ActSwish: acc = acc / (static_cast<" << accum
       << ">(1.0f) + precise::exp(-acc)); break;\n";
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
