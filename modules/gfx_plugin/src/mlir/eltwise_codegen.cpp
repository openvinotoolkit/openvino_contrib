// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen_common.hpp"

#include <sstream>
#include <unordered_set>

#include "openvino/core/except.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace ov {
namespace gfx_plugin {
namespace {

struct EltwisePattern {
    mlir::scf::ForOp loop;
    mlir::Operation* arith_op = nullptr;
};

EltwisePattern find_eltwise_pattern(mlir::func::FuncOp func) {
    EltwisePattern p{};
    func.walk([&](mlir::scf::ForOp op) {
        if (!p.loop) p.loop = op;
    });
    return p;
}

std::string op_to_msl(EltwiseKind k) {
    switch (k) {
        case EltwiseKind::Add: return " + ";
        case EltwiseKind::Sub: return " - ";
        case EltwiseKind::Mul: return " * ";
        case EltwiseKind::Div: return " / ";
        case EltwiseKind::Pow: return "pow";  // handled specially
        case EltwiseKind::Mod: return "fmod";  // handled specially
        case EltwiseKind::FloorMod: return "floormod";  // placeholder
        case EltwiseKind::Min: return "min";
        case EltwiseKind::Max: return "max";
        case EltwiseKind::LogicalAnd: return "&&";
        case EltwiseKind::LogicalOr: return "||";
        case EltwiseKind::Equal: return "==";
        case EltwiseKind::NotEqual: return "!=";
        case EltwiseKind::Less: return "<";
        case EltwiseKind::Greater: return ">";
        case EltwiseKind::LessEqual: return "<=";
        case EltwiseKind::GreaterEqual: return ">=";
        default:
            OPENVINO_THROW("Eltwise codegen: unsupported op kind");
    }
}

std::string activation_expr(ActivationKind kind, float alpha) {
    switch (kind) {
        case ActivationKind::Relu: return "max(x, 0.0f)";
        case ActivationKind::Sigmoid: return "1.0f / (1.0f + exp(-x))";
        case ActivationKind::Tanh: return "tanh(x)";
        case ActivationKind::Prelu: return "(x >= 0.0f) ? x : x * " + std::to_string(alpha);
        case ActivationKind::Gelu:
            return "0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x * x * x)))";
        case ActivationKind::Swish:
            return "(x >= 0.0f) ? (x / (1.0f + exp(-x))) : (x * exp(x) / (1.0f + exp(x)))";
        case ActivationKind::HSwish:
            return "x * clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f";
        case ActivationKind::HSigmoid:
            return "clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f";
        default: return "x";
    }
}

std::string emit_eltwise_msl(const EltwiseCodegenDesc& d,
                             const std::string& scalar_ty,
                             bool is_int,
                             bool is_unsigned,
                             bool use_half,
                             bool is_bf16) {
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    if (is_bf16) {
        ss << "inline float bf16_to_float(ushort v) {\n";
        ss << "    uint u = (uint(v) << 16);\n";
        ss << "    return as_type<float>(u);\n";
        ss << "}\n";
        ss << "inline ushort float_to_bf16(float f) {\n";
        ss << "    uint u = as_type<uint>(f);\n";
        ss << "    uint lsb = (u >> 16) & 1u;\n";
        ss << "    u += 0x7FFFu + lsb;\n";
        ss << "    return ushort(u >> 16);\n";
        ss << "}\n";
    }
    if (is_int && d.eltwise_kind == EltwiseKind::Pow) {
        const std::string exp_ty =
            (scalar_ty == "long" || scalar_ty == "ulong") ? (is_unsigned ? "ulong" : "long")
                                                          : (is_unsigned ? "uint" : "int");
        ss << "inline " << scalar_ty << " ipow(" << scalar_ty << " base, " << scalar_ty
           << " exp) {\n";
        if (!is_unsigned) {
            ss << "    if (exp < 0) return 0;\n";
        }
        ss << "    " << scalar_ty << " result = 1;\n";
        ss << "    " << scalar_ty << " b = base;\n";
        ss << "    " << exp_ty << " e = static_cast<" << exp_ty << ">(exp);\n";
        ss << "    while (e > 0) {\n";
        ss << "        if (e & 1) result *= b;\n";
        ss << "        e >>= 1;\n";
        ss << "        if (e) b *= b;\n";
        ss << "    }\n";
        ss << "    return result;\n";
        ss << "}\n";
    }
    auto load = [&](const std::string& ptr, const std::string& idx) -> std::string {
        if (is_bf16) {
            return "bf16_to_float(" + ptr + "[" + idx + "])";
        }
        return ptr + "[" + idx + "]";
    };
    const bool use_activation = d.has_activation && !is_int;
    auto emit_assign = [&](const std::string& dst, const std::string& value) {
        if (!use_activation) {
            if (is_bf16) {
                ss << "    " << dst << " = float_to_bf16(" << value << ");\n";
            } else {
                ss << "    " << dst << " = " << value << ";\n";
            }
            return;
        }
        ss << "    float x = static_cast<float>(" << value << ");\n";
        ss << "    float y = " << activation_expr(d.activation, d.alpha) << ";\n";
        if (is_bf16) {
            ss << "    " << dst << " = float_to_bf16(y);\n";
        } else if (scalar_ty == "half") {
            ss << "    " << dst << " = half(y);\n";
        } else {
            ss << "    " << dst << " = static_cast<" << scalar_ty << ">(y);\n";
        }
    };
    auto emit_op = [&](const std::string& a, const std::string& b) -> std::string {
        if (d.eltwise_kind == EltwiseKind::Prelu) {
            return "((" + a + ") >= 0 ? (" + a + ") : ((" + a + ") * (" + b + ")))";
        }
        if (d.eltwise_kind == EltwiseKind::SquaredDiff) {
            return "((" + a + " - " + b + ") * (" + a + " - " + b + "))";
        }
        if (d.eltwise_kind == EltwiseKind::Pow) {
            if (is_int) {
                return "ipow(" + a + ", " + b + ")";
            }
            return "pow(" + a + "," + b + ")";
        }
        if (d.eltwise_kind == EltwiseKind::Mod ||
            d.eltwise_kind == EltwiseKind::FloorMod) {
            // Mod/FloorMod are handled in dedicated statement blocks below.
            return "";
        }
        if (d.eltwise_kind == EltwiseKind::Min || d.eltwise_kind == EltwiseKind::Max) {
            return op_to_msl(d.eltwise_kind) + "(" + a + "," + b + ")";
        }
        if (d.eltwise_kind == EltwiseKind::LogicalAnd || d.eltwise_kind == EltwiseKind::LogicalOr) {
            return "(" + a + " " + op_to_msl(d.eltwise_kind) + " " + b + ")";
        }
        if (d.eltwise_kind == EltwiseKind::LogicalXor) {
            return "(" + a + " != " + b + ")";
        }
        if (d.eltwise_kind == EltwiseKind::Equal || d.eltwise_kind == EltwiseKind::NotEqual ||
            d.eltwise_kind == EltwiseKind::Less || d.eltwise_kind == EltwiseKind::Greater ||
            d.eltwise_kind == EltwiseKind::LessEqual || d.eltwise_kind == EltwiseKind::GreaterEqual) {
            return "(" + a + " " + op_to_msl(d.eltwise_kind) + " " + b + ")";
        }
        return a + op_to_msl(d.eltwise_kind) + b;
    };

    auto emit_floor_mod_block = [&](const std::string& a, const std::string& b, const std::string& dst) {
        // FloorMod: result has the sign of the divisor (numpy/TF semantics).
        if (is_int) {
            ss << "    " << scalar_ty << " _a = " << a << ";\n";
            ss << "    " << scalar_ty << " _b = " << b << ";\n";
            ss << "    " << scalar_ty << " _r = _a % _b;\n";
            ss << "    if (_r != 0 && ((_r < 0) != (_b < 0))) _r += _b;\n";
            emit_assign(dst, "_r");
        } else {
            ss << "    float _a = " << a << ";\n";
            ss << "    float _b = " << b << ";\n";
            ss << "    float _r = fmod(_a, _b);\n";
            ss << "    if (_r != 0.0f && ((_r < 0.0f) != (_b < 0.0f))) _r += _b;\n";
            emit_assign(dst, "_r");
        }
    };

    auto emit_mod_block = [&](const std::string& a, const std::string& b, const std::string& dst) {
        // Mod (fmod) semantics: result keeps the sign of the dividend (C/C++ fmod/%).
        if (is_int) {
            ss << "    " << scalar_ty << " _a = " << a << ";\n";
            ss << "    " << scalar_ty << " _b = " << b << ";\n";
            ss << "    " << scalar_ty << " _r = _a % _b;\n";
            emit_assign(dst, "_r");
        } else {
            ss << "    float _a = " << a << ";\n";
            ss << "    float _b = " << b << ";\n";
            ss << "    float _r = fmod(_a, _b);\n";
            emit_assign(dst, "_r");
        }
    };

    ss << "kernel void eltwise_kernel(\n";
    ss << "  device const " << scalar_ty << "* A [[buffer(0)]],\n";
    ss << "  device const " << scalar_ty << "* B [[buffer(1)]],\n";
    ss << "  device " << scalar_ty << "* C [[buffer(2)]],\n";
    ss << "  constant uint& NUM_ELEMS [[buffer(3)]],\n";
    ss << "  constant uint& RANK [[buffer(4)]],\n";
    ss << "  constant int* out_dims [[buffer(5)]],\n";
    ss << "  constant int* stride0 [[buffer(6)]],\n";
    ss << "  constant int* stride1 [[buffer(7)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    if (gid >= NUM_ELEMS) return;\n";
    if (!d.is_broadcast) {
        if (d.eltwise_kind == EltwiseKind::FloorMod) {
            emit_floor_mod_block(load("A", "gid"), load("B", "gid"), "C[gid]");
        } else if (d.eltwise_kind == EltwiseKind::Mod) {
            emit_mod_block(load("A", "gid"), load("B", "gid"), "C[gid]");
        } else if (!is_int && use_half && d.eltwise_kind == EltwiseKind::Div) {
            ss << "    half a = half(" << load("A", "gid") << ");\n";
            ss << "    half b = half(" << load("B", "gid") << ");\n";
            emit_assign("C[gid]", "float(a / b)");
        } else if (is_int && d.eltwise_kind == EltwiseKind::Div) {
            ss << "    " << scalar_ty << " b = " << load("B", "gid") << ";\n";
            emit_assign("C[gid]", "b == 0 ? 0 : (" + load("A", "gid") + " / b)");
        } else {
            emit_assign("C[gid]", emit_op(load("A", "gid"), load("B", "gid")));
        }
        ss << "}\n";
        return ss.str();
    }
    ss << "    uint idx = gid;\n";
    ss << "    int off0 = 0; int off1 = 0;\n";
    ss << "    for (int d = (int)RANK - 1; d >= 0; --d) {\n";
    ss << "        int coord = idx % out_dims[d];\n";
    ss << "        idx /= out_dims[d];\n";
    ss << "        off0 += (stride0[d] == 0 ? 0 : coord * stride0[d]);\n";
    ss << "        off1 += (stride1[d] == 0 ? 0 : coord * stride1[d]);\n";
    ss << "    }\n";
    if (d.eltwise_kind == EltwiseKind::FloorMod) {
        emit_floor_mod_block(load("A", "off0"), load("B", "off1"), "C[gid]");
    } else if (d.eltwise_kind == EltwiseKind::Mod) {
        emit_mod_block(load("A", "off0"), load("B", "off1"), "C[gid]");
    } else if (!is_int && use_half && d.eltwise_kind == EltwiseKind::Div) {
        ss << "    half a = half(" << load("A", "off0") << ");\n";
        ss << "    half b = half(" << load("B", "off1") << ");\n";
        emit_assign("C[gid]", "float(a / b)");
    } else if (is_int && d.eltwise_kind == EltwiseKind::Div) {
        ss << "    " << scalar_ty << " b = " << load("B", "off1") << ";\n";
        emit_assign("C[gid]", "b == 0 ? 0 : (" + load("A", "off0") + " / b)");
    } else {
        emit_assign("C[gid]", emit_op(load("A", "off0"), load("B", "off1")));
    }
    ss << "}\n";
    return ss.str();
}

}  // namespace

std::string generate_msl_for_eltwise(const EltwiseCodegenDesc& d, mlir::ModuleOp module) {
    std::string scalar_ty = "float";
    if (auto func = get_entry_func(module)) {
        auto ft = func.getFunctionType();
        if (ft.getNumInputs() >= 1) {
            scalar_ty = msl_type_from_mlir(ft.getInput(0));
        }
    } else {
        const bool is_int32 = d.element_type == ov::element::i32;
        const bool is_int64 = d.element_type == ov::element::i64;
        const bool is_f16 = d.element_type == ov::element::f16;
        scalar_ty = is_int64 ? "long" : (is_int32 ? "int" : (is_f16 ? "half" : "float"));
    }
    const bool is_bf16 = (d.element_type == ov::element::bf16);
    if (is_bf16) {
        scalar_ty = "ushort";
    }
    bool is_int = false;
    bool is_unsigned = false;
    if (d.element_type != ov::element::dynamic) {
        is_int = d.element_type.is_integral_number() && d.element_type != ov::element::boolean;
        is_unsigned = is_int && !d.element_type.is_signed();
    } else {
        is_int = (scalar_ty == "char" || scalar_ty == "uchar" || scalar_ty == "short" ||
                  scalar_ty == "ushort" || scalar_ty == "int" || scalar_ty == "uint" ||
                  scalar_ty == "long" || scalar_ty == "ulong");
        is_unsigned = (scalar_ty == "uchar" || scalar_ty == "ushort" || scalar_ty == "uint" ||
                       scalar_ty == "ulong");
    }
    const bool use_half = !is_bf16 && (d.use_half_compute || (scalar_ty == "half"));
    if (module) {
        auto func_it = module.getOps<mlir::func::FuncOp>().begin();
        if (func_it != module.getOps<mlir::func::FuncOp>().end()) {
            auto pat = find_eltwise_pattern(*func_it);
            (void)pat;
        }
    }
    return emit_eltwise_msl(d, scalar_ty, is_int, is_unsigned, use_half, is_bf16);
}

}  // namespace gfx_plugin
}  // namespace ov
