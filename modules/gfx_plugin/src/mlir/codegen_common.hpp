// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>

#include "kernel_ir/gfx_codegen_desc.hpp"
#include "openvino/core/type/element_type.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Casting.h"

namespace ov {
namespace gfx_plugin {

// Utility helpers to extract types from MLIR modules for lightweight codegen.
inline mlir::func::FuncOp get_entry_func(mlir::ModuleOp module) {
    for (auto func : module.getOps<mlir::func::FuncOp>()) {
        return func;
    }
    return {};
}

inline mlir::Type get_tensor_element_type(mlir::Type type) {
    if (auto ranked = llvm::dyn_cast<mlir::RankedTensorType>(type)) {
        return ranked.getElementType();
    }
    if (auto unranked = llvm::dyn_cast<mlir::UnrankedTensorType>(type)) {
        return unranked.getElementType();
    }
    if (auto memref = llvm::dyn_cast<mlir::MemRefType>(type)) {
        return memref.getElementType();
    }
    if (auto unranked_memref = llvm::dyn_cast<mlir::UnrankedMemRefType>(type)) {
        return unranked_memref.getElementType();
    }
    return type;
}

inline std::string msl_type_from_mlir(mlir::Type type) {
    type = get_tensor_element_type(type);
    if (llvm::isa<mlir::Float16Type>(type)) {
        return "half";
    }
    if (llvm::isa<mlir::Float32Type>(type)) {
        return "float";
    }
    if (llvm::isa<mlir::IndexType>(type)) {
        return "uint";
    }
    if (auto it = llvm::dyn_cast<mlir::IntegerType>(type)) {
        const bool is_signed = it.isSigned() || it.isSignless();
        switch (it.getWidth()) {
            case 1: return "bool";
            case 8: return is_signed ? "char" : "uchar";
            case 16: return is_signed ? "short" : "ushort";
            case 32: return is_signed ? "int" : "uint";
            case 64: return is_signed ? "long" : "ulong";
            default: return is_signed ? "int" : "uint";
        }
    }
    return "float";
}

inline std::string msl_type_from_element(const ov::element::Type& type) {
    if (type == ov::element::f16) {
        return "half";
    }
    if (type == ov::element::f32) {
        return "float";
    }
    if (type == ov::element::i8) {
        return "char";
    }
    if (type == ov::element::u8) {
        return "uchar";
    }
    if (type == ov::element::i16) {
        return "short";
    }
    if (type == ov::element::u16) {
        return "ushort";
    }
    if (type == ov::element::i32) {
        return "int";
    }
    if (type == ov::element::u32) {
        return "uint";
    }
    if (type == ov::element::i64) {
        return "long";
    }
    if (type == ov::element::u64) {
        return "ulong";
    }
    if (type == ov::element::boolean) {
        return "bool";
    }
    return "float";
}

// Per-op emitters (MSL generation; MLIR module is used when pattern-based codegen is available).
std::string generate_msl_for_matmul(const MatMulCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_conv2d(const Conv2DCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_conv3d(const Conv3DCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_eltwise(const EltwiseCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_maxpool2d(const Pool2DCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_avgpool2d(const Pool2DCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_softmax(const SoftmaxCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_concat(const ConcatCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_interpolate(const InterpolateCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_split(const SplitCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_transpose(const TransposeCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_convert(const ConvertCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_slice_generic(const ConvertCodegenDesc& desc, mlir::ModuleOp module); // reuse ConvertCodegenDesc for dtype only
std::string generate_msl_for_gather(const GatherCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_gathernd(const GatherNDCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_gather_elements(const GatherElementsCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_depth_to_space(const DepthToSpaceCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_space_to_depth(const SpaceToDepthCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_scatter_elements_update(const ScatterElementsUpdateCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_scatter_nd_update(const ScatterNDUpdateCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_shapeof(const ShapeOfCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_batchnorm2d(const BatchNorm2DCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_unary(const UnaryCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_select(mlir::ModuleOp module, ov::element::Type et);
std::string generate_msl_for_reduce(const ReduceCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_pad(const PadCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_tile(const TileCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_broadcast(const BroadcastCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_range(const RangeCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_reverse(const ReverseCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_topk(const TopKCodegenDesc& desc, mlir::ModuleOp module);

// Per-op wrappers to keep call sites uniform without central dispatch.
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const MatMulCodegenDesc& desc) {
    return generate_msl_for_matmul(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const Conv2DCodegenDesc& desc) {
    return generate_msl_for_conv2d(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const Conv3DCodegenDesc& desc) {
    return generate_msl_for_conv3d(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const EltwiseCodegenDesc& desc) {
    return generate_msl_for_eltwise(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const Pool2DCodegenDesc& desc) {
    if (desc.is_avg) {
        return generate_msl_for_avgpool2d(desc, module);
    }
    return generate_msl_for_maxpool2d(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const SoftmaxCodegenDesc& desc) {
    return generate_msl_for_softmax(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const UnaryCodegenDesc& desc) {
    return generate_msl_for_unary(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const BatchNorm2DCodegenDesc& desc) {
    return generate_msl_for_batchnorm2d(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const ConcatCodegenDesc& desc) {
    return generate_msl_for_concat(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const InterpolateCodegenDesc& desc) {
    return generate_msl_for_interpolate(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const SplitCodegenDesc& desc) {
    return generate_msl_for_split(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const TransposeCodegenDesc& desc) {
    return generate_msl_for_transpose(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const ConvertCodegenDesc& desc) {
    return generate_msl_for_convert(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const GatherCodegenDesc& desc) {
    return generate_msl_for_gather(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const GatherNDCodegenDesc& desc) {
    return generate_msl_for_gathernd(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const GatherElementsCodegenDesc& desc) {
    return generate_msl_for_gather_elements(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const DepthToSpaceCodegenDesc& desc) {
    return generate_msl_for_depth_to_space(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const SpaceToDepthCodegenDesc& desc) {
    return generate_msl_for_space_to_depth(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const ScatterElementsUpdateCodegenDesc& desc) {
    return generate_msl_for_scatter_elements_update(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const ScatterNDUpdateCodegenDesc& desc) {
    return generate_msl_for_scatter_nd_update(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const ShapeOfCodegenDesc& desc) {
    return generate_msl_for_shapeof(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, ov::element::Type et, int /*select_tag*/) {
    return generate_msl_for_select(module, et);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const ReduceCodegenDesc& desc) {
    return generate_msl_for_reduce(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const PadCodegenDesc& desc) {
    return generate_msl_for_pad(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const TileCodegenDesc& desc) {
    return generate_msl_for_tile(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const BroadcastCodegenDesc& desc) {
    return generate_msl_for_broadcast(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const RangeCodegenDesc& desc) {
    return generate_msl_for_range(desc, module);
}

inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const ReverseCodegenDesc& desc) {
    return generate_msl_for_reverse(desc, module);
}

inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const TopKCodegenDesc& desc) {
    return generate_msl_for_topk(desc, module);
}

}  // namespace gfx_plugin
}  // namespace ov
