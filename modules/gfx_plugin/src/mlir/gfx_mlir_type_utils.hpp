// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/SmallVector.h"

#include "openvino/core/type/element_type.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/partial_shape.hpp"

namespace ov {
namespace gfx_plugin {

inline mlir::Type to_mlir_type(ov::element::Type et,
                               mlir::MLIRContext& ctx,
                               bool fallback_f32 = false,
                               bool allow_unsigned = false,
                               bool allow_small_ints = false,
                               bool allow_bf16 = false,
                               bool allow_boolean = false,
                               bool signless_integers = false) {
    switch (et) {
        case ov::element::f16: return mlir::Float16Type::get(&ctx);
        case ov::element::f32: return mlir::Float32Type::get(&ctx);
        case ov::element::bf16:
            if (allow_bf16) {
                return mlir::BFloat16Type::get(&ctx);
            }
            break;
        case ov::element::boolean:
            if (allow_boolean) {
                return mlir::IntegerType::get(&ctx, 1);
            }
            break;
        case ov::element::i8:
            if (allow_small_ints) {
                return signless_integers ? mlir::IntegerType::get(&ctx, 8)
                                         : mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Signed);
            }
            break;
        case ov::element::i16:
            if (allow_small_ints) {
                return signless_integers ? mlir::IntegerType::get(&ctx, 16)
                                         : mlir::IntegerType::get(&ctx, 16, mlir::IntegerType::Signed);
            }
            break;
        case ov::element::i32:
            return signless_integers ? mlir::IntegerType::get(&ctx, 32)
                                     : mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);
        case ov::element::i64:
            return signless_integers ? mlir::IntegerType::get(&ctx, 64)
                                     : mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Signed);
        case ov::element::u8:
            if (allow_small_ints) {
                if (signless_integers) {
                    return mlir::IntegerType::get(&ctx, 8);
                }
                return allow_unsigned ? mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Unsigned)
                                      : mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Signed);
            }
            break;
        case ov::element::u16:
            if (allow_small_ints) {
                if (signless_integers) {
                    return mlir::IntegerType::get(&ctx, 16);
                }
                return allow_unsigned ? mlir::IntegerType::get(&ctx, 16, mlir::IntegerType::Unsigned)
                                      : mlir::IntegerType::get(&ctx, 16, mlir::IntegerType::Signed);
            }
            break;
        case ov::element::u32:
            if (signless_integers) {
                return mlir::IntegerType::get(&ctx, 32);
            }
            if (allow_unsigned) {
                return mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Unsigned);
            }
            break;
        case ov::element::u64:
            if (signless_integers) {
                return mlir::IntegerType::get(&ctx, 64);
            }
            if (allow_unsigned) {
                return mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Unsigned);
            }
            break;
        default:
            break;
    }
    if (fallback_f32) {
        return mlir::Float32Type::get(&ctx);
    }
    OPENVINO_THROW("GFX MLIR: unsupported element type");
}

inline mlir::SmallVector<int64_t> to_shape(const ov::PartialShape& ps) {
    mlir::SmallVector<int64_t> dims;
    dims.reserve(ps.rank().get_length());
    for (const auto& d : ps) {
        dims.push_back(d.is_dynamic() ? mlir::ShapedType::kDynamic
                                      : static_cast<int64_t>(d.get_length()));
    }
    return dims;
}

inline llvm::SmallVector<mlir::Value> materialize_dynamic_dims_from_tensor(mlir::OpBuilder& builder,
                                                                           mlir::Location loc,
                                                                           mlir::Value tensor,
                                                                           llvm::ArrayRef<int64_t> shape) {
    llvm::SmallVector<mlir::Value> dyn_dims;
    dyn_dims.reserve(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == mlir::ShapedType::kDynamic) {
            dyn_dims.push_back(builder.create<mlir::tensor::DimOp>(loc, tensor, static_cast<int64_t>(i)).getResult());
        }
    }
    return dyn_dims;
}

}  // namespace gfx_plugin
}  // namespace ov
