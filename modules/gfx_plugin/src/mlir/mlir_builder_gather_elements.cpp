// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "kernel_ir/gfx_codegen_desc.hpp"
#include "mlir/gfx_mlir_type_utils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/gather_elements.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

mlir::Value load_param_as_index(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value params, size_t field_index) {
  auto offset = builder.create<mlir::arith::ConstantIndexOp>(
      loc, static_cast<int64_t>(field_index));
  auto value =
      builder
          .create<mlir::memref::LoadOp>(loc, params, mlir::ValueRange{offset})
          .getResult();
  return builder
      .create<mlir::arith::IndexCastOp>(loc, builder.getIndexType(), value)
      .getResult();
}

mlir::IntegerType gather_axis_normalization_type(mlir::Value value,
                                                 mlir::MLIRContext &ctx) {
  if (auto int_type = mlir::dyn_cast<mlir::IntegerType>(value.getType())) {
    return mlir::IntegerType::get(&ctx, int_type.getWidth() > 32 ? 64 : 32);
  }
  if (value.getType().isIndex()) {
    return mlir::IntegerType::get(&ctx, 32);
  }
  OPENVINO_THROW("GatherElements MLIR: indices must be integral");
}

mlir::Value cast_signed_index_to_type(mlir::OpBuilder &builder,
                                      mlir::Location loc, mlir::Value value,
                                      mlir::IntegerType target_type) {
  if (value.getType() == target_type) {
    return value;
  }
  if (value.getType().isIndex()) {
    return builder.create<mlir::arith::IndexCastOp>(loc, target_type, value)
        .getResult();
  }
  if (auto int_type = mlir::dyn_cast<mlir::IntegerType>(value.getType())) {
    if (int_type.getWidth() < target_type.getWidth()) {
      if (int_type.isUnsigned()) {
        return builder.create<mlir::arith::ExtUIOp>(loc, target_type, value)
            .getResult();
      }
      return builder.create<mlir::arith::ExtSIOp>(loc, target_type, value)
          .getResult();
    }
    if (int_type.getWidth() > target_type.getWidth()) {
      return builder.create<mlir::arith::TruncIOp>(loc, target_type, value)
          .getResult();
    }
  }
  OPENVINO_THROW("GatherElements MLIR: indices must be integral");
}

mlir::Value normalize_gather_axis_index(mlir::OpBuilder &builder,
                                        mlir::Location loc,
                                        mlir::Value raw_index,
                                        mlir::Value axis_dim,
                                        mlir::MLIRContext &ctx) {
  auto norm_type = gather_axis_normalization_type(raw_index, ctx);
  auto int_constant = [&](int64_t value) {
    return builder
        .create<mlir::arith::ConstantOp>(
            loc, norm_type, builder.getIntegerAttr(norm_type, value))
        .getResult();
  };
  auto zero = int_constant(0);
  auto one = int_constant(1);
  auto ix_int = cast_signed_index_to_type(builder, loc, raw_index, norm_type);
  auto axis_dim_int =
      builder.create<mlir::arith::IndexCastOp>(loc, norm_type, axis_dim)
          .getResult();

  auto is_negative = builder
                         .create<mlir::arith::CmpIOp>(
                             loc, mlir::arith::CmpIPredicate::slt, ix_int, zero)
                         .getResult();
  auto wrapped = builder.create<mlir::arith::AddIOp>(loc, ix_int, axis_dim_int)
                     .getResult();
  auto normalized =
      builder.create<mlir::arith::SelectOp>(loc, is_negative, wrapped, ix_int)
          .getResult();

  auto below_zero =
      builder
          .create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
                                       normalized, zero)
          .getResult();
  normalized =
      builder.create<mlir::arith::SelectOp>(loc, below_zero, zero, normalized)
          .getResult();
  auto max_index =
      builder.create<mlir::arith::SubIOp>(loc, axis_dim_int, one).getResult();
  auto above_axis =
      builder
          .create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt,
                                       normalized, max_index)
          .getResult();
  normalized =
      builder
          .create<mlir::arith::SelectOp>(loc, above_axis, max_index, normalized)
          .getResult();
  return builder
      .create<mlir::arith::IndexCastOp>(loc, builder.getIndexType(), normalized)
      .getResult();
}

} // namespace

mlir::ModuleOp build_mlir_gather_elements_from_model(
    const std::shared_ptr<const ov::Model> &model, mlir::MLIRContext &ctx) {
  ctx.loadDialect<mlir::gpu::GPUDialect, mlir::memref::MemRefDialect,
                  mlir::arith::ArithDialect, mlir::scf::SCFDialect>();

  std::shared_ptr<const ov::op::v6::GatherElements> gather;
  for (const auto &node : model->get_ordered_ops()) {
    if (auto g = ov::as_type_ptr<const ov::op::v6::GatherElements>(node)) {
      OPENVINO_ASSERT(
          !gather,
          "GatherElements MLIR builder: expected single GatherElements");
      gather = g;
    }
  }
  OPENVINO_ASSERT(gather,
                  "GatherElements MLIR builder: GatherElements op not found");

  const auto data_shape = gather->get_input_shape(0);
  const auto idx_shape = gather->get_input_shape(1);
  const auto out_shape = gather->get_output_shape(0);
  OPENVINO_ASSERT(
      data_shape.size() == idx_shape.size() &&
          data_shape.size() == out_shape.size(),
      "GatherElements MLIR: data, indices and output ranks must match");
  int64_t axis = gather->get_axis();
  if (axis < 0) {
    axis += static_cast<int64_t>(out_shape.size());
  }
  OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < out_shape.size(),
                  "GatherElements MLIR: axis out of range");
  const size_t axis_pos = static_cast<size_t>(axis);

  auto data_ty = to_mlir_type(gather->get_input_element_type(0), ctx,
                              /*fallback_f32=*/false,
                              /*allow_unsigned=*/true,
                              /*allow_small_ints=*/true,
                              /*allow_bf16=*/false,
                              /*allow_boolean=*/true,
                              /*signless_integers=*/true);
  auto idx_ty = to_mlir_type(gather->get_input_element_type(1), ctx,
                             /*fallback_f32=*/false,
                             /*allow_unsigned=*/true,
                             /*allow_small_ints=*/true,
                             /*allow_bf16=*/false,
                             /*allow_boolean=*/false,
                             /*signless_integers=*/true);
  auto out_ty = to_mlir_type(gather->get_output_element_type(0), ctx,
                             /*fallback_f32=*/false,
                             /*allow_unsigned=*/true,
                             /*allow_small_ints=*/true,
                             /*allow_bf16=*/false,
                             /*allow_boolean=*/true,
                             /*signless_integers=*/true);
  OPENVINO_ASSERT(
      data_ty == out_ty,
      "GatherElements MLIR: data and output element types must match");

  mlir::OpBuilder mb(&ctx);
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
  module->setAttr(mlir::gpu::GPUDialect::getContainerModuleAttrName(),
                  mb.getUnitAttr());
  module->setAttr("gfx.parallel_dispatch", mb.getBoolAttr(true));
  module->setAttr("gfx.parallel_loop_dims",
                  mb.getIndexAttr(static_cast<int64_t>(1)));
  module->setAttr("gfx.prefer_parallel", mb.getBoolAttr(true));
  mb.setInsertionPointToStart(module.getBody());

  auto flat_data_ty =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, data_ty);
  auto flat_indices_ty =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, idx_ty);
  auto flat_output_ty =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, out_ty);
  auto params_ty = mlir::MemRefType::get(
      {static_cast<int64_t>(GatherElementsCodegenDesc::kParamU32Count)},
      mb.getI32Type());
  auto func_type = mb.getFunctionType(
      {flat_data_ty, flat_indices_ty, flat_output_ty, params_ty}, {});
  auto gpu_module = mb.create<mlir::gpu::GPUModuleOp>(
      mlir::UnknownLoc::get(&ctx), "gfx_kernels");
  mlir::OpBuilder gpu_builder =
      mlir::OpBuilder::atBlockBegin(gpu_module.getBody());
  auto func = gpu_builder.create<mlir::gpu::GPUFuncOp>(
      mlir::UnknownLoc::get(&ctx), "gather_scatter_indexed", func_type,
      mlir::TypeRange{}, mlir::TypeRange{});
  func->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(),
                mb.getUnitAttr());

  auto *entry = &func.getBody().front();
  mlir::OpBuilder b(entry, entry->begin());
  auto loc = mlir::UnknownLoc::get(&ctx);

  auto total = load_param_as_index(b, loc, func.getArgument(3),
                                   GatherElementsCodegenDesc::kTotalOffset);

  auto block = b.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::x);
  auto block_dim =
      b.create<mlir::gpu::BlockDimOp>(loc, mlir::gpu::Dimension::x);
  auto thread = b.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::x);
  auto gid = b.create<mlir::arith::AddIOp>(
      loc, b.create<mlir::arith::MulIOp>(loc, block, block_dim), thread);
  auto active = b.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ult, gid, total);
  auto active_if = b.create<mlir::scf::IfOp>(loc, active,
                                             /*withElseRegion=*/false);
  {
    auto lb = active_if.getThenBodyBuilder();
    auto raw_index = lb.create<mlir::memref::LoadOp>(loc, func.getArgument(1),
                                                     mlir::ValueRange{gid})
                         .getResult();
    auto axis_dim = load_param_as_index(
        lb, loc, func.getArgument(3),
        GatherElementsCodegenDesc::kDataDimsOffset + axis_pos);
    auto axis_index =
        normalize_gather_axis_index(lb, loc, raw_index, axis_dim, ctx);

    mlir::Value residual = gid;
    mlir::Value data_linear = lb.create<mlir::arith::ConstantIndexOp>(loc, 0);
    for (size_t d = 0; d < out_shape.size(); ++d) {
      const auto out_stride =
          load_param_as_index(lb, loc, func.getArgument(3),
                              GatherElementsCodegenDesc::kOutStridesOffset + d);
      auto coord = lb.create<mlir::arith::DivUIOp>(loc, residual, out_stride)
                       .getResult();
      residual = lb.create<mlir::arith::RemUIOp>(loc, residual, out_stride)
                     .getResult();
      if (d == axis_pos) {
        coord = axis_index;
      }
      const auto data_stride = load_param_as_index(
          lb, loc, func.getArgument(3),
          GatherElementsCodegenDesc::kDataStridesOffset + d);
      auto scaled =
          lb.create<mlir::arith::MulIOp>(loc, coord, data_stride).getResult();
      data_linear =
          lb.create<mlir::arith::AddIOp>(loc, data_linear, scaled).getResult();
    }

    auto value = lb.create<mlir::memref::LoadOp>(loc, func.getArgument(0),
                                                 mlir::ValueRange{data_linear})
                     .getResult();
    lb.create<mlir::memref::StoreOp>(loc, value, func.getArgument(2),
                                     mlir::ValueRange{gid});
  }
  b.setInsertionPointAfter(active_if);
  b.create<mlir::gpu::ReturnOp>(loc);
  return module;
}

} // namespace gfx_plugin
} // namespace ov
