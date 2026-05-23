// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transforms/conv_parallel_lowering.hpp"

#include <array>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>

#include "mlir/gfx_mlir_debug.hpp"
#include "runtime/gfx_activation.hpp"
#include "runtime/gfx_logger.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringRef.h"

namespace ov {
namespace gfx_plugin {

namespace {

bool is_positive_extent(int64_t value) { return value > 0; }

mlir::Value build_conv_tile_input_interior_condition(
    mlir::OpBuilder &b, mlir::Location loc, mlir::Value oh_base,
    mlir::Value ow_base, mlir::Value tile_h, mlir::Value tile_w,
    mlir::Value stride_h, mlir::Value stride_w, mlir::Value dil_h,
    mlir::Value dil_w, mlir::Value kernel_h, mlir::Value kernel_w,
    mlir::Value pad_h, mlir::Value pad_w, mlir::Value input_h,
    mlir::Value input_w, mlir::Value c0, mlir::Value c1) {
  auto tile_h_last = b.create<mlir::arith::SubIOp>(loc, tile_h, c1).getResult();
  auto tile_w_last = b.create<mlir::arith::SubIOp>(loc, tile_w, c1).getResult();
  auto kernel_h_last =
      b.create<mlir::arith::SubIOp>(loc, kernel_h, c1).getResult();
  auto kernel_w_last =
      b.create<mlir::arith::SubIOp>(loc, kernel_w, c1).getResult();

  auto oh_last =
      b.create<mlir::arith::AddIOp>(loc, oh_base, tile_h_last).getResult();
  auto ow_last =
      b.create<mlir::arith::AddIOp>(loc, ow_base, tile_w_last).getResult();

  auto ih_min =
      b.create<mlir::arith::SubIOp>(
           loc, b.create<mlir::arith::MulIOp>(loc, oh_base, stride_h), pad_h)
          .getResult();
  auto iw_min =
      b.create<mlir::arith::SubIOp>(
           loc, b.create<mlir::arith::MulIOp>(loc, ow_base, stride_w), pad_w)
          .getResult();
  auto ih_max =
      b.create<mlir::arith::SubIOp>(
           loc,
           b.create<mlir::arith::AddIOp>(
               loc, b.create<mlir::arith::MulIOp>(loc, oh_last, stride_h),
               b.create<mlir::arith::MulIOp>(loc, kernel_h_last, dil_h)),
           pad_h)
          .getResult();
  auto iw_max =
      b.create<mlir::arith::SubIOp>(
           loc,
           b.create<mlir::arith::AddIOp>(
               loc, b.create<mlir::arith::MulIOp>(loc, ow_last, stride_w),
               b.create<mlir::arith::MulIOp>(loc, kernel_w_last, dil_w)),
           pad_w)
          .getResult();

  auto ih_min_ge0 = b.create<mlir::arith::CmpIOp>(
                         loc, mlir::arith::CmpIPredicate::sge, ih_min, c0)
                        .getResult();
  auto iw_min_ge0 = b.create<mlir::arith::CmpIOp>(
                         loc, mlir::arith::CmpIPredicate::sge, iw_min, c0)
                        .getResult();
  auto ih_max_lt = b.create<mlir::arith::CmpIOp>(
                        loc, mlir::arith::CmpIPredicate::slt, ih_max, input_h)
                       .getResult();
  auto iw_max_lt = b.create<mlir::arith::CmpIOp>(
                        loc, mlir::arith::CmpIPredicate::slt, iw_max, input_w)
                       .getResult();
  auto interior =
      b.create<mlir::arith::AndIOp>(loc, ih_min_ge0, iw_min_ge0).getResult();
  interior =
      b.create<mlir::arith::AndIOp>(loc, interior, ih_max_lt).getResult();
  return b.create<mlir::arith::AndIOp>(loc, interior, iw_max_lt).getResult();
}

std::optional<ActivationKind> parse_activation_kind(mlir::Operation *op) {
  if (!op) {
    return std::nullopt;
  }
  auto attr = op->getAttrOfType<mlir::StringAttr>("gfx.activation_kind");
  if (!attr) {
    return std::nullopt;
  }
  const auto name = attr.getValue();
  if (name == "Relu")
    return ActivationKind::Relu;
  if (name == "Sigmoid")
    return ActivationKind::Sigmoid;
  if (name == "Tanh")
    return ActivationKind::Tanh;
  if (name == "Elu")
    return ActivationKind::Elu;
  if (name == "Prelu")
    return ActivationKind::Prelu;
  if (name == "Gelu")
    return ActivationKind::Gelu;
  if (name == "Swish")
    return ActivationKind::Swish;
  if (name == "HSwish")
    return ActivationKind::HSwish;
  if (name == "HSigmoid")
    return ActivationKind::HSigmoid;
  if (name == "Abs")
    return ActivationKind::Abs;
  if (name == "Sign")
    return ActivationKind::Sign;
  return std::nullopt;
}

std::optional<llvm::StringRef> module_string_attr(mlir::Operation *op,
                                                  llvm::StringRef name) {
  if (!op) {
    return std::nullopt;
  }
  auto module = op->getParentOfType<mlir::ModuleOp>();
  if (!module) {
    return std::nullopt;
  }
  auto attr = module->getAttrOfType<mlir::StringAttr>(name);
  if (!attr) {
    return std::nullopt;
  }
  return attr.getValue();
}

std::optional<int64_t> module_int_attr(mlir::Operation *op,
                                       llvm::StringRef name) {
  if (!op) {
    return std::nullopt;
  }
  auto module = op->getParentOfType<mlir::ModuleOp>();
  if (!module) {
    return std::nullopt;
  }
  auto attr = module->getAttrOfType<mlir::IntegerAttr>(name);
  if (!attr) {
    return std::nullopt;
  }
  return attr.getInt();
}

bool module_bool_attr(mlir::Operation *op, llvm::StringRef name) {
  if (!op) {
    return false;
  }
  auto module = op->getParentOfType<mlir::ModuleOp>();
  if (!module) {
    return false;
  }
  auto attr = module->getAttrOfType<mlir::BoolAttr>(name);
  return attr && attr.getValue();
}

mlir::Value apply_activation(mlir::OpBuilder &b, mlir::Location loc,
                             mlir::Value x, ActivationKind kind, float alpha,
                             mlir::Type elem_ty) {
  auto make_float_attr = [&](double v) {
    return mlir::FloatAttr::get(elem_ty, v);
  };
  auto zero = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(0.0));
  switch (kind) {
  case ActivationKind::Relu:
    return b.create<mlir::arith::MaximumFOp>(loc, x, zero);
  case ActivationKind::Sigmoid: {
    auto neg = b.create<mlir::arith::NegFOp>(loc, x);
    auto exp = b.create<mlir::math::ExpOp>(loc, neg);
    auto one = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(1.0));
    auto denom = b.create<mlir::arith::AddFOp>(loc, one, exp);
    return b.create<mlir::arith::DivFOp>(loc, one, denom);
  }
  case ActivationKind::Tanh:
    return b.create<mlir::math::TanhOp>(loc, x);
  case ActivationKind::Elu: {
    auto alpha_c =
        b.create<mlir::arith::ConstantOp>(loc, make_float_attr(alpha));
    auto exp = b.create<mlir::math::ExpOp>(loc, x);
    auto one = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(1.0));
    auto expm1 = b.create<mlir::arith::SubFOp>(loc, exp, one);
    auto neg_branch = b.create<mlir::arith::MulFOp>(loc, alpha_c, expm1);
    auto cond = b.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OGT, x, zero);
    return b.create<mlir::arith::SelectOp>(loc, cond, x, neg_branch);
  }
  case ActivationKind::Prelu: {
    auto alpha_c =
        b.create<mlir::arith::ConstantOp>(loc, make_float_attr(alpha));
    auto neg_branch = b.create<mlir::arith::MulFOp>(loc, alpha_c, x);
    auto cond = b.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OGT, x, zero);
    return b.create<mlir::arith::SelectOp>(loc, cond, x, neg_branch);
  }
  case ActivationKind::Gelu: {
    auto half = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(0.5));
    auto one = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(1.0));
    auto c0 =
        b.create<mlir::arith::ConstantOp>(loc, make_float_attr(0.79788456));
    auto c1 = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(0.044715));
    auto x2 = b.create<mlir::arith::MulFOp>(loc, x, x);
    auto x3 = b.create<mlir::arith::MulFOp>(loc, x2, x);
    auto inner = b.create<mlir::arith::AddFOp>(
        loc, x, b.create<mlir::arith::MulFOp>(loc, c1, x3));
    auto tanh_arg = b.create<mlir::arith::MulFOp>(loc, c0, inner);
    auto tanh = b.create<mlir::math::TanhOp>(loc, tanh_arg);
    auto term = b.create<mlir::arith::AddFOp>(loc, one, tanh);
    auto mul = b.create<mlir::arith::MulFOp>(
        loc, half, b.create<mlir::arith::MulFOp>(loc, x, term));
    return mul;
  }
  case ActivationKind::Swish: {
    auto one = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(1.0));
    auto cond = b.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OGE, x, zero);
    auto neg = b.create<mlir::arith::NegFOp>(loc, x);
    auto exp_neg = b.create<mlir::math::ExpOp>(loc, neg);
    auto pos_denom = b.create<mlir::arith::AddFOp>(loc, one, exp_neg);
    auto pos = b.create<mlir::arith::DivFOp>(loc, x, pos_denom);
    auto exp_pos = b.create<mlir::math::ExpOp>(loc, x);
    auto neg_denom = b.create<mlir::arith::AddFOp>(loc, one, exp_pos);
    auto neg_num = b.create<mlir::arith::MulFOp>(loc, x, exp_pos);
    auto neg_res = b.create<mlir::arith::DivFOp>(loc, neg_num, neg_denom);
    return b.create<mlir::arith::SelectOp>(loc, cond, pos, neg_res);
  }
  case ActivationKind::HSwish:
  case ActivationKind::HSigmoid: {
    auto three = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(3.0));
    auto six = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(6.0));
    auto inv6 =
        b.create<mlir::arith::ConstantOp>(loc, make_float_attr(1.0 / 6.0));
    auto x_plus = b.create<mlir::arith::AddFOp>(loc, x, three);
    auto max0 = b.create<mlir::arith::MaximumFOp>(loc, x_plus, zero);
    auto min6 = b.create<mlir::arith::MinimumFOp>(loc, max0, six);
    auto hsig = b.create<mlir::arith::MulFOp>(loc, min6, inv6);
    if (kind == ActivationKind::HSwish) {
      return b.create<mlir::arith::MulFOp>(loc, x, hsig);
    }
    return hsig;
  }
  case ActivationKind::Abs: {
    auto cond = b.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OLT, x, zero);
    auto neg = b.create<mlir::arith::NegFOp>(loc, x);
    return b.create<mlir::arith::SelectOp>(loc, cond, neg, x);
  }
  case ActivationKind::Sign: {
    auto one = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(1.0));
    auto neg_one =
        b.create<mlir::arith::ConstantOp>(loc, make_float_attr(-1.0));
    auto gt = b.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OGT, x, zero);
    auto lt = b.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OLT, x, zero);
    auto pos = b.create<mlir::arith::SelectOp>(loc, gt, one, zero);
    return b.create<mlir::arith::SelectOp>(loc, lt, neg_one, pos);
  }
  default:
    break;
  }
  return x;
}

struct BnGlobals {
  mlir::Value scale;
  mlir::Value bias;
};

mlir::Value append_func_arg(mlir::func::FuncOp func, mlir::Type type,
                            mlir::Location loc) {
  auto fn_type = func.getFunctionType();
  llvm::SmallVector<mlir::Type, 8> inputs(fn_type.getInputs().begin(),
                                          fn_type.getInputs().end());
  inputs.push_back(type);
  auto new_type =
      mlir::FunctionType::get(func.getContext(), inputs, fn_type.getResults());
  func.setType(new_type);
  return func.getBody().front().addArgument(type, loc);
}

std::optional<mlir::Value>
prepare_bias_global(mlir::linalg::Conv2DNchwFchwOp op,
                    mlir::IRRewriter &rewriter, mlir::Type elem_ty) {
  int64_t channels = -1;
  if (auto channels_attr =
          op->getAttrOfType<mlir::IntegerAttr>("gfx.bias_channels")) {
    channels = channels_attr.getInt();
  } else if (auto bias_attr =
                 op->getAttrOfType<mlir::DenseFPElementsAttr>("gfx.bias")) {
    auto bias_type =
        mlir::dyn_cast<mlir::RankedTensorType>(bias_attr.getType());
    if (!bias_type || bias_type.getRank() != 1) {
      return std::nullopt;
    }
    channels = bias_type.getDimSize(0);
  } else {
    return std::nullopt;
  }
  if (channels <= 0) {
    return std::nullopt;
  }
  auto func = op->getParentOfType<mlir::func::FuncOp>();
  if (!func) {
    return std::nullopt;
  }
  auto loc = op.getLoc();
  auto memref_type = mlir::MemRefType::get({channels}, elem_ty);
  return append_func_arg(func, memref_type, loc);
}

std::optional<BnGlobals> prepare_bn_globals(mlir::linalg::Conv2DNchwFchwOp op,
                                            mlir::IRRewriter &rewriter,
                                            mlir::Type elem_ty) {
  auto scale_attr =
      op->getAttrOfType<mlir::DenseFPElementsAttr>("gfx.bn_scale");
  auto bias_attr = op->getAttrOfType<mlir::DenseFPElementsAttr>("gfx.bn_bias");
  if (!scale_attr || !bias_attr) {
    return std::nullopt;
  }
  auto scale_type =
      mlir::dyn_cast<mlir::RankedTensorType>(scale_attr.getType());
  auto bias_type = mlir::dyn_cast<mlir::RankedTensorType>(bias_attr.getType());
  if (!scale_type || !bias_type ||
      scale_type.getShape() != bias_type.getShape()) {
    return std::nullopt;
  }
  auto func = op->getParentOfType<mlir::func::FuncOp>();
  if (!func) {
    return std::nullopt;
  }
  auto loc = op.getLoc();
  auto memref_type = mlir::MemRefType::get(scale_type.getShape(), elem_ty);
  auto scale_arg = append_func_arg(func, memref_type, loc);
  auto bias_arg = append_func_arg(func, memref_type, loc);
  return BnGlobals{scale_arg, bias_arg};
}

bool extract_hw(mlir::DenseIntElementsAttr attr, int64_t &h, int64_t &w) {
  if (!attr) {
    return false;
  }
  const auto count = static_cast<size_t>(attr.getNumElements());
  if (count < 2) {
    return false;
  }
  auto it = attr.getValues<int64_t>().begin();
  for (size_t i = 0; i + 2 < count; ++i) {
    ++it;
  }
  h = *it++;
  w = *it++;
  return true;
}

bool extract_addi_offset(mlir::Value value, int64_t &offset) {
  if (auto add = value.getDefiningOp<mlir::arith::AddIOp>()) {
    if (auto cst = add.getLhs().getDefiningOp<mlir::arith::ConstantIndexOp>()) {
      offset = cst.value();
      return true;
    }
    if (auto cst = add.getRhs().getDefiningOp<mlir::arith::ConstantIndexOp>()) {
      offset = cst.value();
      return true;
    }
  }
  return false;
}

mlir::Value strip_memref_casts(mlir::Value value) {
  mlir::Value current = value;
  while (current) {
    if (auto cast = current.getDefiningOp<mlir::memref::CastOp>()) {
      current = cast.getSource();
      continue;
    }
    if (auto subview = current.getDefiningOp<mlir::memref::SubViewOp>()) {
      current = subview.getSource();
      continue;
    }
    if (auto view = current.getDefiningOp<mlir::memref::ViewOp>()) {
      current = view.getSource();
      continue;
    }
    break;
  }
  return current;
}

bool is_before_in_same_block(mlir::Operation *candidate,
                             mlir::Operation *anchor) {
  return candidate && anchor && candidate->getBlock() == anchor->getBlock() &&
         candidate->isBeforeInBlock(anchor);
}

bool lower_conv2d_op(mlir::linalg::Conv2DNchwFchwOp op,
                     mlir::IRRewriter &rewriter) {
  const bool debug = gfx_mlir_debug_enabled();
  auto fail = [&](const char *reason) {
    if (debug) {
      llvm::errs() << "[GFX][MLIR] Conv2D lowering skip: " << reason << "\n";
    }
    return false;
  };

  if (op.getInputs().size() < 2 || op.getOutputs().empty()) {
    return fail("expected 2 inputs and 1 output");
  }

  mlir::Value input = op.getInputs()[0];
  mlir::Value filter = op.getInputs()[1];
  mlir::Value output = op.getOutputs()[0];

  auto in_type = mlir::dyn_cast<mlir::MemRefType>(input.getType());
  auto w_type = mlir::dyn_cast<mlir::MemRefType>(filter.getType());
  auto out_type = mlir::dyn_cast<mlir::MemRefType>(output.getType());
  if (!in_type || !w_type || !out_type) {
    return fail("non-memref operands (not bufferized)");
  }
  if (in_type.getRank() != 4 || w_type.getRank() != 4 ||
      out_type.getRank() != 4) {
    return fail("non-4D shapes");
  }

  auto elem_ty = out_type.getElementType();
  if (!mlir::isa<mlir::FloatType>(elem_ty)) {
    return fail("non-float element type");
  }

  mlir::Value conv_input = input;
  mlir::Value conv_output = output;
  mlir::Value input_base = strip_memref_casts(input);
  mlir::Value output_base = strip_memref_casts(output);
  int64_t pad_h = 0;
  int64_t pad_w = 0;
  int64_t pad_end_h = 0;
  int64_t pad_end_w = 0;
  mlir::Operation *pad_fill_loop = nullptr;
  mlir::Operation *pad_copy_loop = nullptr;
  const bool has_pad_begin_attr = op->getAttr("gfx.pad_begin") != nullptr;
  if (auto pad_attr =
          op->getAttrOfType<mlir::DenseIntElementsAttr>("gfx.pad_begin")) {
    (void)extract_hw(pad_attr, pad_h, pad_w);
  }
  if (auto pad_attr =
          op->getAttrOfType<mlir::DenseIntElementsAttr>("gfx.pad_end")) {
    (void)extract_hw(pad_attr, pad_end_h, pad_end_w);
  }

  bool found_copy = false;
  auto find_outer_loop = [](mlir::Operation *op) -> mlir::Operation * {
    mlir::Operation *outer = nullptr;
    for (auto *cur = op; cur; cur = cur->getParentOp()) {
      if (mlir::isa<mlir::scf::ForOp, mlir::scf::ParallelOp>(cur)) {
        outer = cur;
      }
    }
    return outer;
  };
  if (auto func = op->getParentOfType<mlir::func::FuncOp>()) {
    func.walk([&](mlir::memref::StoreOp store) {
      if (strip_memref_casts(store.getMemRef()) != input_base) {
        return;
      }
      if (!found_copy) {
        if (auto load =
                store.getValue().getDefiningOp<mlir::memref::LoadOp>()) {
          conv_input = strip_memref_casts(load.getMemRef());
          auto indices = store.getIndices();
          if (!has_pad_begin_attr && indices.size() >= 4) {
            (void)extract_addi_offset(indices[2], pad_h);
            (void)extract_addi_offset(indices[3], pad_w);
          }
          pad_copy_loop = find_outer_loop(store);
          found_copy = true;
          return;
        }
      }
      if (!pad_fill_loop) {
        if (auto cst =
                store.getValue().getDefiningOp<mlir::arith::ConstantOp>()) {
          if (auto fattr = mlir::dyn_cast<mlir::FloatAttr>(cst.getValue())) {
            if (fattr.getValueAsDouble() == 0.0) {
              pad_fill_loop = find_outer_loop(store);
            }
          }
        }
      }
    });
  }

  if (auto func = op->getParentOfType<mlir::func::FuncOp>()) {
    auto padded_type = mlir::dyn_cast<mlir::MemRefType>(input_base.getType());
    if (padded_type && padded_type.getRank() == 4) {
      auto padded_shape = padded_type.getShape();
      if (padded_shape[2] != mlir::ShapedType::kDynamic &&
          padded_shape[3] != mlir::ShapedType::kDynamic) {
        const int64_t orig_h = padded_shape[2] - pad_h - pad_end_h;
        const int64_t orig_w = padded_shape[3] - pad_w - pad_end_w;
        for (auto arg : func.getArguments()) {
          auto arg_type = mlir::dyn_cast<mlir::MemRefType>(arg.getType());
          if (!arg_type || arg_type.getRank() != 4) {
            continue;
          }
          if (arg_type.getElementType() != padded_type.getElementType()) {
            continue;
          }
          auto arg_shape = arg_type.getShape();
          if (arg_shape[0] != padded_shape[0] ||
              arg_shape[1] != padded_shape[1]) {
            continue;
          }
          if (arg_shape[2] == orig_h && arg_shape[3] == orig_w) {
            conv_input = strip_memref_casts(arg);
            break;
          }
        }
      }
    }
  }

  const bool input_is_padded =
      (pad_fill_loop != nullptr || pad_copy_loop != nullptr);
  const bool using_padded_input =
      input_is_padded && (strip_memref_casts(conv_input) == input_base);
  bool prefer_parallel = true;
  if (auto module = op->getParentOfType<mlir::ModuleOp>()) {
    if (auto attr =
            module->getAttrOfType<mlir::BoolAttr>("gfx.prefer_parallel")) {
      prefer_parallel = attr.getValue();
    }
  }
  const bool has_explicit_padding =
      (pad_h != 0 || pad_w != 0 || pad_end_h != 0 || pad_end_w != 0 ||
       pad_fill_loop != nullptr || pad_copy_loop != nullptr);
  if (!prefer_parallel && !has_explicit_padding) {
    return fail("prefer_parallel disabled and no padding");
  }
  mlir::linalg::FillOp fill_op;
  mlir::Value zero_init;
  for (auto *user : output_base.getUsers()) {
    auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(user);
    if (!fill || !is_before_in_same_block(fill, op)) {
      continue;
    }
    if (fill.getInputs().empty()) {
      continue;
    }
    auto cst = fill.getInputs()[0].getDefiningOp<mlir::arith::ConstantOp>();
    if (!cst) {
      continue;
    }
    if (auto fattr = mlir::dyn_cast<mlir::FloatAttr>(cst.getValue())) {
      if (fattr.getValueAsDouble() == 0.0) {
        fill_op = fill;
        zero_init = fill.getInputs()[0];
        break;
      }
    }
  }

  int64_t stride_h = 0, stride_w = 0;
  int64_t dil_h = 0, dil_w = 0;
  if (!extract_hw(op.getStrides(), stride_h, stride_w) ||
      !extract_hw(op.getDilations(), dil_h, dil_w)) {
    return fail("missing strides/dilations");
  }

  if (gfx_log_debug_enabled()) {
    gfx_log_debug("MLIR") << "Conv2D pad detect: conv_input="
                          << (using_padded_input ? "padded" : "orig")
                          << " pad_h=" << pad_h << " pad_w=" << pad_w
                          << " pad_end_h=" << pad_end_h
                          << " pad_end_w=" << pad_end_w << " conv_output="
                          << (conv_output == output ? "alloc" : "arg");
  }
  if (debug) {
    llvm::errs() << "[GFX][MLIR] Conv2D operands: in=" << input.getType()
                 << " w=" << filter.getType() << " out=" << output.getType()
                 << " pad_begin=(" << pad_h << "," << pad_w << ")"
                 << " pad_end=(" << pad_end_h << "," << pad_end_w << ")\n";
  }

  if (op->getNumResults() > 0) {
    if (op->getNumResults() != 1 ||
        op->getResult(0).getType() != output.getType()) {
      return fail("unexpected result count/type");
    }
  }

  const auto loc = op.getLoc();
  rewriter.setInsertionPoint(op);
  auto c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto c4 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 4);
  const auto explicit_thread_h = module_int_attr(op, "gfx.dispatch_threads_h");
  const auto explicit_thread_w = module_int_attr(op, "gfx.dispatch_threads_w");
  const auto explicit_tile_h = module_int_attr(op, "gfx.dispatch_tile_h");
  const auto explicit_tile_w = module_int_attr(op, "gfx.dispatch_tile_w");
  const auto explicit_channel_block =
      module_int_attr(op, "gfx.dispatch_channel_block");
  const auto explicit_channel_block_accumulation =
      module_string_attr(op, "gfx.dispatch_channel_block_accumulation");
  const bool has_explicit_dispatch =
      explicit_thread_h.has_value() && explicit_thread_w.has_value();
  int64_t thread_h = explicit_thread_h.value_or(4);
  int64_t thread_w = explicit_thread_w.value_or(4);
  int64_t channel_block =
      std::max<int64_t>(1, explicit_channel_block.value_or(1));
  const int64_t static_c_out = out_type.getDimSize(1);
  if (static_c_out == mlir::ShapedType::kDynamic || static_c_out <= 0 ||
      static_c_out % channel_block != 0) {
    channel_block = 1;
  }
  const bool serial_channel_block_accumulation =
      channel_block > 1 && explicit_channel_block_accumulation.has_value() &&
      *explicit_channel_block_accumulation == "serial";
  const bool weights_packed_oc4 =
      module_bool_attr(op, "gfx.conv2d_weights_packed_oc4");
  // Keep micro-tiling minimal unless a shared algorithm plan explicitly
  // tells lowering that a wider structural family is selected.
  int64_t micro_h = 1;
  int64_t micro_w = 1;
  if (!has_explicit_dispatch) {
    if (auto kind = module_string_attr(op, "gfx.conv_algorithm_kind")) {
      if (*kind == "direct_1x1") {
        thread_h = 8;
        thread_w = 8;
      } else if (*kind == "direct_3x3_stride2") {
        thread_h = 4;
        thread_w = 8;
      } else if (*kind == "depthwise_direct") {
        thread_h = 8;
        thread_w = 4;
      }
    }
  }
  const int64_t kThreadH = thread_h;
  const int64_t kThreadW = thread_w;
  const int64_t requested_tile_h =
      std::max<int64_t>(explicit_tile_h.value_or(kThreadH * micro_h), kThreadH);
  const int64_t requested_tile_w =
      std::max<int64_t>(explicit_tile_w.value_or(kThreadW * micro_w), kThreadW);
  micro_h = std::max<int64_t>(1, (requested_tile_h + kThreadH - 1) / kThreadH);
  micro_w = std::max<int64_t>(1, (requested_tile_w + kThreadW - 1) / kThreadW);
  const int64_t tile_h = kThreadH * micro_h;
  const int64_t tile_w = kThreadW * micro_w;
  if (auto module = op->getParentOfType<mlir::ModuleOp>()) {
    auto *ctx = module.getContext();
    module->setAttr("gfx.dispatch_tile_h",
                    mlir::IntegerAttr::get(mlir::IndexType::get(ctx), tile_h));
    module->setAttr("gfx.dispatch_tile_w",
                    mlir::IntegerAttr::get(mlir::IndexType::get(ctx), tile_w));
    module->setAttr(
        "gfx.dispatch_threads_h",
        mlir::IntegerAttr::get(mlir::IndexType::get(ctx), kThreadH));
    module->setAttr(
        "gfx.dispatch_threads_w",
        mlir::IntegerAttr::get(mlir::IndexType::get(ctx), kThreadW));
    module->setAttr(
        "gfx.dispatch_channel_block",
        mlir::IntegerAttr::get(mlir::IndexType::get(ctx), channel_block));
    module->setAttr("gfx.dispatch_channel_block_accumulation",
                    mlir::StringAttr::get(ctx, serial_channel_block_accumulation
                                                   ? "serial"
                                                   : "fused"));
    module->setAttr("gfx.parallel_loop_dims",
                    mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), 5));
  }
  auto tileH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_h);
  auto tileW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_w);
  auto threadH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, kThreadH);
  auto threadW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, kThreadW);
  auto microH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, micro_h);
  auto microW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, micro_w);
  auto tileH_minus1 =
      rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_h - 1);
  auto tileW_minus1 =
      rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_w - 1);
  auto strideH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, stride_h);
  auto strideW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, stride_w);
  auto dilH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, dil_h);
  auto dilW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, dil_w);
  auto channelBlock =
      rewriter.create<mlir::arith::ConstantIndexOp>(loc, channel_block);
  auto channelBlockMinus1 =
      rewriter.create<mlir::arith::ConstantIndexOp>(loc, channel_block - 1);

  auto get_dim = [&](mlir::Value value, int64_t dim) -> mlir::Value {
    auto mem_ty = mlir::dyn_cast<mlir::MemRefType>(value.getType());
    if (mem_ty && dim < mem_ty.getRank()) {
      const int64_t sz = mem_ty.getDimSize(dim);
      if (sz != mlir::ShapedType::kDynamic) {
        return rewriter.create<mlir::arith::ConstantIndexOp>(loc, sz);
      }
    }
    return rewriter.create<mlir::memref::DimOp>(loc, value, dim);
  };

  auto N = get_dim(conv_output, 0);
  auto C_out = get_dim(conv_output, 1);
  auto H_out = get_dim(conv_output, 2);
  auto W_out = get_dim(conv_output, 3);
  auto C_in = get_dim(conv_input, 1);
  auto H_in = get_dim(conv_input, 2);
  auto W_in = get_dim(conv_input, 3);
  auto kH = get_dim(filter, 2);
  auto kW = get_dim(filter, 3);
  mlir::Value packed_filter_linear;
  if (weights_packed_oc4) {
    if (!w_type.hasStaticShape()) {
      return fail("packed OC4 weights require static filter shape");
    }
    const int64_t filter_elements = w_type.getNumElements();
    if (filter_elements <= 0) {
      return fail("packed OC4 weights require positive filter elements");
    }
    auto flat_filter_layout =
        mlir::StridedLayoutAttr::get(w_type.getContext(), 0, {1});
    auto flat_filter_type =
        mlir::MemRefType::get({filter_elements}, w_type.getElementType(),
                              flat_filter_layout, w_type.getMemorySpace());
    llvm::SmallVector<int64_t, 1> flat_sizes{filter_elements};
    llvm::SmallVector<int64_t, 1> flat_strides{1};
    packed_filter_linear =
        rewriter
            .create<mlir::memref::ReinterpretCastOp>(
                loc, flat_filter_type, filter, 0, flat_sizes, flat_strides)
            .getResult();
  }
  auto load_filter = [&](mlir::OpBuilder &b_load, mlir::Location load_loc,
                         mlir::Value oc, mlir::Value ic, mlir::Value kh,
                         mlir::Value kw) -> mlir::Value {
    if (!weights_packed_oc4) {
      return b_load
          .create<mlir::memref::LoadOp>(load_loc, filter,
                                        mlir::ValueRange{oc, ic, kh, kw})
          .getResult();
    }

    auto oc_block =
        b_load.create<mlir::arith::DivSIOp>(load_loc, oc, c4).getResult();
    auto oc_lane =
        b_load.create<mlir::arith::RemSIOp>(load_loc, oc, c4).getResult();
    auto packed_flat =
        b_load.create<mlir::arith::MulIOp>(load_loc, oc_block, C_in)
            .getResult();
    packed_flat = b_load.create<mlir::arith::AddIOp>(load_loc, packed_flat, ic)
                      .getResult();
    packed_flat = b_load.create<mlir::arith::MulIOp>(load_loc, packed_flat, kH)
                      .getResult();
    packed_flat = b_load.create<mlir::arith::AddIOp>(load_loc, packed_flat, kh)
                      .getResult();
    packed_flat = b_load.create<mlir::arith::MulIOp>(load_loc, packed_flat, kW)
                      .getResult();
    packed_flat = b_load.create<mlir::arith::AddIOp>(load_loc, packed_flat, kw)
                      .getResult();
    packed_flat = b_load.create<mlir::arith::MulIOp>(load_loc, packed_flat, c4)
                      .getResult();
    packed_flat =
        b_load.create<mlir::arith::AddIOp>(load_loc, packed_flat, oc_lane)
            .getResult();
    return b_load
        .create<mlir::memref::LoadOp>(load_loc, packed_filter_linear,
                                      mlir::ValueRange{packed_flat})
        .getResult();
  };

  // Map parallel loops to output C/H/W tiles with thread-level micro-tiles,
  // so Vulkan dispatch grid maps blocks to [C_out, H_tiles, W_tiles].
  const auto activation = parse_activation_kind(op);
  float activation_alpha = 0.0f;
  if (auto alpha_attr =
          op->getAttrOfType<mlir::FloatAttr>("gfx.activation_alpha")) {
    activation_alpha = static_cast<float>(alpha_attr.getValueAsDouble());
  }
  auto bias_global = prepare_bias_global(op, rewriter, elem_ty);
  auto bn_globals = prepare_bn_globals(op, rewriter, elem_ty);

  auto h_tiles_num =
      rewriter.create<mlir::arith::AddIOp>(loc, H_out, tileH_minus1);
  auto w_tiles_num =
      rewriter.create<mlir::arith::AddIOp>(loc, W_out, tileW_minus1);
  auto H_tiles = rewriter.create<mlir::arith::DivSIOp>(loc, h_tiles_num, tileH);
  auto W_tiles = rewriter.create<mlir::arith::DivSIOp>(loc, w_tiles_num, tileW);
  auto c_blocks_num =
      rewriter.create<mlir::arith::AddIOp>(loc, C_out, channelBlockMinus1);
  auto C_out_blocks =
      rewriter.create<mlir::arith::DivSIOp>(loc, c_blocks_num, channelBlock);

  auto par = rewriter.create<mlir::scf::ParallelOp>(
      loc, mlir::ValueRange{c0, c0, c0, c0, c0},
      mlir::ValueRange{C_out_blocks, H_tiles, W_tiles, threadH, threadW},
      mlir::ValueRange{c1, c1, c1, c1, c1});

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(par.getBody()->getTerminator());

  auto ivs = par.getInductionVars();
  auto iv_oc_base =
      rewriter.create<mlir::arith::MulIOp>(loc, ivs[0], channelBlock);
  auto iv_oh_base = rewriter.create<mlir::arith::MulIOp>(loc, ivs[1], tileH);
  auto iv_ow_base = rewriter.create<mlir::arith::MulIOp>(loc, ivs[2], tileW);
  auto iv_th = ivs[3];
  auto iv_tw = ivs[4];
  auto iv_oh_off = rewriter.create<mlir::arith::MulIOp>(loc, iv_th, microH);
  auto iv_ow_off = rewriter.create<mlir::arith::MulIOp>(loc, iv_tw, microW);
  llvm::SmallVector<mlir::Value, 4> oh_vals;
  llvm::SmallVector<mlir::Value, 4> oh_in_vals;
  oh_vals.reserve(static_cast<size_t>(micro_h));
  oh_in_vals.reserve(static_cast<size_t>(micro_h));
  for (int64_t mh = 0; mh < micro_h; ++mh) {
    mlir::Value off = iv_oh_off;
    if (mh != 0) {
      off = rewriter.create<mlir::arith::AddIOp>(
          loc, iv_oh_off,
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, mh));
    }
    auto oh = rewriter.create<mlir::arith::AddIOp>(loc, iv_oh_base, off);
    oh_vals.push_back(oh);
    oh_in_vals.push_back(rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::slt, oh, H_out));
  }
  llvm::SmallVector<mlir::Value, 8> ow_vals;
  llvm::SmallVector<mlir::Value, 8> ow_in_vals;
  ow_vals.reserve(static_cast<size_t>(micro_w));
  ow_in_vals.reserve(static_cast<size_t>(micro_w));
  for (int64_t mw = 0; mw < micro_w; ++mw) {
    mlir::Value off = iv_ow_off;
    if (mw != 0) {
      off = rewriter.create<mlir::arith::AddIOp>(
          loc, iv_ow_off,
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, mw));
    }
    auto ow = rewriter.create<mlir::arith::AddIOp>(loc, iv_ow_base, off);
    ow_vals.push_back(ow);
    ow_in_vals.push_back(rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::slt, ow, W_out));
  }
  const bool needs_bounds = has_explicit_padding && !using_padded_input;
  const int64_t effective_pad_h = needs_bounds ? pad_h : 0;
  const int64_t effective_pad_w = needs_bounds ? pad_w : 0;
  auto padH =
      rewriter.create<mlir::arith::ConstantIndexOp>(loc, effective_pad_h);
  auto padW =
      rewriter.create<mlir::arith::ConstantIndexOp>(loc, effective_pad_w);
  rewriter.create<mlir::scf::ForOp>(
      loc, c0, N, c1, mlir::ValueRange{},
      [&](mlir::OpBuilder &b, mlir::Location body_loc, mlir::Value iv_n,
          mlir::ValueRange) {
        const int64_t lane_count = micro_h * micro_w;
        llvm::SmallVector<mlir::Value, 8> lane_in;
        llvm::SmallVector<mlir::Value, 8> lane_oh;
        llvm::SmallVector<mlir::Value, 8> lane_ow;
        lane_in.reserve(static_cast<size_t>(lane_count));
        lane_oh.reserve(static_cast<size_t>(lane_count));
        lane_ow.reserve(static_cast<size_t>(lane_count));
        for (int64_t mh = 0; mh < micro_h; ++mh) {
          for (int64_t mw = 0; mw < micro_w; ++mw) {
            auto in = b.create<mlir::arith::AndIOp>(body_loc, oh_in_vals[mh],
                                                    ow_in_vals[mw]);
            lane_in.push_back(in);
            lane_oh.push_back(oh_vals[mh]);
            lane_ow.push_back(ow_vals[mw]);
          }
        }
        mlir::Value tile_in = lane_in.front();
        for (size_t i = 1; i < lane_in.size(); ++i) {
          tile_in = b.create<mlir::arith::OrIOp>(body_loc, tile_in, lane_in[i]);
        }

        auto emit_tile_body = [&](mlir::OpBuilder &b_tile,
                                  mlir::Location tile_loc, bool guard_lanes,
                                  bool guard_input_bounds) {
          auto zero = b_tile.create<mlir::arith::ConstantOp>(
              tile_loc, elem_ty, b_tile.getFloatAttr(elem_ty, 0.0));
          // Convolution accumulates into a fresh zero-initialized tile.
          // Reloading the destination buffer here can leak stale values
          // once larger execution windows start reusing GPU buffers.
          const int64_t acc_count = lane_count * channel_block;
          auto acc_index = [&](int64_t oc_lane, int64_t lane) -> size_t {
            return static_cast<size_t>(oc_lane * lane_count + lane);
          };
          auto oc_value = [&](mlir::OpBuilder &builder,
                              mlir::Location value_loc,
                              int64_t oc_lane) -> mlir::Value {
            if (oc_lane == 0) {
              return iv_oc_base;
            }
            auto lane_offset = builder.create<mlir::arith::ConstantIndexOp>(
                value_loc, oc_lane);
            return builder
                .create<mlir::arith::AddIOp>(value_loc, iv_oc_base, lane_offset)
                .getResult();
          };
          llvm::SmallVector<mlir::Value, 8> lane_ih_base;
          llvm::SmallVector<mlir::Value, 8> lane_iw_base;
          lane_ih_base.reserve(static_cast<size_t>(lane_count));
          lane_iw_base.reserve(static_cast<size_t>(lane_count));
          for (int64_t i = 0; i < lane_count; ++i) {
            auto oh_mul = b_tile.create<mlir::arith::MulIOp>(
                tile_loc, lane_oh[i], strideH);
            auto ow_mul = b_tile.create<mlir::arith::MulIOp>(
                tile_loc, lane_ow[i], strideW);
            mlir::Value ih_base = oh_mul.getResult();
            mlir::Value iw_base = ow_mul.getResult();
            if (needs_bounds) {
              ih_base = b_tile
                            .create<mlir::arith::SubIOp>(tile_loc, ih_base,
                                                          padH)
                            .getResult();
              iw_base = b_tile
                            .create<mlir::arith::SubIOp>(tile_loc, iw_base,
                                                          padW)
                            .getResult();
            }
            lane_ih_base.push_back(ih_base);
            lane_iw_base.push_back(iw_base);
          }
          auto input_coord = [&](mlir::OpBuilder &builder,
                                 mlir::Location value_loc, int64_t lane,
                                 mlir::Value kh_mul,
                                 mlir::Value kw_mul) -> std::array<mlir::Value, 2> {
            return {builder
                        .create<mlir::arith::AddIOp>(
                            value_loc, lane_ih_base[lane], kh_mul)
                        .getResult(),
                    builder
                        .create<mlir::arith::AddIOp>(
                            value_loc, lane_iw_base[lane], kw_mul)
                        .getResult()};
          };
          auto load_input_lane =
              [&](mlir::OpBuilder &b_acc, mlir::Location acc_loc,
                  int64_t lane, mlir::Value iv_ic, mlir::Value ih,
                  mlir::Value iw) -> mlir::Value {
            mlir::Value valid = guard_lanes ? lane_in[lane] : mlir::Value{};
            if (needs_bounds && guard_input_bounds) {
              auto ge_h = b_acc.create<mlir::arith::CmpIOp>(
                  acc_loc, mlir::arith::CmpIPredicate::sge, ih, c0);
              auto lt_h = b_acc.create<mlir::arith::CmpIOp>(
                  acc_loc, mlir::arith::CmpIPredicate::slt, ih, H_in);
              auto ge_w = b_acc.create<mlir::arith::CmpIOp>(
                  acc_loc, mlir::arith::CmpIPredicate::sge, iw, c0);
              auto lt_w = b_acc.create<mlir::arith::CmpIOp>(
                  acc_loc, mlir::arith::CmpIPredicate::slt, iw, W_in);
              auto in_h =
                  b_acc.create<mlir::arith::AndIOp>(acc_loc, ge_h, lt_h);
              auto in_w =
                  b_acc.create<mlir::arith::AndIOp>(acc_loc, ge_w, lt_w);
              auto in_bounds =
                  b_acc.create<mlir::arith::AndIOp>(acc_loc, in_h, in_w);
              valid = valid ? b_acc
                                  .create<mlir::arith::AndIOp>(
                                      acc_loc, valid, in_bounds)
                                  .getResult()
                            : in_bounds.getResult();
            }
            if (valid) {
              auto ifop = b_acc.create<mlir::scf::IfOp>(
                  acc_loc, zero.getType(), valid, /*withElse=*/true);
              {
                mlir::OpBuilder::InsertionGuard guard(b_acc);
                b_acc.setInsertionPointToStart(&ifop.getThenRegion().front());
                auto in_val =
                    b_acc
                        .create<mlir::memref::LoadOp>(
                            acc_loc, conv_input,
                            mlir::ValueRange{iv_n, iv_ic, ih, iw})
                        .getResult();
                b_acc.create<mlir::scf::YieldOp>(acc_loc,
                                                 mlir::ValueRange{in_val});
              }
              {
                mlir::OpBuilder::InsertionGuard guard(b_acc);
                b_acc.setInsertionPointToStart(&ifop.getElseRegion().front());
                b_acc.create<mlir::scf::YieldOp>(acc_loc,
                                                 mlir::ValueRange{zero});
              }
              return ifop.getResult(0);
            }
            return b_acc
                .create<mlir::memref::LoadOp>(
                    acc_loc, conv_input,
                    mlir::ValueRange{iv_n, iv_ic, ih, iw})
                .getResult();
          };
          const int64_t static_kW = w_type.getDimSize(3);
          struct WidthReuseCoord {
            int64_t mh = 0;
            int64_t input_w_offset = 0;
          };
          llvm::SmallVector<WidthReuseCoord, 8> width_reuse_coords;
          llvm::SmallVector<size_t, 16> width_reuse_map;
          const bool can_try_width_reuse =
              static_kW != mlir::ShapedType::kDynamic && static_kW > 1 &&
              micro_w > 1 && stride_w > 0 && dil_w > 0;
          if (can_try_width_reuse) {
            std::map<std::pair<int64_t, int64_t>, size_t> coord_to_index;
            width_reuse_map.reserve(
                static_cast<size_t>(lane_count * static_kW));
            for (int64_t lane = 0; lane < lane_count; ++lane) {
              const int64_t mh = lane / micro_w;
              const int64_t mw = lane % micro_w;
              for (int64_t kw = 0; kw < static_kW; ++kw) {
                const int64_t input_w_offset = mw * stride_w + kw * dil_w;
                const auto key = std::make_pair(mh, input_w_offset);
                auto it = coord_to_index.find(key);
                if (it == coord_to_index.end()) {
                  const size_t index = width_reuse_coords.size();
                  coord_to_index.emplace(key, index);
                  width_reuse_coords.push_back({mh, input_w_offset});
                  width_reuse_map.push_back(index);
                } else {
                  width_reuse_map.push_back(it->second);
                }
              }
            }
            if (width_reuse_coords.size() >=
                static_cast<size_t>(lane_count * static_kW)) {
              width_reuse_coords.clear();
              width_reuse_map.clear();
            } else if (auto module = op->getParentOfType<mlir::ModuleOp>()) {
              auto *module_ctx = module.getContext();
              module->setAttr("gfx.conv_spatial_input_reuse",
                              mlir::StringAttr::get(module_ctx, "width"));
              module->setAttr(
                  "gfx.conv_spatial_input_reuse_lanes",
                  mlir::IntegerAttr::get(
                      mlir::IntegerType::get(module_ctx, 64),
                      static_cast<int64_t>(lane_count)));
              module->setAttr(
                  "gfx.conv_spatial_input_reuse_unique_width_loads",
                  mlir::IntegerAttr::get(
                      mlir::IntegerType::get(module_ctx, 64),
                      static_cast<int64_t>(width_reuse_coords.size())));
            }
          }
          const bool use_width_reuse =
              !width_reuse_coords.empty() && !guard_lanes &&
              !guard_input_bounds && !serial_channel_block_accumulation;
          if (serial_channel_block_accumulation) {
            auto apply_post_serial =
                [&](mlir::OpBuilder &b_post, mlir::Location post_loc,
                    mlir::Value acc, mlir::Value oc) -> mlir::Value {
              if (bn_globals.has_value()) {
                auto scale =
                    b_post
                        .create<mlir::memref::LoadOp>(
                            post_loc, bn_globals->scale, mlir::ValueRange{oc})
                        .getResult();
                auto bias =
                    b_post
                        .create<mlir::memref::LoadOp>(
                            post_loc, bn_globals->bias, mlir::ValueRange{oc})
                        .getResult();
                auto mul =
                    b_post.create<mlir::arith::MulFOp>(post_loc, acc, scale)
                        .getResult();
                acc = b_post.create<mlir::arith::AddFOp>(post_loc, mul, bias)
                          .getResult();
              }
              if (bias_global.has_value()) {
                auto bias =
                    b_post
                        .create<mlir::memref::LoadOp>(post_loc, *bias_global,
                                                      mlir::ValueRange{oc})
                        .getResult();
                acc = b_post.create<mlir::arith::AddFOp>(post_loc, acc, bias)
                          .getResult();
              }
              if (activation.has_value()) {
                acc = apply_activation(b_post, post_loc, acc, *activation,
                                       activation_alpha, elem_ty);
              }
              return acc;
            };
            auto store_lane_serial = [&](mlir::OpBuilder &b_store,
                                         mlir::Location store_loc,
                                         mlir::Value acc, mlir::Value oc,
                                         int64_t lane) {
              if (!guard_lanes) {
                b_store.create<mlir::memref::StoreOp>(
                    store_loc, acc, conv_output,
                    mlir::ValueRange{iv_n, oc, lane_oh[lane], lane_ow[lane]});
                return;
              }
              auto if_store = b_store.create<mlir::scf::IfOp>(
                  store_loc, lane_in[lane], /*withElse=*/false);
              {
                mlir::OpBuilder::InsertionGuard guard(b_store);
                b_store.setInsertionPointToStart(
                    &if_store.getThenRegion().front());
                b_store.create<mlir::memref::StoreOp>(
                    store_loc, acc, conv_output,
                    mlir::ValueRange{iv_n, oc, lane_oh[lane], lane_ow[lane]});
              }
            };
            b_tile.create<mlir::scf::ForOp>(
                tile_loc, c0, channelBlock, c1, mlir::ValueRange{},
                [&](mlir::OpBuilder &b_oc, mlir::Location oc_loc,
                    mlir::Value iv_oc_lane, mlir::ValueRange) {
                  auto oc = b_oc.create<mlir::arith::AddIOp>(oc_loc, iv_oc_base,
                                                             iv_oc_lane)
                                .getResult();
                  llvm::SmallVector<mlir::Value, 8> acc_init(
                      static_cast<size_t>(lane_count), zero);
                  auto for_ic = b_oc.create<mlir::scf::ForOp>(
                      oc_loc, c0, C_in, c1, acc_init,
                      [&](mlir::OpBuilder &b3, mlir::Location loc3,
                          mlir::Value iv_ic, mlir::ValueRange iter_args) {
                        llvm::SmallVector<mlir::Value, 8> acc_ic(
                            iter_args.begin(), iter_args.end());
                        auto emit_kernel_step =
                            [&](mlir::OpBuilder &b5, mlir::Location loc5,
                                mlir::Value step_ic, mlir::Value step_kh,
                                mlir::Value step_kw,
                                const llvm::SmallVectorImpl<mlir::Value>
                                    &acc_kw) {
                              auto kh_mul = b5.create<mlir::arith::MulIOp>(
                                  loc5, step_kh, dilH);
                              auto kw_mul = b5.create<mlir::arith::MulIOp>(
                                  loc5, step_kw, dilW);
                              auto w_val = load_filter(b5, loc5, oc, step_ic,
                                                       step_kh, step_kw);
                              llvm::SmallVector<mlir::Value, 8> next_accs;
                              next_accs.reserve(acc_kw.size());
                              for (int64_t i = 0; i < lane_count; ++i) {
                                auto coord = input_coord(b5, loc5, i, kh_mul,
                                                         kw_mul);
                                auto in_val = load_input_lane(
                                    b5, loc5, i, step_ic, coord[0], coord[1]);
                                auto mul =
                                    b5.create<mlir::arith::MulFOp>(
                                          loc5, in_val, w_val)
                                        .getResult();
                                next_accs.push_back(
                                    b5.create<mlir::arith::AddFOp>(
                                          loc5, acc_kw[i], mul)
                                        .getResult());
                              }
                              return next_accs;
                            };
                        auto for_kh = b3.create<mlir::scf::ForOp>(
                            loc3, c0, kH, c1, acc_ic,
                            [&](mlir::OpBuilder &b4, mlir::Location loc4,
                                mlir::Value iv_kh,
                                mlir::ValueRange iter_args2) {
                              llvm::SmallVector<mlir::Value, 8> acc_kh(
                                  iter_args2.begin(), iter_args2.end());
                              auto for_kw = b4.create<mlir::scf::ForOp>(
                                  loc4, c0, kW, c1, acc_kh,
                                  [&](mlir::OpBuilder &b5, mlir::Location loc5,
                                      mlir::Value iv_kw,
                                      mlir::ValueRange iter_args3) {
                                    llvm::SmallVector<mlir::Value, 8> acc_kw(
                                        iter_args3.begin(), iter_args3.end());
                                    auto next_accs = emit_kernel_step(
                                        b5, loc5, iv_ic, iv_kh, iv_kw, acc_kw);
                                    b5.create<mlir::scf::YieldOp>(loc5,
                                                                  next_accs);
                                  });
                              b4.create<mlir::scf::YieldOp>(
                                  loc4, for_kw.getResults());
                            });
                        b3.create<mlir::scf::YieldOp>(loc3,
                                                      for_kh.getResults());
                      });

                  llvm::SmallVector<mlir::Value, 8> acc_final(
                      for_ic.getResults().begin(), for_ic.getResults().end());
                  for (int64_t i = 0; i < lane_count; ++i) {
                    auto acc =
                        apply_post_serial(b_oc, oc_loc, acc_final[i], oc);
                    store_lane_serial(b_oc, oc_loc, acc, oc, i);
                  }
                  b_oc.create<mlir::scf::YieldOp>(oc_loc);
                });
            return;
          }
          llvm::SmallVector<mlir::Value, 8> acc_init(
              static_cast<size_t>(acc_count), zero);
          auto for_ic = b_tile.create<mlir::scf::ForOp>(
              tile_loc, c0, C_in, c1, acc_init,
              [&](mlir::OpBuilder &b3, mlir::Location loc3, mlir::Value iv_ic,
                  mlir::ValueRange iter_args) {
                llvm::SmallVector<mlir::Value, 8> acc_ic(iter_args.begin(),
                                                         iter_args.end());
                auto emit_kernel_width_reuse =
                    [&](mlir::OpBuilder &b4, mlir::Location loc4,
                        mlir::Value step_ic, mlir::Value step_kh,
                        const llvm::SmallVectorImpl<mlir::Value> &acc_kh) {
                      auto kh_mul = b4.create<mlir::arith::MulIOp>(
                          loc4, step_kh, dilH);
                      llvm::SmallVector<mlir::Value, 8> cached_inputs;
                      cached_inputs.reserve(width_reuse_coords.size());
                      for (const auto &coord : width_reuse_coords) {
                        const int64_t representative_lane = coord.mh * micro_w;
                        auto input_h =
                            b4.create<mlir::arith::AddIOp>(
                                  loc4, lane_ih_base[representative_lane],
                                  kh_mul)
                                .getResult();
                        mlir::Value input_w =
                            lane_iw_base[representative_lane];
                        if (coord.input_w_offset != 0) {
                          input_w =
                              b4.create<mlir::arith::AddIOp>(
                                    loc4, input_w,
                                    b4.create<mlir::arith::ConstantIndexOp>(
                                        loc4, coord.input_w_offset))
                                  .getResult();
                        }
                        cached_inputs.push_back(load_input_lane(
                            b4, loc4, representative_lane, step_ic, input_h,
                            input_w));
                      }

                      llvm::SmallVector<mlir::Value, 8> next_accs(
                          acc_kh.begin(), acc_kh.end());
                      for (int64_t kw = 0; kw < static_kW; ++kw) {
                        auto step_kw =
                            b4.create<mlir::arith::ConstantIndexOp>(loc4, kw);
                        for (int64_t oc_lane = 0; oc_lane < channel_block;
                             ++oc_lane) {
                          auto oc = oc_value(b4, loc4, oc_lane);
                          auto w_val = load_filter(b4, loc4, oc, step_ic,
                                                   step_kh, step_kw);
                          for (int64_t lane = 0; lane < lane_count; ++lane) {
                            const auto acc_idx = acc_index(oc_lane, lane);
                            const size_t reuse_idx =
                                width_reuse_map[static_cast<size_t>(
                                    lane * static_kW + kw)];
                            auto mul = b4.create<mlir::arith::MulFOp>(
                                             loc4, cached_inputs[reuse_idx],
                                             w_val)
                                           .getResult();
                            next_accs[acc_idx] =
                                b4.create<mlir::arith::AddFOp>(
                                      loc4, next_accs[acc_idx], mul)
                                    .getResult();
                          }
                        }
                      }
                      return next_accs;
                    };
                auto emit_kernel_step =
                    [&](mlir::OpBuilder &b5, mlir::Location loc5,
                        mlir::Value step_ic, mlir::Value step_kh,
                        mlir::Value step_kw,
                        const llvm::SmallVectorImpl<mlir::Value> &acc_kw) {
                      auto kh_mul = b5.create<mlir::arith::MulIOp>(
                          loc5, step_kh, dilH);
                      auto kw_mul = b5.create<mlir::arith::MulIOp>(
                          loc5, step_kw, dilW);
                      llvm::SmallVector<mlir::Value, 8> next_accs;
                      next_accs.reserve(acc_kw.size());
                      llvm::SmallVector<mlir::Value, 8> lane_inputs;
                      lane_inputs.reserve(static_cast<size_t>(lane_count));
                      for (int64_t i = 0; i < lane_count; ++i) {
                        auto coord =
                            input_coord(b5, loc5, i, kh_mul, kw_mul);
                        lane_inputs.push_back(load_input_lane(
                            b5, loc5, i, step_ic, coord[0], coord[1]));
                      }
                      for (int64_t oc_lane = 0; oc_lane < channel_block;
                           ++oc_lane) {
                        auto oc = oc_value(b5, loc5, oc_lane);
                        auto w_val = load_filter(b5, loc5, oc, step_ic,
                                                 step_kh, step_kw);
                        for (int64_t i = 0; i < lane_count; ++i) {
                          const auto acc_idx = acc_index(oc_lane, i);
                          auto mul = b5.create<mlir::arith::MulFOp>(
                                           loc5, lane_inputs[i], w_val)
                                         .getResult();
                          next_accs.push_back(
                              b5.create<mlir::arith::AddFOp>(
                                    loc5, acc_kw[acc_idx], mul)
                                  .getResult());
                        }
                      }
                      return next_accs;
                    };
                auto for_kh = b3.create<mlir::scf::ForOp>(
                    loc3, c0, kH, c1, acc_ic,
                    [&](mlir::OpBuilder &b4, mlir::Location loc4,
                        mlir::Value iv_kh, mlir::ValueRange iter_args2) {
                      llvm::SmallVector<mlir::Value, 8> acc_kh(
                          iter_args2.begin(), iter_args2.end());
                      if (use_width_reuse) {
                        auto next_accs = emit_kernel_width_reuse(
                            b4, loc4, iv_ic, iv_kh, acc_kh);
                        b4.create<mlir::scf::YieldOp>(loc4, next_accs);
                        return;
                      }
                      auto for_kw = b4.create<mlir::scf::ForOp>(
                          loc4, c0, kW, c1, acc_kh,
                          [&](mlir::OpBuilder &b5, mlir::Location loc5,
                              mlir::Value iv_kw, mlir::ValueRange iter_args3) {
                            llvm::SmallVector<mlir::Value, 8> acc_kw(
                                iter_args3.begin(), iter_args3.end());
                            auto next_accs = emit_kernel_step(
                                b5, loc5, iv_ic, iv_kh, iv_kw, acc_kw);
                            b5.create<mlir::scf::YieldOp>(loc5, next_accs);
                          });
                      b4.create<mlir::scf::YieldOp>(loc4, for_kw.getResults());
                    });
                b3.create<mlir::scf::YieldOp>(loc3, for_kh.getResults());
              });

          llvm::SmallVector<mlir::Value, 8> acc_final(
              for_ic.getResults().begin(), for_ic.getResults().end());
          auto apply_post = [&](mlir::Value acc,
                                mlir::Value oc) -> mlir::Value {
            if (bn_globals.has_value()) {
              auto scale =
                  b.create<mlir::memref::LoadOp>(body_loc, bn_globals->scale,
                                                 mlir::ValueRange{oc})
                      .getResult();
              auto bias = b.create<mlir::memref::LoadOp>(
                               body_loc, bn_globals->bias, mlir::ValueRange{oc})
                              .getResult();
              auto mul = b.create<mlir::arith::MulFOp>(body_loc, acc, scale)
                             .getResult();
              acc = b.create<mlir::arith::AddFOp>(body_loc, mul, bias)
                        .getResult();
            }
            if (bias_global.has_value()) {
              auto bias = b.create<mlir::memref::LoadOp>(body_loc, *bias_global,
                                                         mlir::ValueRange{oc})
                              .getResult();
              acc = b.create<mlir::arith::AddFOp>(body_loc, acc, bias)
                        .getResult();
            }
            if (activation.has_value()) {
              acc = apply_activation(b, body_loc, acc, *activation,
                                     activation_alpha, elem_ty);
            }
            return acc;
          };

          for (int64_t oc_lane = 0; oc_lane < channel_block; ++oc_lane) {
            auto oc = oc_value(b, body_loc, oc_lane);
            for (int64_t i = 0; i < lane_count; ++i) {
              const auto acc_idx = acc_index(oc_lane, i);
              acc_final[acc_idx] = apply_post(acc_final[acc_idx], oc);
            }
          }

          if (!guard_lanes) {
            for (int64_t oc_lane = 0; oc_lane < channel_block; ++oc_lane) {
              auto oc = oc_value(b_tile, tile_loc, oc_lane);
              for (int64_t i = 0; i < lane_count; ++i) {
                b_tile.create<mlir::memref::StoreOp>(
                    tile_loc, acc_final[acc_index(oc_lane, i)], conv_output,
                    mlir::ValueRange{iv_n, oc, lane_oh[i], lane_ow[i]});
              }
            }
            return;
          }

          for (int64_t oc_lane = 0; oc_lane < channel_block; ++oc_lane) {
            auto oc = oc_value(b_tile, tile_loc, oc_lane);
            for (int64_t i = 0; i < lane_count; ++i) {
              auto if_store = b_tile.create<mlir::scf::IfOp>(
                  tile_loc, lane_in[i], /*withElse=*/false);
              {
                mlir::OpBuilder::InsertionGuard guard(b_tile);
                b_tile.setInsertionPointToStart(
                    &if_store.getThenRegion().front());
                b_tile.create<mlir::memref::StoreOp>(
                    tile_loc, acc_final[acc_index(oc_lane, i)], conv_output,
                    mlir::ValueRange{iv_n, oc, lane_oh[i], lane_ow[i]});
              }
            }
          }
        };

        const auto oh_tile_end =
            b.create<mlir::arith::AddIOp>(body_loc, iv_oh_base, tileH)
                .getResult();
        const auto ow_tile_end =
            b.create<mlir::arith::AddIOp>(body_loc, iv_ow_base, tileW)
                .getResult();
        const auto full_tile_h =
            b.create<mlir::arith::CmpIOp>(
                 body_loc, mlir::arith::CmpIPredicate::sle, oh_tile_end, H_out)
                .getResult();
        const auto full_tile_w =
            b.create<mlir::arith::CmpIOp>(
                 body_loc, mlir::arith::CmpIPredicate::sle, ow_tile_end, W_out)
                .getResult();
        const auto full_tile =
            b.create<mlir::arith::AndIOp>(body_loc, full_tile_h, full_tile_w)
                .getResult();
        if (!needs_bounds) {
          auto if_full_tile =
              b.create<mlir::scf::IfOp>(body_loc, full_tile, /*withElse=*/true);
          {
            mlir::OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(&if_full_tile.getThenRegion().front());
            emit_tile_body(b, body_loc, /*guard_lanes=*/false,
                           /*guard_input_bounds=*/false);
          }
          {
            mlir::OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(&if_full_tile.getElseRegion().front());
            auto if_tile = b.create<mlir::scf::IfOp>(body_loc, tile_in,
                                                     /*withElse=*/false);
            {
              mlir::OpBuilder::InsertionGuard inner_guard(b);
              b.setInsertionPointToStart(&if_tile.getThenRegion().front());
              emit_tile_body(b, body_loc, /*guard_lanes=*/true,
                             /*guard_input_bounds=*/false);
            }
          }
        } else {
          const auto input_interior_tile =
              build_conv_tile_input_interior_condition(
                  b, body_loc, iv_oh_base, iv_ow_base, tileH, tileW, strideH,
                  strideW, dilH, dilW, kH, kW, padH, padW, H_in, W_in, c0, c1);
          const auto full_interior_tile =
              b.create<mlir::arith::AndIOp>(body_loc, input_interior_tile,
                                            full_tile)
                  .getResult();
          auto if_interior_tile = b.create<mlir::scf::IfOp>(
              body_loc, full_interior_tile, /*withElse=*/true);
          {
            mlir::OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(
                &if_interior_tile.getThenRegion().front());
            emit_tile_body(b, body_loc, /*guard_lanes=*/false,
                           /*guard_input_bounds=*/false);
          }
          {
            mlir::OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(
                &if_interior_tile.getElseRegion().front());
            auto if_full_tile = b.create<mlir::scf::IfOp>(body_loc, full_tile,
                                                          /*withElse=*/true);
            {
              mlir::OpBuilder::InsertionGuard full_guard(b);
              b.setInsertionPointToStart(&if_full_tile.getThenRegion().front());
              emit_tile_body(b, body_loc, /*guard_lanes=*/false,
                             /*guard_input_bounds=*/true);
            }
            {
              mlir::OpBuilder::InsertionGuard full_guard(b);
              b.setInsertionPointToStart(&if_full_tile.getElseRegion().front());
              auto if_tile = b.create<mlir::scf::IfOp>(body_loc, tile_in,
                                                       /*withElse=*/false);
              {
                mlir::OpBuilder::InsertionGuard inner_guard(b);
                b.setInsertionPointToStart(&if_tile.getThenRegion().front());
                emit_tile_body(b, body_loc, /*guard_lanes=*/true,
                               /*guard_input_bounds=*/true);
              }
            }
          }
        }
        b.create<mlir::scf::YieldOp>(body_loc);
      });

  if (op->getNumResults() > 0) {
    rewriter.replaceOp(op, conv_output);
  } else {
    rewriter.eraseOp(op);
  }
  if (fill_op) {
    rewriter.eraseOp(fill_op);
  }
  if (!using_padded_input) {
    auto input_alloc = input_base.getDefiningOp<mlir::memref::AllocOp>();
    auto input_alloca = input_base.getDefiningOp<mlir::memref::AllocaOp>();
    if (pad_copy_loop && pad_copy_loop->getParentOp()) {
      rewriter.eraseOp(pad_copy_loop);
    }
    if (pad_fill_loop && pad_fill_loop->getParentOp() &&
        pad_fill_loop != pad_copy_loop) {
      rewriter.eraseOp(pad_fill_loop);
    }
    if (input_alloc && input_alloc->use_empty()) {
      rewriter.eraseOp(input_alloc);
    } else if (input_alloca && input_alloca->use_empty()) {
      rewriter.eraseOp(input_alloca);
    }
  }
  return true;
}

bool lower_depthwise_group_conv_generic(mlir::linalg::GenericOp op,
                                        mlir::IRRewriter &rewriter) {
  const bool debug = gfx_mlir_debug_enabled();
  if (debug) {
    llvm::errs() << "[GFX][MLIR] Depthwise GroupConv lowering begin\n";
  }
  auto fail = [&](const char *reason) {
    if (debug) {
      llvm::errs() << "[GFX][MLIR] Depthwise GroupConv lowering skip: "
                   << reason << "\n";
    }
    return false;
  };

  auto module = op->getParentOfType<mlir::ModuleOp>();
  if (!module) {
    return fail("missing module");
  }
  auto algorithm =
      module->getAttrOfType<mlir::StringAttr>("gfx.conv_algorithm_kind");
  const bool depthwise_route =
      algorithm && algorithm.getValue() == "depthwise_direct";
  const auto backend_domain =
      module_string_attr(op, "gfx.stage_manifest.backend_domain");
  const bool apple_msl_custom_kernel =
      backend_domain && *backend_domain == "apple_msl";
  if (!depthwise_route && !apple_msl_custom_kernel) {
    return fail("not a depthwise route");
  }
  auto direct_attr =
      op->getAttrOfType<mlir::BoolAttr>("gfx.depthwise_nchw_direct");
  if (!direct_attr || !direct_attr.getValue()) {
    return fail("not direct NCHW depthwise generic");
  }
  bool prefer_parallel = true;
  if (auto attr =
          module->getAttrOfType<mlir::BoolAttr>("gfx.prefer_parallel")) {
    prefer_parallel = attr.getValue();
  }
  if (!prefer_parallel && !apple_msl_custom_kernel) {
    return fail("prefer_parallel disabled");
  }
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      op.getNumLoops() != 6) {
    return fail("unexpected generic arity/loop count");
  }
  const auto iters = op.getIteratorTypesArray();
  if (iters.size() != 6 || iters[0] != mlir::utils::IteratorType::parallel ||
      iters[1] != mlir::utils::IteratorType::parallel ||
      iters[2] != mlir::utils::IteratorType::parallel ||
      iters[3] != mlir::utils::IteratorType::parallel ||
      iters[4] != mlir::utils::IteratorType::reduction ||
      iters[5] != mlir::utils::IteratorType::reduction) {
    return fail("unexpected iterator types");
  }
  if (op->getAttr("gfx.bias_channels") || op->getAttr("gfx.bias") ||
      op->getAttr("gfx.bn_scale") || op->getAttr("gfx.bn_bias")) {
    return fail("fused bias/bn not supported for depthwise generic");
  }

  mlir::Value input = strip_memref_casts(op.getDpsInputs()[0]);
  mlir::Value filter = strip_memref_casts(op.getDpsInputs()[1]);
  mlir::Value output = strip_memref_casts(op.getDpsInits()[0]);

  int64_t pad_h = 0;
  int64_t pad_w = 0;
  (void)extract_hw(
      op->getAttrOfType<mlir::DenseIntElementsAttr>("gfx.pad_begin"), pad_h,
      pad_w);
  int64_t pad_end_h = 0;
  int64_t pad_end_w = 0;
  (void)extract_hw(op->getAttrOfType<mlir::DenseIntElementsAttr>("gfx.pad_end"),
                   pad_end_h, pad_end_w);

  mlir::Value padded_input = input;
  mlir::Value compute_input = input;
  mlir::Operation *pad_fill_loop = nullptr;
  mlir::Operation *pad_copy_loop = nullptr;
  mlir::Operation *pad_subview_op = nullptr;
  auto find_outer_loop = [](mlir::Operation *nested) -> mlir::Operation * {
    mlir::Operation *outer = nullptr;
    for (auto *cur = nested; cur; cur = cur->getParentOp()) {
      if (mlir::isa<mlir::scf::ForOp, mlir::scf::ParallelOp>(cur)) {
        outer = cur;
      }
    }
    return outer;
  };
  if (auto func = op->getParentOfType<mlir::func::FuncOp>()) {
    func.walk([&](mlir::linalg::MapOp map) {
      if (!pad_fill_loop && strip_memref_casts(map.getInit()) == padded_input) {
        pad_fill_loop = map;
      }
    });
    func.walk([&](mlir::memref::CopyOp copy) {
      if (pad_copy_loop) {
        return;
      }
      auto target = copy.getTarget();
      auto target_base = strip_memref_casts(target);
      if (target_base != padded_input) {
        return;
      }
      auto source = strip_memref_casts(copy.getSource());
      auto source_type = mlir::dyn_cast<mlir::MemRefType>(source.getType());
      if (source == padded_input || !source_type ||
          source_type.getRank() != 4) {
        return;
      }
      if (auto subview = target.getDefiningOp<mlir::memref::SubViewOp>()) {
        auto offsets = subview.getStaticOffsets();
        if (offsets.size() >= 4) {
          if (offsets[2] != mlir::ShapedType::kDynamic) {
            pad_h = offsets[2];
          }
          if (offsets[3] != mlir::ShapedType::kDynamic) {
            pad_w = offsets[3];
          }
        }
        pad_subview_op = subview;
      }
      compute_input = source;
      pad_copy_loop = copy;
    });
    func.walk([&](mlir::memref::StoreOp store) {
      if (strip_memref_casts(store.getMemRef()) != padded_input) {
        return;
      }
      if (!pad_copy_loop) {
        if (auto load =
                store.getValue().getDefiningOp<mlir::memref::LoadOp>()) {
          auto source = strip_memref_casts(load.getMemRef());
          auto source_type = mlir::dyn_cast<mlir::MemRefType>(source.getType());
          if (source != padded_input && source_type &&
              source_type.getRank() == 4) {
            auto indices = store.getIndices();
            if (indices.size() >= 4) {
              (void)extract_addi_offset(indices[2], pad_h);
              (void)extract_addi_offset(indices[3], pad_w);
            }
            compute_input = source;
            pad_copy_loop = find_outer_loop(store);
            return;
          }
        }
      }
      if (!pad_fill_loop) {
        if (auto cst =
                store.getValue().getDefiningOp<mlir::arith::ConstantOp>()) {
          if (auto fattr = mlir::dyn_cast<mlir::FloatAttr>(cst.getValue())) {
            if (fattr.getValueAsDouble() == 0.0) {
              pad_fill_loop = find_outer_loop(store);
            }
          }
        }
      }
    });
  }
  if (compute_input == padded_input) {
    if (auto func = op->getParentOfType<mlir::func::FuncOp>()) {
      auto padded_type =
          mlir::dyn_cast<mlir::MemRefType>(padded_input.getType());
      auto output_base = strip_memref_casts(output);
      if (padded_type && padded_type.getRank() == 4) {
        const auto padded_shape = padded_type.getShape();
        const bool static_padded_hw =
            padded_shape[2] != mlir::ShapedType::kDynamic &&
            padded_shape[3] != mlir::ShapedType::kDynamic;
        if (static_padded_hw) {
          for (auto arg : func.getArguments()) {
            auto arg_base = strip_memref_casts(arg);
            if (arg_base == output_base) {
              continue;
            }
            auto arg_type =
                mlir::dyn_cast<mlir::MemRefType>(arg_base.getType());
            if (!arg_type || arg_type.getRank() != 4 ||
                arg_type.getElementType() != padded_type.getElementType()) {
              continue;
            }
            const auto arg_shape = arg_type.getShape();
            if (arg_shape[0] != padded_shape[0] ||
                arg_shape[1] != padded_shape[1] ||
                arg_shape[2] == mlir::ShapedType::kDynamic ||
                arg_shape[3] == mlir::ShapedType::kDynamic) {
              continue;
            }
            const int64_t inferred_pad_end_h =
                padded_shape[2] - arg_shape[2] - pad_h;
            const int64_t inferred_pad_end_w =
                padded_shape[3] - arg_shape[3] - pad_w;
            if (inferred_pad_end_h < 0 || inferred_pad_end_w < 0) {
              continue;
            }
            if ((pad_end_h != 0 && pad_end_h != inferred_pad_end_h) ||
                (pad_end_w != 0 && pad_end_w != inferred_pad_end_w)) {
              continue;
            }
            pad_end_h = inferred_pad_end_h;
            pad_end_w = inferred_pad_end_w;
            compute_input = arg_base;
            break;
          }
        }
      }
    }
  }
  if (compute_input != padded_input && (!pad_copy_loop || !pad_fill_loop)) {
    if (auto func = op->getParentOfType<mlir::func::FuncOp>()) {
      func.walk([&](mlir::linalg::MapOp map) {
        if (!pad_fill_loop &&
            strip_memref_casts(map.getInit()) == padded_input) {
          pad_fill_loop = map;
        }
      });
      func.walk([&](mlir::memref::CopyOp copy) {
        if (pad_copy_loop) {
          return;
        }
        auto target = copy.getTarget();
        if (strip_memref_casts(target) != padded_input ||
            strip_memref_casts(copy.getSource()) != compute_input) {
          return;
        }
        if (auto subview = target.getDefiningOp<mlir::memref::SubViewOp>()) {
          pad_subview_op = subview;
        }
        pad_copy_loop = copy;
      });
      func.walk([&](mlir::memref::StoreOp store) {
        if (strip_memref_casts(store.getMemRef()) != padded_input) {
          return;
        }
        if (!pad_copy_loop) {
          if (auto load =
                  store.getValue().getDefiningOp<mlir::memref::LoadOp>()) {
            if (strip_memref_casts(load.getMemRef()) == compute_input) {
              pad_copy_loop = find_outer_loop(store);
              return;
            }
          }
        }
        if (!pad_fill_loop) {
          if (auto cst =
                  store.getValue().getDefiningOp<mlir::arith::ConstantOp>()) {
            if (auto fattr = mlir::dyn_cast<mlir::FloatAttr>(cst.getValue())) {
              if (fattr.getValueAsDouble() == 0.0) {
                pad_fill_loop = find_outer_loop(store);
              }
            }
          }
        }
      });
    }
  }
  const bool use_unpadded_input = compute_input != padded_input;
  input = compute_input;
  auto in_type = mlir::dyn_cast<mlir::MemRefType>(input.getType());
  auto w_type = mlir::dyn_cast<mlir::MemRefType>(filter.getType());
  auto out_type = mlir::dyn_cast<mlir::MemRefType>(output.getType());
  if (!in_type || !w_type || !out_type) {
    return fail("non-memref operands");
  }
  if (in_type.getRank() != 4 || w_type.getRank() != 5 ||
      out_type.getRank() != 4) {
    return fail("unexpected operand ranks");
  }
  auto elem_ty = out_type.getElementType();
  if (!mlir::isa<mlir::FloatType>(elem_ty)) {
    return fail("non-float element type");
  }
  if (w_type.getDimSize(1) != 1 || w_type.getDimSize(2) != 1) {
    return fail("not depthwise [G,1,1,KH,KW] weights");
  }
  const int64_t out_c_static = out_type.getDimSize(1);
  const int64_t groups_static = w_type.getDimSize(0);
  if (out_c_static != mlir::ShapedType::kDynamic &&
      groups_static != mlir::ShapedType::kDynamic &&
      out_c_static != groups_static) {
    return fail("output channels != groups");
  }

  int64_t stride_h = 0;
  int64_t stride_w = 0;
  int64_t dil_h = 0;
  int64_t dil_w = 0;
  if (!extract_hw(op->getAttrOfType<mlir::DenseIntElementsAttr>("gfx.strides"),
                  stride_h, stride_w) ||
      !extract_hw(
          op->getAttrOfType<mlir::DenseIntElementsAttr>("gfx.dilations"), dil_h,
          dil_w)) {
    return fail("missing strides/dilations");
  }

  const auto loc = op.getLoc();
  rewriter.setInsertionPoint(op);
  auto c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
  const auto explicit_thread_h = module_int_attr(op, "gfx.dispatch_threads_h");
  const auto explicit_thread_w = module_int_attr(op, "gfx.dispatch_threads_w");
  const auto explicit_tile_h = module_int_attr(op, "gfx.dispatch_tile_h");
  const auto explicit_tile_w = module_int_attr(op, "gfx.dispatch_tile_w");
  int64_t thread_h = std::max<int64_t>(1, explicit_thread_h.value_or(8));
  int64_t thread_w = std::max<int64_t>(1, explicit_thread_w.value_or(4));
  const int64_t tile_h =
      std::max<int64_t>(explicit_tile_h.value_or(thread_h), thread_h);
  const int64_t tile_w =
      std::max<int64_t>(explicit_tile_w.value_or(thread_w), thread_w);
  int64_t micro_h = std::max<int64_t>(1, (tile_h + thread_h - 1) / thread_h);
  int64_t micro_w = std::max<int64_t>(1, (tile_w + thread_w - 1) / thread_w);
  const int64_t effective_tile_h = thread_h * micro_h;
  const int64_t effective_tile_w = thread_w * micro_w;
  auto *ctx = module.getContext();
  module->setAttr(
      "gfx.dispatch_tile_h",
      mlir::IntegerAttr::get(mlir::IndexType::get(ctx), effective_tile_h));
  module->setAttr(
      "gfx.dispatch_tile_w",
      mlir::IntegerAttr::get(mlir::IndexType::get(ctx), effective_tile_w));
  module->setAttr("gfx.dispatch_threads_h",
                  mlir::IntegerAttr::get(mlir::IndexType::get(ctx), thread_h));
  module->setAttr("gfx.dispatch_threads_w",
                  mlir::IntegerAttr::get(mlir::IndexType::get(ctx), thread_w));
  module->setAttr("gfx.parallel_loop_dims",
                  mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), 5));

  auto tileH =
      rewriter.create<mlir::arith::ConstantIndexOp>(loc, effective_tile_h);
  auto tileW =
      rewriter.create<mlir::arith::ConstantIndexOp>(loc, effective_tile_w);
  auto threadH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, thread_h);
  auto threadW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, thread_w);
  auto microH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, micro_h);
  auto microW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, micro_w);
  auto tileHMinus1 =
      rewriter.create<mlir::arith::ConstantIndexOp>(loc, effective_tile_h - 1);
  auto tileWMinus1 =
      rewriter.create<mlir::arith::ConstantIndexOp>(loc, effective_tile_w - 1);
  auto strideH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, stride_h);
  auto strideW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, stride_w);
  auto dilH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, dil_h);
  auto dilW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, dil_w);
  auto padH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, pad_h);
  auto padW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, pad_w);

  auto get_dim = [&](mlir::Value value, int64_t dim) -> mlir::Value {
    auto mem_ty = mlir::dyn_cast<mlir::MemRefType>(value.getType());
    if (mem_ty && dim < mem_ty.getRank()) {
      const int64_t sz = mem_ty.getDimSize(dim);
      if (sz != mlir::ShapedType::kDynamic) {
        return rewriter.create<mlir::arith::ConstantIndexOp>(loc, sz);
      }
    }
    return rewriter.create<mlir::memref::DimOp>(loc, value, dim);
  };

  auto N = get_dim(output, 0);
  auto C_out = get_dim(output, 1);
  auto H_out = get_dim(output, 2);
  auto W_out = get_dim(output, 3);
  auto H_in = get_dim(input, 2);
  auto W_in = get_dim(input, 3);
  auto kH = get_dim(filter, 3);
  auto kW = get_dim(filter, 4);
  auto hTilesNum =
      rewriter.create<mlir::arith::AddIOp>(loc, H_out, tileHMinus1);
  auto wTilesNum =
      rewriter.create<mlir::arith::AddIOp>(loc, W_out, tileWMinus1);
  auto HTiles = rewriter.create<mlir::arith::DivSIOp>(loc, hTilesNum, tileH);
  auto WTiles = rewriter.create<mlir::arith::DivSIOp>(loc, wTilesNum, tileW);

  mlir::linalg::FillOp fill_op;
  for (auto *user : output.getUsers()) {
    auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(user);
    if (!fill || !is_before_in_same_block(fill, op) ||
        fill.getInputs().empty()) {
      continue;
    }
    auto cst = fill.getInputs()[0].getDefiningOp<mlir::arith::ConstantOp>();
    if (!cst) {
      continue;
    }
    if (auto fattr = mlir::dyn_cast<mlir::FloatAttr>(cst.getValue())) {
      if (fattr.getValueAsDouble() == 0.0) {
        fill_op = fill;
        break;
      }
    }
  }

  if (debug) {
    llvm::errs() << "[GFX][MLIR] Depthwise GroupConv lowering emit parallel"
                 << " use_unpadded_input="
                 << (use_unpadded_input ? "true" : "false")
                 << " tile=" << effective_tile_h << "x" << effective_tile_w
                 << " threads=" << thread_h << "x" << thread_w << "\n";
  }

  auto par = rewriter.create<mlir::scf::ParallelOp>(
      loc, mlir::ValueRange{c0, c0, c0, c0, c0},
      mlir::ValueRange{C_out, HTiles, WTiles, threadH, threadW},
      mlir::ValueRange{c1, c1, c1, c1, c1});

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(par.getBody()->getTerminator());
  auto ivs = par.getInductionVars();
  auto iv_c = ivs[0];
  auto oh_base = rewriter.create<mlir::arith::MulIOp>(loc, ivs[1], tileH);
  auto ow_base = rewriter.create<mlir::arith::MulIOp>(loc, ivs[2], tileW);
  auto oh_off = rewriter.create<mlir::arith::MulIOp>(loc, ivs[3], microH);
  auto ow_off = rewriter.create<mlir::arith::MulIOp>(loc, ivs[4], microW);

  llvm::SmallVector<mlir::Value, 4> oh_vals;
  llvm::SmallVector<mlir::Value, 4> oh_in_vals;
  for (int64_t mh = 0; mh < micro_h; ++mh) {
    mlir::Value off = oh_off;
    if (mh != 0) {
      off = rewriter.create<mlir::arith::AddIOp>(
          loc, oh_off, rewriter.create<mlir::arith::ConstantIndexOp>(loc, mh));
    }
    auto oh = rewriter.create<mlir::arith::AddIOp>(loc, oh_base, off);
    oh_vals.push_back(oh);
    oh_in_vals.push_back(rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::slt, oh, H_out));
  }
  llvm::SmallVector<mlir::Value, 4> ow_vals;
  llvm::SmallVector<mlir::Value, 4> ow_in_vals;
  for (int64_t mw = 0; mw < micro_w; ++mw) {
    mlir::Value off = ow_off;
    if (mw != 0) {
      off = rewriter.create<mlir::arith::AddIOp>(
          loc, ow_off, rewriter.create<mlir::arith::ConstantIndexOp>(loc, mw));
    }
    auto ow = rewriter.create<mlir::arith::AddIOp>(loc, ow_base, off);
    ow_vals.push_back(ow);
    ow_in_vals.push_back(rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::slt, ow, W_out));
  }

  rewriter.create<mlir::scf::ForOp>(
      loc, c0, N, c1, mlir::ValueRange{},
      [&](mlir::OpBuilder &b, mlir::Location body_loc, mlir::Value iv_n,
          mlir::ValueRange) {
        const int64_t lane_count = micro_h * micro_w;
        llvm::SmallVector<mlir::Value, 8> lane_in;
        llvm::SmallVector<mlir::Value, 8> lane_oh;
        llvm::SmallVector<mlir::Value, 8> lane_ow;
        lane_in.reserve(static_cast<size_t>(lane_count));
        lane_oh.reserve(static_cast<size_t>(lane_count));
        lane_ow.reserve(static_cast<size_t>(lane_count));
        for (int64_t mh = 0; mh < micro_h; ++mh) {
          for (int64_t mw = 0; mw < micro_w; ++mw) {
            lane_in.push_back(b.create<mlir::arith::AndIOp>(
                body_loc, oh_in_vals[mh], ow_in_vals[mw]));
            lane_oh.push_back(oh_vals[mh]);
            lane_ow.push_back(ow_vals[mw]);
          }
        }
        mlir::Value any_lane = lane_in.front();
        for (size_t i = 1; i < lane_in.size(); ++i) {
          any_lane =
              b.create<mlir::arith::OrIOp>(body_loc, any_lane, lane_in[i]);
        }
        auto if_tile =
            b.create<mlir::scf::IfOp>(body_loc, any_lane, /*withElse=*/false);
        {
          mlir::OpBuilder::InsertionGuard if_guard(b);
          b.setInsertionPointToStart(&if_tile.getThenRegion().front());
          auto zero = b.create<mlir::arith::ConstantOp>(
              body_loc, elem_ty, b.getFloatAttr(elem_ty, 0.0));
          llvm::SmallVector<mlir::Value, 8> acc_init(
              static_cast<size_t>(lane_count), zero);
          auto for_kh = b.create<mlir::scf::ForOp>(
              body_loc, c0, kH, c1, acc_init,
              [&](mlir::OpBuilder &b2, mlir::Location loc2, mlir::Value iv_kh,
                  mlir::ValueRange iter_args) {
                llvm::SmallVector<mlir::Value, 8> acc_kh(iter_args.begin(),
                                                         iter_args.end());
                auto for_kw = b2.create<mlir::scf::ForOp>(
                    loc2, c0, kW, c1, acc_kh,
                    [&](mlir::OpBuilder &b3, mlir::Location loc3,
                        mlir::Value iv_kw, mlir::ValueRange iter_args2) {
                      llvm::SmallVector<mlir::Value, 8> next_accs;
                      next_accs.reserve(iter_args2.size());
                      auto kh_mul =
                          b3.create<mlir::arith::MulIOp>(loc3, iv_kh, dilH);
                      auto kw_mul =
                          b3.create<mlir::arith::MulIOp>(loc3, iv_kw, dilW);
                      auto w_val =
                          b3.create<mlir::memref::LoadOp>(
                                loc3, filter,
                                mlir::ValueRange{iv_c, c0, c0, iv_kh, iv_kw})
                              .getResult();
                      for (int64_t i = 0; i < lane_count; ++i) {
                        auto oh_mul = b3.create<mlir::arith::MulIOp>(
                            loc3, lane_oh[i], strideH);
                        auto ow_mul = b3.create<mlir::arith::MulIOp>(
                            loc3, lane_ow[i], strideW);
                        mlir::Value ih = b3.create<mlir::arith::AddIOp>(
                            loc3, oh_mul, kh_mul);
                        mlir::Value iw = b3.create<mlir::arith::AddIOp>(
                            loc3, ow_mul, kw_mul);
                        mlir::Value lane_valid = lane_in[i];
                        if (use_unpadded_input) {
                          if (pad_h != 0) {
                            ih = b3.create<mlir::arith::SubIOp>(loc3, ih, padH);
                          }
                          if (pad_w != 0) {
                            iw = b3.create<mlir::arith::SubIOp>(loc3, iw, padW);
                          }
                          auto ih_ge0 =
                              b3.create<mlir::arith::CmpIOp>(
                                    loc3, mlir::arith::CmpIPredicate::sge, ih,
                                    c0)
                                  .getResult();
                          auto iw_ge0 =
                              b3.create<mlir::arith::CmpIOp>(
                                    loc3, mlir::arith::CmpIPredicate::sge, iw,
                                    c0)
                                  .getResult();
                          auto ih_lt =
                              b3.create<mlir::arith::CmpIOp>(
                                    loc3, mlir::arith::CmpIPredicate::slt, ih,
                                    H_in)
                                  .getResult();
                          auto iw_lt =
                              b3.create<mlir::arith::CmpIOp>(
                                    loc3, mlir::arith::CmpIPredicate::slt, iw,
                                    W_in)
                                  .getResult();
                          auto h_valid = b3.create<mlir::arith::AndIOp>(
                                               loc3, ih_ge0, ih_lt)
                                             .getResult();
                          auto w_valid = b3.create<mlir::arith::AndIOp>(
                                               loc3, iw_ge0, iw_lt)
                                             .getResult();
                          auto in_bounds = b3.create<mlir::arith::AndIOp>(
                                                 loc3, h_valid, w_valid)
                                               .getResult();
                          lane_valid = b3.create<mlir::arith::AndIOp>(
                                             loc3, lane_valid, in_bounds)
                                           .getResult();
                        }
                        auto if_lane = b3.create<mlir::scf::IfOp>(
                            loc3, iter_args2[i].getType(), lane_valid,
                            /*withElse=*/true);
                        {
                          mlir::OpBuilder::InsertionGuard then_guard(b3);
                          b3.setInsertionPointToStart(
                              &if_lane.getThenRegion().front());
                          auto in_val =
                              b3.create<mlir::memref::LoadOp>(
                                    loc3, input,
                                    mlir::ValueRange{iv_n, iv_c, ih, iw})
                                  .getResult();
                          auto mul = b3.create<mlir::arith::MulFOp>(
                              loc3, in_val, w_val);
                          auto add = b3.create<mlir::arith::AddFOp>(
                                           loc3, iter_args2[i], mul)
                                         .getResult();
                          b3.create<mlir::scf::YieldOp>(loc3,
                                                        mlir::ValueRange{add});
                        }
                        {
                          mlir::OpBuilder::InsertionGuard else_guard(b3);
                          b3.setInsertionPointToStart(
                              &if_lane.getElseRegion().front());
                          b3.create<mlir::scf::YieldOp>(
                              loc3, mlir::ValueRange{iter_args2[i]});
                        }
                        next_accs.push_back(if_lane.getResult(0));
                      }
                      b3.create<mlir::scf::YieldOp>(loc3, next_accs);
                    });
                b2.create<mlir::scf::YieldOp>(loc2, for_kw.getResults());
              });
          llvm::SmallVector<mlir::Value, 8> acc_final(
              for_kh.getResults().begin(), for_kh.getResults().end());
          for (int64_t i = 0; i < lane_count; ++i) {
            auto if_lane = b.create<mlir::scf::IfOp>(body_loc, lane_in[i],
                                                     /*withElse=*/false);
            {
              mlir::OpBuilder::InsertionGuard lane_guard(b);
              b.setInsertionPointToStart(&if_lane.getThenRegion().front());
              b.create<mlir::memref::StoreOp>(
                  body_loc, acc_final[i], output,
                  mlir::ValueRange{iv_n, iv_c, lane_oh[i], lane_ow[i]});
            }
          }
        }
        b.create<mlir::scf::YieldOp>(body_loc);
      });

  if (debug) {
    llvm::errs()
        << "[GFX][MLIR] Depthwise GroupConv lowering rewrite original\n";
  }
  if (op->getNumResults() > 0) {
    rewriter.replaceOp(op, output);
  } else {
    rewriter.eraseOp(op);
  }
  if (debug) {
    llvm::errs() << "[GFX][MLIR] Depthwise GroupConv lowering cleanup begin\n";
  }
  if (fill_op) {
    if (debug) {
      llvm::errs() << "[GFX][MLIR] Depthwise GroupConv erase output fill\n";
    }
    rewriter.eraseOp(fill_op);
  }
  if (use_unpadded_input) {
    auto padded_alloc = padded_input.getDefiningOp<mlir::memref::AllocOp>();
    auto padded_alloca = padded_input.getDefiningOp<mlir::memref::AllocaOp>();
    if (pad_copy_loop && pad_copy_loop->getBlock()) {
      if (debug) {
        llvm::errs() << "[GFX][MLIR] Depthwise GroupConv erase pad copy\n";
      }
      rewriter.eraseOp(pad_copy_loop);
    }
    if (pad_subview_op && pad_subview_op->getBlock() &&
        pad_subview_op->use_empty()) {
      if (debug) {
        llvm::errs() << "[GFX][MLIR] Depthwise GroupConv erase pad subview\n";
      }
      rewriter.eraseOp(pad_subview_op);
    }
    if (pad_fill_loop && pad_fill_loop != pad_copy_loop &&
        pad_fill_loop->getBlock()) {
      if (debug) {
        llvm::errs() << "[GFX][MLIR] Depthwise GroupConv erase pad fill\n";
      }
      rewriter.eraseOp(pad_fill_loop);
    }
    if (padded_alloc && padded_alloc->use_empty()) {
      if (debug) {
        llvm::errs() << "[GFX][MLIR] Depthwise GroupConv erase pad alloc\n";
      }
      rewriter.eraseOp(padded_alloc);
    } else if (padded_alloca && padded_alloca->use_empty()) {
      if (debug) {
        llvm::errs() << "[GFX][MLIR] Depthwise GroupConv erase pad alloca\n";
      }
      rewriter.eraseOp(padded_alloca);
    }
  }
  if (debug) {
    llvm::errs() << "[GFX][MLIR] Depthwise GroupConv lowering done\n";
  }
  return true;
}

} // namespace

namespace detail {

bool is_conv_tile_input_h_interior(int64_t oh_base, int64_t tile_h,
                                   int64_t stride_h, int64_t dil_h,
                                   int64_t kernel_h, int64_t pad_h,
                                   int64_t input_h) {
  if (!is_positive_extent(tile_h) || !is_positive_extent(stride_h) ||
      !is_positive_extent(dil_h) || !is_positive_extent(kernel_h) ||
      !is_positive_extent(input_h)) {
    return false;
  }
  const int64_t oh_last = oh_base + tile_h - 1;
  const int64_t ih_min = oh_base * stride_h - pad_h;
  const int64_t ih_max = oh_last * stride_h + (kernel_h - 1) * dil_h - pad_h;
  return ih_min >= 0 && ih_max < input_h;
}

bool is_conv_tile_input_w_interior(int64_t ow_base, int64_t tile_w,
                                   int64_t stride_w, int64_t dil_w,
                                   int64_t kernel_w, int64_t pad_w,
                                   int64_t input_w) {
  if (!is_positive_extent(tile_w) || !is_positive_extent(stride_w) ||
      !is_positive_extent(dil_w) || !is_positive_extent(kernel_w) ||
      !is_positive_extent(input_w)) {
    return false;
  }
  const int64_t ow_last = ow_base + tile_w - 1;
  const int64_t iw_min = ow_base * stride_w - pad_w;
  const int64_t iw_max = ow_last * stride_w + (kernel_w - 1) * dil_w - pad_w;
  return iw_min >= 0 && iw_max < input_w;
}

bool is_conv_tile_input_interior(int64_t oh_base, int64_t ow_base,
                                 int64_t tile_h, int64_t tile_w,
                                 int64_t stride_h, int64_t stride_w,
                                 int64_t dil_h, int64_t dil_w, int64_t kernel_h,
                                 int64_t kernel_w, int64_t pad_h, int64_t pad_w,
                                 int64_t input_h, int64_t input_w) {
  if (!is_positive_extent(tile_h) || !is_positive_extent(tile_w) ||
      !is_positive_extent(stride_h) || !is_positive_extent(stride_w) ||
      !is_positive_extent(dil_h) || !is_positive_extent(dil_w) ||
      !is_positive_extent(kernel_h) || !is_positive_extent(kernel_w) ||
      !is_positive_extent(input_h) || !is_positive_extent(input_w)) {
    return false;
  }
  return is_conv_tile_input_h_interior(oh_base, tile_h, stride_h, dil_h,
                                       kernel_h, pad_h, input_h) &&
         is_conv_tile_input_w_interior(ow_base, tile_w, stride_w, dil_w,
                                       kernel_w, pad_w, input_w);
}

} // namespace detail

void run_conv2d_parallel_lowering(mlir::ModuleOp module) {
  if (!module) {
    return;
  }
  if (auto skip =
          module->getAttrOfType<mlir::BoolAttr>("gfx.skip_conv_parallel")) {
    if (skip.getValue()) {
      if (gfx_log_debug_enabled()) {
        gfx_log_debug("MLIR")
            << "Conv2D parallel lowering skipped by module attr";
      }
      return;
    }
  }
  mlir::IRRewriter rewriter(module.getContext());
  llvm::SmallVector<mlir::linalg::Conv2DNchwFchwOp, 8> convs;
  llvm::SmallVector<mlir::linalg::GenericOp, 8> depthwise_group_convs;
  module.walk([&](mlir::linalg::Conv2DNchwFchwOp op) { convs.push_back(op); });
  module.walk([&](mlir::linalg::GenericOp op) {
    if (auto attr =
            op->getAttrOfType<mlir::BoolAttr>("gfx.depthwise_nchw_direct")) {
      if (attr.getValue()) {
        depthwise_group_convs.push_back(op);
      }
    }
  });
  size_t rewritten = 0;
  for (auto op : convs) {
    if (!op || !op->getParentOp()) {
      continue;
    }
    if (lower_conv2d_op(op, rewriter)) {
      ++rewritten;
    }
  }
  size_t rewritten_depthwise = 0;
  for (auto op : depthwise_group_convs) {
    if (!op || !op->getParentOp()) {
      continue;
    }
    if (lower_depthwise_group_conv_generic(op, rewriter)) {
      ++rewritten_depthwise;
    }
  }
  if (gfx_log_debug_enabled()) {
    gfx_log_debug("MLIR") << "Conv2D parallel lowering: convs=" << convs.size()
                          << " rewritten=" << rewritten
                          << " depthwise_group_convs="
                          << depthwise_group_convs.size()
                          << " depthwise_rewritten=" << rewritten_depthwise;
  }
  if (!convs.empty() && rewritten == 0) {
    if (auto skip =
            module->getAttrOfType<mlir::BoolAttr>("gfx.skip_conv_parallel")) {
      if (skip.getValue())
        return;
    }
    throw std::runtime_error("Conv2D parallel lowering failed");
  }
  if (!depthwise_group_convs.empty() && rewritten_depthwise == 0) {
    if (auto skip =
            module->getAttrOfType<mlir::BoolAttr>("gfx.skip_conv_parallel")) {
      if (skip.getValue())
        return;
    }
    throw std::runtime_error(
        "Depthwise GroupConvolution parallel lowering failed");
  }
}

} // namespace gfx_plugin
} // namespace ov
