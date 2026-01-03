// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_passes.hpp"

#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/Verifier.h"

#include <functional>
#include <stdexcept>

#include "transforms/conv_parallel_lowering.hpp"
#include "transforms/parallel_fill_fusion.hpp"

#ifndef GFX_MLIR_DEBUG
#define GFX_MLIR_DEBUG 0
#endif

namespace ov {
namespace gfx_plugin {

namespace {

// Control function: allow fusion when producer is linalg::Conv* and consumer is elementwise generic.
static bool allowConvIntoEltwise(mlir::OpOperand* fusedOperand) {
    if (!fusedOperand)
        return false;
    auto* producer = fusedOperand->get().getDefiningOp();
    if (!producer)
        return false;
    return llvm::isa<mlir::linalg::ConvolutionOpInterface>(producer);
}

// Run conv->eltwise greedy fusion directly over the module to avoid RTTI issues with a custom pass type.
static void runConvEltwiseFusion(mlir::ModuleOp module) {
    mlir::RewritePatternSet patterns(module.getContext());
    mlir::linalg::populateElementwiseOpsFusionPatterns(patterns, allowConvIntoEltwise);
    mlir::GreedyRewriteConfig cfg;
    cfg.setMaxIterations(3);
    size_t beforeConv = 0, beforeGeneric = 0;
    module.walk([&](mlir::Operation* op) {
        if (llvm::isa<mlir::linalg::ConvolutionOpInterface>(op)) beforeConv++;
        if (llvm::isa<mlir::linalg::GenericOp>(op)) beforeGeneric++;
    });

    if (mlir::failed(mlir::applyPatternsGreedily(module, std::move(patterns), cfg))) {
        throw std::runtime_error("Conv->Eltwise fusion failed");
    }

    size_t afterConv = 0, afterGeneric = 0;
    module.walk([&](mlir::Operation* op) {
        if (llvm::isa<mlir::linalg::ConvolutionOpInterface>(op)) afterConv++;
        if (llvm::isa<mlir::linalg::GenericOp>(op)) afterGeneric++;
    });

#if GFX_MLIR_DEBUG
    llvm::errs() << "[GFX][MLIR] Fusion stats: conv " << beforeConv << " -> " << afterConv
                 << ", linalg.generic " << beforeGeneric << " -> " << afterGeneric << "\n";
#endif
}

static void strip_strided_func_layouts(mlir::ModuleOp module) {
    bool updated = false;
    module.walk([&](mlir::func::FuncOp func) {
        if (func.isExternal()) {
            return;
        }
        auto fn_type = func.getFunctionType();
        llvm::SmallVector<mlir::Type, 8> inputs;
        inputs.reserve(fn_type.getNumInputs());

        bool changed = false;
        for (auto type : fn_type.getInputs()) {
            if (auto memref = mlir::dyn_cast<mlir::MemRefType>(type)) {
                auto plain = mlir::MemRefType::get(memref.getShape(),
                                                   memref.getElementType(),
                                                   mlir::AffineMap(),
                                                   memref.getMemorySpace());
                inputs.push_back(plain);
                changed |= (plain != memref);
            } else {
                inputs.push_back(type);
            }
        }

        if (!changed) {
            return;
        }

        auto new_type = mlir::FunctionType::get(func.getContext(), inputs, fn_type.getResults());
        func.setType(new_type);
        auto& entry = func.getBody().front();
        for (size_t i = 0; i < inputs.size(); ++i) {
            entry.getArgument(static_cast<unsigned>(i)).setType(inputs[i]);
        }
        updated = true;
    });

    if (updated) {
#if GFX_MLIR_DEBUG
        llvm::errs() << "[GFX][MLIR] Stripped strided layouts from func arguments\n";
#endif
    }
}

static void fix_subview_memory_spaces(mlir::ModuleOp module) {
    module.walk([&](mlir::memref::SubViewOp subview) {
        auto src_type = mlir::dyn_cast<mlir::MemRefType>(subview.getSource().getType());
        auto res_type = mlir::dyn_cast<mlir::MemRefType>(subview.getType());
        if (!src_type || !res_type) {
            return;
        }
        if (src_type.getMemorySpace() == res_type.getMemorySpace()) {
            return;
        }
        auto fixed =
            mlir::MemRefType::get(res_type.getShape(),
                                  res_type.getElementType(),
                                  res_type.getLayout(),
                                  src_type.getMemorySpace());
        subview.getResult().setType(fixed);
    });

    module.walk([&](mlir::memref::ReinterpretCastOp cast_op) {
        auto src_type = mlir::dyn_cast<mlir::MemRefType>(cast_op.getSource().getType());
        auto res_type = mlir::dyn_cast<mlir::MemRefType>(cast_op.getType());
        if (!src_type || !res_type) {
            return;
        }
        if (src_type.getMemorySpace() == res_type.getMemorySpace()) {
            return;
        }
        auto fixed =
            mlir::MemRefType::get(res_type.getShape(),
                                  res_type.getElementType(),
                                  res_type.getLayout(),
                                  src_type.getMemorySpace());
        cast_op.getResult().setType(fixed);
    });

    module.walk([&](mlir::memref::CastOp cast_op) {
        auto src_type = mlir::dyn_cast<mlir::MemRefType>(cast_op.getSource().getType());
        auto res_type = mlir::dyn_cast<mlir::MemRefType>(cast_op.getType());
        if (!src_type || !res_type) {
            return;
        }
        if (src_type.getMemorySpace() == res_type.getMemorySpace()) {
            return;
        }
        auto fixed =
            mlir::MemRefType::get(res_type.getShape(),
                                  res_type.getElementType(),
                                  res_type.getLayout(),
                                  src_type.getMemorySpace());
        cast_op.getResult().setType(fixed);
    });

    module.walk([&](mlir::memref::ViewOp view_op) {
        auto src_type = mlir::dyn_cast<mlir::MemRefType>(view_op.getSource().getType());
        auto res_type = mlir::dyn_cast<mlir::MemRefType>(view_op.getType());
        if (!src_type || !res_type) {
            return;
        }
        if (src_type.getMemorySpace() == res_type.getMemorySpace()) {
            return;
        }
        auto fixed =
            mlir::MemRefType::get(res_type.getShape(),
                                  res_type.getElementType(),
                                  res_type.getLayout(),
                                  src_type.getMemorySpace());
        view_op.getResult().setType(fixed);
    });
}

static void fix_shape_cast_memory_spaces(mlir::ModuleOp module) {
    module.walk([&](mlir::memref::CollapseShapeOp op) {
        auto src_type = mlir::dyn_cast<mlir::MemRefType>(op.getSrcType());
        auto res_type = mlir::dyn_cast<mlir::MemRefType>(op.getType());
        if (!src_type || !res_type) {
            return;
        }
        if (src_type.getMemorySpace() == res_type.getMemorySpace()) {
            return;
        }
        auto fixed = mlir::MemRefType::get(res_type.getShape(),
                                           res_type.getElementType(),
                                           res_type.getLayout(),
                                           src_type.getMemorySpace());
        op.getResult().setType(fixed);
    });

    module.walk([&](mlir::memref::ExpandShapeOp op) {
        auto src_type = mlir::dyn_cast<mlir::MemRefType>(op.getSrcType());
        auto res_type = mlir::dyn_cast<mlir::MemRefType>(op.getType());
        if (!src_type || !res_type) {
            return;
        }
        if (src_type.getMemorySpace() == res_type.getMemorySpace()) {
            return;
        }
        auto fixed = mlir::MemRefType::get(res_type.getShape(),
                                           res_type.getElementType(),
                                           res_type.getLayout(),
                                           src_type.getMemorySpace());
        op.getResult().setType(fixed);
    });
}

static void inline_simple_subviews(mlir::ModuleOp module) {
    llvm::SmallVector<mlir::memref::SubViewOp, 8> to_erase;
    module.walk([&](mlir::memref::SubViewOp subview) {
        auto strides = subview.getStaticStrides();
        for (auto stride : strides) {
            if (stride == mlir::ShapedType::kDynamic || stride != 1) {
                return;
            }
        }
        auto offsets = subview.getMixedOffsets();
        auto make_offset_value = [&](mlir::OpFoldResult ofr,
                                     mlir::Location loc,
                                     mlir::OpBuilder& b) -> mlir::Value {
            if (auto attr = ofr.dyn_cast<mlir::Attribute>()) {
                auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(attr);
                if (!int_attr) {
                    return {};
                }
                return b.create<mlir::arith::ConstantIndexOp>(loc, int_attr.getInt());
            }
            return ofr.dyn_cast<mlir::Value>();
        };

        bool handled = true;
        for (auto* user : llvm::make_early_inc_range(subview->getUsers())) {
            if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(user)) {
                if (load.getMemRef() != subview.getResult()) {
                    handled = false;
                    break;
                }
                mlir::OpBuilder b(load);
                llvm::SmallVector<mlir::Value, 4> new_indices;
                auto idxs = load.getIndices();
                if (idxs.size() != offsets.size()) {
                    handled = false;
                    break;
                }
                for (size_t i = 0; i < idxs.size(); ++i) {
                    auto off_val = make_offset_value(offsets[i], load.getLoc(), b);
                    if (!off_val) {
                        handled = false;
                        break;
                    }
                    if (auto const_op = off_val.getDefiningOp<mlir::arith::ConstantIndexOp>();
                        const_op && const_op.value() == 0) {
                        new_indices.push_back(idxs[i]);
                    } else {
                        new_indices.push_back(b.create<mlir::arith::AddIOp>(load.getLoc(),
                                                                            idxs[i],
                                                                            off_val));
                    }
                }
                if (!handled) {
                    break;
                }
                auto repl = b.create<mlir::memref::LoadOp>(load.getLoc(), subview.getSource(), new_indices);
                load.replaceAllUsesWith(repl.getResult());
                load.erase();
            } else if (auto store = llvm::dyn_cast<mlir::memref::StoreOp>(user)) {
                if (store.getMemRef() != subview.getResult()) {
                    handled = false;
                    break;
                }
                mlir::OpBuilder b(store);
                llvm::SmallVector<mlir::Value, 4> new_indices;
                auto idxs = store.getIndices();
                if (idxs.size() != offsets.size()) {
                    handled = false;
                    break;
                }
                for (size_t i = 0; i < idxs.size(); ++i) {
                    auto off_val = make_offset_value(offsets[i], store.getLoc(), b);
                    if (!off_val) {
                        handled = false;
                        break;
                    }
                    if (auto const_op = off_val.getDefiningOp<mlir::arith::ConstantIndexOp>();
                        const_op && const_op.value() == 0) {
                        new_indices.push_back(idxs[i]);
                    } else {
                        new_indices.push_back(b.create<mlir::arith::AddIOp>(store.getLoc(),
                                                                            idxs[i],
                                                                            off_val));
                    }
                }
                if (!handled) {
                    break;
                }
                b.create<mlir::memref::StoreOp>(store.getLoc(),
                                                store.getValue(),
                                                subview.getSource(),
                                                new_indices);
                store.erase();
            } else {
                handled = false;
                break;
            }
        }

        if (handled && subview.use_empty()) {
            to_erase.push_back(subview);
        }
    });

    for (auto subview : to_erase) {
        if (subview.use_empty()) {
            subview.erase();
        }
    }
}

static void fold_out_param_allocs(mlir::ModuleOp module) {
    module.walk([&](mlir::func::FuncOp func) {
        if (func.isExternal()) {
            return;
        }
        for (auto copy : llvm::make_early_inc_range(func.getOps<mlir::memref::CopyOp>())) {
            auto alloc = copy.getSource().getDefiningOp<mlir::memref::AllocOp>();
            auto alloca = copy.getSource().getDefiningOp<mlir::memref::AllocaOp>();
            if (!alloc && !alloca) {
                continue;
            }
            auto dest_arg = mlir::dyn_cast<mlir::BlockArgument>(copy.getTarget());
            if (!dest_arg) {
                continue;
            }
            const auto src_type = alloc ? alloc.getType() : alloca.getType();
            if (src_type != dest_arg.getType()) {
                continue;
            }
            if (alloc) {
                alloc.replaceAllUsesWith(dest_arg);
            } else {
                alloca.replaceAllUsesWith(dest_arg);
            }
            copy.erase();
            if (alloc) {
                alloc.erase();
            } else {
                alloca.erase();
            }
        }
    });
}

static void lower_memref_copies_to_loops(mlir::ModuleOp module) {
    module.walk([&](mlir::memref::CopyOp copy) {
        auto src_type = mlir::dyn_cast<mlir::MemRefType>(copy.getSource().getType());
        auto dst_type = mlir::dyn_cast<mlir::MemRefType>(copy.getTarget().getType());
        if (!src_type || !dst_type) {
            return;
        }
        if (src_type.getRank() != dst_type.getRank()) {
            return;
        }
        const int64_t rank = src_type.getRank();
        mlir::OpBuilder b(copy);
        mlir::Location loc = copy.getLoc();
        auto zero = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
        auto one = b.create<mlir::arith::ConstantIndexOp>(loc, 1);

        llvm::SmallVector<mlir::Value, 4> ivs;
        std::function<void(int64_t)> build_loop = [&](int64_t dim) {
            mlir::Value upper;
            if (dst_type.isDynamicDim(dim)) {
                upper = b.create<mlir::memref::DimOp>(loc, copy.getTarget(), dim);
            } else {
                upper = b.create<mlir::arith::ConstantIndexOp>(loc, dst_type.getDimSize(dim));
            }
            auto loop = b.create<mlir::scf::ForOp>(loc, zero, upper, one);
            b.setInsertionPointToStart(loop.getBody());
            ivs.push_back(loop.getInductionVar());
            if (dim + 1 < rank) {
                build_loop(dim + 1);
            } else {
                auto val = b.create<mlir::memref::LoadOp>(loc, copy.getSource(), ivs);
                b.create<mlir::memref::StoreOp>(loc, val, copy.getTarget(), ivs);
            }
            ivs.pop_back();
            b.setInsertionPointAfter(loop);
        };

        if (rank == 0) {
            auto val = b.create<mlir::memref::LoadOp>(loc, copy.getSource(), mlir::ValueRange{});
            b.create<mlir::memref::StoreOp>(loc, val, copy.getTarget(), mlir::ValueRange{});
        } else {
            build_loop(0);
        }
        copy.erase();
    });
}

static void lower_allocs_to_alloca(mlir::ModuleOp module) {
    module.walk([&](mlir::memref::AllocOp alloc) {
        auto type = alloc.getType();
        if (!type.hasStaticShape()) {
            return;
        }
        auto* ctx = alloc.getContext();
        auto mem_space = mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), 6);  // Function
        auto new_type = mlir::MemRefType::get(type.getShape(),
                                              type.getElementType(),
                                              type.getLayout(),
                                              mem_space);
        mlir::OpBuilder b(alloc);
        auto alloca = b.create<mlir::memref::AllocaOp>(alloc.getLoc(), new_type);
        if (auto align = alloc->getAttr("alignment")) {
            alloca->setAttr("alignment", align);
        }
        alloc.replaceAllUsesWith(alloca.getResult());
        alloc.erase();
    });
}

}  // namespace

void run_mlir_pipeline(mlir::ModuleOp module, bool use_alloca, bool use_parallel_loops) {
    auto* ctx = module.getContext();
    mlir::DialectRegistry registry;
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    ctx->appendDialectRegistry(registry);

    ctx->loadDialect<mlir::bufferization::BufferizationDialect,
                    mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect,
                    mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect>();

    const bool debug = (GFX_MLIR_DEBUG != 0);
    const bool debug_pre = debug;

    if (mlir::failed(mlir::verify(module))) {
        llvm::errs() << "[GFX][MLIR] Module verification failed before pipeline\n";
        module.dump();
        throw std::runtime_error("MLIR module verification failed");
    }

    if (debug) {
        module.dump();
    }
    {
        mlir::PassManager pm_pre(ctx);
        // Canonicalize and CSE before fusion to expose larger elementwise regions.
        pm_pre.addPass(mlir::createCanonicalizerPass());
        pm_pre.addPass(mlir::createCSEPass());

        // Fuse conv -> eltwise chains (bias/add/unary) at the tensor level.
        // Run it directly before bufferization to maximize fusion opportunities.
        // (Runs outside the pass manager to avoid plugin RTTI issues.)
        // Generic linalg elementwise fusion still happens via the standard pass.
        runConvEltwiseFusion(module);
        pm_pre.addPass(mlir::createLinalgElementwiseOpFusionPass());

        if (debug_pre) {
            llvm::errs() << "[GFX][MLIR] Module before bufferization:\n";
            module.dump();
        }

        // Run cleanup again to simplify the fused op bodies prior to bufferization.
        pm_pre.addPass(mlir::createCanonicalizerPass());
        pm_pre.addPass(mlir::createCSEPass());
        mlir::bufferization::OneShotBufferizePassOptions opts;
        opts.bufferizeFunctionBoundaries = true;
        pm_pre.addPass(mlir::bufferization::createOneShotBufferizePass(opts));

        if (mlir::failed(pm_pre.run(module))) {
            throw std::runtime_error("MLIR pipeline failed (pre-bufferization)");
        }
    }

    if (use_parallel_loops) {
        run_conv2d_parallel_lowering(module);
        if (mlir::failed(mlir::verify(module))) {
            throw std::runtime_error("MLIR module verification failed after Conv2D parallel lowering");
        }
    }

    {
        mlir::PassManager pm_post(ctx);
        if (use_parallel_loops) {
            pm_post.addPass(mlir::createConvertLinalgToParallelLoopsPass());
        } else if (!use_parallel_loops) {
            pm_post.addPass(mlir::createConvertLinalgToLoopsPass());
        }
        pm_post.addPass(mlir::createLowerAffinePass());
        pm_post.addPass(mlir::memref::createNormalizeMemRefsPass());
        pm_post.addPass(mlir::createCanonicalizerPass());
        pm_post.addPass(mlir::createCSEPass());

        if (mlir::failed(pm_post.run(module))) {
            throw std::runtime_error("MLIR pipeline failed (post-bufferization)");
        }
    }
    if (use_parallel_loops) {
        run_parallel_fill_fusion(module);
    }

    {
        mlir::bufferization::BufferResultsToOutParamsOpts out_opts;
        out_opts.hoistStaticAllocs = true;
        if (mlir::failed(mlir::bufferization::promoteBufferResultsToOutParams(module, out_opts))) {
            throw std::runtime_error("MLIR buffer results to out params failed");
        }
    }

    if (use_alloca) {
        lower_allocs_to_alloca(module);
    }
    fold_out_param_allocs(module);
    lower_memref_copies_to_loops(module);
    fix_subview_memory_spaces(module);
    fix_shape_cast_memory_spaces(module);
    inline_simple_subviews(module);
    {
        mlir::PassManager cleanup_pm(ctx);
        cleanup_pm.addPass(mlir::createCanonicalizerPass());
        cleanup_pm.addPass(mlir::createCSEPass());
        if (mlir::failed(cleanup_pm.run(module))) {
            throw std::runtime_error("MLIR post-normalization cleanup failed");
        }
    }
    fix_subview_memory_spaces(module);
    fix_shape_cast_memory_spaces(module);

    if (debug) {
        module.dump();
    }
}

}  // namespace gfx_plugin
}  // namespace ov
