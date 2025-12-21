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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/Verifier.h"

#include <stdexcept>
#include <cstdlib>

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

    if (std::getenv("GFX_MLIR_FUSION_LOG")) {
        llvm::errs() << "[GFX][MLIR] Fusion stats: conv " << beforeConv << " -> " << afterConv
                     << ", linalg.generic " << beforeGeneric << " -> " << afterGeneric << "\n";
    }
}

}  // namespace

void run_mlir_pipeline(mlir::ModuleOp module) {
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

    const auto env_debug = std::getenv("GFX_MLIR_DEBUG");
    const bool debug = env_debug && std::string(env_debug) != "0";
    const auto env_pre = std::getenv("GFX_MLIR_DEBUG_PRE_BUFFERIZE");
    const bool debug_pre = env_pre && std::string(env_pre) != "0";

    if (mlir::failed(mlir::verify(module))) {
        llvm::errs() << "[GFX][MLIR] Module verification failed before pipeline\n";
        module.dump();
        throw std::runtime_error("MLIR module verification failed");
    }

    mlir::PassManager pm(ctx);
    if (debug) {
        module.dump();
    }
    // Canonicalize and CSE before fusion to expose larger elementwise regions.
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    // Fuse conv -> eltwise chains (bias/add/unary) at the tensor level.
    // Run it directly before bufferization to maximize fusion opportunities.
    // (Runs outside the pass manager to avoid plugin RTTI issues.)
    // Generic linalg elementwise fusion still happens via the standard pass.
    runConvEltwiseFusion(module);
    pm.addPass(mlir::createLinalgElementwiseOpFusionPass());

    if (debug_pre) {
        llvm::errs() << "[GFX][MLIR] Module before bufferization:\n";
        module.dump();
    }

    // Run cleanup again to simplify the fused op bodies prior to bufferization.
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    mlir::bufferization::OneShotBufferizePassOptions opts;
    opts.bufferizeFunctionBoundaries = true;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(opts));
    pm.addPass(mlir::createConvertLinalgToLoopsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    if (mlir::failed(pm.run(module))) {
        throw std::runtime_error("MLIR pipeline failed");
    }

    if (debug) {
        module.dump();
    }
}

}  // namespace gfx_plugin
}  // namespace ov
