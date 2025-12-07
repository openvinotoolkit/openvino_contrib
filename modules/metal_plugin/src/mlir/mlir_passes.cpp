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
namespace metal_plugin {

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

    if (mlir::failed(mlir::verify(module))) {
        llvm::errs() << "[METAL][MLIR] Module verification failed before pipeline\n";
        module.dump();
        throw std::runtime_error("MLIR module verification failed");
    }

    mlir::PassManager pm(ctx);
    if (std::getenv("METAL_MLIR_DEBUG")) {
        module.dump();
    }
    mlir::bufferization::OneShotBufferizePassOptions opts;
    opts.bufferizeFunctionBoundaries = true;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(opts));
    pm.addPass(mlir::createConvertLinalgToLoopsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    if (mlir::failed(pm.run(module))) {
        throw std::runtime_error("MLIR pipeline failed");
    }

    if (std::getenv("METAL_MLIR_DEBUG")) {
        module.dump();
    }
}

}  // namespace metal_plugin
}  // namespace ov
