// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/gfx_mlir_transform_pipeline.hpp"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

namespace ov {
namespace gfx_plugin {

void populate_gfx_pre_bufferization_pipeline(mlir::PassManager& pm) {
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createLinalgElementwiseOpFusionPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
}

void populate_gfx_post_bufferization_pipeline(mlir::PassManager& pm,
                                              const GfxMlirTransformPipelineOptions& options) {
    if (options.use_parallel_loops) {
        pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
    } else {
        pm.addPass(mlir::createConvertLinalgToLoopsPass());
    }
    pm.addPass(mlir::createLowerAffinePass());
    if (!options.preserve_compact_memref_abi) {
        pm.addPass(mlir::memref::createNormalizeMemRefsPass());
    }
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
}

void populate_gfx_post_normalization_cleanup_pipeline(mlir::PassManager& pm) {
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
}

}  // namespace gfx_plugin
}  // namespace ov
