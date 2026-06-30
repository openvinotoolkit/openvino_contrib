// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "mlir/Pass/PassManager.h"

namespace ov {
namespace gfx_plugin {

struct GfxMlirTransformPipelineOptions {
    bool use_parallel_loops = false;
    bool preserve_compact_memref_abi = false;
};

// Shared pass sequencing for backend-neutral MLIR lowering phases.
void populate_gfx_pre_bufferization_pipeline(mlir::PassManager& pm);
void populate_gfx_post_bufferization_pipeline(mlir::PassManager& pm,
                                              const GfxMlirTransformPipelineOptions& options);
void populate_gfx_post_normalization_cleanup_pipeline(mlir::PassManager& pm);

}  // namespace gfx_plugin
}  // namespace ov
