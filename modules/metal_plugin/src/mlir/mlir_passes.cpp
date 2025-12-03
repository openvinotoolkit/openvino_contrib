// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_passes.hpp"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include <stdexcept>

namespace ov {
namespace metal_plugin {

void run_mlir_pipeline(mlir::ModuleOp module) {
    mlir::PassManager pm(module.getContext());
    pm.addPass(mlir::createCanonicalizerPass());

    if (mlir::failed(pm.run(module))) {
        throw std::runtime_error("MLIR pipeline failed");
    }
}

}  // namespace metal_plugin
}  // namespace ov

