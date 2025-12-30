// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_support.hpp"

#include "mlir/gfx_mlir_kernel_builder.hpp"

namespace ov {
namespace gfx_plugin {

bool mlir_supports_node(const std::shared_ptr<const ov::Node>& node) {
    if (!node) {
        return false;
    }
    mlir::MLIRContext ctx;
    auto module = build_mlir_for_node(node, ctx);
    return module != nullptr;
}

}  // namespace gfx_plugin
}  // namespace ov
