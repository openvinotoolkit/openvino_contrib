// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_support.hpp"

namespace ov {
namespace gfx_plugin {

bool metal_supports_node(const std::shared_ptr<const ov::Node>& node) {
    return mlir_supports_node(node);
}

}  // namespace gfx_plugin
}  // namespace ov
