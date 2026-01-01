// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/op_factory.hpp"

namespace ov {
namespace gfx_plugin {

bool metal_supports_node(const std::shared_ptr<const ov::Node>& node) {
    auto op = MetalOpFactory::create(node);
    return op != nullptr;
}

}  // namespace gfx_plugin
}  // namespace ov
