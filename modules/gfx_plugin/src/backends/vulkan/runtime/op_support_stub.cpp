// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "openvino/core/node.hpp"

namespace ov {
namespace gfx_plugin {

bool vulkan_supports_node(const std::shared_ptr<const ov::Node>& /*node*/) {
    return false;
}

}  // namespace gfx_plugin
}  // namespace ov
