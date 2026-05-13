// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>

#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace gfx_plugin {

std::string
generate_static_msl_for_slice(const std::shared_ptr<const ov::Node> &node,
                              const ov::element::Type &storage_type);

} // namespace gfx_plugin
} // namespace ov
