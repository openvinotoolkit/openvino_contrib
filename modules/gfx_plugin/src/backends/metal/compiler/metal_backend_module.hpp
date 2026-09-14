// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "compiler/backend_registry.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

std::shared_ptr<const BackendModule>
make_metal_backend_module(BackendTarget target);
std::shared_ptr<const BackendModule> make_metal_backend_module();

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
