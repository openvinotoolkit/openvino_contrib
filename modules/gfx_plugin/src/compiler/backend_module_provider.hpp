// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "compiler/backend_registry.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

std::vector<std::shared_ptr<const BackendModule>>
make_default_backend_modules();

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
