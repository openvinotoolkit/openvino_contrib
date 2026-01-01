// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "plugin/backend_state.hpp"

namespace ov {
namespace gfx_plugin {

BackendResources get_backend_resources(const BackendState* state);
bool backend_has_const_manager(const BackendState* state);

}  // namespace gfx_plugin
}  // namespace ov
