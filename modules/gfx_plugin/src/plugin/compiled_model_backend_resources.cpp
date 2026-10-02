// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "plugin/compiled_model_backend_resources.hpp"

namespace ov {
namespace gfx_plugin {

BackendResources get_backend_resources(const BackendState* state) {
    return state ? state->resources() : BackendResources{};
}

bool backend_has_const_manager(const BackendState* state) {
    if (!state) {
        return false;
    }
    if (!state->requires_const_manager()) {
        return true;
    }
    return state->has_const_manager();
}

}  // namespace gfx_plugin
}  // namespace ov
