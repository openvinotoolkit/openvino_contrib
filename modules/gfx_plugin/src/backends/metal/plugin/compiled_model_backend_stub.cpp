// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/plugin/compiled_model_backend.hpp"

#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<MetalBackendState> create_metal_backend_state(const ov::AnyMap&,
                                                              const ov::SoPtr<ov::IRemoteContext>&) {
    OPENVINO_THROW("GFX Metal backend is not available in this build");
}

}  // namespace gfx_plugin
}  // namespace ov
