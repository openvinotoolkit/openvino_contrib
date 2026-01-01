// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/backend_factory.hpp"

#include "openvino/core/except.hpp"
#include "runtime/gfx_backend_utils.hpp"
#include "backends/metal/plugin/compiled_model_backend.hpp"
#include "backends/vulkan/plugin/compiled_model_backend.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<BackendState> create_backend_state(GpuBackend backend,
                                                   const ov::AnyMap& properties,
                                                   const ov::SoPtr<ov::IRemoteContext>& context) {
    if (!backend_supported(backend)) {
        OPENVINO_THROW("GFX ", backend_to_string(backend), " backend is not available in this build.");
    }
    switch (backend) {
        case GpuBackend::Metal:
            return create_metal_backend_state(properties, context);
        case GpuBackend::Vulkan:
            return create_vulkan_backend_state();
        default:
            break;
    }
    OPENVINO_THROW("GFX: unsupported backend for compiled model");
}

}  // namespace gfx_plugin
}  // namespace ov
