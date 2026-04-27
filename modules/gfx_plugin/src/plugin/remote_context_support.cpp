// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/remote_context_support.hpp"

#include "openvino/core/except.hpp"
#include "plugin/gfx_property_utils.hpp"
#include "runtime/gfx_remote_context.hpp"
#include "runtime/gfx_backend_utils.hpp"

namespace ov {
namespace gfx_plugin {
ov::SoPtr<ov::IRemoteContext> create_metal_remote_context(const std::string& resolved_name,
                                                          const RemoteContextParams& params);
ov::SoPtr<ov::IRemoteContext> create_vulkan_remote_context(const std::string& resolved_name,
                                                           const RemoteContextParams& params);
}  // namespace gfx_plugin
}  // namespace ov

namespace ov {
namespace gfx_plugin {

int get_remote_device_id(const ov::SoPtr<ov::IRemoteContext>& context) {
    auto gfx_ctx = std::dynamic_pointer_cast<GfxRemoteContext>(context._ptr);
    OPENVINO_ASSERT(gfx_ctx, "GFX: remote context type mismatch");
    return gfx_ctx->device_id();
}

std::string get_remote_backend(const ov::SoPtr<ov::IRemoteContext>& context) {
    auto gfx_ctx = std::dynamic_pointer_cast<GfxRemoteContext>(context._ptr);
    OPENVINO_ASSERT(gfx_ctx, "GFX: remote context type mismatch");
    return gfx_ctx->backend_name();
}

ov::SoPtr<ov::IRemoteContext> make_gfx_remote_context(const std::string& device_name,
                                                      const ov::AnyMap& remote_properties) {
    auto params = normalize_remote_context_params(remote_properties);
    if (!backend_supported(params.backend)) {
        OPENVINO_THROW("GFX: backend '", params.backend_name, "' is not available for remote context");
    }
    const std::string resolved_name = device_name.empty() ? "GFX" : device_name;
    switch (params.backend) {
        case GpuBackend::Metal: {
            return create_metal_remote_context(resolved_name, params);
        }
        case GpuBackend::Vulkan: {
            return create_vulkan_remote_context(resolved_name, params);
        }
        default:
            break;
    }
    OPENVINO_THROW("GFX: unsupported remote backend: ", params.backend_name);
}

}  // namespace gfx_plugin
}  // namespace ov
