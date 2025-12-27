// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/remote_context_support.hpp"

#include "openvino/core/except.hpp"
#include "plugin/gfx_property_utils.hpp"
#include "plugin/remote_stub.hpp"
#include "runtime/gfx_backend_utils.hpp"

#if GFX_BACKEND_METAL_AVAILABLE
#include "backends/metal/runtime/memory.hpp"
#endif
#if GFX_BACKEND_VULKAN_AVAILABLE
#include "backends/vulkan/runtime/backend.hpp"
#endif

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
#if GFX_BACKEND_METAL_AVAILABLE
            auto handle = metal_get_device_by_id(params.device_id);
            OPENVINO_ASSERT(handle, "GFX: failed to resolve device for remote context");
            return ov::SoPtr<ov::IRemoteContext>{
                std::make_shared<GfxRemoteContext>(resolved_name,
                                                   params.device_id,
                                                   GpuBackend::Metal,
                                                   handle,
                                                   params.backend_name,
                                                   params.merged),
                nullptr};
#else
            OPENVINO_THROW("GFX: Metal backend is not available for remote context");
#endif
        }
        case GpuBackend::Vulkan: {
#if GFX_BACKEND_VULKAN_AVAILABLE
            auto& ctx = VulkanContext::instance();
            auto handle = reinterpret_cast<GpuDeviceHandle>(ctx.device());
            return ov::SoPtr<ov::IRemoteContext>{
                std::make_shared<GfxRemoteContext>(resolved_name,
                                                   /*device_id=*/0,
                                                   GpuBackend::Vulkan,
                                                   handle,
                                                   params.backend_name,
                                                   params.merged),
                nullptr};
#else
            OPENVINO_THROW("GFX: Vulkan backend is not available for remote context");
#endif
        }
        default:
            break;
    }
    OPENVINO_THROW("GFX: unsupported remote backend: ", params.backend_name);
}

}  // namespace gfx_plugin
}  // namespace ov
