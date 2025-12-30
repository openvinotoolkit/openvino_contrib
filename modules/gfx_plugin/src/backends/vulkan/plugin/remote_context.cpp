// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_remote_context.hpp"

#include "plugin/gfx_property_utils.hpp"
#include "backends/vulkan/runtime/vulkan_backend.hpp"

namespace ov {
namespace gfx_plugin {

ov::SoPtr<ov::IRemoteContext> create_vulkan_remote_context(const std::string& resolved_name,
                                                           const RemoteContextParams& params) {
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
}

}  // namespace gfx_plugin
}  // namespace ov
