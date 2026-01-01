// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_remote_context.hpp"

#include "plugin/gfx_property_utils.hpp"
#include "backends/vulkan/runtime/vulkan_backend.hpp"
#include "backends/vulkan/plugin/remote_tensor.hpp"

namespace ov {
namespace gfx_plugin {

class VulkanRemoteContext final : public GfxRemoteContext {
public:
    using GfxRemoteContext::GfxRemoteContext;

protected:
    RemoteTensorCreateResult create_remote_tensor(const ov::element::Type& type,
                                                  const ov::Shape& shape,
                                                  const ov::AnyMap& params,
                                                  size_t bytes) override {
        return create_vulkan_remote_tensor(type, shape, params, device_handle(), bytes);
    }
};

ov::SoPtr<ov::IRemoteContext> create_vulkan_remote_context(const std::string& resolved_name,
                                                           const RemoteContextParams& params) {
    auto& ctx = VulkanContext::instance();
    auto handle = reinterpret_cast<GpuDeviceHandle>(ctx.device());
    return ov::SoPtr<ov::IRemoteContext>{
        std::make_shared<VulkanRemoteContext>(resolved_name,
                                              /*device_id=*/0,
                                              GpuBackend::Vulkan,
                                              handle,
                                              params.backend_name,
                                              params.merged),
        nullptr};
}

}  // namespace gfx_plugin
}  // namespace ov
