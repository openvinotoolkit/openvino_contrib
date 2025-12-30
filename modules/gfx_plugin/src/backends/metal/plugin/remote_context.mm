// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_remote_context.hpp"

#include "openvino/core/except.hpp"
#include "plugin/gfx_property_utils.hpp"
#include "backends/metal/runtime/metal_memory.hpp"

namespace ov {
namespace gfx_plugin {

ov::SoPtr<ov::IRemoteContext> create_metal_remote_context(const std::string& resolved_name,
                                                          const RemoteContextParams& params) {
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
}

}  // namespace gfx_plugin
}  // namespace ov
