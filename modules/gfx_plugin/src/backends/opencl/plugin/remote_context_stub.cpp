// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "openvino/core/except.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "plugin/gfx_property_utils.hpp"

namespace ov {
namespace gfx_plugin {

ov::SoPtr<ov::IRemoteContext> create_opencl_remote_context(
    const std::string& /*resolved_name*/,
    const RemoteContextParams& /*params*/) {
    OPENVINO_THROW("GFX OpenCL backend is not available for remote context in this build");
}

}  // namespace gfx_plugin
}  // namespace ov
