// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/except.hpp"
#include "backends/metal/plugin/remote_tensor.hpp"

namespace ov {
namespace gfx_plugin {

RemoteTensorCreateResult create_metal_remote_tensor(const ov::element::Type& /*type*/,
                                                    const ov::Shape& /*shape*/,
                                                    const ov::AnyMap& /*params*/,
                                                    GpuDeviceHandle /*device*/,
                                                    size_t /*bytes*/) {
    OPENVINO_THROW("GFX: Metal backend is not available for remote tensor");
}

}  // namespace gfx_plugin
}  // namespace ov
