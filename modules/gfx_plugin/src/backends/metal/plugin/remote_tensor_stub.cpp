// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/except.hpp"
#include "runtime/gfx_remote_context.hpp"

namespace ov {
namespace gfx_plugin {

RemoteTensorCreateResult create_metal_remote_tensor(const ov::element::Type& /*type*/,
                                                    const ov::Shape& /*shape*/,
                                                    const ov::AnyMap& /*params*/,
                                                    GpuDeviceHandle /*device*/,
                                                    size_t /*bytes*/) {
    OPENVINO_THROW("GFX: Metal backend is not available for remote tensor");
}

void release_metal_remote_tensor(GpuTensor& /*tensor*/, bool /*owns_buffer*/) {}

}  // namespace gfx_plugin
}  // namespace ov
