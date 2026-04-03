// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "runtime/gfx_remote_tensor.hpp"

namespace ov {
namespace gfx_plugin {

RemoteTensorCreateResult create_vulkan_remote_tensor(const ov::element::Type& type,
                                                     const ov::Shape& shape,
                                                     const ov::AnyMap& params,
                                                     GpuDeviceHandle device,
                                                     size_t bytes);

}  // namespace gfx_plugin
}  // namespace ov
