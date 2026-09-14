// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "backends/opencl/runtime/opencl_api.hpp"
#include "runtime/gfx_remote_tensor.hpp"

namespace ov {
namespace gfx_plugin {

RemoteTensorCreateResult create_opencl_remote_tensor(
    const ov::element::Type& type,
    const ov::Shape& shape,
    const ov::AnyMap& params,
    const std::shared_ptr<OpenClRuntimeContext>& context,
    size_t bytes);

}  // namespace gfx_plugin
}  // namespace ov
