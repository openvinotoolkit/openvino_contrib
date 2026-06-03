// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/core/any.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "runtime/backend_runtime.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<BackendState> create_opencl_backend_state(
    const ov::AnyMap& properties,
    const ov::SoPtr<ov::IRemoteContext>& context);

}  // namespace gfx_plugin
}  // namespace ov
