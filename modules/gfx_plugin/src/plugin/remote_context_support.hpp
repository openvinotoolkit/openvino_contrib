// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>

#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace gfx_plugin {

int get_remote_device_id(const ov::SoPtr<ov::IRemoteContext>& context);
std::string get_remote_backend(const ov::SoPtr<ov::IRemoteContext>& context);
ov::SoPtr<ov::IRemoteContext> make_gfx_remote_context(const std::string& device_name,
                                                      const ov::AnyMap& remote_properties);

}  // namespace gfx_plugin
}  // namespace ov
