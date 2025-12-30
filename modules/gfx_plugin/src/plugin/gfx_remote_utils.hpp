// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "openvino/core/any.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace gfx_plugin {

void* any_to_ptr(const ov::Any& value);
bool any_to_bool(const ov::Any& value, bool fallback);
void* find_any_ptr(const ov::AnyMap& params, std::initializer_list<const char*> keys);
bool find_any_bool(const ov::AnyMap& params,
                   std::initializer_list<const char*> keys,
                   bool fallback);

}  // namespace gfx_plugin
}  // namespace ov
