// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "set_device_name.hpp"

namespace ov {
namespace test {

// Test runner calls this to append device suffix; GFX backend uses fixed name,
// so keep no-op to satisfy linker on non-Metal builds.
void set_device_suffix(const std::string& /*suffix*/) {}

}  // namespace test
}  // namespace ov
