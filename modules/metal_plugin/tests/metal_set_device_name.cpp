// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/convolution.hpp"
#include "set_device_name.hpp"

namespace ov {
namespace test {

// Test runner calls this to append device suffix; METAL backend uses fixed name,
// so keep no-op to satisfy linker.
void set_device_suffix(const std::string& /*suffix*/) {}

}  // namespace test
}  // namespace ov
