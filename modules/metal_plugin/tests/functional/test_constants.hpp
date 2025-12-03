// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../metal_test_constants.hpp"

namespace ov {
namespace test {
namespace utils {

// DEVICE_METAL is provided by metal_test_constants.hpp as const char*.
// Reference backend for comparisons. Switch to "TEMPLATE" when available.
static const char* DEVICE_REF = "CPU";

}  // namespace utils
}  // namespace test
}  // namespace ov
