// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdlib>
#include <string>

#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace utils {

// Primary device names used across METAL functional tests.
static const char* DEVICE_METAL = "METAL";

// Reference device for numerical comparisons. CPU handles mixed precisions
// (including f16 inputs) without pointer-representability issues seen in TEMPLATE.
static const char* DEVICE_REF = ov::test::utils::DEVICE_CPU;

// Always enable METAL functional tests; use env only for debug logging.
inline bool metal_tests_enabled() {
    return true;
}

inline bool metal_tests_debug_enabled() {
    const char* env = std::getenv("OV_METAL_TEST_DEBUG");
    return env && std::string(env) == "1";
}

}  // namespace utils
}  // namespace test
}  // namespace ov
