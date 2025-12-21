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

// Primary device names used across GFX functional tests.
static const char* DEVICE_GFX = "GFX";

// Reference device for numerical comparisons. CPU handles mixed precisions
// (including f16 inputs) without pointer-representability issues seen in TEMPLATE.
static const char* DEVICE_REF = ov::test::utils::DEVICE_CPU;

// Always enable GFX functional tests; use env only for debug logging.
inline bool gfx_tests_enabled() {
    return true;
}

inline bool gfx_tests_debug_enabled() {
    const char* env = std::getenv("OV_GFX_TEST_DEBUG");
    return env && std::string(env) == "1";
}

}  // namespace utils
}  // namespace test
}  // namespace ov
