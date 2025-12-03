// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdlib>
#include <string>

#include <gtest/gtest.h>

#include "openvino/openvino.hpp"
#include "openvino/core/except.hpp"

#include "test_constants.hpp"

namespace ov {
namespace test {
namespace utils {

inline bool metal_tests_enabled() {
    const char* flag = std::getenv("OV_METAL_RUN_FUNCTIONAL_TESTS");
    return flag != nullptr && std::string(flag) == "1";
}

// Try compile_model on METAL; if it throws ov::Exception, skip the test gracefully.
template <typename Fn>
void metal_try_compile_or_skip(Fn&& fn) {
    try {
        fn();
    } catch (const ov::Exception& e) {
        GTEST_SKIP() << "METAL unsupported: " << e.what();
    }
}

// Helper wrapper that skips tests on METAL unless explicitly enabled.
template <class Base>
class MetalSkippedTests : public Base {
protected:
    void SetUp() override {
        if (!metal_tests_enabled()) {
            GTEST_SKIP() << "METAL functional test disabled by default; set OV_METAL_RUN_FUNCTIONAL_TESTS=1 to run.";
        }
        // Probe a tiny supported model; if METAL rejects it, skip the whole test
        metal_try_compile_or_skip([&]() {
            ov::Core core;
            auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
            auto res = std::make_shared<ov::op::v0::Result>(param);
            auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "metal_probe");
            core.compile_model(model, ov::test::utils::DEVICE_METAL);
        });
        Base::SetUp();
    }
};

}  // namespace utils
}  // namespace test
}  // namespace ov
