// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gfx_runtime_model_runner.hpp"

#include <memory>

namespace ov::test::gfx {
namespace {

class InlineRuntimeExecutionPolicy final : public RuntimeExecutionPolicy {
public:
    void run(const std::function<void()>& fn, int timeout_seconds) const override {
        (void)timeout_seconds;
        fn();
    }
};

}  // namespace

std::unique_ptr<RuntimeExecutionPolicy> make_runtime_execution_policy() {
    return std::make_unique<InlineRuntimeExecutionPolicy>();
}

}  // namespace ov::test::gfx
