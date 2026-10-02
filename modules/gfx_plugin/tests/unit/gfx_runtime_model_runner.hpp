// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "openvino/openvino.hpp"

namespace ov::test::gfx {

class RuntimeExecutionPolicy {
public:
    virtual ~RuntimeExecutionPolicy() = default;

    virtual void run(const std::function<void()>& fn, int timeout_seconds) const = 0;
};

std::unique_ptr<RuntimeExecutionPolicy> make_runtime_execution_policy();

class RuntimeModelRunner final {
public:
    explicit RuntimeModelRunner(std::unique_ptr<RuntimeExecutionPolicy> execution = make_runtime_execution_policy());

    void with_gfx_core(const std::function<void(ov::Core&)>& fn, int timeout_seconds) const;

    void compare_model(const std::shared_ptr<ov::Model>& model,
                       const std::vector<ov::Tensor>& inputs,
                       int timeout_seconds,
                       float atol = 1e-5f,
                       float rtol = 0.f) const;

    void compare_model_repeated_infer(const std::shared_ptr<ov::Model>& model,
                                      const std::vector<ov::Tensor>& inputs,
                                      size_t infer_count,
                                      int timeout_seconds,
                                      float atol = 1e-5f,
                                      float rtol = 0.f) const;

private:
    std::unique_ptr<RuntimeExecutionPolicy> execution_;
};

ov::AnyMap fp16_compile_config();

}  // namespace ov::test::gfx
