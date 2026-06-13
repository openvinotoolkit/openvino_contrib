// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "openvino/openvino.hpp"

namespace ov::test::gfx {

class RuntimeScenario {
public:
    virtual ~RuntimeScenario() = default;

    virtual std::string name() const = 0;
    virtual std::shared_ptr<ov::Model> make_model() const = 0;
    virtual std::vector<ov::Tensor> make_inputs() const = 0;
    virtual int timeout_seconds() const = 0;
    virtual float atol() const = 0;
    virtual float rtol() const = 0;
    virtual size_t infer_count() const = 0;
};

using RuntimeModelBuilder = std::function<std::shared_ptr<ov::Model>()>;
using RuntimeInputBuilder = std::function<std::vector<ov::Tensor>()>;

class LambdaRuntimeScenario final : public RuntimeScenario {
public:
    LambdaRuntimeScenario(std::string scenario_name,
                          RuntimeModelBuilder model_builder,
                          RuntimeInputBuilder input_builder,
                          int timeout,
                          float abs_tolerance,
                          float rel_tolerance = 0.f,
                          size_t repeats = 1)
        : m_name(std::move(scenario_name)),
          m_model_builder(std::move(model_builder)),
          m_input_builder(std::move(input_builder)),
          m_timeout(timeout),
          m_atol(abs_tolerance),
          m_rtol(rel_tolerance),
          m_repeats(repeats) {}

    std::string name() const override {
        return m_name;
    }

    std::shared_ptr<ov::Model> make_model() const override {
        return m_model_builder();
    }

    std::vector<ov::Tensor> make_inputs() const override {
        return m_input_builder();
    }

    int timeout_seconds() const override {
        return m_timeout;
    }

    float atol() const override {
        return m_atol;
    }

    float rtol() const override {
        return m_rtol;
    }

    size_t infer_count() const override {
        return m_repeats;
    }

private:
    std::string m_name;
    RuntimeModelBuilder m_model_builder;
    RuntimeInputBuilder m_input_builder;
    int m_timeout;
    float m_atol;
    float m_rtol;
    size_t m_repeats;
};

inline std::shared_ptr<const RuntimeScenario> runtime_scenario(std::string name,
                                                               RuntimeModelBuilder model_builder,
                                                               RuntimeInputBuilder input_builder,
                                                               int timeout = 20,
                                                               float atol = 1e-4f,
                                                               float rtol = 0.f,
                                                               size_t infer_count = 1) {
    return std::make_shared<LambdaRuntimeScenario>(std::move(name),
                                                   std::move(model_builder),
                                                   std::move(input_builder),
                                                   timeout,
                                                   atol,
                                                   rtol,
                                                   infer_count);
}

}  // namespace ov::test::gfx
