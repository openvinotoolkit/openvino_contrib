// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gfx_runtime_model_runner.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

#include "common_test_utils/ov_plugin_cache.hpp"
#include "../gfx_test_utils.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov::test::gfx {
namespace {

void register_gfx_plugin_or_throw(ov::Core& core) {
    try {
        const char* env_path = std::getenv("GFX_PLUGIN_PATH");
        const char* path = (env_path && *env_path) ? env_path : GFX_PLUGIN_PATH;
        core.register_plugin(path, "GFX");
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        if (msg.find("already registered") == std::string::npos) {
            throw std::runtime_error(std::string("GFX plugin unavailable: ") + e.what());
        }
    }

    try {
        const auto backend = core.get_property("GFX", "GFX_BACKEND").as<std::string>();
        if (backend.empty()) {
            throw std::runtime_error("GFX backend not available");
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("GFX backend property unavailable: ") + e.what());
    }

    if (!ov::test::utils::ensure_template_plugin(core)) {
        throw std::runtime_error("TEMPLATE plugin unavailable");
    }
}

std::string reference_device(const ov::Core& core) {
    try {
        (void)core.get_property("TEMPLATE", ov::supported_properties);
        return "TEMPLATE";
    } catch (...) {
        throw std::runtime_error("TEMPLATE reference device not available");
    }
}

std::shared_ptr<ov::Model> make_runtime_probe_model() {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 1});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 1});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, true);
    auto res = std::make_shared<ov::op::v0::Result>(matmul);
    return std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{lhs, rhs}, "gfx_runtime_probe");
}

void require_gfx_runtime_available_or_throw() {
    ov::Core core;
    register_gfx_plugin_or_throw(core);
    (void)core.compile_model(make_runtime_probe_model(), "GFX", fp16_compile_config());
}

void require_allclose(const ov::Tensor& a, const ov::Tensor& b, float atol, float rtol) {
    if (a.get_element_type() != ov::element::f32 || b.get_element_type() != ov::element::f32) {
        throw std::runtime_error("expected f32 tensors");
    }
    if (a.get_byte_size() != b.get_byte_size() || a.get_shape() != b.get_shape()) {
        throw std::runtime_error("tensor shape mismatch");
    }
    auto* pa = a.data<const float>();
    auto* pb = b.data<const float>();
    const size_t count = a.get_size();
    for (size_t i = 0; i < count; ++i) {
        const float diff = std::abs(pa[i] - pb[i]);
        const float thresh = std::max(atol, rtol * std::abs(pa[i]));
        if (diff > thresh) {
            throw std::runtime_error("tensor mismatch at index " + std::to_string(i) +
                                     ": ref=" + std::to_string(pa[i]) +
                                     " gfx=" + std::to_string(pb[i]) +
                                     " diff=" + std::to_string(diff) +
                                     " thresh=" + std::to_string(thresh));
        }
    }
}

void require_f16_allclose(const ov::Tensor& a, const ov::Tensor& b, float atol, float rtol) {
    if (a.get_element_type() != ov::element::f16 || b.get_element_type() != ov::element::f16) {
        throw std::runtime_error("expected f16 tensors");
    }
    if (a.get_byte_size() != b.get_byte_size() || a.get_shape() != b.get_shape()) {
        throw std::runtime_error("tensor shape mismatch");
    }
    auto* pa = a.data<const ov::float16>();
    auto* pb = b.data<const ov::float16>();
    const size_t count = a.get_size();
    for (size_t i = 0; i < count; ++i) {
        const float ref = static_cast<float>(pa[i]);
        const float actual = static_cast<float>(pb[i]);
        const float diff = std::abs(ref - actual);
        const float thresh = std::max(atol, rtol * std::abs(ref));
        if (diff > thresh) {
            throw std::runtime_error("tensor mismatch at index " + std::to_string(i) +
                                     ": ref=" + std::to_string(ref) +
                                     " gfx=" + std::to_string(actual) +
                                     " diff=" + std::to_string(diff) +
                                     " thresh=" + std::to_string(thresh));
        }
    }
}

void require_bool_equal(const ov::Tensor& a, const ov::Tensor& b) {
    if (a.get_element_type() != ov::element::boolean || b.get_element_type() != ov::element::boolean) {
        throw std::runtime_error("expected boolean tensors");
    }
    if (a.get_byte_size() != b.get_byte_size() || a.get_shape() != b.get_shape()) {
        throw std::runtime_error("tensor shape mismatch");
    }
    auto* pa = a.data<const uint8_t>();
    auto* pb = b.data<const uint8_t>();
    const size_t count = a.get_size();
    for (size_t i = 0; i < count; ++i) {
        const uint8_t va = static_cast<uint8_t>(pa[i] != 0);
        const uint8_t vb = static_cast<uint8_t>(pb[i] != 0);
        if (va != vb) {
            throw std::runtime_error("boolean tensor mismatch at index " + std::to_string(i) +
                                     ": ref=" + std::to_string(va) +
                                     " gfx=" + std::to_string(vb));
        }
    }
}

void require_tensor_match(const ov::Tensor& a, const ov::Tensor& b, float atol, float rtol) {
    if (a.get_element_type() != b.get_element_type()) {
        throw std::runtime_error("tensor element type mismatch: ref=" + a.get_element_type().to_string() +
                                 " gfx=" + b.get_element_type().to_string());
    }
    if (a.get_element_type() == ov::element::boolean) {
        require_bool_equal(a, b);
        return;
    }
    if (a.get_element_type() == ov::element::f16) {
        require_f16_allclose(a, b, atol, rtol);
        return;
    }
    require_allclose(a, b, atol, rtol);
}

}  // namespace

RuntimeModelRunner::RuntimeModelRunner(std::unique_ptr<RuntimeExecutionPolicy> execution)
    : execution_(std::move(execution)) {}

void RuntimeModelRunner::with_gfx_core(const std::function<void(ov::Core&)>& fn, int timeout_seconds) const {
    try {
        require_gfx_runtime_available_or_throw();
    } catch (const std::exception& e) {
        GTEST_SKIP() << e.what();
    }

    execution_->run([&] {
        ov::Core child_core;
        register_gfx_plugin_or_throw(child_core);
        fn(child_core);
    }, timeout_seconds);
}

void RuntimeModelRunner::compare_model(const std::shared_ptr<ov::Model>& model,
                                       const std::vector<ov::Tensor>& inputs,
                                       int timeout_seconds,
                                       float atol,
                                       float rtol) const {
    with_gfx_core([&](ov::Core& child_core) {
        const auto ref_dev = reference_device(child_core);
        auto ref_cm = child_core.compile_model(model, ref_dev, fp16_compile_config());
        auto gfx_cm = child_core.compile_model(model, "GFX", fp16_compile_config());
        auto ref_req = ref_cm.create_infer_request();
        auto gfx_req = gfx_cm.create_infer_request();
        for (size_t i = 0; i < inputs.size(); ++i) {
            ref_req.set_input_tensor(i, inputs[i]);
            gfx_req.set_input_tensor(i, inputs[i]);
        }
        ref_req.infer();
        gfx_req.infer();
        for (size_t i = 0; i < ref_cm.outputs().size(); ++i) {
            const auto ref_out = ref_req.get_output_tensor(i);
            const auto gfx_out = gfx_req.get_output_tensor(i);
            try {
                require_tensor_match(ref_out, gfx_out, atol, rtol);
            } catch (const std::exception& ex) {
                throw std::runtime_error("output[" + std::to_string(i) + "] " + ex.what());
            }
        }
    }, timeout_seconds);
}

void RuntimeModelRunner::compare_model_repeated_infer(const std::shared_ptr<ov::Model>& model,
                                                      const std::vector<ov::Tensor>& inputs,
                                                      size_t infer_count,
                                                      int timeout_seconds,
                                                      float atol,
                                                      float rtol) const {
    with_gfx_core([&](ov::Core& child_core) {
        const auto ref_dev = reference_device(child_core);
        auto ref_cm = child_core.compile_model(model, ref_dev, fp16_compile_config());
        auto gfx_cm = child_core.compile_model(model, "GFX", fp16_compile_config());
        auto ref_req = ref_cm.create_infer_request();
        auto gfx_req = gfx_cm.create_infer_request();
        for (size_t i = 0; i < inputs.size(); ++i) {
            ref_req.set_input_tensor(i, inputs[i]);
            gfx_req.set_input_tensor(i, inputs[i]);
        }
        for (size_t infer_index = 0; infer_index < infer_count; ++infer_index) {
            ref_req.infer();
            gfx_req.infer();
            for (size_t output_index = 0; output_index < ref_cm.outputs().size(); ++output_index) {
                const auto ref_out = ref_req.get_output_tensor(output_index);
                const auto gfx_out = gfx_req.get_output_tensor(output_index);
                try {
                    require_tensor_match(ref_out, gfx_out, atol, rtol);
                } catch (const std::exception& ex) {
                    throw std::runtime_error("infer[" + std::to_string(infer_index) +
                                             "] output[" + std::to_string(output_index) + "] " + ex.what());
                }
            }
        }
    }, timeout_seconds);
}

ov::AnyMap fp16_compile_config() {
    ov::AnyMap config;
    config[ov::hint::inference_precision.name()] = ov::element::f16;
    return config;
}

}  // namespace ov::test::gfx
