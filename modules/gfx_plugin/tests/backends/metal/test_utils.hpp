// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "openvino/openvino.hpp"
#include "openvino/core/except.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "transformations/convert_precision.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/remote_tensor.hpp"

#include "plugin/gfx_backend_config.hpp"
#include "backends/metal/runtime/dtype.hpp"
#include "backends/metal/runtime/metal_memory.hpp"

#include "integration/test_constants.hpp"
#include "gfx_test_utils.hpp"

namespace ov {
namespace test {
namespace utils {

// Try compile_model on GFX; fail the test if backend is unavailable.
template <typename Fn>
void gfx_try_compile_or_fail(Fn&& fn) {
    try {
        fn();
    } catch (const std::exception& e) {
        FAIL() << "GFX backend unavailable for this test: " << e.what();
    } catch (...) {
        FAIL() << "GFX backend unavailable for this test";
    }
}

// Helper wrapper that skips tests on GFX unless explicitly enabled.
template <class Base>
class GfxSkippedTests : public Base {
protected:
    void SetUp() override {
        ov::test::utils::require_gfx_backend();
        // Probe a tiny supported model; if GFX rejects it, skip the whole test
        gfx_try_compile_or_fail([&]() {
            ov::Core core;
            auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
            auto res = std::make_shared<ov::op::v0::Result>(param);
            auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "gfx_probe");
            core.compile_model(model, ov::test::utils::DEVICE_GFX);
        });
        Base::SetUp();
    }
};

// Fixture that compiles a model on GFX and TEMPLATE and compares outputs.
class GfxVsTemplateFixture {
protected:
    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
    std::string device_gfx = ov::test::utils::DEVICE_GFX;
    std::string device_ref = ov::test::utils::DEVICE_REF;
    // Keep compiled models / requests alive so GFX shared outputs remain valid (no CPU copies).
    std::vector<ov::CompiledModel> keep_alive_models;
    std::vector<ov::InferRequest> keep_alive_requests;

    std::vector<ov::Tensor> run_on_device(const std::shared_ptr<ov::Model>& model,
                                          const std::string& device,
                                          const std::vector<ov::Tensor>& inputs,
                                          const ov::AnyMap& config = {}) {
        auto compiled = core->compile_model(model, device, config);
        auto req = compiled.create_infer_request();
        for (size_t i = 0; i < inputs.size(); ++i) {
            req.set_input_tensor(i, inputs[i]);
        }
        req.infer();
        std::vector<ov::Tensor> outputs;
        const auto output_count = model->outputs().size();
        outputs.reserve(output_count);
        for (size_t i = 0; i < output_count; ++i) {
            outputs.emplace_back(req.get_output_tensor(i));
        }
        // Preserve lifetimes for shared output buffers (GFX).
        keep_alive_requests.emplace_back(req);
        keep_alive_models.emplace_back(compiled);
        return outputs;
    }

    std::shared_ptr<ov::Model> make_static_model(const std::shared_ptr<ov::Model>& model,
                                                 const std::vector<ov::Shape>& static_shapes) {
        auto clone = model->clone();
        const auto& params = clone->get_parameters();
        OPENVINO_ASSERT(params.size() == static_shapes.size(),
                        "Mismatch between parameters count and provided static shapes");
        for (size_t i = 0; i < params.size(); ++i) {
            params[i]->set_partial_shape(static_shapes[i]);
        }
        clone->validate_nodes_and_infer_types();
        return clone;
    }

    static std::vector<ov::Tensor> clone_inputs(const std::vector<ov::Tensor>& src) {
        std::vector<ov::Tensor> dst;
        dst.reserve(src.size());
        for (const auto& t : src) {
            ov::Tensor copy{t.get_element_type(), t.get_shape()};
            std::memcpy(copy.data(), t.data(), t.get_byte_size());
            dst.emplace_back(std::move(copy));
        }
        return dst;
    }

    void compare_gfx_vs_template(const std::shared_ptr<ov::Model>& model,
                                 const std::vector<ov::Tensor>& inputs,
                                 const ov::AnyMap& gfx_config,
                                 float abs_tol = 1e-5f,
                                 float rel_tol = 1e-5f) {
        ov::test::utils::require_gfx_backend();
        ov::AnyMap ref_config;

        std::vector<ov::Tensor> out_ref;
        bool ref_failed = false;

        auto try_run = [&](std::shared_ptr<ov::Model>& m, std::vector<ov::Tensor>& in) -> bool {
            try {
                out_ref = run_on_device(m, device_ref, in, ref_config);
                return true;
            } catch (...) {
                return false;
            }
        };

        auto has_fp16_input = [&]() {
            return std::any_of(inputs.begin(), inputs.end(), [](const ov::Tensor& t) {
                auto et = t.get_element_type();
                return et == ov::element::f16 || et == ov::element::bf16;
            });
        };

        // Some reference backends convert FP16 models to FP32; to avoid pointer
        // representability issues in tests, run reference in FP32 when inputs are FP16/BF16.
        auto ref_model = model;
        std::vector<ov::Tensor> ref_inputs = clone_inputs(inputs);

        // If FP16/BF16 present, convert reference model+inputs to F32 to avoid pointer issues.
        if (has_fp16_input()) {
            ov::pass::Manager m;
            m.register_pass<ov::pass::ConvertPrecision>(ov::element::f16, ov::element::f32);
            m.register_pass<ov::pass::ConvertPrecision>(ov::element::bf16, ov::element::f32);
            ref_model = model->clone();
            m.run_passes(ref_model);
            ref_model->validate_nodes_and_infer_types();
            for (auto& t : ref_inputs) {
                auto et = t.get_element_type();
                if (et == ov::element::f16 || et == ov::element::bf16) {
                    t = ov::gfx_plugin::to_float32_tensor(t);
                }
            }
        }

        if (ov::test::utils::gfx_tests_debug_enabled()) {
            std::cerr << "[GFX TEST] inputs dump\n";
            for (size_t i = 0; i < ref_inputs.size(); ++i) {
                const auto& t = ref_inputs[i];
                std::cerr << "  input[" << i << "] et=" << t.get_element_type().get_type_name()
                          << " shape=";
                auto s = t.get_shape();
                std::cerr << "[";
                for (size_t j = 0; j < s.size(); ++j) {
                    if (j) std::cerr << ",";
                    std::cerr << s[j];
                }
                std::cerr << "] first:";
                auto as_f32 = ov::gfx_plugin::to_float32_tensor(t);
                auto old_precision = std::cerr.precision();
                std::cerr.setf(std::ios::fixed, std::ios::floatfield);
                std::cerr << std::setprecision(10);
                const float* p = as_f32.data<const float>();
                size_t n = std::min<size_t>(8, as_f32.get_size());
                for (size_t j = 0; j < n; ++j) std::cerr << " " << p[j];
                std::cerr << "\n";
                std::cerr.precision(old_precision);
                std::cerr.unsetf(std::ios::floatfield);
            }
        }

        if (!try_run(ref_model, ref_inputs)) {
            ref_failed = true;
        }
        if (ref_failed) {
            FAIL() << "Reference device execution failed";
        }

        auto out_metal = run_on_device(model, device_gfx, clone_inputs(inputs), gfx_config);

        if (ov::test::utils::gfx_tests_debug_enabled() && !out_ref.empty() && !out_metal.empty()) {
            auto dump = [](const ov::Tensor& t, const char* tag) {
                std::cerr << "[GFX TEST] " << tag << " shape=";
                auto s = t.get_shape();
                std::cerr << "[";
                for (size_t i = 0; i < s.size(); ++i) {
                    if (i) std::cerr << ",";
                    std::cerr << s[i];
                }
                std::cerr << "] first:";
                auto as_f32 = ov::gfx_plugin::to_float32_tensor(t);
                const float* p = as_f32.data<const float>();
                size_t n = std::min<size_t>(8, as_f32.get_size());
                for (size_t i = 0; i < n; ++i) std::cerr << " " << p[i];
                std::cerr << "\n";
            };
            dump(out_ref[0], "ref_out0");
            dump(out_metal[0], "gfx_out0");
        }

        ASSERT_EQ(out_ref.size(), out_metal.size())
            << "Reference device returned " << out_ref.size() << " outputs, GFX returned " << out_metal.size();

        bool uses_fp16 = std::any_of(inputs.begin(), inputs.end(), [](const ov::Tensor& t) {
            auto et = t.get_element_type();
            return et == ov::element::f16 || et == ov::element::bf16;
        });
        float local_abs = abs_tol;
        float local_rel = rel_tol;
        if (uses_fp16) {
            local_abs = std::max(local_abs, 5e-4f);
            local_rel = std::max(local_rel, 5e-4f);
        }

        for (size_t i = 0; i < out_ref.size(); ++i) {
            const auto ref_t = out_ref[i].get_element_type() == ov::element::f32
                                   ? out_ref[i]
                                   : ov::gfx_plugin::to_float32_tensor(out_ref[i]);
            const auto metal_t = out_metal[i].get_element_type() == ov::element::f32
                                     ? out_metal[i]
                                     : ov::gfx_plugin::to_float32_tensor(out_metal[i]);
            ov::test::utils::compare(ref_t, metal_t, local_abs, local_rel);
        }
    }
};

// Base for single-layer shared tests: inherits shared fixture and runs GFX vs TEMPLATE.
template <class Base>
class GfxVsTemplateLayerTest : public Base, protected GfxVsTemplateFixture {
protected:
    void SetUp() override {
        Base::SetUp();
        if (this->targetStaticShapes.empty()) {
            std::vector<ov::Shape> shapes;
            for (const auto& param : this->function->get_parameters()) {
                shapes.emplace_back(param->get_shape());
            }
            this->targetStaticShapes.push_back(shapes);
        }
    }

    void run_compare() {
        ov::test::utils::require_gfx_backend();
        for (const auto& static_shapes : this->targetStaticShapes) {
            const auto param_count = this->function->get_parameters().size();
            if (static_shapes.size() < param_count) {
                FAIL() << "Static shapes provided (" << static_shapes.size()
                       << ") less than parameter count (" << param_count << ")";
            }
            std::vector<ov::Shape> param_shapes(static_shapes.begin(),
                                                static_shapes.begin() + static_cast<std::ptrdiff_t>(param_count));

            this->generate_inputs(param_shapes);
            std::vector<ov::Tensor> inputs;
            inputs.reserve(this->function->inputs().size());
            for (const auto& input : this->function->inputs()) {
                inputs.emplace_back(this->inputs.at(input.get_node_shared_ptr()));
            }
            auto static_model = make_static_model(this->function, param_shapes);

            compare_gfx_vs_template(static_model,
                                    inputs,
                                    this->configuration,
                                    this->abs_threshold,
                                    this->rel_threshold);
        }
    }
};

}  // namespace utils
}  // namespace test
}  // namespace ov
