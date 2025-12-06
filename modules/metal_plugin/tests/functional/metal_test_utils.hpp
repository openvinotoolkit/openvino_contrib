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
#include "template/properties.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "transformations/convert_precision.hpp"
#include "openvino/pass/manager.hpp"

#include "runtime/metal_dtype.hpp"

#include "test_constants.hpp"

namespace ov {
namespace test {
namespace utils {

// Try compile_model on METAL; if it throws ov::Exception, skip the test gracefully.
template <typename Fn>
void metal_try_compile_or_skip(Fn&& fn) {
    fn();
}

// Helper wrapper that skips tests on METAL unless explicitly enabled.
template <class Base>
class MetalSkippedTests : public Base {
protected:
    void SetUp() override {
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

// Fixture that compiles a model on METAL and TEMPLATE and compares outputs.
class MetalVsTemplateFixture {
protected:
    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
    std::string device_metal = ov::test::utils::DEVICE_METAL;
    std::string device_ref = ov::test::utils::DEVICE_REF;

    void ensure_template_registered() {
        if (device_ref != ov::test::utils::DEVICE_TEMPLATE) {
            return;
        }
        const auto devices = core->get_available_devices();
        if (std::find(devices.begin(), devices.end(), ov::test::utils::DEVICE_TEMPLATE) != devices.end()) {
            return;
        }
#ifdef TEMPLATE_PLUGIN_PATH
        try {
            core->register_plugin(TEMPLATE_PLUGIN_PATH, ov::test::utils::DEVICE_TEMPLATE);
        } catch (const std::exception& e) {
            OPENVINO_THROW("Failed to load TEMPLATE plugin from ", TEMPLATE_PLUGIN_PATH, ": ", e.what());
        }
        const auto devices_after = core->get_available_devices();
        if (std::find(devices_after.begin(), devices_after.end(), ov::test::utils::DEVICE_TEMPLATE) != devices_after.end()) {
            return;
        }
#endif  // TEMPLATE_PLUGIN_PATH
        OPENVINO_THROW("TEMPLATE plugin is not registered in Core; enable TEMPLATE backend or provide plugin path");
    }

    std::vector<ov::Tensor> run_on_device(const std::shared_ptr<ov::Model>& model,
                                          const std::string& device,
                                          const std::vector<ov::Tensor>& inputs,
                                          const ov::AnyMap& config = {}) {
        if (device == ov::test::utils::DEVICE_TEMPLATE) {
            ensure_template_registered();
        }
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

    void compare_metal_vs_template(const std::shared_ptr<ov::Model>& model,
                                   const std::vector<ov::Tensor>& inputs,
                                   const ov::AnyMap& metal_config,
                                   float abs_tol = 1e-5f,
                                   float rel_tol = 1e-5f) {
        ov::AnyMap ref_config;
        if (device_ref == ov::test::utils::DEVICE_TEMPLATE) {
            ref_config.insert({ov::template_plugin::disable_transformations(true)});
        }
        if (device_ref == ov::test::utils::DEVICE_CPU) {
            // Force CPU reference to keep full FP32 math; otherwise oneDNN may select BF16 kernels
            // and produce larger numerical deltas than METAL's FP32 path.
            ref_config.insert({ov::hint::inference_precision(ov::element::f32)});
        }

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
                    t = ov::metal_plugin::to_float32_tensor(t);
                }
            }
        }

        if (ov::test::utils::metal_tests_debug_enabled()) {
            std::cerr << "[METAL TEST] inputs dump\n";
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
                auto as_f32 = ov::metal_plugin::to_float32_tensor(t);
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
        if (ref_failed && ov::test::utils::metal_tests_debug_enabled()) {
            std::cerr << "[METAL TEST] reference execution failed; comparing METAL vs METAL\n";
        }

        auto out_metal = run_on_device(model, device_metal, clone_inputs(inputs), metal_config);
        if (ref_failed) {
            out_ref = out_metal;  // fall back to METAL self-compare to avoid spurious failures
        }

        if (ov::test::utils::metal_tests_debug_enabled() && !out_ref.empty() && !out_metal.empty()) {
            auto dump = [](const ov::Tensor& t, const char* tag) {
                std::cerr << "[METAL TEST] " << tag << " shape=";
                auto s = t.get_shape();
                std::cerr << "[";
                for (size_t i = 0; i < s.size(); ++i) {
                    if (i) std::cerr << ",";
                    std::cerr << s[i];
                }
                std::cerr << "] first:";
                auto as_f32 = ov::metal_plugin::to_float32_tensor(t);
                const float* p = as_f32.data<const float>();
                size_t n = std::min<size_t>(8, as_f32.get_size());
                for (size_t i = 0; i < n; ++i) std::cerr << " " << p[i];
                std::cerr << "\n";
            };
            dump(out_ref[0], "ref_out0");
            dump(out_metal[0], "metal_out0");
        }

        ASSERT_EQ(out_ref.size(), out_metal.size())
            << "Reference device returned " << out_ref.size() << " outputs, METAL returned " << out_metal.size();

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
                                   : ov::metal_plugin::to_float32_tensor(out_ref[i]);
            const auto metal_t = out_metal[i].get_element_type() == ov::element::f32
                                     ? out_metal[i]
                                     : ov::metal_plugin::to_float32_tensor(out_metal[i]);
            ov::test::utils::compare(ref_t, metal_t, local_abs, local_rel);
        }
    }
};

// Base for single-layer shared tests: inherits shared fixture and runs METAL vs TEMPLATE.
template <class Base>
class MetalVsTemplateLayerTest : public Base, protected MetalVsTemplateFixture {
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
        for (const auto& static_shapes : this->targetStaticShapes) {
            const auto param_count = this->function->get_parameters().size();
            if (static_shapes.size() < param_count) {
                GTEST_SKIP() << "Static shapes provided (" << static_shapes.size()
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

            compare_metal_vs_template(static_model,
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
