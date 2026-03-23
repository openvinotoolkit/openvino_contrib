#include <openvino/openvino.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "common_test_utils/ov_plugin_cache.hpp"

namespace {

const char* resolve_gfx_plugin_path() {
    if (const char* env_path = std::getenv("GFX_PLUGIN_PATH")) {
        if (*env_path) {
            return env_path;
        }
    }
#ifdef GFX_PLUGIN_PATH
    return GFX_PLUGIN_PATH;
#else
    return nullptr;
#endif
}

void register_gfx_plugin(ov::Core& core) {
    if (const char* path = resolve_gfx_plugin_path()) {
        try {
            core.register_plugin(path, "GFX");
        } catch (...) {
        }
    }
}

std::string reference_device(const ov::Core& core) {
    try {
        ov::test::utils::register_template_plugin(const_cast<ov::Core&>(core));
    } catch (...) {
    }
    const auto devices = core.get_available_devices();
    if (std::find(devices.begin(), devices.end(), "TEMPLATE") != devices.end()) {
        return "TEMPLATE";
    }
    throw std::runtime_error("TEMPLATE reference device not available");
}

template <typename T>
void fill_tensor_data(ov::Tensor& tensor) {
    T* data = tensor.data<T>();
    const size_t count = tensor.get_size();
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<T>(((static_cast<int64_t>(i) % 251) - 125) / 32.0f);
    }
}

void fill_tensor(ov::Tensor& tensor) {
    switch (tensor.get_element_type()) {
        case ov::element::f32:
            fill_tensor_data<float>(tensor);
            return;
        case ov::element::f16:
            fill_tensor_data<ov::float16>(tensor);
            return;
        case ov::element::i32:
            fill_tensor_data<int32_t>(tensor);
            return;
        case ov::element::i64:
            fill_tensor_data<int64_t>(tensor);
            return;
        case ov::element::u8:
            fill_tensor_data<uint8_t>(tensor);
            return;
        default:
            throw std::runtime_error("unsupported input type: " + tensor.get_element_type().to_string());
    }
}

struct DiffStats {
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    size_t elements = 0;
    size_t max_index = 0;
    double lhs_at_max = 0.0;
    double rhs_at_max = 0.0;
};

template <typename T>
DiffStats compare_typed(const ov::Tensor& a, const ov::Tensor& b) {
    const T* lhs = a.data<const T>();
    const T* rhs = b.data<const T>();
    DiffStats stats;
    stats.elements = a.get_size();
    for (size_t i = 0; i < stats.elements; ++i) {
        const double da = static_cast<double>(lhs[i]);
        const double db = static_cast<double>(rhs[i]);
        const double abs_diff = std::abs(da - db);
        const double denom = std::max({std::abs(da), std::abs(db), 1e-12});
        const double rel_diff = abs_diff / denom;
        if (abs_diff > stats.max_abs_diff) {
            stats.max_abs_diff = abs_diff;
            stats.max_index = i;
            stats.lhs_at_max = da;
            stats.rhs_at_max = db;
        }
        stats.max_rel_diff = std::max(stats.max_rel_diff, rel_diff);
    }
    return stats;
}

DiffStats compare_tensors(const ov::Tensor& a, const ov::Tensor& b) {
    if (a.get_element_type() != b.get_element_type()) {
        throw std::runtime_error("output type mismatch");
    }
    if (a.get_shape() != b.get_shape()) {
        throw std::runtime_error("output shape mismatch");
    }
    switch (a.get_element_type()) {
        case ov::element::f32:
            return compare_typed<float>(a, b);
        case ov::element::f16:
            return compare_typed<ov::float16>(a, b);
        case ov::element::i32:
            return compare_typed<int32_t>(a, b);
        case ov::element::i64:
            return compare_typed<int64_t>(a, b);
        case ov::element::u8:
            return compare_typed<uint8_t>(a, b);
        default:
            throw std::runtime_error("unsupported output type: " + a.get_element_type().to_string());
    }
}

double median_ms(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if ((values.size() & 1u) != 0u) {
        return values[mid];
    }
    return 0.5 * (values[mid - 1] + values[mid]);
}

ov::Shape make_static_shape(const ov::PartialShape& ps, int64_t fallback = 1) {
    if (ps.is_static()) {
        return ps.to_shape();
    }
    ov::Shape shape;
    if (ps.rank().is_static()) {
        shape.reserve(ps.rank().get_length());
    }
    for (const auto& dim : ps) {
        shape.push_back(static_cast<size_t>(dim.is_static() ? dim.get_length() : fallback));
    }
    if (shape.empty()) {
        shape.push_back(1);
    }
    return shape;
}

double benchmark_request(ov::InferRequest& request, size_t iterations) {
    std::vector<double> samples;
    samples.reserve(iterations);
    for (size_t i = 0; i < iterations; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        request.infer();
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return median_ms(samples);
}

bool is_debug_skippable_node(const std::shared_ptr<ov::Node>& node) {
    return ov::is_type<ov::op::v0::Parameter>(node) || ov::is_type<ov::op::v0::Constant>(node) ||
           ov::is_type<ov::op::v0::Result>(node);
}

struct CompareOptions {
    size_t iterations = 10;
    bool per_op = false;
    bool print_ops = false;
    bool time_per_op = false;
    size_t start_op = 0;
    std::optional<size_t> stop_after_op;
    double abs_threshold = 1e-4;
    double rel_threshold = 1e-4;
};

ov::AnyMap make_compile_config();
ov::InferRequest make_request(ov::CompiledModel& compiled_model,
                              const std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>& inputs);

ov::AnyMap make_compile_config() {
    ov::AnyMap config;
    config[ov::hint::inference_precision.name()] = ov::element::f16;
    return config;
}

ov::InferRequest make_request(ov::CompiledModel& compiled_model,
                              const std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>& inputs) {
    auto request = compiled_model.create_infer_request();
    for (const auto& [port, tensor] : inputs) {
        request.set_tensor(port, tensor);
    }
    return request;
}

std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>> make_inputs(const std::shared_ptr<ov::Model>& model) {
    std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>> inputs;
    inputs.reserve(model->inputs().size());
    for (const auto& input : model->inputs()) {
        ov::Tensor tensor(input.get_element_type(), make_static_shape(input.get_partial_shape()));
        fill_tensor(tensor);
        inputs.emplace_back(input, std::move(tensor));
    }
    return inputs;
}

DiffStats compare_model_outputs(ov::CompiledModel& ref_model,
                                ov::CompiledModel& gfx_model,
                                const std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>& inputs,
                                bool print_outputs) {
    auto ref_req = make_request(ref_model, inputs);
    auto gfx_req = make_request(gfx_model, inputs);

    ref_req.infer();
    gfx_req.infer();

    DiffStats total;
    for (size_t i = 0; i < ref_model.outputs().size(); ++i) {
        const auto ref_tensor = ref_req.get_tensor(ref_model.output(i));
        const auto gfx_tensor = gfx_req.get_tensor(gfx_model.output(i));
        const auto stats = compare_tensors(ref_tensor, gfx_tensor);
        if (stats.max_abs_diff > total.max_abs_diff) {
            total.max_abs_diff = stats.max_abs_diff;
            total.max_index = stats.max_index;
            total.lhs_at_max = stats.lhs_at_max;
            total.rhs_at_max = stats.rhs_at_max;
        }
        total.max_rel_diff = std::max(total.max_rel_diff, stats.max_rel_diff);
        total.elements += stats.elements;
        if (print_outputs) {
            std::cout << "output[" << i << "]"
                      << " elements=" << stats.elements
                      << " max_abs_diff=" << std::setprecision(10) << stats.max_abs_diff
                      << " max_rel_diff=" << stats.max_rel_diff << '\n';
        }
    }
    return total;
}

std::optional<ov::Tensor> evaluate_constant_source_tensor(const ov::Output<ov::Node>& source) {
    auto node = source.get_node_shared_ptr();
    if (!node) {
        return std::nullopt;
    }
    if (auto constant = std::dynamic_pointer_cast<const ov::op::v0::Constant>(node)) {
        return constant->get_tensor_view();
    }
    if (!node->has_evaluate()) {
        return std::nullopt;
    }

    ov::TensorVector inputs;
    inputs.reserve(node->get_input_size());
    for (const auto& input_value : node->input_values()) {
        auto input_tensor = evaluate_constant_source_tensor(input_value);
        if (!input_tensor.has_value()) {
            return std::nullopt;
        }
        inputs.push_back(*input_tensor);
    }

    ov::TensorVector outputs;
    outputs.reserve(node->get_output_size());
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        if (node->get_output_partial_shape(i).is_dynamic()) {
            return std::nullopt;
        }
        outputs.emplace_back(node->get_output_element_type(i), node->get_output_shape(i));
    }
    if (!node->evaluate(outputs, inputs)) {
        return std::nullopt;
    }
    return outputs.at(source.get_index());
}

std::optional<ov::Tensor> evaluate_source_tensor(
    const ov::Output<ov::Node>& source,
    const std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>& inputs) {
    auto node = source.get_node_shared_ptr();
    if (!node) {
        return std::nullopt;
    }

    if (ov::is_type<ov::op::v0::Parameter>(node)) {
        for (const auto& [port, tensor] : inputs) {
            if (port.get_node_shared_ptr() == node && port.get_index() == source.get_index()) {
                return tensor;
            }
        }
        return std::nullopt;
    }

    if (auto constant = std::dynamic_pointer_cast<const ov::op::v0::Constant>(node)) {
        return constant->get_tensor_view();
    }

    if (!node->has_evaluate()) {
        return std::nullopt;
    }

    ov::TensorVector eval_inputs;
    eval_inputs.reserve(node->get_input_size());
    for (const auto& input_value : node->input_values()) {
        auto input_tensor = evaluate_source_tensor(input_value, inputs);
        if (!input_tensor.has_value()) {
            return std::nullopt;
        }
        eval_inputs.push_back(*input_tensor);
    }

    ov::TensorVector outputs;
    outputs.reserve(node->get_output_size());
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        if (node->get_output_partial_shape(i).is_dynamic()) {
            return std::nullopt;
        }
        outputs.emplace_back(node->get_output_element_type(i), node->get_output_shape(i));
    }
    if (!node->evaluate(outputs, eval_inputs)) {
        return std::nullopt;
    }
    return outputs.at(source.get_index());
}

std::optional<ov::Tensor> evaluate_source_tensor_with_reference(
    ov::Core& core,
    const std::string& ref_device,
    const ov::Output<ov::Node>& source,
    const std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>& inputs) {
    ov::ParameterVector params;
    params.reserve(inputs.size());
    for (const auto& [port, _] : inputs) {
        auto param_const = std::dynamic_pointer_cast<const ov::op::v0::Parameter>(port.get_node_shared_ptr());
        if (!param_const) {
            return std::nullopt;
        }
        params.push_back(std::const_pointer_cast<ov::op::v0::Parameter>(param_const));
    }

    ov::OutputVector outputs;
    outputs.push_back(std::make_shared<ov::op::v0::Result>(source));
    auto submodel = std::make_shared<ov::Model>(outputs, params, source.get_any_name());
    auto compiled = core.compile_model(submodel, ref_device, make_compile_config());
    auto request = make_request(compiled, inputs);
    request.infer();
    return request.get_tensor(compiled.output(0));
}

void print_conv_probe(ov::Core& core,
                      const std::string& ref_device,
                      const std::shared_ptr<ov::Node>& node,
                      const std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>& inputs,
                      const DiffStats& stats) {
    auto conv = std::dynamic_pointer_cast<ov::op::v1::Convolution>(node);
    if (!conv || stats.elements == 0) {
        return;
    }
    if (conv->get_input_size() != 2 || conv->get_output_shape(0).size() != 4 || inputs.empty()) {
        return;
    }
    auto input_tensor = evaluate_source_tensor(conv->input_value(0), inputs);
    auto weights_tensor = evaluate_source_tensor(conv->input_value(1), inputs);
    if (!input_tensor.has_value()) {
        input_tensor = evaluate_source_tensor_with_reference(core, ref_device, conv->input_value(0), inputs);
    }
    if (!weights_tensor.has_value()) {
        weights_tensor = evaluate_source_tensor_with_reference(core, ref_device, conv->input_value(1), inputs);
    }
    if (!input_tensor.has_value() || !weights_tensor.has_value()) {
        return;
    }
    if (input_tensor->get_element_type() != ov::element::f32 || weights_tensor->get_element_type() != ov::element::f32) {
        return;
    }
    if (input_tensor->get_shape() != conv->get_input_shape(0) || weights_tensor->get_shape() != conv->get_input_shape(1)) {
        return;
    }

    const auto* input_ptr = input_tensor->data<const float>();
    const auto* weight_ptr = weights_tensor->data<const float>();
    const auto& out_shape = conv->get_output_shape(0);
    const size_t spatial = out_shape[2] * out_shape[3];
    const size_t channel_stride = spatial;
    const size_t batch_stride = out_shape[1] * channel_stride;
    const size_t n = stats.max_index / batch_stride;
    const size_t rem0 = stats.max_index % batch_stride;
    const size_t oc = rem0 / channel_stride;
    const size_t rem1 = rem0 % channel_stride;
    const size_t oh = rem1 / out_shape[3];
    const size_t ow = rem1 % out_shape[3];

    const auto& in_shape = conv->get_input_shape(0);
    const auto& w_shape = conv->get_input_shape(1);
    const auto& strides = conv->get_strides();
    const auto& dilations = conv->get_dilations();
    const auto& pads_begin = conv->get_pads_begin();

    auto f16_round = [](float v) -> float {
        return static_cast<float>(ov::float16(v));
    };

    float acc_f16 = 0.0f;
    float acc_f16_accum = 0.0f;
    float acc_f32 = 0.0f;
    float acc_hwio = 0.0f;
    for (size_t ic = 0; ic < in_shape[1]; ++ic) {
        for (size_t kh = 0; kh < w_shape[2]; ++kh) {
            const int64_t ih = static_cast<int64_t>(oh * strides[0]) - static_cast<int64_t>(pads_begin[0]) +
                               static_cast<int64_t>(kh * dilations[0]);
            if (ih < 0 || ih >= static_cast<int64_t>(in_shape[2])) {
                continue;
            }
            for (size_t kw = 0; kw < w_shape[3]; ++kw) {
                const int64_t iw = static_cast<int64_t>(ow * strides[1]) - static_cast<int64_t>(pads_begin[1]) +
                                   static_cast<int64_t>(kw * dilations[1]);
                if (iw < 0 || iw >= static_cast<int64_t>(in_shape[3])) {
                    continue;
                }
                const size_t in_idx = ((n * in_shape[1] + ic) * in_shape[2] + static_cast<size_t>(ih)) * in_shape[3] +
                                      static_cast<size_t>(iw);
                const size_t w_idx = ((oc * w_shape[1] + ic) * w_shape[2] + kh) * w_shape[3] + kw;
                const size_t w_idx_hwio = ((kh * w_shape[3] + kw) * w_shape[1] + ic) * w_shape[0] + oc;
                const float in_v = input_ptr[in_idx];
                const float w_v = weight_ptr[w_idx];
                const float prod_f16 = f16_round(in_v) * f16_round(w_v);
                acc_f16 += prod_f16;
                acc_f16_accum = f16_round(acc_f16_accum + prod_f16);
                acc_f32 += in_v * w_v;
                acc_hwio += in_v * weight_ptr[w_idx_hwio];
            }
        }
    }

    std::cout << "CONV_PROBE n=" << n
              << " oc=" << oc
              << " oh=" << oh
              << " ow=" << ow
              << " naive_f16=" << std::setprecision(10) << acc_f16
              << " naive_f16_accum=" << acc_f16_accum
              << " naive_f32=" << acc_f32
              << " naive_hwio=" << acc_hwio
              << '\n';
}

int run_per_op_compare(ov::Core& core,
                       const std::shared_ptr<ov::Model>& model,
                       const std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>& inputs,
                       const CompareOptions& options) {
    const auto ordered_ops = model->get_ordered_ops();
    const auto ref_device = reference_device(core);
    size_t checked = 0;
    for (size_t idx = 0; idx < ordered_ops.size(); ++idx) {
        const auto& node = ordered_ops[idx];
        if (idx < options.start_op) {
            continue;
        }
        if (is_debug_skippable_node(node)) {
            continue;
        }
        if (options.stop_after_op.has_value() && checked >= *options.stop_after_op) {
            break;
        }

        ov::OutputVector outputs;
        outputs.reserve(node->get_output_size());
        for (const auto& output : node->outputs()) {
            outputs.push_back(std::make_shared<ov::op::v0::Result>(output));
        }
        auto submodel = std::make_shared<ov::Model>(outputs, model->get_parameters(), node->get_friendly_name());

        ov::CompiledModel ref_submodel;
        ov::CompiledModel gfx_submodel;
        try {
            ref_submodel = core.compile_model(submodel, ref_device, make_compile_config());
            gfx_submodel = core.compile_model(submodel, "GFX", make_compile_config());
        } catch (const std::exception& ex) {
            std::cout << "[op " << idx << "] " << node->get_friendly_name() << " (" << node->get_type_name()
                      << ") compile_skip=" << ex.what() << '\n';
            ++checked;
            continue;
        }

        DiffStats stats;
        try {
            stats = compare_model_outputs(ref_submodel, gfx_submodel, inputs, false);
        } catch (const std::exception& ex) {
            std::cout << "[op " << idx << "] " << node->get_friendly_name() << " (" << node->get_type_name()
                      << ") infer_skip=" << ex.what() << '\n';
            ++checked;
            continue;
        }

        std::cout << "[op " << idx << "] " << node->get_friendly_name() << " (" << node->get_type_name()
                  << ") max_abs_diff=" << std::setprecision(10) << stats.max_abs_diff
                  << " max_rel_diff=" << stats.max_rel_diff;
        if (options.time_per_op) {
            auto ref_req = make_request(ref_submodel, inputs);
            auto gfx_req = make_request(gfx_submodel, inputs);
            const double ref_ms = benchmark_request(ref_req, options.iterations);
            const double gfx_ms = benchmark_request(gfx_req, options.iterations);
            std::cout << " ref_ms=" << ref_ms
                      << " gfx_ms=" << gfx_ms;
        }
        std::cout << '\n';
        if (stats.max_abs_diff > options.abs_threshold && stats.max_rel_diff > options.rel_threshold) {
            print_conv_probe(core, ref_device, node, inputs, stats);
            std::cout << "FIRST_MISMATCH op_index=" << idx
                      << " name=" << node->get_friendly_name()
                      << " type=" << node->get_type_name()
                      << " max_index=" << stats.max_index
                      << " ref=" << std::setprecision(10) << stats.lhs_at_max
                      << " gfx=" << stats.rhs_at_max << '\n';
            return 3;
        }
        ++checked;
    }
    std::cout << "PER_OP_MATCH\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 2) {
        std::cerr << "usage: ov_gfx_compare_runner <model.xml> [iterations] [--per-op] [--print-ops] "
                     "[--time-per-op] [--start-op N] "
                     "[--stop-after-op N] [--abs-threshold V] [--rel-threshold V]\n";
        return 2;
    }

    const std::string model_path = argv[1];
    CompareOptions options;
    bool positional_iterations_consumed = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--per-op") {
            options.per_op = true;
            continue;
        }
        if (arg == "--print-ops") {
            options.print_ops = true;
            continue;
        }
        if (arg == "--time-per-op") {
            options.time_per_op = true;
            continue;
        }
        if (arg == "--start-op") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--start-op requires a value");
            }
            options.start_op = static_cast<size_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg == "--stop-after-op") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--stop-after-op requires a value");
            }
            options.stop_after_op = static_cast<size_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg == "--abs-threshold") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--abs-threshold requires a value");
            }
            options.abs_threshold = std::stod(argv[++i]);
            continue;
        }
        if (arg == "--rel-threshold") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--rel-threshold requires a value");
            }
            options.rel_threshold = std::stod(argv[++i]);
            continue;
        }
        if (!positional_iterations_consumed) {
            options.iterations = static_cast<size_t>(std::stoul(arg));
            positional_iterations_consumed = true;
            continue;
        }
        throw std::runtime_error("unknown argument: " + arg);
    }

    ov::Core core;
    register_gfx_plugin(core);
    ov::test::utils::register_template_plugin(core);

    auto model = core.read_model(model_path);
    const auto inputs = make_inputs(model);

    if (options.print_ops) {
        const auto ordered_ops = model->get_ordered_ops();
        for (size_t idx = 0; idx < ordered_ops.size(); ++idx) {
            const auto& node = ordered_ops[idx];
            if (is_debug_skippable_node(node)) {
                continue;
            }
            std::cout << "[op " << idx << "] "
                      << node->get_friendly_name()
                      << " (" << node->get_type_name() << ")\n";
        }
        return 0;
    }

    if (options.per_op) {
        return run_per_op_compare(core, model, inputs, options);
    }

    const auto ref_device = reference_device(core);
    auto ref_model = core.compile_model(model, ref_device, make_compile_config());
    auto gfx_model = core.compile_model(model, "GFX", make_compile_config());
    auto ref_req = make_request(ref_model, inputs);
    auto gfx_req = make_request(gfx_model, inputs);

    ref_req.infer();
    gfx_req.infer();

    double global_max_abs = 0.0;
    double global_max_rel = 0.0;
    for (const auto& output : model->outputs()) {
        const auto ref_tensor = ref_req.get_tensor(output.get_any_name());
        const auto gfx_tensor = gfx_req.get_tensor(output.get_any_name());
        const auto stats = compare_tensors(ref_tensor, gfx_tensor);
        global_max_abs = std::max(global_max_abs, stats.max_abs_diff);
        global_max_rel = std::max(global_max_rel, stats.max_rel_diff);
        std::cout << output.get_any_name()
                  << " elements=" << stats.elements
                  << " max_abs_diff=" << std::setprecision(10) << stats.max_abs_diff
                  << " max_rel_diff=" << stats.max_rel_diff
                  << '\n';
    }

    const double ref_ms = benchmark_request(ref_req, options.iterations);
    const double gfx_ms = benchmark_request(gfx_req, options.iterations);
    const double ref_fps = ref_ms > 0.0 ? (1000.0 / ref_ms) : 0.0;
    const double gfx_fps = gfx_ms > 0.0 ? (1000.0 / gfx_ms) : 0.0;

    std::cout << "GLOBAL max_abs_diff=" << std::setprecision(10) << global_max_abs
              << " max_rel_diff=" << global_max_rel << '\n';
    std::cout << "REF[" << ref_device << "] median_ms=" << ref_ms << " fps=" << ref_fps << '\n';
    std::cout << "GFX median_ms=" << gfx_ms << " fps=" << gfx_fps << '\n';
    if (gfx_ms > 0.0) {
        std::cout << "speedup_vs_ref=" << (ref_ms / gfx_ms) << '\n';
    }
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "ERROR: " << ex.what() << '\n';
    return 1;
}
