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
#include <unordered_map>
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

void register_reference_plugin(ov::Core& core,
                               const std::string& reference_device_name,
                               const std::string& reference_plugin_path) {
    if (!reference_plugin_path.empty()) {
        const auto devices = core.get_available_devices();
        if (std::find(devices.begin(), devices.end(), reference_device_name) == devices.end()) {
            core.register_plugin(reference_plugin_path, reference_device_name);
        }
    }
    if (reference_device_name == "TEMPLATE") {
        try {
            ov::test::utils::register_template_plugin(core);
        } catch (...) {
        }
    }
}

std::string reference_device(const ov::Core& core, const std::string& requested_device) {
    if (requested_device == "CPU") {
        throw std::runtime_error("CPU reference device is not supported; use TEMPLATE");
    }
    const auto devices = core.get_available_devices();
    if (std::find(devices.begin(), devices.end(), requested_device) != devices.end()) {
        return requested_device;
    }
    throw std::runtime_error(requested_device + " reference device not available");
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

bool is_debug_skippable_node(const std::shared_ptr<ov::Node>& node) {
    return ov::is_type<ov::op::v0::Parameter>(node) || ov::is_type<ov::op::v0::Constant>(node) ||
           ov::is_type<ov::op::v0::Result>(node);
}

struct CompareOptions {
    bool per_op = false;
    bool per_op_all = false;
    bool print_ops = false;
    bool gfx_only = false;
    std::string reference_device = "TEMPLATE";
    std::string reference_plugin_path;
    size_t start_op = 0;
    size_t window_size = 1;
    std::optional<size_t> stop_after_op;
    double abs_threshold = 1e-4;
    double rel_threshold = 1e-4;
};

struct OutputKey {
    const ov::Node* node = nullptr;
    size_t port = 0;
    bool operator==(const OutputKey& other) const {
        return node == other.node && port == other.port;
    }
};

struct OutputKeyHash {
    size_t operator()(const OutputKey& key) const {
        size_t h1 = std::hash<const ov::Node*>()(key.node);
        size_t h2 = std::hash<size_t>()(key.port);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

ov::AnyMap make_compile_config(bool for_gfx);
ov::InferRequest make_request(ov::CompiledModel& compiled_model,
                              const std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>& inputs);
void maybe_print_gfx_profile(const ov::CompiledModel& compiled_model);

struct TensorSummary {
    size_t elements = 0;
    size_t finite_count = 0;
    size_t nan_count = 0;
    size_t inf_count = 0;
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double l2 = 0.0;
};

ov::AnyMap make_compile_config(bool for_gfx) {
    ov::AnyMap config;
    config[ov::hint::inference_precision.name()] = ov::element::f16;
    if (for_gfx) {
        if (const char* profiling_level = std::getenv("OV_GFX_PROFILING_LEVEL")) {
            if (*profiling_level) {
                config["GFX_PROFILING_LEVEL"] = std::string(profiling_level);
                config[ov::enable_profiling.name()] = true;
                config["PERF_COUNT"] = true;
            }
        }
        if (const char* disable_fusion = std::getenv("OV_GFX_DISABLE_FUSION")) {
        if (std::string(disable_fusion) != "0" && !std::string(disable_fusion).empty()) {
            config["GFX_ENABLE_FUSION"] = false;
        }
        }
    }
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

void maybe_print_gfx_profile(const ov::CompiledModel& compiled_model) {
    const char* dump_profile = std::getenv("OV_GFX_DUMP_PROFILE");
    if (!dump_profile || !*dump_profile) {
        return;
    }
    try {
        const auto profile_json = compiled_model.get_property("GFX_PROFILING_REPORT").as<std::string>();
        std::cout << "GFX_PROFILE " << profile_json << '\n';
    } catch (const std::exception& ex) {
        std::cout << "GFX_PROFILE_ERROR " << ex.what() << '\n';
    }
}

template <typename T>
TensorSummary summarize_typed_tensor(const ov::Tensor& tensor) {
    const T* data = tensor.data<const T>();
    TensorSummary summary;
    summary.elements = tensor.get_size();
    if (summary.elements == 0) {
        return summary;
    }

    double min_value = 0.0;
    double max_value = 0.0;
    double sum = 0.0;
    double sum_sq = 0.0;
    bool initialized = false;
    for (size_t i = 0; i < summary.elements; ++i) {
        const double value = static_cast<double>(data[i]);
        if (std::isnan(value)) {
            ++summary.nan_count;
            continue;
        }
        if (!std::isfinite(value)) {
            ++summary.inf_count;
            continue;
        }
        ++summary.finite_count;
        if (!initialized) {
            min_value = value;
            max_value = value;
            initialized = true;
        } else {
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
        }
        sum += value;
        sum_sq += value * value;
    }

    if (initialized) {
        summary.min = min_value;
        summary.max = max_value;
    }
    if (summary.finite_count > 0) {
        summary.mean = sum / static_cast<double>(summary.finite_count);
        summary.l2 = std::sqrt(sum_sq);
    }
    return summary;
}

TensorSummary summarize_tensor(const ov::Tensor& tensor) {
    switch (tensor.get_element_type()) {
        case ov::element::f32:
            return summarize_typed_tensor<float>(tensor);
        case ov::element::f16:
            return summarize_typed_tensor<ov::float16>(tensor);
        case ov::element::i32:
            return summarize_typed_tensor<int32_t>(tensor);
        case ov::element::i64:
            return summarize_typed_tensor<int64_t>(tensor);
        case ov::element::u8:
            return summarize_typed_tensor<uint8_t>(tensor);
        default:
            throw std::runtime_error("unsupported output type: " + tensor.get_element_type().to_string());
    }
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
                                bool print_outputs,
                                std::vector<ov::Tensor>* ref_outputs = nullptr) {
    auto ref_req = make_request(ref_model, inputs);
    auto gfx_req = make_request(gfx_model, inputs);

    ref_req.infer();
    gfx_req.infer();

    DiffStats total;
    if (ref_outputs) {
        ref_outputs->clear();
        ref_outputs->reserve(ref_model.outputs().size());
    }
    for (size_t i = 0; i < ref_model.outputs().size(); ++i) {
        const auto ref_tensor = ref_req.get_tensor(ref_model.output(i));
        const auto gfx_tensor = gfx_req.get_tensor(gfx_model.output(i));
        if (ref_outputs) {
            ref_outputs->push_back(ref_tensor);
        }
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
    auto source_node = source.get_node_shared_ptr();
    const std::string submodel_name = source_node ? (source_node->get_friendly_name() + "/out_" + std::to_string(source.get_index()))
                                                  : std::string("ref_source");
    auto submodel = std::make_shared<ov::Model>(outputs, params, submodel_name);
    auto compiled = core.compile_model(submodel, ref_device, make_compile_config(false));
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

struct FullGraphOutputDesc {
    size_t op_index = 0;
    std::string name;
    std::string type;
};

std::shared_ptr<ov::Model> make_full_graph_debug_model(const std::shared_ptr<ov::Model>& model,
                                                       std::vector<FullGraphOutputDesc>& output_descs) {
    auto cloned_model = model->clone();
    const auto cloned_ops = cloned_model->get_ordered_ops();
    ov::ResultVector results;
    output_descs.clear();
    for (size_t idx = 0; idx < cloned_ops.size(); ++idx) {
        const auto& node = cloned_ops[idx];
        if (is_debug_skippable_node(node)) {
            continue;
        }
        for (size_t port = 0; port < node->get_output_size(); ++port) {
            results.push_back(std::make_shared<ov::op::v0::Result>(node->output(port)));
            FullGraphOutputDesc desc;
            desc.op_index = idx;
            desc.type = node->get_type_name();
            desc.name = node->get_friendly_name();
            if (node->get_output_size() > 1) {
                desc.name += "/out_" + std::to_string(port);
            }
            output_descs.push_back(std::move(desc));
        }
    }
    return std::make_shared<ov::Model>(results, cloned_model->get_parameters(), cloned_model->get_friendly_name() + "_per_op");
}

int run_full_graph_per_op_compare(ov::Core& core,
                                  const std::shared_ptr<ov::Model>& model,
                                  const CompareOptions& options) {
    const auto ref_device = reference_device(core, options.reference_device);
    std::vector<FullGraphOutputDesc> output_descs;
    auto debug_model = make_full_graph_debug_model(model, output_descs);
    const auto inputs = make_inputs(debug_model);

    auto ref_model = core.compile_model(debug_model, ref_device, make_compile_config(false));
    auto gfx_model = core.compile_model(debug_model, "GFX", make_compile_config(true));
    auto ref_req = make_request(ref_model, inputs);
    auto gfx_req = make_request(gfx_model, inputs);

    ref_req.infer();
    gfx_req.infer();

    for (size_t i = 0; i < output_descs.size(); ++i) {
        const auto ref_tensor = ref_req.get_tensor(ref_model.output(i));
        const auto gfx_tensor = gfx_req.get_tensor(gfx_model.output(i));
        const auto stats = compare_tensors(ref_tensor, gfx_tensor);
        const auto& desc = output_descs[i];
        std::cout << "[op " << desc.op_index << "] " << desc.name << " (" << desc.type << ")"
                  << " max_abs_diff=" << std::setprecision(10) << stats.max_abs_diff
                  << " max_rel_diff=" << stats.max_rel_diff << '\n';
    }
    std::cout << "PER_OP_MATCH\n";
    return 0;
}

int run_per_op_compare(ov::Core& core,
                       const std::shared_ptr<ov::Model>& model,
                       const std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>& inputs,
                       const CompareOptions& options) {
    const auto ordered_ops = model->get_ordered_ops();
    const auto ref_device = reference_device(core, options.reference_device);
    std::vector<size_t> relevant_indices;
    relevant_indices.reserve(ordered_ops.size());
    std::unordered_map<const ov::Node*, size_t> relevant_pos;
    for (size_t idx = 0; idx < ordered_ops.size(); ++idx) {
        const auto& node = ordered_ops[idx];
        if (is_debug_skippable_node(node)) {
            continue;
        }
        relevant_pos.emplace(node.get(), relevant_indices.size());
        relevant_indices.push_back(idx);
    }
    size_t checked = 0;
    std::unordered_map<OutputKey, ov::Tensor, OutputKeyHash> cached_reference_outputs;
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

        const auto pos_it = relevant_pos.find(node.get());
        OPENVINO_ASSERT(pos_it != relevant_pos.end(), "per-op debug: node position missing");
        const size_t pos = pos_it->second;
        const size_t window_begin_pos = (options.window_size > 0 && pos + 1 > options.window_size)
                                            ? (pos + 1 - options.window_size)
                                            : 0;

        ov::ParameterVector isolated_params;
        std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>> isolated_input_tensors;
        std::unordered_map<OutputKey, ov::Output<ov::Node>, OutputKeyHash> cloned_outputs;
        std::unordered_map<OutputKey, ov::Output<ov::Node>, OutputKeyHash> external_values;
        std::shared_ptr<ov::Node> cloned_target;

        auto materialize_external = [&](const ov::Output<ov::Node>& source) -> std::optional<ov::Output<ov::Node>> {
            OutputKey key{source.get_node(), source.get_index()};
            if (auto it = external_values.find(key); it != external_values.end()) {
                return it->second;
            }
            if (auto const_tensor = evaluate_constant_source_tensor(source)) {
                auto constant = std::make_shared<ov::op::v0::Constant>(const_tensor->get_element_type(),
                                                                       const_tensor->get_shape(),
                                                                       const_tensor->data());
                constant->set_friendly_name((source.get_node_shared_ptr() ? source.get_node_shared_ptr()->get_friendly_name()
                                                                          : std::string("const")) +
                                            "/isolated_const_" + std::to_string(source.get_index()));
                auto out = constant->output(0);
                external_values.emplace(key, out);
                return out;
            }
            auto cached_it = cached_reference_outputs.find(key);
            std::optional<ov::Tensor> tensor;
            if (cached_it != cached_reference_outputs.end()) {
                tensor = cached_it->second;
            } else {
                tensor = evaluate_source_tensor(source, inputs);
            }
            if (!tensor.has_value()) {
                tensor = evaluate_source_tensor_with_reference(core, ref_device, source, inputs);
            }
            if (!tensor.has_value()) {
                return std::nullopt;
            }
            cached_reference_outputs.insert_or_assign(key, *tensor);
            auto param = std::make_shared<ov::op::v0::Parameter>(tensor->get_element_type(), tensor->get_shape());
            param->set_friendly_name((source.get_node_shared_ptr() ? source.get_node_shared_ptr()->get_friendly_name()
                                                                   : std::string("param")) +
                                     "/isolated_input_" + std::to_string(source.get_index()));
            isolated_params.push_back(param);
            isolated_input_tensors.emplace_back(param->output(0), *tensor);
            auto out = param->output(0);
            external_values.emplace(key, out);
            return out;
        };

        bool missing_input = false;
        for (size_t p = window_begin_pos; p <= pos; ++p) {
            const auto& stage_node = ordered_ops[relevant_indices[p]];
            ov::OutputVector cloned_inputs;
            cloned_inputs.reserve(stage_node->get_input_size());
            for (const auto& source : stage_node->input_values()) {
                OutputKey key{source.get_node(), source.get_index()};
                if (auto it = cloned_outputs.find(key); it != cloned_outputs.end()) {
                    cloned_inputs.push_back(it->second);
                    continue;
                }
                auto ext = materialize_external(source);
                if (!ext.has_value()) {
                    std::cout << "[op " << idx << "] " << node->get_friendly_name() << " (" << node->get_type_name()
                              << ") infer_skip=failed to materialize isolated input from "
                              << stage_node->get_friendly_name() << '\n';
                    missing_input = true;
                    break;
                }
                cloned_inputs.push_back(*ext);
            }
            if (missing_input) {
                break;
            }
            auto cloned = stage_node->clone_with_new_inputs(cloned_inputs);
            for (size_t out_idx = 0; out_idx < cloned->get_output_size(); ++out_idx) {
                cloned_outputs[{stage_node.get(), out_idx}] = cloned->output(out_idx);
            }
            cloned_target = cloned;
        }
        if (missing_input) {
            ++checked;
            continue;
        }
        OPENVINO_ASSERT(cloned_target, "per-op debug: isolated target is null");
        ov::OutputVector outputs;
        outputs.reserve(cloned_target->get_output_size());
        for (const auto& output : cloned_target->outputs()) {
            outputs.push_back(std::make_shared<ov::op::v0::Result>(output));
        }
        auto submodel = std::make_shared<ov::Model>(outputs, isolated_params, node->get_friendly_name());

        ov::CompiledModel ref_submodel;
        ov::CompiledModel gfx_submodel;
        try {
            ref_submodel = core.compile_model(submodel, ref_device, make_compile_config(false));
            gfx_submodel = core.compile_model(submodel, "GFX", make_compile_config(true));
        } catch (const std::exception& ex) {
            std::cout << "[op " << idx << "] " << node->get_friendly_name() << " (" << node->get_type_name()
                      << ") compile_skip=" << ex.what() << '\n';
            ++checked;
            continue;
        }

        DiffStats stats;
        std::vector<ov::Tensor> ref_outputs;
        try {
            stats = compare_model_outputs(ref_submodel, gfx_submodel, isolated_input_tensors, false, &ref_outputs);
        } catch (const std::exception& ex) {
            std::cout << "[op " << idx << "] " << node->get_friendly_name() << " (" << node->get_type_name()
                      << ") infer_skip=" << ex.what() << '\n';
            ++checked;
            continue;
        }
        for (size_t out_idx = 0; out_idx < ref_outputs.size(); ++out_idx) {
            cached_reference_outputs.insert_or_assign({node.get(), out_idx}, ref_outputs[out_idx]);
        }

        std::cout << "[op " << idx << "] " << node->get_friendly_name() << " (" << node->get_type_name()
                  << ") max_abs_diff=" << std::setprecision(10) << stats.max_abs_diff
                  << " max_rel_diff=" << stats.max_rel_diff;
        std::cout << '\n';
        if (!options.per_op_all && stats.max_abs_diff > options.abs_threshold && stats.max_rel_diff > options.rel_threshold) {
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
        std::cerr << "usage: ov_gfx_compare_runner <model.xml> [--per-op] [--per-op-all] [--print-ops] [--gfx-only] "
                     "[--reference-device NAME] [--reference-plugin PATH] "
                     "[--start-op N] "
                     "[--window-size N] "
                     "[--stop-after-op N] [--abs-threshold V] [--rel-threshold V]\n";
        return 2;
    }

    const std::string model_path = argv[1];
    CompareOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--per-op") {
            options.per_op = true;
            continue;
        }
        if (arg == "--per-op-all") {
            options.per_op = true;
            options.per_op_all = true;
            continue;
        }
        if (arg == "--print-ops") {
            options.print_ops = true;
            continue;
        }
        if (arg == "--gfx-only") {
            options.gfx_only = true;
            continue;
        }
        if (arg == "--reference-device") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--reference-device requires a value");
            }
            options.reference_device = argv[++i];
            continue;
        }
        if (arg == "--reference-plugin") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--reference-plugin requires a value");
            }
            options.reference_plugin_path = argv[++i];
            continue;
        }
        if (arg == "--start-op") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--start-op requires a value");
            }
            options.start_op = static_cast<size_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg == "--window-size") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--window-size requires a value");
            }
            options.window_size = std::max<size_t>(1, static_cast<size_t>(std::stoul(argv[++i])));
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
        throw std::runtime_error("unknown argument: " + arg);
    }

    ov::Core core;
    register_gfx_plugin(core);
    if (options.reference_device == "CPU") {
        throw std::runtime_error("CPU reference device is not supported; use TEMPLATE");
    }
    register_reference_plugin(core, options.reference_device, options.reference_plugin_path);

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

    if (options.per_op_all) {
        return run_full_graph_per_op_compare(core, model, options);
    }

    if (options.per_op) {
        return run_per_op_compare(core, model, inputs, options);
    }

    if (options.gfx_only) {
        auto gfx_model = core.compile_model(model, "GFX", make_compile_config(true));
        auto gfx_req = make_request(gfx_model, inputs);
        gfx_req.infer();
        maybe_print_gfx_profile(gfx_model);

        std::cout << "GFX_ONLY\n";
        for (const auto& output : model->outputs()) {
            const auto tensor = gfx_req.get_tensor(output.get_any_name());
            const auto summary = summarize_tensor(tensor);
            std::cout << output.get_any_name()
                      << " elements=" << summary.elements
                      << " finite=" << summary.finite_count
                      << " nan=" << summary.nan_count
                      << " inf=" << summary.inf_count
                      << " min=" << std::setprecision(10) << summary.min
                      << " max=" << summary.max
                      << " mean=" << summary.mean
                      << " l2=" << summary.l2
                      << '\n';
        }
        return 0;
    }

    const auto ref_device = reference_device(core, options.reference_device);
    std::cout << "REFERENCE device=" << ref_device << '\n';
    auto ref_model = core.compile_model(model, ref_device, make_compile_config(false));
    auto gfx_model = core.compile_model(model, "GFX", make_compile_config(true));
    auto ref_req = make_request(ref_model, inputs);
    auto gfx_req = make_request(gfx_model, inputs);

    ref_req.infer();
    gfx_req.infer();
    maybe_print_gfx_profile(gfx_model);

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

    std::cout << "GLOBAL max_abs_diff=" << std::setprecision(10) << global_max_abs
              << " max_rel_diff=" << global_max_rel << '\n';
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "ERROR: " << ex.what() << '\n';
    return 1;
}
