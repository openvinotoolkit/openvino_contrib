#include <openvino/openvino.hpp>
#include <openvino/opsets/opset8.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "ov_gfx_microbench_common.hpp"
#include "ov_gfx_microbench_calibration.hpp"

#if defined(__APPLE__)
#    include "backends/metal/runtime/metal_memory.hpp"
#else
#    include "backends/vulkan/runtime/vulkan_backend.hpp"
#    include <vulkan/vulkan.h>
#endif

namespace {

struct DeviceFingerprint {
    std::string backend;
    std::string device_name;
    std::string full_name;
    std::string platform;
    std::string vendor_id;
    std::string device_id;
    std::string driver_version;
    std::string architecture;
};

struct BenchResult {
    std::string name;
    std::string kind;
    std::string model_desc;
    std::string actual_backend;
    double compile_ms = 0.0;
    double first_infer_ms = 0.0;
    double median_infer_ms = 0.0;
    double min_infer_ms = 0.0;
    double max_infer_ms = 0.0;
    std::string profile_json;
};

struct WorkloadEstimate {
    uint64_t bytes_in = 0;
    uint64_t bytes_out = 0;
    uint64_t bytes_moved = 0;
    uint64_t macs_est = 0;
    uint64_t flops_est = 0;
    double arithmetic_intensity = 0.0;
    std::string note;
};

struct ProfileDigest {
    bool has_extended = false;
    bool counters_supported = false;
    bool counters_used = false;
    uint64_t total_gpu_us = 0;
    uint64_t total_cpu_us = 0;
    uint64_t total_wall_us = 0;
    uint64_t total_h2d_bytes = 0;
    uint64_t total_d2h_bytes = 0;
    uint64_t wait_cpu_us = 0;
    uint64_t submit_cpu_us = 0;
    uint64_t barrier_cpu_us = 0;
    uint64_t upload_cpu_us = 0;
    uint64_t download_cpu_us = 0;
    uint64_t final_fence_wait_cpu_us = 0;
    uint64_t submit_count = 0;
    uint64_t barrier_count = 0;
    uint64_t descriptor_update_count = 0;
    uint64_t pipeline_creation_count = 0;
    bool sync_heavy = false;
    bool transfer_heavy = false;
    bool compile_in_infer = false;
    bool binding_prepare_in_infer = false;
    bool final_fence_wait_seen = false;
    bool cross_submit_barrier_seen = false;
};

struct BenchDerived {
    WorkloadEstimate workload;
    ProfileDigest profile;
    double fixed_overhead_us = 0.0;
    double fixed_overhead_share = 0.0;
    double overhead_subtracted_ms = 0.0;
    double e2e_tflops = 0.0;
    double e2e_gbps = 0.0;
    double adjusted_tflops = 0.0;
    double adjusted_gbps = 0.0;
    double gpu_tflops = 0.0;
    double gpu_gbps = 0.0;
    double gpu_share_of_wall = 0.0;
    double wait_share_of_wall = 0.0;
    double transfer_share_of_wall = 0.0;
    double first_to_steady_ratio = 0.0;
    std::vector<std::string> hints;
};

struct Options {
    std::string backend = "auto";
    size_t warmup = 3;
    size_t iterations = 10;
    std::string output_path;
    std::string calibration_output_path;
    std::string calibration_input_path;
};

struct LoadedCalibrationSummary {
    bool provided = false;
    std::string path;
    ov::gfx_plugin::microbench::CalibrationArtifact artifact;
    bool device_key_match = false;
    bool backend_match = false;
    bool schema_match = false;
};

std::string hex_u32(uint32_t value) {
    std::ostringstream oss;
    oss << "0x" << std::hex << std::uppercase << value;
    return oss.str();
}

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

double median_of(std::vector<double> values) {
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

std::pair<double, double> minmax_of(const std::vector<double>& values) {
    if (values.empty()) {
        return {0.0, 0.0};
    }
    const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    return {*min_it, *max_it};
}

double safe_div(double num, double denom) {
    return denom > 0.0 ? (num / denom) : 0.0;
}

size_t shape_elements(const ov::Shape& shape) {
    size_t total = 1;
    for (const size_t dim : shape) {
        total *= dim;
    }
    return total;
}

template <typename T>
void fill_tensor_data(ov::Tensor& tensor) {
    T* data = tensor.data<T>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
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
        default:
            throw std::runtime_error("unsupported tensor type: " + tensor.get_element_type().to_string());
    }
}

std::vector<ov::float16> make_fp16_constant_data(size_t count, float scale = 1.0f) {
    std::vector<ov::float16> values(count);
    for (size_t i = 0; i < count; ++i) {
        const float value = scale * static_cast<float>(((static_cast<int64_t>(i) % 37) - 18) / 19.0f);
        values[i] = ov::float16(value);
    }
    return values;
}

WorkloadEstimate estimate_workload(std::string_view bench_name) {
    constexpr uint64_t kFp16Bytes = 2;
    WorkloadEstimate estimate;
    if (bench_name == "MB1") {
        const uint64_t elems = 1ull * 1ull * 1024ull * 1024ull;
        estimate.bytes_in = elems * kFp16Bytes;
        estimate.bytes_out = elems * kFp16Bytes;
        estimate.bytes_moved = estimate.bytes_in + estimate.bytes_out;
        estimate.flops_est = elems;
        estimate.note = "ReLU is treated as one elementwise FLOP per output element for quick triage.";
    } else if (bench_name == "MB2") {
        const uint64_t elems = 1ull * 4ull * 1024ull * 1024ull;
        estimate.bytes_in = elems * kFp16Bytes * 2ull;
        estimate.bytes_out = elems * kFp16Bytes;
        estimate.bytes_moved = estimate.bytes_in + estimate.bytes_out;
        estimate.flops_est = elems;
        estimate.note = "Add reads one FP16 input tensor and one FP16 constant tensor, then writes one FP16 output tensor.";
    } else if (bench_name == "MB3") {
        const uint64_t m = 1024;
        const uint64_t n = 1024;
        const uint64_t k = 1024;
        estimate.bytes_in = ((m * k) + (k * n)) * kFp16Bytes;
        estimate.bytes_out = (m * n) * kFp16Bytes;
        estimate.bytes_moved = estimate.bytes_in + estimate.bytes_out;
        estimate.macs_est = m * n * k;
        estimate.flops_est = estimate.macs_est * 2ull;
        estimate.note = "MatMul FLOP estimate assumes 2 FLOPs per MAC and excludes cache-reuse effects.";
    }
    estimate.arithmetic_intensity = safe_div(static_cast<double>(estimate.flops_est), static_cast<double>(estimate.bytes_moved));
    return estimate;
}

std::shared_ptr<ov::Model> make_mb1_model() {
    using namespace ov::opset8;
    const ov::Shape shape{1, 1, 1024, 1024};
    auto input = std::make_shared<Parameter>(ov::element::f16, shape);
    auto relu = std::make_shared<Relu>(input);
    auto result = std::make_shared<Result>(relu);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "gfx_mb1_copy_dispatch");
}

std::shared_ptr<ov::Model> make_mb2_model() {
    using namespace ov::opset8;
    const ov::Shape shape{1, 4, 1024, 1024};
    auto input = std::make_shared<Parameter>(ov::element::f16, shape);
    auto weights = Constant::create(ov::element::f16, shape, make_fp16_constant_data(shape_elements(shape), 0.25f));
    auto add = std::make_shared<Add>(input, weights);
    auto result = std::make_shared<Result>(add);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "gfx_mb2_bandwidth");
}

std::shared_ptr<ov::Model> make_mb3_model() {
    using namespace ov::opset8;
    const ov::Shape lhs_shape{1, 1024, 1024};
    const ov::Shape rhs_shape{1, 1024, 1024};
    auto input = std::make_shared<Parameter>(ov::element::f16, lhs_shape);
    auto weights =
        Constant::create(ov::element::f16, rhs_shape, make_fp16_constant_data(shape_elements(rhs_shape), 0.125f));
    auto matmul = std::make_shared<MatMul>(input, weights, false, false);
    auto result = std::make_shared<Result>(matmul);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "gfx_mb3_gemm");
}

ov::AnyMap make_compile_config(const std::string& backend) {
    ov::AnyMap config;
    config[ov::hint::inference_precision.name()] = ov::element::f16;
    config[ov::enable_profiling.name()] = true;
    config["GFX_PROFILING_LEVEL"] = 2;
    if (backend != "auto" && !backend.empty()) {
        config["GFX_BACKEND"] = backend;
    }
    return config;
}

ov::InferRequest make_request(ov::CompiledModel& compiled_model) {
    ov::InferRequest request = compiled_model.create_infer_request();
    for (const auto& input : compiled_model.inputs()) {
        ov::Tensor tensor(input.get_element_type(), input.get_shape());
        fill_tensor(tensor);
        request.set_tensor(input, tensor);
    }
    return request;
}

double infer_once_ms(ov::InferRequest& request) {
    const auto start = std::chrono::steady_clock::now();
    request.infer();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

BenchResult run_model_bench(ov::Core& core,
                            const Options& options,
                            std::string name,
                            std::string kind,
                            std::string model_desc,
                            const std::shared_ptr<ov::Model>& model) {
    BenchResult result;
    result.name = std::move(name);
    result.kind = std::move(kind);
    result.model_desc = std::move(model_desc);

    const auto compile_start = std::chrono::steady_clock::now();
    ov::CompiledModel compiled = core.compile_model(model, "GFX", make_compile_config(options.backend));
    const auto compile_stop = std::chrono::steady_clock::now();
    result.compile_ms = std::chrono::duration<double, std::milli>(compile_stop - compile_start).count();
    result.actual_backend = compiled.get_property("GFX_BACKEND").as<std::string>();

    ov::InferRequest request = make_request(compiled);
    result.first_infer_ms = infer_once_ms(request);

    for (size_t i = 0; i < options.warmup; ++i) {
        (void)infer_once_ms(request);
    }

    std::vector<double> times_ms;
    times_ms.reserve(options.iterations);
    for (size_t i = 0; i < options.iterations; ++i) {
        times_ms.push_back(infer_once_ms(request));
    }
    result.median_infer_ms = median_of(times_ms);
    const auto [min_ms, max_ms] = minmax_of(times_ms);
    result.min_infer_ms = min_ms;
    result.max_infer_ms = max_ms;
    result.profile_json = compiled.get_property("GFX_PROFILING_REPORT").as<std::string>();
    return result;
}

Mb0Result run_mb0(const std::string& backend, size_t warmup, size_t iterations) {
#if defined(__APPLE__)
    if (backend == "vulkan") {
        throw std::runtime_error("MB0 Vulkan is not available in the macOS tool build");
    }
    return run_metal_mb0(warmup, iterations);
#else
    if (backend == "metal") {
        throw std::runtime_error("MB0 Metal is not available in the Vulkan tool build");
    }
    return run_vulkan_mb0(warmup, iterations);
#endif
}

std::string escape_json(std::string_view value) {
    std::ostringstream oss;
    for (const char ch : value) {
        switch (ch) {
            case '\\':
                oss << "\\\\";
                break;
            case '"':
                oss << "\\\"";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                oss << ch;
                break;
        }
    }
    return oss.str();
}

std::string_view extract_json_object_field(std::string_view json, std::string_view field_name) {
    const std::string needle = "\"" + std::string(field_name) + "\":{";
    const size_t key_pos = json.find(needle);
    if (key_pos == std::string_view::npos) {
        return {};
    }
    const size_t object_start = key_pos + needle.size() - 1;
    size_t depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t i = object_start; i < json.size(); ++i) {
        const char c = json[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '{') {
            depth += 1;
            continue;
        }
        if (c == '}') {
            depth -= 1;
            if (depth == 0) {
                return json.substr(object_start, i - object_start + 1);
            }
        }
    }
    return {};
}

std::string_view extract_json_array_field(std::string_view json, std::string_view field_name) {
    const std::string needle = "\"" + std::string(field_name) + "\":[";
    const size_t key_pos = json.find(needle);
    if (key_pos == std::string_view::npos) {
        return {};
    }
    const size_t array_start = key_pos + needle.size() - 1;
    size_t depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t i = array_start; i < json.size(); ++i) {
        const char c = json[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '[') {
            depth += 1;
            continue;
        }
        if (c == ']') {
            depth -= 1;
            if (depth == 0) {
                return json.substr(array_start, i - array_start + 1);
            }
        }
    }
    return {};
}

std::vector<std::string_view> split_top_level_json_array_objects(std::string_view array_json) {
    std::vector<std::string_view> objects;
    if (array_json.size() < 2 || array_json.front() != '[' || array_json.back() != ']') {
        return objects;
    }
    bool in_string = false;
    bool escaped = false;
    size_t object_start = std::string_view::npos;
    size_t depth = 0;
    for (size_t i = 1; i + 1 < array_json.size(); ++i) {
        const char c = array_json[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '{') {
            if (depth == 0) {
                object_start = i;
            }
            depth += 1;
            continue;
        }
        if (c == '}') {
            if (depth == 0) {
                continue;
            }
            depth -= 1;
            if (depth == 0 && object_start != std::string_view::npos) {
                objects.push_back(array_json.substr(object_start, i - object_start + 1));
                object_start = std::string_view::npos;
            }
        }
    }
    return objects;
}

uint64_t extract_json_uint_field(std::string_view object_json, std::string_view field_name, uint64_t fallback = 0) {
    const std::string needle = "\"" + std::string(field_name) + "\":";
    const size_t pos = object_json.find(needle);
    if (pos == std::string_view::npos) {
        return fallback;
    }
    size_t begin = pos + needle.size();
    while (begin < object_json.size() && std::isspace(static_cast<unsigned char>(object_json[begin]))) {
        ++begin;
    }
    size_t end = begin;
    while (end < object_json.size() && std::isdigit(static_cast<unsigned char>(object_json[end]))) {
        ++end;
    }
    if (end == begin) {
        return fallback;
    }
    return static_cast<uint64_t>(std::strtoull(std::string(object_json.substr(begin, end - begin)).c_str(), nullptr, 10));
}

bool extract_json_bool_field(std::string_view object_json, std::string_view field_name, bool fallback = false) {
    const std::string needle = "\"" + std::string(field_name) + "\":";
    const size_t pos = object_json.find(needle);
    if (pos == std::string_view::npos) {
        return fallback;
    }
    size_t begin = pos + needle.size();
    while (begin < object_json.size() && std::isspace(static_cast<unsigned char>(object_json[begin]))) {
        ++begin;
    }
    if (object_json.substr(begin, 4) == "true") {
        return true;
    }
    if (object_json.substr(begin, 5) == "false") {
        return false;
    }
    return fallback;
}

std::string extract_json_string_field(std::string_view object_json, std::string_view field_name) {
    const std::string needle = "\"" + std::string(field_name) + "\":\"";
    const size_t pos = object_json.find(needle);
    if (pos == std::string_view::npos) {
        return {};
    }
    size_t begin = pos + needle.size();
    std::string out;
    bool escaped = false;
    for (size_t i = begin; i < object_json.size(); ++i) {
        const char c = object_json[i];
        if (escaped) {
            out.push_back(c);
            escaped = false;
            continue;
        }
        if (c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            return out;
        }
        out.push_back(c);
    }
    return {};
}

uint64_t extract_json_object_uint_value(std::string_view object_json, std::string_view field_name, uint64_t fallback = 0) {
    return extract_json_uint_field(object_json, field_name, fallback);
}

ProfileDigest digest_profile_json(std::string_view profile_json) {
    ProfileDigest digest;
    const auto extended = extract_json_object_field(profile_json, "extended");
    if (extended.empty()) {
        return digest;
    }
    digest.has_extended = true;
    digest.counters_supported = extract_json_bool_field(extended, "counters_supported");
    digest.counters_used = extract_json_bool_field(extended, "counters_used");
    digest.total_gpu_us = extract_json_uint_field(extended, "total_gpu_us");
    digest.total_cpu_us = extract_json_uint_field(extended, "total_cpu_us");
    digest.total_wall_us = extract_json_uint_field(extended, "total_wall_us");
    digest.total_h2d_bytes = extract_json_uint_field(extended, "total_h2d_bytes");
    digest.total_d2h_bytes = extract_json_uint_field(extended, "total_d2h_bytes");

    const auto summary = extract_json_object_field(extended, "summary");
    if (summary.empty()) {
        return digest;
    }

    const auto phase_totals = extract_json_array_field(summary, "phase_totals");
    for (const auto& phase_object : split_top_level_json_array_objects(phase_totals)) {
        const auto phase = extract_json_string_field(phase_object, "phase");
        const uint64_t cpu_us = extract_json_uint_field(phase_object, "cpu_us");
        if (phase == "wait") {
            digest.wait_cpu_us = cpu_us;
        } else if (phase == "submit") {
            digest.submit_cpu_us = cpu_us;
        } else if (phase == "barrier") {
            digest.barrier_cpu_us = cpu_us;
        } else if (phase == "upload") {
            digest.upload_cpu_us = cpu_us;
        } else if (phase == "download") {
            digest.download_cpu_us = cpu_us;
        }
    }

    const auto counter_map = extract_json_object_field(summary, "counter_map");
    if (!counter_map.empty()) {
        digest.submit_count = std::max(extract_json_object_uint_value(counter_map, "submit_count"),
                                       extract_json_object_uint_value(counter_map, "vkQueueSubmit_count"));
        digest.barrier_count = extract_json_object_uint_value(counter_map, "barrier_count") +
                               extract_json_object_uint_value(counter_map, "cross_submit_barrier_count");
        digest.descriptor_update_count = extract_json_object_uint_value(counter_map, "descriptor_update_count");
        digest.pipeline_creation_count = extract_json_object_uint_value(counter_map, "pipeline_creation_count");
    }

    const auto segments = extract_json_array_field(extended, "segments");
    for (const auto& segment_object : split_top_level_json_array_objects(segments)) {
        const auto name = extract_json_string_field(segment_object, "name");
        if (name == "final_fence_wait") {
            digest.final_fence_wait_seen = true;
            digest.final_fence_wait_cpu_us += extract_json_uint_field(segment_object, "cpu_us");
        } else if (name == "cross_submit_memory_barrier") {
            digest.cross_submit_barrier_seen = true;
        }
    }

    const auto diagnostics = extract_json_array_field(summary, "diagnostics");
    if (diagnostics.find("sync-heavy") != std::string_view::npos) {
        digest.sync_heavy = true;
    }
    if (diagnostics.find("Upload/download CPU overhead") != std::string_view::npos) {
        digest.transfer_heavy = true;
    }
    if (diagnostics.find("Compilation/setup work was recorded inside the profiled infer path.") != std::string_view::npos) {
        digest.compile_in_infer = true;
    }
    if (diagnostics.find("binding preparation ran inside infer") != std::string_view::npos) {
        digest.binding_prepare_in_infer = true;
    }
    return digest;
}

BenchDerived analyze_benchmark(const BenchResult& result, const Mb0Result& mb0) {
    BenchDerived analysis;
    analysis.workload = estimate_workload(result.name);
    analysis.profile = digest_profile_json(result.profile_json);
    analysis.fixed_overhead_us = mb0.median_wall_us;
    analysis.fixed_overhead_share = safe_div(mb0.median_wall_us / 1000.0, result.median_infer_ms);
    analysis.overhead_subtracted_ms = std::max(result.median_infer_ms - (mb0.median_wall_us / 1000.0), 0.0);
    analysis.e2e_tflops = safe_div(static_cast<double>(analysis.workload.flops_est), result.median_infer_ms * 1.0e9);
    analysis.e2e_gbps = safe_div(static_cast<double>(analysis.workload.bytes_moved), result.median_infer_ms * 1.0e6);
    analysis.adjusted_tflops =
        safe_div(static_cast<double>(analysis.workload.flops_est), analysis.overhead_subtracted_ms * 1.0e9);
    analysis.adjusted_gbps =
        safe_div(static_cast<double>(analysis.workload.bytes_moved), analysis.overhead_subtracted_ms * 1.0e6);
    analysis.gpu_tflops = safe_div(static_cast<double>(analysis.workload.flops_est), static_cast<double>(analysis.profile.total_gpu_us) * 1.0e6);
    analysis.gpu_gbps = safe_div(static_cast<double>(analysis.workload.bytes_moved), static_cast<double>(analysis.profile.total_gpu_us) * 1.0e3);
    analysis.gpu_share_of_wall =
        safe_div(static_cast<double>(analysis.profile.total_gpu_us), static_cast<double>(analysis.profile.total_wall_us));
    analysis.wait_share_of_wall =
        safe_div(static_cast<double>(analysis.profile.wait_cpu_us), static_cast<double>(analysis.profile.total_wall_us));
    analysis.transfer_share_of_wall =
        safe_div(static_cast<double>(analysis.profile.upload_cpu_us + analysis.profile.download_cpu_us),
                 static_cast<double>(analysis.profile.total_wall_us));
    analysis.first_to_steady_ratio = safe_div(result.first_infer_ms, result.median_infer_ms);

    if (analysis.fixed_overhead_share >= 0.25) {
        analysis.hints.push_back("fixed_overhead_dominates_small_workload");
    }
    if (analysis.profile.sync_heavy || analysis.wait_share_of_wall >= 0.20 || analysis.profile.final_fence_wait_seen) {
        analysis.hints.push_back("sync_heavy");
    }
    if (analysis.profile.transfer_heavy || analysis.transfer_share_of_wall >= 0.10) {
        analysis.hints.push_back("transfer_heavy");
    }
    if (analysis.profile.cross_submit_barrier_seen || analysis.profile.barrier_count > 0) {
        analysis.hints.push_back("barrier_activity_visible");
    }
    if (analysis.profile.submit_count > 3) {
        analysis.hints.push_back("multi_submit_steady_state");
    }
    if (analysis.first_to_steady_ratio >= 2.0 || analysis.profile.pipeline_creation_count > 0 || analysis.profile.compile_in_infer) {
        analysis.hints.push_back("first_run_or_lazy_compile_visible");
    }
    if (analysis.profile.binding_prepare_in_infer || analysis.profile.descriptor_update_count > 0) {
        analysis.hints.push_back("binding_or_descriptor_churn");
    }
    if (analysis.workload.arithmetic_intensity < 4.0) {
        analysis.hints.push_back("memory_pressure_candidate");
    } else if (analysis.workload.arithmetic_intensity >= 16.0) {
        analysis.hints.push_back("compute_pressure_candidate");
    }
    return analysis;
}

std::string device_key_for_autotune(const DeviceFingerprint& device) {
    return device.vendor_id + ":" + device.device_id + ":" + device.driver_version;
}

std::vector<std::string> top_level_triage_hints(const Mb0Result& mb0,
                                                const BenchDerived& mb1,
                                                const BenchDerived& mb2,
                                                const BenchDerived& mb3) {
    std::vector<std::string> triage_hints;
    if (mb0.median_wall_us >= 200.0) {
        triage_hints.push_back("mb0_high_fixed_overhead");
    }
    if (mb1.fixed_overhead_share >= 0.25 || mb2.fixed_overhead_share >= 0.25) {
        triage_hints.push_back("small_workloads_likely_cpu_competitive");
    }
    if (mb3.wait_share_of_wall >= 0.20 || mb3.profile.final_fence_wait_seen) {
        triage_hints.push_back("mb3_sync_bound");
    }
    if (mb2.transfer_share_of_wall >= 0.10 || mb2.profile.transfer_heavy) {
        triage_hints.push_back("mb2_transfer_heavy");
    }
    if (mb2.profile.barrier_count > 0 || mb3.profile.cross_submit_barrier_seen) {
        triage_hints.push_back("cross_submit_barrier_activity");
    }
    if (mb3.first_to_steady_ratio >= 2.0 || mb3.profile.pipeline_creation_count > 0) {
        triage_hints.push_back("lazy_compile_or_pipeline_creation_visible");
    }
    return triage_hints;
}

std::vector<std::string> calibration_assumptions() {
    return {
        "fixed_overhead_us is taken from MB0 median wall time with explicit sync.",
        "bandwidth_estimate_gbps uses MB2 synthetic bytes_moved and subtracts MB0 median overhead.",
        "compute_estimate_tflops uses MB3 synthetic FLOP estimate and subtracts MB0 median overhead.",
        "gpu_* estimates require extended.total_gpu_us and remain observational rather than hardware-peak-calibrated."
    };
}

ov::gfx_plugin::microbench::CalibrationProbe make_calibration_probe(const BenchResult& result,
                                                                    const BenchDerived& derived) {
    ov::gfx_plugin::microbench::CalibrationProbe probe;
    probe.name = result.name;
    probe.actual_backend = result.actual_backend;
    probe.arithmetic_intensity = derived.workload.arithmetic_intensity;
    probe.overhead_subtracted_ms = derived.overhead_subtracted_ms;
    probe.adjusted_gbps = derived.adjusted_gbps;
    probe.adjusted_tflops = derived.adjusted_tflops;
    probe.gpu_gbps = derived.gpu_gbps;
    probe.gpu_tflops = derived.gpu_tflops;
    probe.first_to_steady_ratio = derived.first_to_steady_ratio;
    probe.wait_share_of_wall = derived.wait_share_of_wall;
    probe.transfer_share_of_wall = derived.transfer_share_of_wall;
    probe.submit_count = derived.profile.submit_count;
    probe.barrier_count = derived.profile.barrier_count;
    probe.hints = derived.hints;
    return probe;
}

ov::gfx_plugin::microbench::CalibrationArtifact make_calibration_artifact(const DeviceFingerprint& device,
                                                                          const Mb0Result& mb0,
                                                                          const BenchResult& mb1,
                                                                          const BenchDerived& mb1_analysis,
                                                                          const BenchResult& mb2,
                                                                          const BenchDerived& mb2_analysis,
                                                                          const BenchResult& mb3,
                                                                          const BenchDerived& mb3_analysis) {
    ov::gfx_plugin::microbench::CalibrationArtifact artifact;
    artifact.device_key = device_key_for_autotune(device);
    artifact.backend = device.backend;
    artifact.device_name = device.device_name;
    artifact.platform = device.platform;
    artifact.vendor_id = device.vendor_id;
    artifact.device_id = device.device_id;
    artifact.driver_version = device.driver_version;
    artifact.fixed_overhead_us = mb0.median_wall_us;
    artifact.bandwidth_estimate_gbps = mb2_analysis.adjusted_gbps;
    artifact.compute_estimate_tflops = mb3_analysis.adjusted_tflops;
    artifact.gpu_bandwidth_estimate_gbps = mb2_analysis.gpu_gbps;
    artifact.gpu_compute_estimate_tflops = mb3_analysis.gpu_tflops;
    artifact.triage_hints = top_level_triage_hints(mb0, mb1_analysis, mb2_analysis, mb3_analysis);
    artifact.assumptions = calibration_assumptions();
    artifact.probes.push_back(make_calibration_probe(mb1, mb1_analysis));
    artifact.probes.push_back(make_calibration_probe(mb2, mb2_analysis));
    artifact.probes.push_back(make_calibration_probe(mb3, mb3_analysis));
    return artifact;
}

void append_workload_json(std::ostringstream& json, const WorkloadEstimate& workload) {
    json << "{";
    json << "\"bytes_in\":" << workload.bytes_in << ",";
    json << "\"bytes_out\":" << workload.bytes_out << ",";
    json << "\"bytes_moved\":" << workload.bytes_moved << ",";
    json << "\"macs_est\":" << workload.macs_est << ",";
    json << "\"flops_est\":" << workload.flops_est << ",";
    json << "\"arithmetic_intensity\":" << workload.arithmetic_intensity << ",";
    json << "\"note\":\"" << escape_json(workload.note) << "\"";
    json << "}";
}

void append_profile_digest_json(std::ostringstream& json, const ProfileDigest& profile) {
    json << "{";
    json << "\"has_extended\":" << (profile.has_extended ? "true" : "false") << ",";
    json << "\"counters_supported\":" << (profile.counters_supported ? "true" : "false") << ",";
    json << "\"counters_used\":" << (profile.counters_used ? "true" : "false") << ",";
    json << "\"total_gpu_us\":" << profile.total_gpu_us << ",";
    json << "\"total_cpu_us\":" << profile.total_cpu_us << ",";
    json << "\"total_wall_us\":" << profile.total_wall_us << ",";
    json << "\"total_h2d_bytes\":" << profile.total_h2d_bytes << ",";
    json << "\"total_d2h_bytes\":" << profile.total_d2h_bytes << ",";
    json << "\"wait_cpu_us\":" << profile.wait_cpu_us << ",";
    json << "\"submit_cpu_us\":" << profile.submit_cpu_us << ",";
    json << "\"barrier_cpu_us\":" << profile.barrier_cpu_us << ",";
    json << "\"upload_cpu_us\":" << profile.upload_cpu_us << ",";
    json << "\"download_cpu_us\":" << profile.download_cpu_us << ",";
    json << "\"final_fence_wait_cpu_us\":" << profile.final_fence_wait_cpu_us << ",";
    json << "\"submit_count\":" << profile.submit_count << ",";
    json << "\"barrier_count\":" << profile.barrier_count << ",";
    json << "\"descriptor_update_count\":" << profile.descriptor_update_count << ",";
    json << "\"pipeline_creation_count\":" << profile.pipeline_creation_count << ",";
    json << "\"sync_heavy\":" << (profile.sync_heavy ? "true" : "false") << ",";
    json << "\"transfer_heavy\":" << (profile.transfer_heavy ? "true" : "false") << ",";
    json << "\"compile_in_infer\":" << (profile.compile_in_infer ? "true" : "false") << ",";
    json << "\"binding_prepare_in_infer\":" << (profile.binding_prepare_in_infer ? "true" : "false") << ",";
    json << "\"final_fence_wait_seen\":" << (profile.final_fence_wait_seen ? "true" : "false") << ",";
    json << "\"cross_submit_barrier_seen\":" << (profile.cross_submit_barrier_seen ? "true" : "false");
    json << "}";
}

void append_calibration_json(std::ostringstream& json, const ov::gfx_plugin::microbench::CalibrationArtifact& artifact) {
    json << ov::gfx_plugin::microbench::calibration_artifact_to_json(artifact);
}

void append_loaded_calibration_summary_json(std::ostringstream& json, const LoadedCalibrationSummary& summary) {
    json << "{";
    json << "\"provided\":" << (summary.provided ? "true" : "false");
    if (summary.provided) {
        json << ",";
        json << "\"path\":\"" << escape_json(summary.path) << "\",";
        json << "\"device_key_match\":" << (summary.device_key_match ? "true" : "false") << ",";
        json << "\"backend_match\":" << (summary.backend_match ? "true" : "false") << ",";
        json << "\"schema_match\":" << (summary.schema_match ? "true" : "false") << ",";
        json << "\"loaded_device_key\":\"" << escape_json(summary.artifact.device_key) << "\",";
        json << "\"loaded_backend\":\"" << escape_json(summary.artifact.backend) << "\"";
    }
    json << "}";
}

void append_derived_json(std::ostringstream& json, const BenchDerived& derived) {
    json << "{";
    json << "\"fixed_overhead_us\":" << derived.fixed_overhead_us << ",";
    json << "\"fixed_overhead_share\":" << derived.fixed_overhead_share << ",";
    json << "\"overhead_subtracted_ms\":" << derived.overhead_subtracted_ms << ",";
    json << "\"e2e_tflops\":" << derived.e2e_tflops << ",";
    json << "\"e2e_gbps\":" << derived.e2e_gbps << ",";
    json << "\"adjusted_tflops\":" << derived.adjusted_tflops << ",";
    json << "\"adjusted_gbps\":" << derived.adjusted_gbps << ",";
    json << "\"gpu_tflops\":" << derived.gpu_tflops << ",";
    json << "\"gpu_gbps\":" << derived.gpu_gbps << ",";
    json << "\"gpu_share_of_wall\":" << derived.gpu_share_of_wall << ",";
    json << "\"wait_share_of_wall\":" << derived.wait_share_of_wall << ",";
    json << "\"transfer_share_of_wall\":" << derived.transfer_share_of_wall << ",";
    json << "\"first_to_steady_ratio\":" << derived.first_to_steady_ratio << ",";
    json << "\"hints\":[";
    for (size_t i = 0; i < derived.hints.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << "\"" << escape_json(derived.hints[i]) << "\"";
    }
    json << "]";
    json << "}";
}

void append_bench_json(std::ostringstream& json, const BenchResult& result, const BenchDerived& derived) {
    json << "{";
    json << "\"name\":\"" << escape_json(result.name) << "\",";
    json << "\"kind\":\"" << escape_json(result.kind) << "\",";
    json << "\"model\":\"" << escape_json(result.model_desc) << "\",";
    json << "\"actual_backend\":\"" << escape_json(result.actual_backend) << "\",";
    json << "\"compile_ms\":" << result.compile_ms << ",";
    json << "\"first_infer_ms\":" << result.first_infer_ms << ",";
    json << "\"median_infer_ms\":" << result.median_infer_ms << ",";
    json << "\"min_infer_ms\":" << result.min_infer_ms << ",";
    json << "\"max_infer_ms\":" << result.max_infer_ms << ",";
    json << "\"workload\":";
    append_workload_json(json, derived.workload);
    json << ",";
    json << "\"derived\":";
    append_derived_json(json, derived);
    json << ",";
    json << "\"profile_digest\":";
    append_profile_digest_json(json, derived.profile);
    json << ",";
    json << "\"profile\":";
    if (result.profile_json.empty()) {
        json << "null";
    } else {
        json << result.profile_json;
    }
    json << "}";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--backend" && i + 1 < argc) {
            options.backend = argv[++i];
        } else if (arg == "--warmup" && i + 1 < argc) {
            options.warmup = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--iterations" && i + 1 < argc) {
            options.iterations = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--output" && i + 1 < argc) {
            options.output_path = argv[++i];
        } else if (arg == "--calibration-output" && i + 1 < argc) {
            options.calibration_output_path = argv[++i];
        } else if (arg == "--calibration-input" && i + 1 < argc) {
            options.calibration_input_path = argv[++i];
        } else if (arg == "--help") {
            std::cout << "Usage: ov_gfx_microbench [--backend auto|metal|vulkan] [--warmup N] [--iterations N] "
                         "[--output path] [--calibration-output path] [--calibration-input path]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    if (options.iterations == 0) {
        throw std::runtime_error("--iterations must be > 0");
    }
    return options;
}

}  // namespace

#if !defined(__APPLE__)
Mb0Result run_vulkan_mb0(size_t warmup, size_t iterations) {
    using namespace ov::gfx_plugin;

    VulkanContext& ctx = VulkanContext::instance();
    VkDevice device = ctx.device();
    VkQueue queue = ctx.queue();

    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = ctx.queue_family_index();
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkResult res = vkCreateCommandPool(device, &pool_info, nullptr, &pool);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("MB0 Vulkan: vkCreateCommandPool failed");
    }

    VkCommandBufferAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    res = vkAllocateCommandBuffers(device, &alloc, &cmd);
    if (res != VK_SUCCESS) {
        vkDestroyCommandPool(device, pool, nullptr);
        throw std::runtime_error("MB0 Vulkan: vkAllocateCommandBuffers failed");
    }

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    res = vkCreateFence(device, &fence_info, nullptr, &fence);
    if (res != VK_SUCCESS) {
        vkFreeCommandBuffers(device, pool, 1, &cmd);
        vkDestroyCommandPool(device, pool, nullptr);
        throw std::runtime_error("MB0 Vulkan: vkCreateFence failed");
    }

    std::vector<double> wall_us;
    wall_us.reserve(iterations);
    const size_t total_iters = warmup + iterations;
    for (size_t i = 0; i < total_iters; ++i) {
        vkResetFences(device, 1, &fence);
        vkResetCommandPool(device, pool, 0);

        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        res = vkBeginCommandBuffer(cmd, &begin);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("MB0 Vulkan: vkBeginCommandBuffer failed");
        }
        res = vkEndCommandBuffer(cmd);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("MB0 Vulkan: vkEndCommandBuffer failed");
        }

        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;

        const auto start = std::chrono::steady_clock::now();
        res = vkQueueSubmit(queue, 1, &submit, fence);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("MB0 Vulkan: vkQueueSubmit failed");
        }
        res = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("MB0 Vulkan: vkWaitForFences failed");
        }
        const auto stop = std::chrono::steady_clock::now();
        if (i >= warmup) {
            wall_us.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
        }
    }

    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, pool, 1, &cmd);
    vkDestroyCommandPool(device, pool, nullptr);

    const auto [min_us, max_us] = minmax_of(wall_us);
    Mb0Result result;
    result.backend = "vulkan";
    result.median_wall_us = median_of(wall_us);
    result.min_wall_us = min_us;
    result.max_wall_us = max_us;
    return result;
}
#endif

namespace {

DeviceFingerprint query_device_fingerprint(const std::string& selected_backend, const std::string& actual_backend) {
    DeviceFingerprint info;
    info.backend = actual_backend.empty() ? selected_backend : actual_backend;
#if defined(__APPLE__)
    info.platform = "macos";
    const auto names = ov::gfx_plugin::metal_get_device_names();
    if (!names.empty()) {
        info.device_name = names.front();
        info.full_name = "GFX (" + info.device_name + ")";
        info.vendor_id = "apple";
        info.device_id = "metal_default";
        info.driver_version = "metal";
        info.architecture = "apple_silicon";
    }
#else
    info.platform = "linux_or_android";
    const auto& ctx = ov::gfx_plugin::VulkanContext::instance();
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx.physical_device(), &props);
    info.device_name = props.deviceName;
    info.full_name = std::string("GFX (") + props.deviceName + ")";
    info.vendor_id = hex_u32(props.vendorID);
    info.device_id = hex_u32(props.deviceID);
    info.driver_version = std::to_string(props.driverVersion);
    info.architecture = "vulkan";
#endif
    return info;
}

void append_device_json(std::ostringstream& json, const DeviceFingerprint& device) {
    json << "{";
    json << "\"backend\":\"" << escape_json(device.backend) << "\",";
    json << "\"device_name\":\"" << escape_json(device.device_name) << "\",";
    json << "\"full_name\":\"" << escape_json(device.full_name) << "\",";
    json << "\"platform\":\"" << escape_json(device.platform) << "\",";
    json << "\"vendor_id\":\"" << escape_json(device.vendor_id) << "\",";
    json << "\"device_id\":\"" << escape_json(device.device_id) << "\",";
    json << "\"driver_version\":\"" << escape_json(device.driver_version) << "\",";
    json << "\"architecture\":\"" << escape_json(device.architecture) << "\"";
    json << "}";
}

void append_top_level_analysis_json(std::ostringstream& json,
                                    const DeviceFingerprint& device,
                                    const Mb0Result& mb0,
                                    const BenchDerived& mb1,
                                    const BenchDerived& mb2,
                                    const BenchDerived& mb3) {
    const auto triage_hints = top_level_triage_hints(mb0, mb1, mb2, mb3);
    const auto assumptions = calibration_assumptions();

    json << "{";
    json << "\"device_key\":\"" << escape_json(device_key_for_autotune(device)) << "\",";
    json << "\"fixed_overhead_us\":" << mb0.median_wall_us << ",";
    json << "\"bandwidth_probe\":\"MB2\",";
    json << "\"compute_probe\":\"MB3\",";
    json << "\"bandwidth_estimate_gbps\":" << mb2.adjusted_gbps << ",";
    json << "\"compute_estimate_tflops\":" << mb3.adjusted_tflops << ",";
    json << "\"gpu_bandwidth_estimate_gbps\":" << mb2.gpu_gbps << ",";
    json << "\"gpu_compute_estimate_tflops\":" << mb3.gpu_tflops << ",";
    json << "\"triage_hints\":[";
    for (size_t i = 0; i < triage_hints.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << "\"" << escape_json(triage_hints[i]) << "\"";
    }
    json << "],";
    json << "\"assumptions\":["
         ;
    for (size_t i = 0; i < assumptions.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << "\"" << escape_json(assumptions[i]) << "\"";
    }
    json << "]";
    json << "}";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        ov::Core core;
        register_gfx_plugin(core);

        const Mb0Result mb0 = run_mb0(options.backend, options.warmup, options.iterations);
        BenchResult mb1 = run_model_bench(core,
                                          options,
                                          "MB1",
                                          "copy_dispatch_model",
                                          "relu<f16>[1x1x1024x1024]",
                                          make_mb1_model());
        BenchResult mb2 = run_model_bench(core,
                                          options,
                                          "MB2",
                                          "bandwidth_model",
                                          "add_const<f16>[1x4x1024x1024]",
                                          make_mb2_model());
        BenchResult mb3 = run_model_bench(core,
                                          options,
                                          "MB3",
                                          "gemm_model",
                                          "matmul_const<f16>[1x1024x1024]x[1x1024x1024]",
                                          make_mb3_model());
        const DeviceFingerprint device = query_device_fingerprint(options.backend, mb1.actual_backend);
        const BenchDerived mb1_analysis = analyze_benchmark(mb1, mb0);
        const BenchDerived mb2_analysis = analyze_benchmark(mb2, mb0);
        const BenchDerived mb3_analysis = analyze_benchmark(mb3, mb0);
        const auto calibration =
            make_calibration_artifact(device, mb0, mb1, mb1_analysis, mb2, mb2_analysis, mb3, mb3_analysis);

        LoadedCalibrationSummary loaded_calibration;
        if (!options.calibration_input_path.empty()) {
            loaded_calibration.provided = true;
            loaded_calibration.path = options.calibration_input_path;
            if (!ov::gfx_plugin::microbench::read_calibration_artifact_file(options.calibration_input_path,
                                                                            loaded_calibration.artifact)) {
                throw std::runtime_error("failed to read calibration artifact: " + options.calibration_input_path);
            }
            loaded_calibration.device_key_match = loaded_calibration.artifact.device_key == calibration.device_key;
            loaded_calibration.backend_match = loaded_calibration.artifact.backend == calibration.backend;
            loaded_calibration.schema_match =
                loaded_calibration.artifact.schema_version == calibration.schema_version &&
                loaded_calibration.artifact.microbench_schema_version == calibration.microbench_schema_version;
        }

        std::ostringstream json;
        json << std::fixed << std::setprecision(3);
        json << "{";
        json << "\"schema_version\":2,";
        json << "\"tool\":\"ov_gfx_microbench\",";
        json << "\"selected_backend\":\"" << escape_json(options.backend) << "\",";
        json << "\"device\":";
        append_device_json(json, device);
        json << ",";
        json << "\"warmup\":" << options.warmup << ",";
        json << "\"iterations\":" << options.iterations << ",";
        json << "\"assumptions\":["
             << "\"MB0 is a raw backend empty submit or empty command-buffer commit measured with sync.\","
             << "\"MB1-MB3 are synthetic FP16 OpenVINO models run through the GFX plugin with detailed profiling enabled.\","
             << "\"MB1 approximates copy+dispatch, MB2 approximates bandwidth pressure, MB3 approximates compute-bound GEMM pressure.\""
             << "],";
        json << "\"mb0\":{";
        json << "\"backend\":\"" << escape_json(mb0.backend) << "\",";
        json << "\"median_wall_us\":" << mb0.median_wall_us << ",";
        json << "\"min_wall_us\":" << mb0.min_wall_us << ",";
        json << "\"max_wall_us\":" << mb0.max_wall_us << ",";
        json << "\"median_gpu_us\":";
        if (mb0.has_gpu_us) {
            json << mb0.median_gpu_us;
        } else {
            json << "null";
        }
        json << "},";
        json << "\"analysis\":";
        append_top_level_analysis_json(json, device, mb0, mb1_analysis, mb2_analysis, mb3_analysis);
        json << ",";
        json << "\"calibration\":";
        append_calibration_json(json, calibration);
        json << ",";
        json << "\"loaded_calibration\":";
        append_loaded_calibration_summary_json(json, loaded_calibration);
        json << ",";
        json << "\"benchmarks\":[";
        append_bench_json(json, mb1, mb1_analysis);
        json << ",";
        append_bench_json(json, mb2, mb2_analysis);
        json << ",";
        append_bench_json(json, mb3, mb3_analysis);
        json << "]";
        json << "}";

        if (!options.output_path.empty()) {
            std::ofstream output(options.output_path);
            if (!output) {
                throw std::runtime_error("failed to open output file: " + options.output_path);
            }
            output << json.str() << '\n';
        }
        if (!options.calibration_output_path.empty() &&
            !ov::gfx_plugin::microbench::write_calibration_artifact_file(calibration, options.calibration_output_path)) {
            throw std::runtime_error("failed to write calibration artifact: " + options.calibration_output_path);
        }

        std::cout << json.str() << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "fatal: " << ex.what() << '\n';
        return 1;
    }
}
