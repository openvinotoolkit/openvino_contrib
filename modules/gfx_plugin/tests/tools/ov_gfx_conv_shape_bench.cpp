#include <openvino/openvino.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

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

struct ShapeCase {
    std::string name;
    ov::Shape input;
    ov::Shape weights;
    ov::Strides strides;
    ov::CoordinateDiff pads_begin;
    ov::CoordinateDiff pads_end;
    ov::Strides dilations;
};

struct Options {
    size_t warmup = 1;
    size_t iterations = 3;
    std::vector<std::string> devices{"CPU", "GFX"};
    std::vector<std::string> case_filters;
    bool list_cases = false;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--warmup" && i + 1 < argc) {
            options.warmup = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--iterations" && i + 1 < argc) {
            options.iterations = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--device" && i + 1 < argc) {
            options.devices = {argv[++i]};
        } else if (arg == "--case" && i + 1 < argc) {
            options.case_filters.push_back(argv[++i]);
        } else if (arg == "--list-cases") {
            options.list_cases = true;
        } else if (arg == "--help") {
            std::cout << "Usage: ov_gfx_conv_shape_bench [--warmup N] [--iterations N]"
                         " [--device CPU|GFX] [--case SUBSTRING] [--list-cases]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.iterations == 0) {
        throw std::runtime_error("--iterations must be > 0");
    }
    return options;
}

bool case_matches_filters(const ShapeCase& c, const std::vector<std::string>& filters) {
    if (filters.empty()) {
        return true;
    }
    return std::any_of(filters.begin(), filters.end(), [&](const std::string& filter) {
        return !filter.empty() && c.name.find(filter) != std::string::npos;
    });
}

std::vector<float> make_data(size_t count, size_t salt, float scale) {
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i) {
        const int value = static_cast<int>((i * 131u + salt * 17u) % 257u) - 128;
        data[i] = static_cast<float>(value) * scale;
    }
    return data;
}

std::shared_ptr<ov::Model> make_conv_model(const ShapeCase& c) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, c.input);
    const auto weight_count = ov::shape_size(c.weights);
    auto weights = ov::op::v0::Constant::create(ov::element::f32,
                                                c.weights,
                                                make_data(weight_count, c.input[1] + c.weights[0], 0.01f));
    auto conv = std::make_shared<ov::op::v1::Convolution>(input,
                                                          weights,
                                                          c.strides,
                                                          c.pads_begin,
                                                          c.pads_end,
                                                          c.dilations);
    const ov::Shape bias_shape{1, c.weights[0], 1, 1};
    auto bias = ov::op::v0::Constant::create(ov::element::f32,
                                             bias_shape,
                                             make_data(c.weights[0], c.weights[0], 0.001f));
    auto add = std::make_shared<ov::op::v1::Add>(conv, bias, ov::op::AutoBroadcastType::NUMPY);
    auto result = std::make_shared<ov::op::v0::Result>(add);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, c.name);
}

ov::AnyMap make_config() {
    ov::AnyMap config;
    config[ov::hint::inference_precision.name()] = ov::element::f16;
    return config;
}

void fill_input(ov::Tensor& tensor) {
    auto* data = tensor.data<float>();
    const size_t count = tensor.get_size();
    for (size_t i = 0; i < count; ++i) {
        const int value = static_cast<int>((i * 97u + 13u) % 251u) - 125;
        data[i] = static_cast<float>(value) / 64.0f;
    }
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

struct RunStats {
    double compile_ms = 0.0;
    double median_infer_ms = 0.0;
    double min_infer_ms = 0.0;
};

RunStats run_case(ov::Core& core, const ShapeCase& c, const std::string& device, const Options& options) {
    auto model = make_conv_model(c);
    const auto compile_start = std::chrono::steady_clock::now();
    auto compiled = core.compile_model(model, device, make_config());
    const auto compile_stop = std::chrono::steady_clock::now();

    auto request = compiled.create_infer_request();
    ov::Tensor input_tensor(model->input().get_element_type(), model->input().get_shape());
    fill_input(input_tensor);
    request.set_input_tensor(input_tensor);

    std::vector<double> infer_ms;
    infer_ms.reserve(options.iterations);
    const size_t total_iterations = options.warmup + options.iterations;
    for (size_t i = 0; i < total_iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        request.infer();
        const auto stop = std::chrono::steady_clock::now();
        if (i >= options.warmup) {
            infer_ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
        }
    }

    RunStats stats;
    stats.compile_ms = std::chrono::duration<double, std::milli>(compile_stop - compile_start).count();
    stats.median_infer_ms = median(infer_ms);
    stats.min_infer_ms = *std::min_element(infer_ms.begin(), infer_ms.end());
    return stats;
}

std::vector<ShapeCase> yolo26x_cases() {
    return {
        {"yolo26x_model_1_conv_s2", {1, 96, 320, 320}, {192, 96, 3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}},
        {"yolo26x_model_3_conv_s2", {1, 384, 160, 160}, {384, 384, 3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}},
        {"yolo26x_model_5_conv_s2", {1, 768, 80, 80}, {768, 768, 3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}},
        {"yolo26x_c2_48_48_k3_160", {1, 48, 160, 160}, {48, 48, 3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
        {"yolo26x_c4_96_96_k3_80", {1, 96, 80, 80}, {96, 96, 3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
        {"yolo26x_c6_192_192_k3_40", {1, 192, 40, 40}, {192, 192, 3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
        {"yolo26x_head_384_96_k3_80", {1, 384, 80, 80}, {96, 384, 3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
        {"yolo26x_pw_48_48_160", {1, 48, 160, 160}, {48, 48, 1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}},
        {"yolo26x_pw_96_96_80", {1, 96, 80, 80}, {96, 96, 1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}},
        {"yolo26x_pw_192_192_160", {1, 192, 160, 160}, {192, 192, 1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}},
        {"yolo26x_pw_384_384_80", {1, 384, 80, 80}, {384, 384, 1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}},
        {"yolo26x_pw_768_768_40", {1, 768, 40, 40}, {768, 768, 1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}},
        {"yolo26x_pw_384_384_160", {1, 384, 160, 160}, {384, 384, 1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}},
        {"yolo26x_pw_1536_384_80", {1, 1536, 80, 80}, {384, 1536, 1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}},
    };
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        ov::Core core;
        register_gfx_plugin(core);
        const auto cases = yolo26x_cases();

        if (options.list_cases) {
            for (const auto& c : cases) {
                std::cout << c.name << "\n";
            }
            return 0;
        }

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "case,device,compile_ms,median_infer_ms,min_infer_ms,fps\n";
        bool ran_any_case = false;
        for (const auto& c : cases) {
            if (!case_matches_filters(c, options.case_filters)) {
                continue;
            }
            ran_any_case = true;
            for (const auto& device : options.devices) {
                const RunStats stats = run_case(core, c, device, options);
                const double fps = stats.median_infer_ms > 0.0 ? 1000.0 / stats.median_infer_ms : 0.0;
                std::cout << c.name << "," << device << "," << stats.compile_ms << ","
                          << stats.median_infer_ms << "," << stats.min_infer_ms << "," << fps << "\n";
            }
        }
        if (!ran_any_case) {
            std::cerr << "fatal: no shape cases matched --case filters\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "fatal: " << ex.what() << "\n";
        return 1;
    }
}
