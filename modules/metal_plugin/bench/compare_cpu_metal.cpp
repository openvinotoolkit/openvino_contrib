// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Small utility to compare CPU vs METAL inference outputs for a given model.
//
// Build (from build directory):
//   c++ -std=c++17 -I${OPENVINO_SRC}/src/core/include -I${OPENVINO_SRC}/src/inference/include \\
//       modules/metal_plugin/bench/compare_cpu_metal.cpp \\
//       -L ./bin/arm64/Release -lopenvino -lopenvino_c_api -o compare_cpu_metal
//
// Run:
//   DYLD_LIBRARY_PATH=./bin/arm64/Release:./temp/Darwin_arm64/tbb/lib \\
//     ./compare_cpu_metal /path/to/model.xml [--iter 1] [--seed 0]
//
#include <openvino/openvino.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>
#include <openvino/op/constant.hpp>

namespace {

int64_t static_dim_or(const ov::Dimension& d, int64_t fallback) {
    if (d.is_static()) return static_cast<int64_t>(d.get_length());
    return fallback;
}

ov::Shape make_static_shape(const ov::PartialShape& ps, int64_t fallback = 1) {
    ov::Shape shp;
    shp.reserve(ps.rank().is_static() ? ps.rank().get_length() : 0);
    for (const auto& dim : ps) {
        shp.push_back(static_cast<size_t>(static_dim_or(dim, fallback)));
    }
    if (shp.empty()) shp.push_back(1);
    return shp;
}

template <typename T>
void fill_tensor(ov::Tensor& t, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    auto* data = t.data<T>();
    for (size_t i = 0; i < t.get_size(); ++i) data[i] = static_cast<T>(dist(gen));
}

struct Stats {
    double max_abs = 0.0;
    double max_rel = 0.0;
    size_t mismatches = 0;
    size_t total = 0;
};

template <typename T>
ov::Tensor to_float_tensor(const ov::Tensor& src) {
    if (src.get_element_type() == ov::element::f32) return src;
    ov::Tensor dst(ov::element::f32, src.get_shape());
    const T* s = src.data<const T>();
    float* d = dst.data<float>();
    for (size_t i = 0; i < src.get_size(); ++i) d[i] = static_cast<float>(s[i]);
    return dst;
}

ov::Tensor convert_to_float(const ov::Tensor& t) {
    switch (t.get_element_type()) {
        case ov::element::f32: return t;
        case ov::element::f16: return to_float_tensor<ov::float16>(t);
        case ov::element::i32: return to_float_tensor<int32_t>(t);
        default: return t;  // best effort
    }
}

Stats compare_tensors(const ov::Tensor& ref, const ov::Tensor& test, double abs_eps = 1e-5, double rel_eps = 1e-5) {
    Stats s;
    s.total = ref.get_size();
    const float* a = ref.data<const float>();
    const float* b = test.data<const float>();
    for (size_t i = 0; i < s.total; ++i) {
        double diff = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        double denom = std::max(std::fabs(static_cast<double>(a[i])), 1e-7);
        double rel = diff / denom;
        s.max_abs = std::max(s.max_abs, diff);
        s.max_rel = std::max(s.max_rel, rel);
        if (diff > abs_eps && rel > rel_eps) ++s.mismatches;
    }
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " model.xml [--iter N] [--count N] [--seed S]\n";
        std::cerr << "       [--abs_eps E] [--rel_eps E] [--per-op]\n";
        std::cerr << "       Add --per-op to stop on first differing op (only with --count 1).\n";
        return 1;
    }
    std::string model_path = argv[1];
    int iterations = 1;
    int count = 1;
    uint64_t seed = 0;
    double abs_eps = 1e-5;
    double rel_eps = 1e-5;
    bool per_op = false;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--iter" && i + 1 < argc) {
            iterations = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--count" && i + 1 < argc) {
            count = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<uint64_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (arg == "--abs_eps" && i + 1 < argc) {
            abs_eps = std::max(0.0, std::strtod(argv[++i], nullptr));
        } else if (arg == "--rel_eps" && i + 1 < argc) {
            rel_eps = std::max(0.0, std::strtod(argv[++i], nullptr));
        } else if (arg == "--per-op") {
            per_op = true;
        }
    }

    if (per_op && count > 1) {
        std::cerr << "--per-op is only supported with --count 1\n";
        return 1;
    }

    ov::Core core;
    auto model = core.read_model(model_path);

    auto compile = [&](const std::string& device) {
        ov::AnyMap cfg;
        cfg[ov::hint::inference_precision.name()] = ov::element::f32;
        return core.compile_model(model, device, cfg);
    };

    auto compiled_cpu = compile("CPU");
    auto compiled_metal = compile("METAL");

    auto make_inputs = [&](uint64_t seed_value) {
        std::mt19937 gen(seed_value);
        std::vector<ov::Tensor> inputs;
        inputs.reserve(compiled_cpu.inputs().size());
        for (const auto& input : compiled_cpu.inputs()) {
            const auto ps = input.get_partial_shape();
            ov::Shape shp = ps.is_static() ? ps.to_shape() : make_static_shape(ps);
            const auto et = input.get_element_type();
            ov::Tensor t(et, shp);
            switch (et) {
                case ov::element::f32: fill_tensor<float>(t, gen); break;
                case ov::element::f16: fill_tensor<ov::float16>(t, gen); break;
                case ov::element::i32: fill_tensor<int32_t>(t, gen); break;
                default:
                    std::cerr << "Unsupported input type: " << et << "\n";
                    inputs.clear();
                    return inputs;
            }
            inputs.push_back(t);
        }
        return inputs;
    };

    auto make_request = [&](ov::CompiledModel& cm, const std::vector<ov::Tensor>& src_inputs) {
        auto req = cm.create_infer_request();
        const auto& ports = cm.inputs();
        for (size_t i = 0; i < ports.size(); ++i) {
            req.set_tensor(ports[i], src_inputs[i]);
        }
        return req;
    };

    auto run_req = [&](ov::InferRequest& req, int n_iter) {
        for (int i = 0; i < n_iter; ++i) req.infer();
    };

    struct RunStats {
        double max_abs = 0.0;
        double max_rel = 0.0;
        size_t mismatches = 0;
        size_t total = 0;
    };

    auto compare_outputs = [&](const ov::OutputVector& outs,
                               ov::CompiledModel& cm_cpu,
                               ov::CompiledModel& cm_metal,
                               const std::vector<ov::Tensor>& inputs,
                               const std::string& tag,
                               bool verbose) -> RunStats {
        auto req_c = make_request(cm_cpu, inputs);
        auto req_m = make_request(cm_metal, inputs);
        run_req(req_c, iterations);
        run_req(req_m, iterations);
        RunStats rs;
        for (size_t i = 0; i < outs.size(); ++i) {
            auto ref = req_c.get_tensor(cm_cpu.output(i));
            auto tst = req_m.get_tensor(cm_metal.output(i));
            auto ref_f = convert_to_float(ref);
            auto tst_f = convert_to_float(tst);
            auto stats = compare_tensors(ref_f, tst_f, abs_eps, rel_eps);
            if (verbose) {
                std::cout << tag << " Output[" << i << "] shape=" << ref.get_shape()
                          << " max_abs=" << stats.max_abs
                          << " max_rel=" << stats.max_rel
                          << " mismatches=" << stats.mismatches << "/" << stats.total
                          << "\n";
            }
            rs.max_abs = std::max(rs.max_abs, stats.max_abs);
            rs.max_rel = std::max(rs.max_rel, stats.max_rel);
            rs.mismatches += stats.mismatches;
            rs.total += stats.total;
        }
        return rs;
    };

    size_t match_count = 0;
    size_t mismatch_count = 0;
    double global_max_abs = 0.0;
    double global_max_rel = 0.0;
    bool all_ok = true;

    for (int run = 0; run < count; ++run) {
        uint64_t run_seed = seed + static_cast<uint64_t>(run);
        auto inputs = make_inputs(run_seed);
        if (inputs.empty()) return 1;

        auto req_cpu = make_request(compiled_cpu, inputs);
        auto req_metal = make_request(compiled_metal, inputs);

        run_req(req_cpu, iterations);
        run_req(req_metal, iterations);

        RunStats rs;
        const auto outputs_cpu = compiled_cpu.outputs();
        for (size_t i = 0; i < outputs_cpu.size(); ++i) {
            const auto& port = outputs_cpu[i];
            ov::Tensor ref = req_cpu.get_tensor(port);
            ov::Tensor tst = req_metal.get_tensor(port);
            ov::Tensor ref_f = convert_to_float(ref);
            ov::Tensor tst_f = convert_to_float(tst);
            auto stats = compare_tensors(ref_f, tst_f, abs_eps, rel_eps);
            if (count == 1) {
                std::cout << "Output[" << i << "] shape=" << ref.get_shape()
                          << " max_abs=" << stats.max_abs
                          << " max_rel=" << stats.max_rel
                          << " mismatches=" << stats.mismatches << "/" << stats.total
                          << "\n";
            }
            rs.max_abs = std::max(rs.max_abs, stats.max_abs);
            rs.max_rel = std::max(rs.max_rel, stats.max_rel);
            rs.mismatches += stats.mismatches;
            rs.total += stats.total;
        }

        bool ok = (rs.mismatches == 0);
        if (ok) {
            ++match_count;
        } else {
            ++mismatch_count;
            all_ok = false;
        }
        global_max_abs = std::max(global_max_abs, rs.max_abs);
        global_max_rel = std::max(global_max_rel, rs.max_rel);

        if (count > 1) {
            std::cout << "Seed " << run_seed << ": " << (ok ? "MATCH" : "MISMATCH")
                      << " max_abs=" << rs.max_abs
                      << " max_rel=" << rs.max_rel
                      << " mismatches=" << rs.mismatches << "/" << rs.total
                      << "\n";
        }
    }

    if (!per_op || count > 1 || all_ok) {
        if (count > 1) {
            std::cout << "Summary: match=" << match_count
                      << " mismatch=" << mismatch_count
                      << " max_abs=" << global_max_abs
                      << " max_rel=" << global_max_rel
                      << "\n";
        }
        std::cout << (all_ok ? "MATCH" : "MISMATCH") << std::endl;
        return all_ok ? 0 : 2;
    }

    // Per-op diff: walk ordered ops, build submodel with current node outputs, compare.
    const auto ordered_ops = model->get_ordered_ops();
    size_t idx = 0;
    for (const auto& node : ordered_ops) {
        if (ov::is_type<ov::op::v0::Parameter>(node) || ov::is_type<ov::op::v0::Constant>(node) ||
            ov::is_type<ov::op::v0::Result>(node)) {
            ++idx;
            continue;
        }
        ov::OutputVector res_outputs;
        for (auto& out : node->outputs()) {
            res_outputs.push_back(std::make_shared<ov::op::v0::Result>(out));
        }
        auto sub = std::make_shared<ov::Model>(res_outputs, model->get_parameters(), node->get_friendly_name());
        ov::CompiledModel sub_cpu, sub_metal;
        try {
            sub_cpu = core.compile_model(sub, "CPU", {{ov::hint::inference_precision.name(), ov::element::f32}});
            sub_metal = core.compile_model(sub, "METAL", {{ov::hint::inference_precision.name(), ov::element::f32}});
        } catch (const std::exception& e) {
            std::cout << "[op " << idx << "] " << node->get_friendly_name() << " (" << node->get_type_name()
                      << ") SKIP: compile failed: " << e.what() << "\n";
            ++idx;
            continue;
        }
        bool ok = true;
        try {
            auto inputs = make_inputs(seed);
            if (inputs.empty()) return 1;
            auto rs = compare_outputs(res_outputs, sub_cpu, sub_metal, inputs,
                                      "[op " + std::to_string(idx) + "] " + node->get_friendly_name() +
                                          " (" + node->get_type_name() + ")",
                                      true);
            ok = (rs.mismatches == 0);
        } catch (const std::exception& e) {
            std::cout << "[op " << idx << "] " << node->get_friendly_name() << " (" << node->get_type_name()
                      << ") SKIP: infer failed: " << e.what() << "\n";
            ++idx;
            continue;
        }
        if (!ok) {
            std::cout << "Mismatch at op #" << idx << " name=" << node->get_friendly_name()
                      << " type=" << node->get_type_name() << "\n";
            all_ok = false;
            std::cout << "Stopping at first mismatch (per-op).\n";
            std::cout << "MISMATCH" << std::endl;
            return 2;
        }
        ++idx;
    }

    std::cout << (all_ok ? "MATCH" : "MISMATCH") << std::endl;
    return all_ok ? 0 : 2;
}
