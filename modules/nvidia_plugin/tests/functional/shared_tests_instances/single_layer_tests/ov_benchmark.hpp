// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cctype>
#include <iostream>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace test {

template <typename BaseLayerTest>
class BenchmarkLayerTest : public BaseLayerTest {
    static_assert(std::is_base_of<SubgraphBaseTest, BaseLayerTest>::value,
                  "BaseLayerTest should inherit from ov::test::SubgraphBaseTest");

public:
    void run(const std::initializer_list<std::string>& names,
             const std::chrono::milliseconds warmupTime = std::chrono::milliseconds(2000),
             const int numAttempts = 100) {
        bench_names_ = names;
        warmup_time_ = warmupTime;
        num_attempts_ = numAttempts;
        SubgraphBaseTest::configuration = {ov::enable_profiling(true)};
        SubgraphBaseTest::run();
    }

    void run(const std::string& name,
             const std::chrono::milliseconds warmupTime = std::chrono::milliseconds(2000),
             const int numAttempts = 100) {
        if (!name.empty()) {
            run({name}, warmupTime, numAttempts);
        } else {
            run({}, warmupTime, numAttempts);
        }
    }

    void validate() override {
        // NOTE: Validation is ignored because we are interested in benchmarks results
    }

protected:
    void infer() override {
        // Operation names search
        std::map<std::string, std::chrono::microseconds> results_us{};
        SubgraphBaseTest::infer();
        const auto& perf_results = SubgraphBaseTest::inferRequest.get_profiling_info();
        for (const auto& name : bench_names_) {
            bool found = false;
            for (const auto& result : perf_results) {
                const auto& res_name = result.node_name;
                const bool should_add =
                    !name.empty() && res_name.find(name) != std::string::npos && res_name.find('_') != std::string::npos;
                // Adding operations with numbers for the case there are several operations of the same type
                if (should_add) {
                    found = true;
                    results_us.emplace(std::make_pair(res_name, std::chrono::microseconds::zero()));
                }
            }
            if (!found) {
                std::cout << "WARNING! Performance count for \"" << name << "\" wasn't found!\n";
            }
        }
        // If no operations were found adding the time of all operations except Parameter and Result
        if (results_us.empty()) {
            for (const auto& result : perf_results) {
                const auto& res_name = result.node_name;
                const bool should_add = (res_name.find("Parameter") == std::string::npos) &&
                                        (res_name.find("Result") == std::string::npos) &&
                                        (res_name.find('_') != std::string::npos);
                if (should_add) {
                    results_us.emplace(std::make_pair(res_name, std::chrono::microseconds::zero()));
                }
            }
        }
        // Warmup
        auto warmCur = std::chrono::steady_clock::now();
        const auto warmEnd = warmCur + warmup_time_;
        while (warmCur < warmEnd) {
            SubgraphBaseTest::infer();
            warmCur = std::chrono::steady_clock::now();
        }
        // Benchmark
        for (int i = 0; i < num_attempts_; ++i) {
            SubgraphBaseTest::infer();
            const auto& perf_results = SubgraphBaseTest::inferRequest.get_profiling_info();
            for (auto& [name, time] : results_us) {
                auto it = std::find_if(perf_results.begin(), perf_results.end(), [&](const ::ov::ProfilingInfo& info) { return info.node_name == name; });
                OPENVINO_ASSERT(it != perf_results.end());
                time += it->real_time;
            }
        }

        std::chrono::microseconds total_us = std::chrono::microseconds::zero();
        for (auto& [name, time] : results_us) {
            time /= num_attempts_;
            total_us += time;
            std::cout << std::fixed << std::setfill('0') << name << ": " << time.count() << " us\n";
        }
        std::cout << std::fixed << std::setfill('0') << "Total time: " << total_us.count() << " us\n";
    }

private:
    std::vector<std::string> bench_names_;
    std::chrono::milliseconds warmup_time_;
    int num_attempts_;
};

}  // namespace test
}  // namespace ov
