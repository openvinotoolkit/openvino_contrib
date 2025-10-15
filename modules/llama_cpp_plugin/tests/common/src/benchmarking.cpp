// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef BENCHMARKING_CPP
#define BENCHMARKING_CPP

#include "benchmarking.hpp"

#include <algorithm>
#include <chrono>

double measure_iterations_per_second(std::function<void(void)> iteration_fn, size_t iterations) {
    std::vector<float> iteration_times_s(iterations);

    for (size_t i = 0; i < iterations; i++) {
        auto start = std::chrono::steady_clock::now();
        iteration_fn();
        auto end = std::chrono::steady_clock::now();
        iteration_times_s[i] = std::chrono::duration<double>(end - start).count();
    }

    std::sort(iteration_times_s.begin(), iteration_times_s.end());
    return 1.0 / iteration_times_s[iteration_times_s.size() / 2];
}

#endif /* BENCHMARKING_CPP */
