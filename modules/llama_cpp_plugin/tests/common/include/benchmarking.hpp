// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef BENCHMARKING_HPP
#define BENCHMARKING_HPP

#include <functional>
#include <vector>

double measure_iterations_per_second(std::function<void(void)> iteration_fn, size_t iterations);

#endif /* BENCHMARKING_HPP */
