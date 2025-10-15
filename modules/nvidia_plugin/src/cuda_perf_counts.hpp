// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <string>

namespace ov {
namespace nvidia_gpu {

static const char PERF_COUNTER_NAME[] = "nvidia_perf_counter";

enum class PerfStages { Preprocess, Postprocess, StartPipeline, WaitPipeline, NumOfStages };

struct PerfCounts {
    std::chrono::microseconds total_duration;
    uint32_t num;
    std::string impl_type;
    std::string runtime_precision;

    PerfCounts() : total_duration{0}, num(0) {}

    uint64_t average() const { return (num == 0) ? 0 : total_duration.count() / num; }
};

}  // namespace nvidia_gpu
}  // namespace ov
