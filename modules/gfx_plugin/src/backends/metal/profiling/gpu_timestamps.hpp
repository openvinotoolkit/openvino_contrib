// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "backends/metal/memory/buffer.hpp"

namespace ov {
namespace gfx_plugin {

class MetalGpuTimestamps {
public:
    using SampleIndex = int32_t;

    MetalGpuTimestamps(MetalDeviceHandle device, bool counters_supported);
    ~MetalGpuTimestamps();

    bool supported() const;

    void begin_frame(size_t expected_samples);

    SampleIndex sample_begin(MetalCommandEncoderHandle encoder);
    SampleIndex sample_end(MetalCommandEncoderHandle encoder);

    void resolve();

    uint64_t get_timestamp(SampleIndex idx) const;

    double gpu_ticks_to_ns_factor() const;

private:
    void update_calibration_if_needed();
    SampleIndex sample_internal(MetalCommandEncoderHandle encoder);

    MetalDeviceHandle m_device = nullptr;
    bool m_supported = false;

    void* m_sample_buffer = nullptr;  // id<MTLCounterSampleBuffer>
    size_t m_sample_capacity = 0;
    size_t m_sample_count = 0;
    bool m_sample_overflow = false;

    std::vector<uint64_t> m_timestamps;

    double m_gpu_ticks_to_ns = 0.0;
    uint64_t m_last_calibration_ns = 0;
};

}  // namespace gfx_plugin
}  // namespace ov
