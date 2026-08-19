// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/profiling/gpu_timestamps.hpp"

namespace ov {
namespace gfx_plugin {

MetalGpuTimestamps::MetalGpuTimestamps(MetalDeviceHandle /*device*/, bool /*counters_supported*/) {}

MetalGpuTimestamps::~MetalGpuTimestamps() = default;

bool MetalGpuTimestamps::supported() const {
    return false;
}

void MetalGpuTimestamps::begin_frame(size_t /*expected_samples*/) {}

MetalGpuTimestamps::SampleIndex MetalGpuTimestamps::sample_begin(MetalCommandEncoderHandle /*encoder*/) {
    return -1;
}

MetalGpuTimestamps::SampleIndex MetalGpuTimestamps::sample_end(MetalCommandEncoderHandle /*encoder*/) {
    return -1;
}

void MetalGpuTimestamps::resolve() {}

uint64_t MetalGpuTimestamps::get_timestamp(SampleIndex /*idx*/) const {
    return 0;
}

double MetalGpuTimestamps::gpu_ticks_to_ns_factor() const {
    return 0.0;
}

}  // namespace gfx_plugin
}  // namespace ov
