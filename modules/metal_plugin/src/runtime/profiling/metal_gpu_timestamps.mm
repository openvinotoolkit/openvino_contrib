// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/profiling/metal_gpu_timestamps.hpp"

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

#include <algorithm>
#include <chrono>

#include <mach/mach_time.h>

namespace ov {
namespace metal_plugin {

namespace {
uint64_t mach_to_ns(uint64_t t) {
    static mach_timebase_info_data_t info = {};
    if (info.denom == 0) {
        (void)mach_timebase_info(&info);
    }
    return t * info.numer / info.denom;
}

#ifdef __OBJC__
id<MTLCounterSet> find_timestamp_counter_set(id<MTLDevice> dev) {
    if (!dev)
        return nil;
    for (id<MTLCounterSet> set in dev.counterSets) {
        if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
            return set;
        }
    }
    return nil;
}
#endif

}  // namespace

MetalGpuTimestamps::MetalGpuTimestamps(MetalDeviceHandle device, bool counters_supported)
    : m_device(device), m_supported(counters_supported) {
#ifdef __OBJC__
    if (!device) {
        m_supported = false;
    }
#else
    (void)device;
    m_supported = false;
#endif
}

MetalGpuTimestamps::~MetalGpuTimestamps() {
#ifdef __OBJC__
    if (m_sample_buffer) {
#if __has_feature(objc_arc)
        id<MTLCounterSampleBuffer> sb = (__bridge_transfer id<MTLCounterSampleBuffer>)m_sample_buffer;
        (void)sb;
#else
        id<MTLCounterSampleBuffer> sb = static_cast<id<MTLCounterSampleBuffer>>(m_sample_buffer);
        [sb release];
#endif
        m_sample_buffer = nullptr;
    }
#endif
}

bool MetalGpuTimestamps::supported() const {
    return m_supported;
}

void MetalGpuTimestamps::begin_frame(size_t expected_samples) {
    if (!m_supported)
        return;

#ifdef __OBJC__
    id<MTLDevice> dev = static_cast<id<MTLDevice>>(m_device);
    if (!dev) {
        m_supported = false;
        return;
    }

    id<MTLCounterSet> timestamp_set = find_timestamp_counter_set(dev);
    if (!timestamp_set) {
        m_supported = false;
        return;
    }

    if (expected_samples < 2) {
        expected_samples = 2;
    }

    if (expected_samples > m_sample_capacity || !m_sample_buffer) {
        if (m_sample_buffer) {
#if __has_feature(objc_arc)
            id<MTLCounterSampleBuffer> old = (__bridge_transfer id<MTLCounterSampleBuffer>)m_sample_buffer;
            (void)old;
#else
            id<MTLCounterSampleBuffer> old = static_cast<id<MTLCounterSampleBuffer>>(m_sample_buffer);
            [old release];
#endif
            m_sample_buffer = nullptr;
        }
        MTLCounterSampleBufferDescriptor* desc = [[MTLCounterSampleBufferDescriptor alloc] init];
        desc.counterSet = timestamp_set;
        desc.label = @"ov.metal.profiling";
        desc.storageMode = MTLStorageModeShared;
        desc.sampleCount = static_cast<NSUInteger>(expected_samples);
        NSError* err = nil;
        id<MTLCounterSampleBuffer> sb = [dev newCounterSampleBufferWithDescriptor:desc error:&err];
        if (!sb || err) {
            m_supported = false;
            return;
        }
#if __has_feature(objc_arc)
        m_sample_buffer = (__bridge_retained void*)sb;
#else
        m_sample_buffer = static_cast<void*>([sb retain]);
#endif
        m_sample_capacity = static_cast<size_t>(sb.sampleCount);
    }

    m_sample_count = 0;
    m_sample_overflow = false;
    m_timestamps.assign(m_sample_capacity, 0);
    update_calibration_if_needed();
#else
    (void)expected_samples;
#endif
}

MetalGpuTimestamps::SampleIndex MetalGpuTimestamps::sample_begin(MetalCommandEncoderHandle encoder) {
    return sample_internal(encoder);
}

MetalGpuTimestamps::SampleIndex MetalGpuTimestamps::sample_end(MetalCommandEncoderHandle encoder) {
    return sample_internal(encoder);
}

MetalGpuTimestamps::SampleIndex MetalGpuTimestamps::sample_internal(MetalCommandEncoderHandle encoder) {
    if (!m_supported || !m_sample_buffer || m_sample_overflow) {
        return -1;
    }

#ifdef __OBJC__
    id<MTLCounterSampleBuffer> sb = (__bridge id<MTLCounterSampleBuffer>)m_sample_buffer;
    if (!sb) {
        return -1;
    }
    if (m_sample_count >= m_sample_capacity) {
        m_sample_overflow = true;
        return -1;
    }

    id enc = static_cast<id>(encoder);
    if (!enc || ![enc respondsToSelector:@selector(sampleCountersInBuffer:atSampleIndex:withBarrier:)]) {
        return -1;
    }

    const auto idx = static_cast<NSUInteger>(m_sample_count++);
    [enc sampleCountersInBuffer:sb atSampleIndex:idx withBarrier:YES];
    return static_cast<SampleIndex>(idx);
#else
    (void)encoder;
    return -1;
#endif
}

void MetalGpuTimestamps::resolve() {
    if (!m_supported || !m_sample_buffer) {
        return;
    }
#ifdef __OBJC__
    id<MTLCounterSampleBuffer> sb = (__bridge id<MTLCounterSampleBuffer>)m_sample_buffer;
    if (!sb || m_sample_count == 0) {
        return;
    }
    NSData* data = [sb resolveCounterRange:NSMakeRange(0, m_sample_count)];
    if (!data || data.length < sizeof(MTLCounterResultTimestamp)) {
        return;
    }
    const size_t count = std::min(m_sample_count, m_sample_capacity);
    const auto* results = reinterpret_cast<const MTLCounterResultTimestamp*>(data.bytes);
    for (size_t i = 0; i < count; ++i) {
        m_timestamps[i] = results[i].timestamp;
    }
#else
    (void)m_sample_buffer;
#endif
}

uint64_t MetalGpuTimestamps::get_timestamp(SampleIndex idx) const {
    if (idx < 0)
        return 0;
    const size_t u = static_cast<size_t>(idx);
    if (u >= m_timestamps.size())
        return 0;
    return m_timestamps[u];
}

double MetalGpuTimestamps::gpu_ticks_to_ns_factor() const {
    return m_gpu_ticks_to_ns > 0.0 ? m_gpu_ticks_to_ns : 0.0;
}

void MetalGpuTimestamps::update_calibration_if_needed() {
#ifdef __OBJC__
    if (!m_supported || !m_device)
        return;

    const uint64_t now_ns = mach_to_ns(mach_absolute_time());
    constexpr uint64_t kMinIntervalNs = 500ULL * 1000ULL * 1000ULL;  // 500ms
    if (m_last_calibration_ns != 0 && (now_ns - m_last_calibration_ns) < kMinIntervalNs) {
        return;
    }

    id<MTLDevice> dev = static_cast<id<MTLDevice>>(m_device);
    MTLTimestamp cpu0 = 0, gpu0 = 0;
    MTLTimestamp cpu1 = 0, gpu1 = 0;
    [dev sampleTimestamps:&cpu0 gpuTimestamp:&gpu0];
    [dev sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];

    const uint64_t cpu_ns0 = mach_to_ns(static_cast<uint64_t>(cpu0));
    const uint64_t cpu_ns1 = mach_to_ns(static_cast<uint64_t>(cpu1));
    const uint64_t cpu_delta = cpu_ns1 > cpu_ns0 ? (cpu_ns1 - cpu_ns0) : 0;
    const uint64_t gpu_delta = gpu1 > gpu0 ? static_cast<uint64_t>(gpu1 - gpu0) : 0;

    if (cpu_delta > 0 && gpu_delta > 0) {
        m_gpu_ticks_to_ns = static_cast<double>(cpu_delta) / static_cast<double>(gpu_delta);
        m_last_calibration_ns = now_ns;
    }
#else
    (void)m_device;
#endif
}

}  // namespace metal_plugin
}  // namespace ov
