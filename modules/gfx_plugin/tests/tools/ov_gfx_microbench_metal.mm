#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ov_gfx_microbench_common.hpp"

namespace {

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

}  // namespace

Mb0Result run_metal_mb0(size_t warmup, size_t iterations) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            throw std::runtime_error("MB0 Metal: no Metal device available");
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            throw std::runtime_error("MB0 Metal: failed to create command queue");
        }

        std::vector<double> wall_us;
        std::vector<double> gpu_us;
        wall_us.reserve(iterations);
        gpu_us.reserve(iterations);
        const size_t total_iters = warmup + iterations;
        for (size_t i = 0; i < total_iters; ++i) {
            const auto start = std::chrono::steady_clock::now();
            id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            const auto stop = std::chrono::steady_clock::now();
            if (i >= warmup) {
                wall_us.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
                if (command_buffer.GPUStartTime > 0.0 && command_buffer.GPUEndTime > command_buffer.GPUStartTime) {
                    gpu_us.push_back((command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1.0e6);
                }
            }
        }

        Mb0Result result;
        result.backend = "metal";
        result.median_wall_us = median_of(wall_us);
        const auto [min_us, max_us] = minmax_of(wall_us);
        result.min_wall_us = min_us;
        result.max_wall_us = max_us;
        if (!gpu_us.empty()) {
            result.has_gpu_us = true;
            result.median_gpu_us = median_of(gpu_us);
        }
        return result;
    }
}
