#pragma once

#include <string>

struct Mb0Result {
    std::string backend;
    double median_wall_us = 0.0;
    double min_wall_us = 0.0;
    double max_wall_us = 0.0;
    double median_gpu_us = 0.0;
    bool has_gpu_us = false;
};

#if defined(__APPLE__)
Mb0Result run_metal_mb0(size_t warmup, size_t iterations);
#endif

#if !defined(__APPLE__)
Mb0Result run_vulkan_mb0(size_t warmup, size_t iterations);
#endif
