// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Runs PointPillars with preprocessing and postprocessing kernels around the
// network. The kernels and network share an OpenCL queue and device memory.

#ifndef __PP_RUNTIME_HPP__
#define __PP_RUNTIME_HPP__

#include <CL/cl.h>

#include <memory>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>

namespace PpExtension {

class PointPillarsRuntime {
public:
    // model_xml is the network model; kernel_dir contains the OpenCL sources.
    PointPillarsRuntime(const std::string& model_xml, const std::string& kernel_dir);
    ~PointPillarsRuntime();

    PointPillarsRuntime(const PointPillarsRuntime&) = delete;
    PointPillarsRuntime& operator=(const PointPillarsRuntime&) = delete;

    // Run inference and return detection records. The pointer stays valid until
    // the next call; a zero score marks the end of the records.
    const float* forward(const float* points, int point_count);

    // Run inference without copying detections to host memory.
    void forward_on_device(const float* points, int point_count);

    static int max_detections();
    static int detection_stride();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace PpExtension

#endif  // __PP_RUNTIME_HPP__
