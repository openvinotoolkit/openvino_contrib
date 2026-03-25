// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/tensor.hpp"

#include "openvino/gfx_plugin/profiling.hpp"
#include "plugin/infer_io_utils.hpp"
#include "plugin/infer_pipeline.hpp"
#include "runtime/gfx_profiler.hpp"
#include "runtime/gpu_buffer.hpp"

namespace ov {
namespace gfx_plugin {

class GfxRemoteTensor;

struct BackendInferState {
    virtual ~BackendInferState() = default;
    std::unique_ptr<GfxProfiler> profiler;
    std::vector<InferStage> reusable_pipeline;
    PreparedInferExecutionPlan reusable_execution_plan;
    PreparedInferOutputPlan reusable_output_plan;
    PreparedInferHostOutputPlan reusable_host_output_plan;
    std::vector<std::vector<BufferHandle>> stage_output_handles;
    std::vector<BufferHandle> input_handles;
    std::vector<BufferHandle> input_staging_handles;
    std::vector<BufferHandle> output_staging_handles;
};

struct InferRequestState {
    std::unique_ptr<BackendInferState> backend;
    std::vector<ov::Tensor> bound_inputs;
    std::vector<std::shared_ptr<GfxRemoteTensor>> bound_remote_inputs;
    std::vector<ov::Tensor> bound_output_hosts;
    std::vector<std::shared_ptr<GfxRemoteTensor>> bound_remote_outputs;

    std::vector<std::pair<std::string, ov::Tensor>> debug_tensors;
    std::vector<GpuBuffer> debug_buffers;

    std::vector<ov::ProfilingInfo> last_profiling;
    GfxProfilerConfig profiler_cfg{};
};

}  // namespace gfx_plugin
}  // namespace ov
