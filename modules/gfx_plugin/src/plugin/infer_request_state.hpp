// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/runtime/tensor.hpp"

#include "openvino/gfx_plugin/profiling.hpp"
#include "plugin/infer_io_utils.hpp"
#include "runtime/backend_request_state.hpp"
#include "runtime/gpu_buffer.hpp"
#include "runtime/stateful_variable_state.hpp"

namespace ov {
namespace gfx_plugin {

class GfxRemoteTensor;

struct InferRequestState {
    BackendRequestState runtime;
    std::vector<ov::Tensor> bound_inputs;
    std::vector<std::shared_ptr<GfxRemoteTensor>> bound_remote_inputs;
    std::vector<ov::Tensor> bound_output_hosts;
    std::vector<std::shared_ptr<GfxRemoteTensor>> bound_remote_outputs;
    StatefulVariableStateMap variable_states;
    std::vector<ov::SoPtr<ov::IVariableState>> variable_state_objects;

    std::vector<std::pair<std::string, ov::Tensor>> debug_tensors;
    std::vector<GpuBuffer> debug_buffers;

    std::vector<ov::ProfilingInfo> last_profiling;
    GfxProfilerConfig profiler_cfg{};
};

}  // namespace gfx_plugin
}  // namespace ov
