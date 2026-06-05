// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/runtime/tensor.hpp"
#include "runtime/gpu_buffer.hpp"
#include "runtime/gpu_stage.hpp"
#include "runtime/gpu_tensor.hpp"
#include "runtime/pipeline_stage_desc.hpp"
#include "runtime/runtime_session.hpp"

namespace ov {
namespace gfx_plugin {

struct InferStage {
    std::shared_ptr<const ov::Node> node;
    std::unique_ptr<GpuStage> stage;
    std::shared_ptr<RuntimeSession> runtime_session;
    size_t runtime_stage_index = PipelineStageDesc::npos;
    std::shared_ptr<const RuntimeStageExecutableDescriptor> runtime_stage_descriptor;
    std::unique_ptr<PreparedKernelExecutable> prepared_executable;
    std::vector<std::unique_ptr<GpuTensor>> outputs;
    std::vector<bool> output_is_model_output;
    std::vector<PipelineStageDesc::InputLink> output_sources;
    std::vector<std::string> direct_stateful_assign_variable_ids;
    std::vector<PipelineStageDesc::InputLink> inputs;
    std::vector<PipelineStageDesc::OutputAlias> output_aliases;
    std::vector<PipelineStageDesc::OutputLifetime> output_lifetimes;
};

struct StageOutputBufferWorkspace {
    static constexpr size_t npos = std::numeric_limits<size_t>::max();

    std::vector<BufferHandle> handles;
    std::vector<std::vector<size_t>> output_slots;
    size_t last_workspace_outputs = 0;
    size_t last_direct_outputs = 0;
    size_t last_slots_used = 0;
    size_t last_peak_live_slots = 0;
};

enum class PreparedStageInputKind {
    None,
    Parameter,
    StageOutput,
};

struct PreparedStageInputRef {
    PreparedStageInputKind kind = PreparedStageInputKind::None;
    size_t index = 0;
    size_t port = 0;
};

struct PreparedStageExecution {
    std::vector<PreparedStageInputRef> input_refs;
    std::vector<GpuTensor*> resolved_inputs;
};

struct PreparedInferExecutionPlan {
    std::vector<PreparedStageExecution> stages;
};

struct OutputSource {
    std::shared_ptr<const ov::Node> node;
    size_t port = 0;
};

enum class PreparedOutputSourceKind {
    None,
    Parameter,
    StageOutput,
};

struct PreparedOutputBinding {
    PreparedOutputSourceKind kind = PreparedOutputSourceKind::None;
    size_t index = 0;
    size_t port = 0;
    OutputSource source;
    ov::Shape static_shape;
    ov::element::Type static_type = ov::element::dynamic;
};

struct PreparedInferOutputPlan {
    std::vector<PreparedOutputBinding> outputs;
};

struct OutputViewInfo {
    OutputSource source;
    ov::Shape shape;
    ov::element::Type type = ov::element::dynamic;
};

struct PreparedHostOutputBinding {
    ov::Shape shape;
    ov::element::Type type = ov::element::dynamic;
    ov::Tensor host;
};

struct PreparedInferHostOutputPlan {
    std::vector<PreparedHostOutputBinding> outputs;
};

}  // namespace gfx_plugin
}  // namespace ov
