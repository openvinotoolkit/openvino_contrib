// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "compiler/pipeline_stage_builder.hpp"
#include "compiler/pipeline_stage_graph_snapshot.hpp"
#include "compiler/stage_compiler_policy.hpp"
#include "runtime/executable_descriptor.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace detail {

using RuntimeStageDescriptorMap =
    std::unordered_map<const ov::Node *,
                       const RuntimeStageExecutableDescriptor *>;

struct RuntimeDescriptorMaterializationDraft {
  std::vector<::ov::gfx_plugin::PipelineStageMaterializationPlan> stage_plans;
  std::vector<::ov::gfx_plugin::PipelineStagePublicOutputDesc> public_outputs;
  ::ov::gfx_plugin::PipelineStageRuntimeOptionsPlan runtime_options;
};

RuntimeStageDescriptorMap build_runtime_stage_descriptor_map(
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
    const RuntimeExecutableDescriptor &runtime_descriptor);

const RuntimeStageExecutableDescriptor *
descriptor_for_node(const RuntimeStageDescriptorMap &descriptors,
                    const std::shared_ptr<const ov::Node> &node);

size_t stage_index_for_node(const RuntimeStageDescriptorMap &descriptors,
                            const std::shared_ptr<const ov::Node> &node);

PipelineStageFusionContract
fusion_contract_for_node(const RuntimeStageDescriptorMap &descriptors,
                         const std::shared_ptr<const ov::Node> &node);

RuntimeDescriptorMaterializationDraft
make_runtime_descriptor_materialization_draft(
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
    const std::vector<PipelineStageMaterializationPlan> &stage_plans,
    const std::vector<PipelineStagePublicOutputSource> &public_outputs,
    const RuntimeStageDescriptorMap &descriptors,
    const StageCompilerPolicy &stage_compiler_policy,
    const std::unordered_map<const ov::Node *, size_t> &param_index,
    const ::ov::gfx_plugin::RuntimeExecutableDescriptor *runtime_descriptor);

std::vector<RuntimePublicOutputDescriptor>
make_runtime_public_output_descriptors(
    const RuntimeDescriptorMaterializationDraft &draft);

} // namespace detail
} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
