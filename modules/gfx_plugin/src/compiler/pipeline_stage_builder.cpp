// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/pipeline_stage_runtime_descriptor_builder_detail.hpp"

#include "compiler/pipeline_stage_materialization_draft.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "compiler/backend_config.hpp"
#include "compiler/backend_registry.hpp"
#include "compiler/stage_compiler_policy.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/gfx_compile_profiling.hpp"
#include "runtime/gfx_logger.hpp"

#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

struct AttentionSequenceStagePlan {
  PipelineStageIoPlan io_plan;
  std::vector<PipelineFusedInnerStagePlan> inner_stages;
};

struct PipelineStageBuildState {
  std::vector<std::shared_ptr<ov::Node>> ordered_ops;
  std::vector<PipelineStageMaterializationPlan> stage_plans;
  StageCompilerPolicy stage_compiler_policy;
  std::unordered_map<const ov::Node *, size_t> param_index;
};

std::optional<AttentionSequenceStagePlan> make_attention_sequence_stage_plan(
    const FusionGroup &group,
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
    const PipelineStagePlanBuilder &stage_plan_builder,
    const PipelineOutputAliasMap &output_aliases,
    const detail::RuntimeStageDescriptorMap &descriptors) {
  const size_t stage_count = group.node_indices.size();
  if (stage_count < 3) {
    return std::nullopt;
  }

  std::unordered_map<const ov::Node *, size_t> fused_stage_index;
  fused_stage_index.reserve(stage_count);
  for (size_t i = 0; i < stage_count; ++i) {
    const auto idx = group.node_indices[i];
    if (idx >= ordered_ops.size()) {
      return std::nullopt;
    }
    fused_stage_index[ordered_ops[idx].get()] = i;
  }

  std::vector<std::vector<size_t>> stage_output_slots(stage_count);
  size_t fused_output_count = 1;
  for (size_t i = 0; i < stage_count; ++i) {
    const size_t idx = group.node_indices[i];
    if (idx >= ordered_ops.size()) {
      return std::nullopt;
    }
    const auto &stage_node = ordered_ops[idx];
    stage_output_slots[i].reserve(stage_node->get_output_size());
    for (size_t port = 0; port < stage_node->get_output_size(); ++port) {
      if (i + 1 == stage_count && port == 0) {
        stage_output_slots[i].push_back(0);
      } else {
        stage_output_slots[i].push_back(fused_output_count++);
      }
    }
  }

  std::unordered_map<PipelineOutputPortKey, size_t, PipelineOutputPortKeyHash>
      external_map;
  std::vector<PipelineStageInputLink> fused_inputs;
  std::vector<PipelineFusedInnerStagePlan> inner_stages;
  inner_stages.reserve(stage_count);
  for (size_t i = 0; i < stage_count; ++i) {
    const size_t idx = group.node_indices[i];
    const auto &stage_node = ordered_ops[idx];
    PipelineFusedInnerStagePlan inner_stage;
    inner_stage.output_indices = stage_output_slots[i];
    inner_stage.inputs.reserve(stage_node->get_input_size());
    for (const auto &iv : stage_node->input_values()) {
      auto src_node = iv.get_node();
      const auto it_stage = fused_stage_index.find(src_node);
      if (it_stage != fused_stage_index.end()) {
        const size_t src_stage = it_stage->second;
        if (iv.get_index() >= stage_output_slots[src_stage].size()) {
          return std::nullopt;
        }
        inner_stage.inputs.push_back(
            {PipelineFusedInputPlan::Kind::Output,
             stage_output_slots[src_stage][iv.get_index()]});
        continue;
      }
      if (ov::as_type_ptr<const ov::op::v0::Constant>(
              iv.get_node_shared_ptr())) {
        inner_stage.inputs.push_back({PipelineFusedInputPlan::Kind::None, 0});
        continue;
      }

      auto remapped = stage_plan_builder.remap_input_link(
          output_aliases, iv.get_node_shared_ptr(), iv.get_index());
      PipelineOutputPortKey key{remapped.node.get(), remapped.port};
      if (external_map.find(key) != external_map.end()) {
        inner_stage.inputs.push_back(
            {PipelineFusedInputPlan::Kind::External, external_map.at(key)});
        continue;
      }
      const size_t ext_idx = fused_inputs.size();
      external_map.emplace(key, ext_idx);
      fused_inputs.push_back(std::move(remapped));
      inner_stage.inputs.push_back(
          {PipelineFusedInputPlan::Kind::External, ext_idx});
    }
    inner_stages.push_back(std::move(inner_stage));
  }

  const auto &final_node = ordered_ops[group.node_indices.back()];
  AttentionSequenceStagePlan result;
  result.io_plan = stage_plan_builder.make_fused_stage_plan(
      final_node, fused_output_count,
      detail::stage_index_for_node(descriptors, final_node));
  result.io_plan.inputs = std::move(fused_inputs);
  result.inner_stages = std::move(inner_stages);

  for (size_t stage_idx = 0; stage_idx < stage_count; ++stage_idx) {
    const size_t node_idx = group.node_indices[stage_idx];
    const auto &out_node = ordered_ops[node_idx];
    for (size_t port = 0; port < out_node->get_output_size(); ++port) {
      const size_t slot = stage_output_slots[stage_idx][port];
      stage_plan_builder.describe_output(result.io_plan, slot, out_node, port);
      stage_plan_builder.append_output_alias(result.io_plan, out_node, port,
                                             slot);
    }
  }

  return result;
}

} // namespace

namespace detail {

RuntimeExecutableDescriptor build_pipeline_stage_runtime_descriptor(
    const PipelineStageBuildRequest &request) {
  OPENVINO_ASSERT(request.graph.valid(),
                  "GFX: pipeline stage builder requires compiler-owned graph "
                  "snapshot");
  OPENVINO_ASSERT(request.runtime_descriptor,
                  "GFX: pipeline stage builder requires compiler-owned runtime "
                  "executable descriptor");

  OPENVINO_ASSERT(
      request.target.backend() != GpuBackend::Unknown,
      "GFX: pipeline stage builder requires concrete BackendTarget");
  OPENVINO_ASSERT(request.runtime_descriptor->target_fingerprint ==
                      request.target.fingerprint(),
                  "GFX: pipeline stage target mismatch: descriptor=",
                  request.runtime_descriptor->target_fingerprint,
                  " request=", request.target.fingerprint());
  OPENVINO_ASSERT(
      request.backend_registry,
      "GFX: pipeline stage builder requires explicit BackendRegistry");
  const auto backend_module = request.backend_registry->resolve(request.target);
  OPENVINO_ASSERT(backend_module, "GFX: backend registry has no module for ",
                  request.target.debug_string());
  const auto &backend_capabilities = backend_module->capabilities();
  const auto &fusion_capabilities = backend_capabilities.fusion();
  const auto stage_compiler_policy =
      compiler::make_stage_compiler_policy_from_capabilities(
          backend_capabilities);

  const std::string backend_name = request.backend_name.empty()
                                       ? request.target.backend_id()
                                       : request.backend_name;
  gfx_log_info("StagePlan") << "Planning pipeline for backend=" << backend_name
                            << " target=" << request.target.debug_string()
                            << " ops=" << request.graph.graph_op_count;
  const auto build_start = request.compile_trace
                               ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
  if (request.compile_trace) {
    request.compile_trace->set_counter(
        "compiler_graph_op_count",
        static_cast<uint64_t>(request.graph.graph_op_count));
  }

  PipelineStageBuildState result;
  result.stage_compiler_policy = stage_compiler_policy;
  result.param_index = request.graph.param_index;

  const auto &model_outputs = request.graph.model_outputs;
  const compiler::PipelineStagePlanBuilder stage_plan_builder(model_outputs);

  result.ordered_ops = request.graph.ordered_ops;
  const auto &ordered_ops = result.ordered_ops;
  const auto descriptors = build_runtime_stage_descriptor_map(
      ordered_ops, *request.runtime_descriptor);
  gfx_log_info("StagePlan") << "Ordered ops count=" << ordered_ops.size();
  result.stage_plans.reserve(ordered_ops.size());

  const bool has_unobserved_stage_edges = [&]() {
    for (const auto &node : ordered_ops) {
      if (!compiler::is_executable_stage_node(node)) {
        continue;
      }
      for (size_t port = 0; port < node->get_output_size(); ++port) {
        if (!compiler::is_model_output_port(model_outputs, node.get(), port) &&
            !node->output(port).get_target_inputs().empty()) {
          return true;
        }
      }
    }
    return false;
  }();

  compiler::PipelineFusionSelectionOptions fusion_options;
  fusion_options.enable_fusion =
      request.graph.fusion_enabled && has_unobserved_stage_edges;
  fusion_options.debug_dump_ir = gfx_log_debug_enabled();
  fusion_options.fusion_capabilities = fusion_capabilities;
  fusion_options.stage_compiler_policy = &stage_compiler_policy;
  fusion_options.fusion_contract_for =
      [&](const std::shared_ptr<const ov::Node> &stage_node) {
        return fusion_contract_for_node(descriptors, stage_node);
      };

  const auto fusion_selection = compiler::plan_pipeline_fusions(
      request.graph.fusion_plan, ordered_ops, model_outputs, fusion_options);
  const auto &planned_fused_indices = fusion_selection.planned_fused_indices;
  const auto &planned_fused_nodes = fusion_selection.planned_fused_nodes;
  const auto &fused_nodes = fusion_selection.fused_nodes;
  auto primary_fusion_group_for = [&](size_t primary_idx) {
    return compiler::primary_fusion_group_for(fusion_selection, primary_idx);
  };

  if (request.compile_trace) {
    request.compile_trace->set_counter(
        "fusion_group_count",
        static_cast<uint64_t>(fusion_selection.fusion_plan.groups.size()));
    if (fusion_selection.precision_sensitive_vendor_allow_count != 0) {
      request.compile_trace->increment_counter(
          "fusion_precision_sensitive_vendor_allow_count",
          static_cast<uint64_t>(
              fusion_selection.precision_sensitive_vendor_allow_count));
    }
    if (fusion_selection.precision_sensitive_arithmetic_skip_count != 0) {
      request.compile_trace->increment_counter(
          "fusion_precision_sensitive_arithmetic_skip_count",
          static_cast<uint64_t>(
              fusion_selection.precision_sensitive_arithmetic_skip_count));
    }
  }
  if (fusion_options.enable_fusion) {
    if (gfx_log_debug_enabled()) {
      for (const auto &group : fusion_selection.fusion_plan.groups) {
        std::string node_list;
        for (size_t i = 0; i < group.node_indices.size(); ++i) {
          const auto node_idx = group.node_indices[i];
          if (node_idx >= ordered_ops.size()) {
            continue;
          }
          const auto &fused_node = ordered_ops[node_idx];
          if (!node_list.empty()) {
            node_list += " | ";
          }
          node_list += "[" + std::to_string(node_idx) + "] " +
                       fused_node->get_friendly_name() + " (" +
                       fused_node->get_type_name() + ")";
        }
        gfx_log_debug("Fusion")
            << "group kind=" << group.kind
            << " size=" << group.node_indices.size() << " nodes=" << node_list;
      }
      gfx_log_debug("Fusion") << "Fusion enabled: groups="
                              << fusion_selection.fusion_plan.groups.size();
    }
  } else if (gfx_log_debug_enabled()) {
    gfx_log_debug("Fusion")
        << (request.graph.fusion_enabled
                ? "Fusion skipped: all stage edges are observable"
                : "Fusion disabled via GFX_ENABLE_FUSION");
  }

  std::unordered_map<const ov::Node *, std::shared_ptr<const ov::Node>>
      rms_residual_adds;
  std::unordered_set<const ov::Node *> rms_residual_add_nodes;
  for (const auto &node : ordered_ops) {
    auto add = compiler::find_rms_residual_add(node, model_outputs,
                                               planned_fused_nodes);
    if (!add) {
      continue;
    }
    if (!compiler::allow_stage_residual_add_fusion(
            fusion_contract_for_node(descriptors, node))) {
      continue;
    }
    rms_residual_adds.emplace(node.get(), add);
    rms_residual_add_nodes.insert(add.get());
  }
  if (request.compile_trace && !rms_residual_adds.empty()) {
    request.compile_trace->set_counter(
        "rms_residual_add_fusion_count",
        static_cast<uint64_t>(rms_residual_adds.size()));
  }
  if (gfx_log_debug_enabled() && !rms_residual_adds.empty()) {
    gfx_log_debug("Fusion")
        << "RMS residual Add fusion candidates=" << rms_residual_adds.size();
  }

  std::unordered_set<size_t> fused_indices;
  fused_indices.reserve(ordered_ops.size());
  compiler::PipelineOutputAliasMap fused_output_port_aliases;

  const auto absorbed_input_transform_plan =
      compiler::plan_absorbed_input_transforms(ordered_ops, model_outputs,
                                               fused_nodes);

  for (size_t op_index = 0; op_index < ordered_ops.size(); ++op_index) {
    const auto &node = ordered_ops[op_index];
    if (!compiler::is_executable_stage_node(node)) {
      continue;
    }

    if (absorbed_input_transform_plan.absorbed_nodes.count(node.get()) != 0) {
      continue;
    }

    if (rms_residual_add_nodes.count(node.get()) != 0) {
      continue;
    }

    if (fused_indices.count(op_index)) {
      continue;
    }
    const auto *primary_group = primary_fusion_group_for(op_index);
    if (planned_fused_indices.count(op_index) && primary_group == nullptr) {
      continue;
    }

    if (primary_group) {
      const auto *group = primary_group;
      if (group->kind == "VendorAttention") {
        auto plan =
            compiler::plan_vendor_attention_subgraph(*group, ordered_ops);
        const auto &final_node = group->node_indices.empty()
                                     ? std::shared_ptr<ov::Node>{}
                                     : ordered_ops[group->node_indices.back()];
        const auto *base_descriptor =
            descriptor_for_node(descriptors, final_node);
        auto artifact =
            plan && base_descriptor
                ? backend_module->materialize_vendor_attention_artifact(
                      base_descriptor->stage_record_key, *plan)
                : PipelineVendorAttentionArtifact{};
        if (plan && artifact.valid() && !group->node_indices.empty()) {
          PipelineStageMaterializationPlan stage_plan;
          stage_plan.kind = PipelineStageMaterializationKind::VendorAttention;
          stage_plan.vendor_attention = *plan;
          stage_plan.vendor_attention_artifact = std::move(artifact);
          stage_plan.fusion_group = *group;
          stage_plan.io_plan = stage_plan_builder.make_fused_stage_plan(
              final_node, 1, stage_index_for_node(descriptors, final_node));
          stage_plan.io_plan.inputs.push_back(
              stage_plan_builder.remap_input_link(fused_output_port_aliases,
                                                  plan->query.node,
                                                  plan->query.port));
          stage_plan.io_plan.inputs.push_back(
              stage_plan_builder.remap_input_link(
                  fused_output_port_aliases, plan->key.node, plan->key.port));
          stage_plan.io_plan.inputs.push_back(
              stage_plan_builder.remap_input_link(fused_output_port_aliases,
                                                  plan->value.node,
                                                  plan->value.port));
          stage_plan.io_plan.outputs[0].shape = plan->output_shape;
          stage_plan.io_plan.outputs[0].type = plan->element_type;
          stage_plan.io_plan.outputs[0].source_node = final_node;
          stage_plan.io_plan.outputs[0].source_port = 0;
          stage_plan_builder.merge_model_outputs(stage_plan.io_plan,
                                                 final_node.get());
          stage_plan_builder.append_output_alias(stage_plan.io_plan, final_node,
                                                 0, 0);
          stage_plan_builder.record_output_alias(fused_output_port_aliases,
                                                 final_node.get(), 0, 0);

          if (request.compile_trace) {
            request.compile_trace->increment_counter("fused_stage_count");
            request.compile_trace->increment_counter(
                "fused_node_count",
                static_cast<uint64_t>(group->node_indices.size()));
            request.compile_trace->increment_counter(
                "vendor_attention_stage_count");
          }

          result.stage_plans.emplace_back(std::move(stage_plan));
          for (const auto node_idx : group->node_indices) {
            if (node_idx < ordered_ops.size()) {
              fused_indices.insert(node_idx);
            }
          }
          continue;
        }
      }
      if (group->kind == "Attention" || group->kind == "AttentionScale" ||
          group->kind == "AttentionScaleMask" || group->kind == "NativeSDPA") {
        auto sequence_plan = make_attention_sequence_stage_plan(
            *group, ordered_ops, stage_plan_builder, fused_output_port_aliases,
            descriptors);
        if (sequence_plan) {
          PipelineStageMaterializationPlan stage_plan;
          stage_plan.kind =
              PipelineStageMaterializationKind::FusedAttentionSequence;
          stage_plan.fusion_group = *group;
          stage_plan.io_plan = std::move(sequence_plan->io_plan);
          stage_plan.fused_inner_stages =
              std::move(sequence_plan->inner_stages);

          result.stage_plans.emplace_back(std::move(stage_plan));
          for (const auto &alias :
               result.stage_plans.back().io_plan.output_aliases) {
            stage_plan_builder.record_output_alias(
                fused_output_port_aliases, alias.node.get(), alias.source_port,
                alias.output_port);
          }
          for (const auto node_idx : group->node_indices) {
            if (node_idx < ordered_ops.size()) {
              fused_indices.insert(node_idx);
            }
          }
          if (request.compile_trace) {
            request.compile_trace->increment_counter("fused_stage_count");
            request.compile_trace->increment_counter(
                "fused_node_count",
                static_cast<uint64_t>(group->node_indices.size()));
          }
          continue;
        }
      }
    }

    if (gfx_log_debug_enabled()) {
      gfx_log_debug("StagePlan")
          << "Planning stage for " << node->get_type_name()
          << " name=" << node->get_friendly_name();
    }
    if (primary_group &&
        compiler::pipeline_fusion_group_has_fp32_precision(*primary_group,
                                                           ordered_ops) &&
        !ov::fp16_compression_is_disabled(node)) {
      ov::disable_fp16_compression(node);
    }

    PipelineStageMaterializationPlan stage_plan;
    stage_plan.kind = PipelineStageMaterializationKind::SingleStage;
    stage_plan.io_plan = stage_plan_builder.make_stage_plan(
        node, stage_index_for_node(descriptors, node));
    const auto residual_it = rms_residual_adds.find(node.get());
    if (residual_it != rms_residual_adds.end() && residual_it->second) {
      stage_plan.residual_add =
          PipelineStageResidualAddFusionPlan{residual_it->second};
      const auto &add = residual_it->second;
      stage_plan.io_plan.inputs.push_back(stage_plan_builder.remap_input_link(
          fused_output_port_aliases, add->input_value(0).get_node_shared_ptr(),
          add->input_value(0).get_index()));
      stage_plan.io_plan.inputs.push_back(stage_plan_builder.remap_input_link(
          fused_output_port_aliases, node->input_value(1).get_node_shared_ptr(),
          node->input_value(1).get_index()));
      stage_plan.io_plan.inputs.push_back(stage_plan_builder.remap_input_link(
          fused_output_port_aliases, add->input_value(1).get_node_shared_ptr(),
          add->input_value(1).get_index()));
    } else {
      const auto absorbed_it =
          absorbed_input_transform_plan.input_transforms.find(node.get());
      for (size_t input_idx = 0; input_idx < node->get_input_size();
           ++input_idx) {
        const auto &iv = node->input_value(input_idx);
        auto linked_node = iv.get_node_shared_ptr();
        size_t linked_port = iv.get_index();
        if (absorbed_it !=
            absorbed_input_transform_plan.input_transforms.end()) {
          auto transform_it = absorbed_it->second.find(input_idx);
          if (transform_it != absorbed_it->second.end()) {
            auto transpose =
                ov::as_type_ptr<const ov::op::v1::Transpose>(linked_node);
            OPENVINO_ASSERT(
                transpose,
                "GFX: absorbed transpose input is not a transpose for ",
                node->get_friendly_name());
            linked_node = transpose->input_value(0).get_node_shared_ptr();
            linked_port = transpose->input_value(0).get_index();
            stage_plan.input_transforms.push_back(
                PipelineStageInputTransformBinding{input_idx,
                                                   transform_it->second});
          }
        }
        stage_plan.io_plan.inputs.push_back(stage_plan_builder.remap_input_link(
            fused_output_port_aliases, linked_node, linked_port));
      }
    }

    const size_t idx = result.stage_plans.size();
    result.stage_plans.emplace_back(std::move(stage_plan));

    if (primary_group) {
      const auto *group = primary_group;
      auto &planned_stage = result.stage_plans[idx];
      auto &io_plan = planned_stage.io_plan;
      auto mark_fused = [&](size_t fused_idx,
                            bool aliases_stage_output) -> bool {
        if (fused_idx >= ordered_ops.size()) {
          return false;
        }
        fused_indices.insert(fused_idx);
        const auto &fused_node = ordered_ops[fused_idx];
        if (aliases_stage_output && fused_node->get_output_size() == 1) {
          stage_plan_builder.merge_model_outputs(io_plan, fused_node.get());
          stage_plan_builder.append_output_alias(io_plan, fused_node, 0, 0);
        }
        return true;
      };

      auto mark_fused_tail = [&](size_t start_idx) -> bool {
        for (size_t i = start_idx; i < group->node_indices.size(); ++i) {
          const bool aliases_stage_output =
              (i + 1 == group->node_indices.size());
          if (!mark_fused(group->node_indices[i], aliases_stage_output)) {
            return false;
          }
        }
        return true;
      };
      if (group->input_activation.has_value()) {
        const size_t input_idx = group->input_activation_input;
        bool input_activation_ok = false;
        const bool input_activation_exclusive =
            io_plan.node &&
            compiler::pipeline_input_activation_has_exclusive_consumer(
                *group, op_index, ordered_ops, model_outputs);
        if (input_activation_exclusive && group->node_indices.size() > 1 &&
            input_idx < io_plan.inputs.size()) {
          const size_t act_idx = group->node_indices[1];
          if (act_idx < ordered_ops.size()) {
            const auto &act_node = ordered_ops[act_idx];
            if (io_plan.inputs[input_idx].node.get() == act_node.get() &&
                act_node->get_input_size() == 1) {
              io_plan.inputs[input_idx].node =
                  act_node->input_value(0).get_node_shared_ptr();
              io_plan.inputs[input_idx].port =
                  act_node->input_value(0).get_index();
              planned_stage.post_ops.input_activation =
                  PipelineStageInputActivationFusionPlan{
                      input_idx, *group->input_activation,
                      group->input_activation_alpha};
              input_activation_ok = mark_fused(act_idx, false);
            }
          }
        }
        if (!input_activation_ok && gfx_log_debug_enabled()) {
          gfx_log_debug("Fusion")
              << "Failed input activation fusion planning for "
              << (node ? node->get_friendly_name() : std::string("<null>"));
        }
      }
      size_t next_post_op_idx = 1;
      bool post_ops_ok = true;
      if (compiler::pipeline_fusion_requires_batchnorm_payload(*group) &&
          !group->batchnorm.has_value()) {
        post_ops_ok = false;
      }
      if (compiler::pipeline_fusion_requires_bias_payload(*group) &&
          !group->bias.has_value()) {
        post_ops_ok = false;
      }
      std::vector<size_t> fused_post_ops;
      if (group->batchnorm.has_value()) {
        if (group->node_indices.size() <= next_post_op_idx) {
          post_ops_ok = false;
        } else {
          planned_stage.post_ops.batchnorm = *group->batchnorm;
          fused_post_ops.push_back(group->node_indices[next_post_op_idx]);
          ++next_post_op_idx;
        }
      }
      if (post_ops_ok && group->bias.has_value()) {
        if (group->node_indices.size() <= next_post_op_idx) {
          post_ops_ok = false;
        } else {
          planned_stage.post_ops.bias = *group->bias;
          fused_post_ops.push_back(group->node_indices[next_post_op_idx]);
          ++next_post_op_idx;
        }
      }
      bool activation_fused = false;
      if (post_ops_ok && group->activation.has_value() &&
          group->node_indices.size() > next_post_op_idx) {
        planned_stage.post_ops.activation = *group->activation;
        planned_stage.post_ops.activation_alpha = group->activation_alpha;
        activation_fused = mark_fused_tail(next_post_op_idx);
        post_ops_ok = activation_fused;
      }
      if (!post_ops_ok) {
        planned_stage.post_ops.batchnorm.reset();
        planned_stage.post_ops.bias.reset();
        planned_stage.post_ops.activation.reset();
      }
      for (size_t i = 0; i < fused_post_ops.size(); ++i) {
        const bool aliases_stage_output =
            !activation_fused && (i + 1 == fused_post_ops.size());
        post_ops_ok =
            mark_fused(fused_post_ops[i], aliases_stage_output) && post_ops_ok;
      }
    }
  }

  if (request.compile_trace) {
    request.compile_trace->set_counter(
        "pipeline_stage_count",
        static_cast<uint64_t>(result.stage_plans.size()));
    request.compile_trace->add_segment(
        "compile", "build_pipeline_stage_runtime_descriptor",
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - build_start)
                .count()));
  }
  gfx_log_info("StagePlan")
      << "Planned GFX " << backend_name << " pipeline with "
      << result.stage_plans.size() << " stages";
  auto materialization_draft = make_runtime_descriptor_materialization_draft(
      result.ordered_ops, result.stage_plans, request.graph.public_outputs,
      descriptors, result.stage_compiler_policy, result.param_index,
      request.runtime_descriptor);
  RuntimeExecutableDescriptor runtime_descriptor = *request.runtime_descriptor;
  runtime_descriptor.materialization_finalized = true;
  runtime_descriptor.materialization_stages =
      std::move(materialization_draft.stage_plans);
  runtime_descriptor.public_outputs =
      make_runtime_public_output_descriptors(materialization_draft);
  runtime_descriptor.runtime_options =
      std::move(materialization_draft.runtime_options);
  return runtime_descriptor;
}

} // namespace detail

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
