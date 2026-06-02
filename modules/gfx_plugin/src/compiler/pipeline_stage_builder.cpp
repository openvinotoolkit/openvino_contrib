// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/pipeline_stage_builder.hpp"

#include <chrono>
#include <string>
#include <unordered_set>
#include <utility>

#include "compiler/backend_config.hpp"
#include "compiler/backend_registry.hpp"
#include "compiler/pipeline_stage_fusion.hpp"
#include "compiler/pipeline_stage_plan.hpp"
#include "compiler/stage_compiler_policy.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/gfx_compile_profiling.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/pipeline_stage_materializer.hpp"

#include "openvino/core/except.hpp"
#include "openvino/op/transpose.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

void apply_pipeline_stage_io_plan(PipelineStageDesc &stage_desc,
                                  compiler::PipelineStageIoPlan stage_plan) {
  static_cast<compiler::PipelineStageIoPlan &>(stage_desc) =
      std::move(stage_plan);
}

} // namespace

PipelineStageBuildResult
build_pipeline_stage_descriptors(const PipelineStageBuildRequest &request) {
  OPENVINO_ASSERT(request.runtime_model,
                  "GFX: pipeline stage builder requires runtime model");
  OPENVINO_ASSERT(request.stage_factory,
                  "GFX: pipeline stage builder requires backend stage factory");
  OPENVINO_ASSERT(request.runtime_descriptor,
                  "GFX: pipeline stage builder requires compiler-owned runtime "
                  "executable descriptor");

  const auto backend_module =
      compiler::BackendRegistry::default_registry().resolve(request.backend);
  OPENVINO_ASSERT(backend_module, "GFX: backend registry has no module for ",
                  backend_to_string(request.backend));
  const auto &backend_capabilities = backend_module->capabilities();
  const auto &fusion_capabilities = backend_capabilities.fusion();
  const auto stage_compiler_policy =
      compiler::make_stage_compiler_policy_from_capabilities(
          backend_capabilities);

  GpuStageRuntimeOptions stage_runtime_options{};
  stage_runtime_options.diagnostic_f32_vendor_image =
      request.diagnostic_f32_vendor_image;
  stage_runtime_options.stage_placement_policy =
      stage_compiler_policy.placement;
  stage_runtime_options.post_op_fusion_capabilities =
      stage_compiler_policy.post_ops;

  gfx_log_info("StageFactory")
      << "Building pipeline for backend=" << request.backend_name
      << " ops=" << request.runtime_model->get_ops().size();
  const auto build_start = request.compile_trace
                               ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
  if (request.compile_trace) {
    request.compile_trace->set_counter(
        "runtime_model_op_count",
        static_cast<uint64_t>(request.runtime_model->get_ops().size()));
  }

  PipelineStageBuildResult result;
  for (size_t i = 0; i < request.runtime_model->inputs().size(); ++i) {
    result.param_index[request.runtime_model->inputs()[i].get_node()] = i;
  }

  const auto model_outputs =
      compiler::collect_model_output_ports(*request.runtime_model);
  const compiler::PipelineStagePlanBuilder stage_plan_builder(model_outputs);

  const auto ordered_ops = request.runtime_model->get_ordered_ops();
  gfx_log_info("StageFactory") << "Ordered ops count=" << ordered_ops.size();
  result.pipeline.reserve(ordered_ops.size());

  const PipelineStageMaterializer stage_materializer(
      *request.stage_factory, ordered_ops, *request.runtime_descriptor,
      stage_runtime_options,
      [&](uint64_t stage_record_key,
          const compiler::PipelineVendorAttentionPlan &plan) {
        return backend_module->materialize_vendor_attention_artifact(
            stage_record_key, plan);
      });

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
      request.enable_fusion && has_unobserved_stage_edges;
  fusion_options.debug_dump_ir = gfx_log_debug_enabled();
  fusion_options.fusion_capabilities = fusion_capabilities;
  fusion_options.stage_compiler_policy = &stage_compiler_policy;
  fusion_options.fusion_contract_for =
      [&](const std::shared_ptr<const ov::Node> &stage_node) {
        return stage_materializer.fusion_contract_for(stage_node);
      };

  const auto fusion_selection = compiler::plan_pipeline_fusions(
      request.runtime_model, ordered_ops, model_outputs, fusion_options);
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
        << (request.enable_fusion
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
            stage_materializer.fusion_contract_for(node))) {
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
        auto stage = plan ? stage_materializer.create_vendor_attention_stage(
                                *plan, final_node)
                          : nullptr;
        if (request.compile_trace) {
          request.compile_trace->increment_counter("stage_create_count");
        }
        if (stage && !group->node_indices.empty()) {
          PipelineStageDesc stage_desc;
          apply_pipeline_stage_io_plan(
              stage_desc, stage_plan_builder.make_fused_stage_plan(
                              final_node, 1,
                              stage_materializer.stage_index_for(final_node)));
          stage_desc.stage = std::move(stage);
          stage_desc.inputs.push_back(stage_plan_builder.remap_input_link(
              fused_output_port_aliases, plan->query.node, plan->query.port));
          stage_desc.inputs.push_back(stage_plan_builder.remap_input_link(
              fused_output_port_aliases, plan->key.node, plan->key.port));
          stage_desc.inputs.push_back(stage_plan_builder.remap_input_link(
              fused_output_port_aliases, plan->value.node, plan->value.port));
          stage_desc.outputs[0].shape = plan->output_shape;
          stage_desc.outputs[0].type = plan->element_type;
          stage_desc.outputs[0].source_node = final_node;
          stage_desc.outputs[0].source_port = 0;
          stage_plan_builder.merge_model_outputs(stage_desc, final_node.get());
          stage_plan_builder.append_output_alias(stage_desc, final_node, 0, 0);
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

          const size_t idx = result.pipeline.size();
          result.pipeline.emplace_back(std::move(stage_desc));
          for (const auto node_idx : group->node_indices) {
            if (node_idx < ordered_ops.size()) {
              fused_indices.insert(node_idx);
              const auto &fused_node = ordered_ops[node_idx];
              result.node_to_stage[fused_node.get()] = idx;
            }
          }
          continue;
        }
      }
      if (group->kind == "Attention" || group->kind == "AttentionScale" ||
          group->kind == "AttentionScaleMask" || group->kind == "NativeSDPA") {
        auto materialized = stage_materializer.create_attention_sequence_stage(
            *group, ordered_ops, stage_plan_builder, fused_output_port_aliases);
        if (materialized) {
          PipelineStageDesc stage_desc;
          apply_pipeline_stage_io_plan(stage_desc,
                                       std::move(materialized->io_plan));
          stage_desc.output_lifetimes =
              std::move(materialized->output_lifetimes);
          stage_desc.stage = std::move(materialized->stage);

          const size_t idx = result.pipeline.size();
          result.pipeline.emplace_back(std::move(stage_desc));
          for (const auto &alias : result.pipeline.back().output_aliases) {
            stage_plan_builder.record_output_alias(
                fused_output_port_aliases, alias.node.get(), alias.source_port,
                alias.output_port);
          }
          for (const auto node_idx : group->node_indices) {
            if (node_idx < ordered_ops.size()) {
              fused_indices.insert(node_idx);
              const auto &fused_node = ordered_ops[node_idx];
              result.node_to_stage[fused_node.get()] = idx;
            }
          }
          if (request.compile_trace) {
            request.compile_trace->increment_counter(
                "stage_create_count", materialized->materialized_stage_count);
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
      gfx_log_debug("StageFactory")
          << "Preparing stage for " << node->get_type_name()
          << " name=" << node->get_friendly_name();
    }
    if (primary_group &&
        compiler::pipeline_fusion_group_has_fp32_precision(*primary_group,
                                                           ordered_ops) &&
        !ov::fp16_compression_is_disabled(node)) {
      ov::disable_fp16_compression(node);
    }
    auto gpu_stage = stage_materializer.create_stage(node);
    if (request.compile_trace) {
      request.compile_trace->increment_counter("stage_create_count");
    }
    OPENVINO_ASSERT(gpu_stage, "GFX: unsupported op in MLIR pipeline: ",
                    node->get_friendly_name(), " (", node->get_type_name(),
                    ")");

    PipelineStageDesc stage_desc;
    apply_pipeline_stage_io_plan(
        stage_desc, stage_plan_builder.make_stage_plan(
                        node, stage_materializer.stage_index_for(node)));
    stage_desc.stage = std::move(gpu_stage);
    const auto residual_it = rms_residual_adds.find(node.get());
    if (residual_it != rms_residual_adds.end() && residual_it->second &&
        stage_desc.stage->fuse_residual_add()) {
      const auto &add = residual_it->second;
      stage_desc.inputs.push_back(stage_plan_builder.remap_input_link(
          fused_output_port_aliases, add->input_value(0).get_node_shared_ptr(),
          add->input_value(0).get_index()));
      stage_desc.inputs.push_back(stage_plan_builder.remap_input_link(
          fused_output_port_aliases, node->input_value(1).get_node_shared_ptr(),
          node->input_value(1).get_index()));
      stage_desc.inputs.push_back(stage_plan_builder.remap_input_link(
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
            GfxInputTransform transform;
            transform.source_shape = transform_it->second.source_shape;
            transform.transpose_permutation =
                transform_it->second.transpose_permutation;
            stage_desc.stage->set_input_transform(input_idx, transform);
          }
        }
        stage_desc.inputs.push_back(stage_plan_builder.remap_input_link(
            fused_output_port_aliases, linked_node, linked_port));
      }
    }
    if (gfx_log_debug_enabled()) {
      gfx_log_debug("StageFactory")
          << "Created GpuStage for " << node->get_type_name()
          << " name=" << node->get_friendly_name();
    }

    const size_t idx = result.pipeline.size();
    result.node_to_stage[node.get()] = idx;
    if (residual_it != rms_residual_adds.end() && residual_it->second) {
      result.node_to_stage[residual_it->second.get()] = idx;
    }
    result.pipeline.emplace_back(std::move(stage_desc));

    if (primary_group) {
      const auto *group = primary_group;
      auto &stage = result.pipeline[idx];
      auto mark_fused = [&](size_t fused_idx,
                            bool aliases_stage_output) -> bool {
        if (fused_idx >= ordered_ops.size()) {
          return false;
        }
        fused_indices.insert(fused_idx);
        const auto &fused_node = ordered_ops[fused_idx];
        result.node_to_stage[fused_node.get()] = idx;
        if (aliases_stage_output && fused_node->get_output_size() == 1) {
          stage_plan_builder.merge_model_outputs(stage, fused_node.get());
          stage_plan_builder.append_output_alias(stage, fused_node, 0, 0);
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
            stage.node &&
            compiler::pipeline_input_activation_has_exclusive_consumer(
                *group, op_index, ordered_ops, model_outputs);
        if (input_activation_exclusive && group->node_indices.size() > 1 &&
            input_idx < stage.inputs.size() &&
            stage.stage->fuse_input_activation(input_idx,
                                               *group->input_activation,
                                               group->input_activation_alpha)) {
          const size_t act_idx = group->node_indices[1];
          if (act_idx < ordered_ops.size()) {
            const auto &act_node = ordered_ops[act_idx];
            if (stage.inputs[input_idx].node.get() == act_node.get() &&
                act_node->get_input_size() == 1) {
              stage.inputs[input_idx].node =
                  act_node->input_value(0).get_node_shared_ptr();
              stage.inputs[input_idx].port =
                  act_node->input_value(0).get_index();
              input_activation_ok = mark_fused(act_idx, false);
            }
          }
        }
        if (!input_activation_ok && gfx_log_debug_enabled()) {
          gfx_log_debug("Fusion")
              << "Failed input activation fusion for "
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
        if (group->node_indices.size() <= next_post_op_idx ||
            !stage.stage->fuse_batchnorm(*group->batchnorm)) {
          post_ops_ok = false;
        } else {
          fused_post_ops.push_back(group->node_indices[next_post_op_idx]);
          ++next_post_op_idx;
        }
      }
      if (post_ops_ok && group->bias.has_value()) {
        if (group->node_indices.size() <= next_post_op_idx ||
            !stage.stage->fuse_bias(*group->bias)) {
          post_ops_ok = false;
        } else {
          fused_post_ops.push_back(group->node_indices[next_post_op_idx]);
          ++next_post_op_idx;
        }
      }
      bool activation_fused = false;
      if (post_ops_ok && group->activation.has_value() &&
          group->node_indices.size() > next_post_op_idx) {
        if (stage.stage->fuse_activation(*group->activation,
                                         group->activation_alpha)) {
          activation_fused = mark_fused_tail(next_post_op_idx);
          post_ops_ok = activation_fused;
        }
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
        "pipeline_stage_count", static_cast<uint64_t>(result.pipeline.size()));
    request.compile_trace->add_segment(
        "compile", "build_op_pipeline",
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - build_start)
                .count()));
  }
  gfx_log_info("StageFactory")
      << "Built GFX " << request.backend_name << " pipeline with "
      << result.pipeline.size() << " stages";
  return result;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
