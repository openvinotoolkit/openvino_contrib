// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/gfx_plugin/compiled_model.hpp"

#include "openvino/gfx_plugin/infer_request.hpp"
#include "openvino/gfx_plugin/plugin.hpp"
#include "openvino/gfx_plugin/profiling.hpp"
#include "openvino/gfx_plugin/properties.hpp"
#include "plugin/backend_factory.hpp"
#include "plugin/backend_state.hpp"
#include "plugin/compiled_model_backend_resources.hpp"
#include "plugin/gfx_backend_config.hpp"
#include "plugin/gfx_profiling_utils.hpp"
#include "plugin/gfx_property_lists.hpp"
#include "plugin/gfx_property_utils.hpp"
#include "plugin/model_serialization.hpp"
#include "runtime/fused_sequence_stage.hpp"
#include "runtime/gfx_backend_utils.hpp"
#include "runtime/gfx_compile_profiling.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_precision.hpp"
#include "runtime/gfx_profiling_report.hpp"
#include "runtime/gfx_remote_context.hpp"
#include "runtime/gfx_stage_policy.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

#include "transforms/fusion_pass.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace ov {
namespace gfx_plugin {

namespace {

std::vector<int64_t> evaluate_constant_i64(const ov::Output<ov::Node> &value) {
  auto constant =
      ov::as_type_ptr<const ov::op::v0::Constant>(value.get_node_shared_ptr());
  OPENVINO_ASSERT(constant, "GFX: expected constant input");
  return constant->cast_vector<int64_t>();
}

bool read_const_f32_values(
    const std::shared_ptr<const ov::op::v0::Constant> &constant,
    std::vector<float> &values) {
  if (!constant) {
    return false;
  }
  const auto count = ov::shape_size(constant->get_shape());
  if (count == 0) {
    return false;
  }
  values.resize(count);
  if (constant->get_element_type() == ov::element::f32) {
    const auto *src = constant->get_data_ptr<float>();
    std::copy(src, src + count, values.begin());
    return true;
  }
  if (constant->get_element_type() == ov::element::f16) {
    const auto *src = constant->get_data_ptr<ov::float16>();
    for (size_t i = 0; i < count; ++i) {
      values[i] = static_cast<float>(src[i]);
    }
    return true;
  }
  return false;
}

bool read_uniform_scale_from_multiply(
    const std::shared_ptr<const ov::op::v1::Multiply> &multiply,
    const std::shared_ptr<const ov::Node> &producer, float &scale) {
  if (!multiply || !producer) {
    return false;
  }
  ov::Output<const ov::Node> scale_value;
  if (multiply->input_value(0).get_node_shared_ptr() == producer) {
    scale_value = multiply->input_value(1);
  } else if (multiply->input_value(1).get_node_shared_ptr() == producer) {
    scale_value = multiply->input_value(0);
  } else {
    return false;
  }
  auto constant =
      ov::as_type_ptr<const ov::op::v0::Constant>(scale_value.get_node_shared_ptr());
  std::vector<float> values;
  if (!read_const_f32_values(constant, values)) {
    return false;
  }
  scale = values.front();
  return std::all_of(values.begin(), values.end(),
                     [&](float value) { return value == scale; });
}

bool extract_scaled_tensor_input(
    const std::shared_ptr<const ov::op::v1::Multiply> &multiply,
    ov::Output<const ov::Node> &tensor, float &scale) {
  if (!multiply || multiply->get_input_size() != 2) {
    return false;
  }
  auto const0 = ov::util::get_constant_from_source(multiply->input_value(0));
  auto const1 = ov::util::get_constant_from_source(multiply->input_value(1));
  ov::Output<const ov::Node> tensor_candidate;
  std::shared_ptr<const ov::op::v0::Constant> scale_const;
  if (const0 && !const1) {
    tensor_candidate = multiply->input_value(1);
    scale_const = const0;
  } else if (const1 && !const0) {
    tensor_candidate = multiply->input_value(0);
    scale_const = const1;
  } else {
    return false;
  }

  std::vector<float> values;
  if (!read_const_f32_values(scale_const, values)) {
    return false;
  }
  scale = values.front();
  if (!std::all_of(values.begin(), values.end(),
                   [&](float value) { return value == scale; })) {
    return false;
  }
  tensor = tensor_candidate;
  return true;
}

bool is_supported_softmax_node(const std::shared_ptr<const ov::Node> &node) {
  return static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Softmax>(node)) ||
         static_cast<bool>(ov::as_type_ptr<const ov::op::v8::Softmax>(node));
}

struct VendorAttentionSubgraphPlan {
  VendorAttentionStageSpec spec;
  ov::Output<const ov::Node> query;
  ov::Output<const ov::Node> key;
  ov::Output<const ov::Node> value;
};

std::optional<VendorAttentionSubgraphPlan> make_vendor_attention_subgraph_plan(
    const FusionGroup &group,
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops) {
  if (group.node_indices.size() != 4) {
    return std::nullopt;
  }
  for (auto idx : group.node_indices) {
    if (idx >= ordered_ops.size()) {
      return std::nullopt;
    }
  }
  const auto first = ordered_ops[group.node_indices[0]];
  const auto second = ordered_ops[group.node_indices[1]];
  auto softmax = ordered_ops[group.node_indices[2]];
  auto matmul2 =
      ov::as_type_ptr<const ov::op::v0::MatMul>(ordered_ops[group.node_indices[3]]);
  auto matmul1 = ov::as_type_ptr<const ov::op::v0::MatMul>(first);
  auto scale = ov::as_type_ptr<const ov::op::v1::Multiply>(second);
  bool pre_scaled_key = false;
  if (!matmul1 || !scale) {
    scale = ov::as_type_ptr<const ov::op::v1::Multiply>(first);
    matmul1 = ov::as_type_ptr<const ov::op::v0::MatMul>(second);
    pre_scaled_key = static_cast<bool>(scale && matmul1);
  }
  if (!matmul1 || !scale || !is_supported_softmax_node(softmax) || !matmul2 ||
      !matmul1->get_transpose_a() || matmul1->get_transpose_b() ||
      matmul2->get_transpose_a() || !matmul2->get_transpose_b()) {
    return std::nullopt;
  }

  const auto q = matmul1->input_value(0);
  ov::Output<const ov::Node> k;
  ov::Output<const ov::Node> value;
  if (matmul2->input_value(0).get_node() == softmax.get()) {
    value = matmul2->input_value(1);
  } else if (matmul2->input_value(1).get_node() == softmax.get()) {
    value = matmul2->input_value(0);
  } else {
    return std::nullopt;
  }
  float attention_scale = 1.0f;
  if (pre_scaled_key) {
    if (softmax->input_value(0).get_node() != matmul1.get() ||
        matmul1->input_value(1).get_node() != scale.get() ||
        !extract_scaled_tensor_input(scale, k, attention_scale)) {
      return std::nullopt;
    }
  } else {
    if (softmax->input_value(0).get_node() != scale.get() ||
        !read_uniform_scale_from_multiply(scale, matmul1->shared_from_this(),
                                          attention_scale)) {
      return std::nullopt;
    }
    k = matmul1->input_value(1);
  }

  if (!q.get_partial_shape().is_static() || !k.get_partial_shape().is_static() ||
      !value.get_partial_shape().is_static() ||
      !matmul2->get_output_partial_shape(0).is_static()) {
    return std::nullopt;
  }

  VendorAttentionStageSpec spec;
  spec.name = matmul2->get_friendly_name();
  spec.element_type = q.get_element_type();
  spec.query_shape = q.get_shape();
  spec.key_shape = k.get_shape();
  spec.value_shape = value.get_shape();
  spec.output_shape = matmul2->get_output_shape(0);
  spec.scale = attention_scale;
  if (spec.element_type != ov::element::f32 &&
      spec.element_type != ov::element::f16) {
    return std::nullopt;
  }
  if (k.get_element_type() != spec.element_type ||
      value.get_element_type() != spec.element_type ||
      matmul2->get_output_element_type(0) != spec.element_type) {
    return std::nullopt;
  }
  if (spec.query_shape.size() != 4 || spec.key_shape.size() != 4 ||
      spec.value_shape.size() != 4 || spec.output_shape.size() != 4) {
    return std::nullopt;
  }
  if (spec.query_shape[0] != spec.key_shape[0] ||
      spec.query_shape[0] != spec.value_shape[0] ||
      spec.query_shape[1] != spec.key_shape[1] ||
      spec.query_shape[1] != spec.value_shape[1] ||
      spec.query_shape[2] != spec.key_shape[2] ||
      spec.key_shape[3] != spec.value_shape[3] ||
      spec.output_shape[0] != spec.query_shape[0] ||
      spec.output_shape[1] != spec.query_shape[1] ||
      spec.output_shape[2] != spec.value_shape[2] ||
      spec.output_shape[3] != spec.query_shape[3]) {
    return std::nullopt;
  }
  VendorAttentionSubgraphPlan plan;
  plan.spec = std::move(spec);
  plan.query = q;
  plan.key = k;
  plan.value = value;
  return plan;
}

bool is_supported_absorbing_consumer(
    const std::shared_ptr<const ov::Node> &node) {
  return ov::is_type<ov::op::v1::Add>(node.get()) ||
         ov::is_type<ov::op::v1::GroupConvolution>(node.get());
}

bool is_supported_absorbing_input(const std::shared_ptr<const ov::Node> &node,
                                  size_t input_idx) {
  if (ov::is_type<ov::op::v1::Add>(node.get())) {
    return input_idx < 2;
  }
  if (ov::is_type<ov::op::v1::GroupConvolution>(node.get())) {
    return input_idx == 0;
  }
  return false;
}

bool is_absorbable_transpose_candidate(
    const std::shared_ptr<const ov::Node> &node,
    const std::unordered_map<const ov::Node *, std::vector<bool>>
        &model_outputs,
    const std::unordered_set<const ov::Node *> &fused_nodes) {
  auto transpose = ov::as_type_ptr<const ov::op::v1::Transpose>(node);
  if (!transpose || transpose->get_output_size() != 1) {
    return false;
  }
  if (fused_nodes.count(node.get()) != 0) {
    return false;
  }
  if (auto it = model_outputs.find(node.get()); it != model_outputs.end()) {
    if (std::any_of(it->second.begin(), it->second.end(),
                    [](bool value) { return value; })) {
      return false;
    }
  }
  if (!transpose->get_input_partial_shape(0).is_static() ||
      !transpose->get_output_partial_shape(0).is_static()) {
    return false;
  }
  auto source = transpose->input_value(0).get_node_shared_ptr();
  if (!source || ov::is_type<ov::op::v0::Constant>(source.get())) {
    return false;
  }
  return true;
}

bool is_model_output_port(
    const std::unordered_map<const ov::Node *, std::vector<bool>>
        &model_outputs,
    const ov::Node *node, size_t port) {
  auto it = model_outputs.find(node);
  if (it == model_outputs.end() || port >= it->second.size()) {
    return false;
  }
  return it->second[port];
}

bool is_executable_stage_node(const std::shared_ptr<ov::Node> &node) {
  if (!node) {
    return false;
  }
  return !ov::as_type_ptr<ov::op::v0::Parameter>(node) &&
         !ov::as_type_ptr<ov::op::v0::Result>(node) &&
         !ov::as_type_ptr<ov::op::v0::Constant>(node) &&
         !ov::is_decompression(node);
}

bool shape_matches_without_broadcast(const ov::PartialShape &input,
                                     const ov::PartialShape &output) {
  if (input.rank().is_dynamic() || output.rank().is_dynamic() ||
      input.rank().get_length() != output.rank().get_length()) {
    return false;
  }
  const auto rank = static_cast<size_t>(input.rank().get_length());
  for (size_t i = 0; i < rank; ++i) {
    if (input[i].is_static() && output[i].is_static() &&
        input[i].get_length() != output[i].get_length()) {
      return false;
    }
  }
  return true;
}

std::shared_ptr<const ov::op::v1::Add> residual_add_for_rms(
    const std::shared_ptr<const ov::Node> &rms,
    const std::unordered_map<const ov::Node *, std::vector<bool>>
        &model_outputs,
    const std::unordered_set<const ov::Node *> &fused_nodes) {
  if (!rms || rms->get_type_name() != std::string("RMS") ||
      rms->get_input_size() != 2 || rms->get_output_size() != 1) {
    return nullptr;
  }
  auto add = ov::as_type_ptr<const ov::op::v1::Add>(
      rms->input_value(0).get_node_shared_ptr());
  if (!add || add->get_output_size() != 1 || add->get_input_size() != 2 ||
      add->output(0).get_target_inputs().size() != 1 ||
      is_model_output_port(model_outputs, add.get(), 0) ||
      fused_nodes.count(add.get()) != 0) {
    return nullptr;
  }
  const auto out_shape = add->get_output_partial_shape(0);
  if (!shape_matches_without_broadcast(add->get_input_partial_shape(0),
                                       out_shape) ||
      !shape_matches_without_broadcast(add->get_input_partial_shape(1),
                                       out_shape)) {
    return nullptr;
  }
  if (!shape_matches_without_broadcast(out_shape,
                                       rms->get_input_partial_shape(0))) {
    return nullptr;
  }
  return add;
}

} // namespace

CompiledModel::CompiledModel(
    const std::shared_ptr<const ov::Model> &model,
    const std::shared_ptr<const ov::IPlugin> &plugin,
    const std::shared_ptr<const ov::Model> &original_model,
    const ov::AnyMap &properties, const ov::SoPtr<ov::IRemoteContext> &context)
    : ov::ICompiledModel(model, plugin, context), m_runtime_model(model),
      m_original_model(original_model ? original_model : model) {
  // GFX exposes f16 as the performance preference, while codegen preserves
  // f32 arithmetic for declared f32 and fp32-sensitive stages.
  if (auto it = properties.find(ov::hint::inference_precision.name());
      it != properties.end()) {
    m_inference_precision =
        parse_inference_precision_property(it->second,
                                           ov::hint::inference_precision.name());
  } else {
    m_inference_precision = gfx_default_inference_precision();
  }
  if (auto it = properties.find(ov::enable_profiling.name());
      it != properties.end()) {
    m_enable_profiling =
        parse_bool_property(it->second, ov::enable_profiling.name());
  } else {
    // Honour legacy PERF_COUNT=true if provided under a different key.
    if (auto it2 = properties.find("PERF_COUNT"); it2 != properties.end()) {
      m_enable_profiling = parse_bool_property(it2->second, "PERF_COUNT");
    }
  }
  if (auto it = properties.find(kGfxEnableFusionProperty);
      it != properties.end()) {
    m_enable_fusion = parse_bool_property(it->second, kGfxEnableFusionProperty);
  }
  for (const auto &kv : properties) {
    if (is_diagnostic_f32_vendor_image_property(kv.first)) {
      m_diagnostic_f32_vendor_image = parse_bool_property(kv.second, kv.first);
      break;
    }
  }
  if (auto it = properties.find(kGfxProfilingLevelProperty);
      it != properties.end()) {
    m_profiling_level = parse_profiling_level(it->second);
    m_profiling_level_set = true;
  }
  if (auto it = properties.find(ov::loaded_from_cache.name());
      it != properties.end()) {
    m_loaded_from_cache = it->second.as<bool>();
  }
  ov::AnyMap resolved_props = properties;
  const auto request = get_backend_request(resolved_props);
  auto resolved = resolve_backend_for_properties(
      resolved_props, /*log_fallback=*/true, "CompiledModel");
  if (context) {
    auto gfx_ctx = std::dynamic_pointer_cast<GfxRemoteContext>(context._ptr);
    OPENVINO_ASSERT(gfx_ctx, "GFX: remote context type mismatch");
    const auto ctx_backend = gfx_ctx->backend();
    if (request.explicit_request && request.kind != ctx_backend) {
      OPENVINO_THROW("GFX: backend mismatch between properties (",
                     backend_to_string(request.kind), ") and remote context (",
                     backend_to_string(ctx_backend), ")");
    }
    resolved.backend = ctx_backend;
    resolved.backend_name = backend_to_string(ctx_backend);
    resolved_props[kGfxBackendProperty] = resolved.backend_name;
  }
  m_backend = resolved.backend;
  m_backend_name = resolved.backend_name;
  const bool capture_compile_profile =
      m_enable_profiling && profiling_level() != ProfilingLevel::Off;
  GfxProfilingTrace compile_trace;
  GfxProfilingTrace *compile_trace_ptr =
      capture_compile_profile ? &compile_trace : nullptr;
  const auto compile_wall_start = capture_compile_profile
                                      ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
  if (compile_trace_ptr) {
    compile_trace_ptr->reset(profiling_level());
    compile_trace_ptr->set_backend(m_backend_name);
    compile_trace_ptr->set_counter_capability(false, false);
    compile_trace_ptr->set_counter("loaded_from_cache",
                                   m_loaded_from_cache ? 1 : 0);
  }

  // Preserve user properties; store inference_precision as ov::element::Type.
  for (const auto &kv : resolved_props) {
    if (kv.first == ov::hint::inference_precision.name()) {
      m_config[kv.first] = m_inference_precision;
    } else if (kv.first == ov::enable_profiling.name() ||
               kv.first == "PERF_COUNT") {
      m_config[kv.first] = m_enable_profiling;
    } else if (kv.first == kGfxProfilingLevelProperty) {
      m_config[kv.first] = kv.second;
    } else if (kv.first == kGfxEnableFusionProperty) {
      m_config[kv.first] = m_enable_fusion;
    } else if (is_diagnostic_f32_vendor_image_property(kv.first)) {
      m_config[kv.first] = m_diagnostic_f32_vendor_image;
    } else {
      m_config[kv.first] = kv.second;
    }
  }
  m_config[kGfxBackendProperty] = m_backend_name;
  gfx_log_info("CompiledModel")
      << "Creating backend state for " << m_backend_name;
  const auto backend_state_start =
      compile_trace_ptr ? std::chrono::steady_clock::now()
                        : std::chrono::steady_clock::time_point{};
  m_backend_state = create_backend_state(m_backend, properties, context);
  if (compile_trace_ptr) {
    compile_trace_ptr->increment_counter("backend_state_create_count");
    compile_trace_ptr->add_segment(
        "compile", "create_backend_state",
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - backend_state_start)
                .count()));
  }
  gfx_log_info("CompiledModel") << "Backend state created";
  // Build GpuStage pipeline eagerly; fail early if unsupported ops encountered.
  gfx_log_info("CompiledModel") << "Building stage pipeline";
  build_op_pipeline(compile_trace_ptr);
  if (compile_trace_ptr) {
    uint64_t total_cpu_us = 0;
    for (const auto &segment : compile_trace_ptr->report().segments) {
      total_cpu_us += segment.cpu_us;
    }
    compile_trace_ptr->set_total_cpu_us(total_cpu_us);
    compile_trace_ptr->set_total_gpu_us(0);
    compile_trace_ptr->set_total_wall_us(static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - compile_wall_start)
            .count()));
    const auto compile_json = compile_trace_ptr->to_json();
    update_compile_profiling_report_json(compile_json);
    update_last_profiling_report_json(build_profiling_report_json(
        m_backend_name, profiling_level(), {}, {}, compile_json));
  }
}

CompiledModel::~CompiledModel() {
  if (m_backend_state) {
    m_backend_state->release();
  }
}

std::shared_ptr<ov::ISyncInferRequest>
CompiledModel::create_sync_infer_request() const {
  return std::make_shared<InferRequest>(shared_from_this());
}

void CompiledModel::export_model(std::ostream &model) const {
  const auto source = m_original_model ? m_original_model : m_runtime_model;
  write_model_to_stream(source, model);
}

void CompiledModel::set_property(const ov::AnyMap &properties) {
  for (const auto &kv : properties) {
    if (kv.first == ov::hint::inference_precision.name()) {
      m_inference_precision =
          parse_inference_precision_property(kv.second, kv.first);
      m_config[kv.first] = m_inference_precision;
    } else if (apply_profiling_property(kv.first, kv.second, m_enable_profiling,
                                        m_profiling_level,
                                        m_profiling_level_set, m_config)) {
      // handled
    } else if (kv.first == ov::cache_dir.name()) {
      m_config[kv.first] = kv.second.as<std::string>();
    } else if (kv.first == kGfxEnableFusionProperty) {
      m_enable_fusion = parse_bool_property(kv.second, kv.first);
      m_config[kv.first] = m_enable_fusion;
    } else if (is_diagnostic_f32_vendor_image_property(kv.first)) {
      m_diagnostic_f32_vendor_image = parse_bool_property(kv.second, kv.first);
      m_config[kv.first] = m_diagnostic_f32_vendor_image;
    } else {
      OPENVINO_THROW("CompiledModel unsupported property: ", kv.first);
    }
  }
}

ProfilingLevel CompiledModel::profiling_level() const {
  if (!m_enable_profiling) {
    return ProfilingLevel::Off;
  }
  if (m_profiling_level_set) {
    return m_profiling_level;
  }
  return ProfilingLevel::Standard;
}

ov::Any CompiledModel::get_property(const std::string &name) const {
  // Read-only properties we currently expose for integration with tools like
  // benchmark_app
  if (ov::model_name == name) {
    return decltype(ov::model_name)::value_type{
        m_runtime_model->get_friendly_name()};
  } else if (ov::execution_devices == name) {
    return decltype(ov::execution_devices)::value_type{
        get_plugin()->get_device_name()};
  } else if (ov::optimal_number_of_infer_requests == name) {
    // Single-stream synchronous execution for now
    return decltype(ov::optimal_number_of_infer_requests)::value_type{1};
  } else if (ov::loaded_from_cache == name) {
    return decltype(ov::loaded_from_cache)::value_type{m_loaded_from_cache};
  } else if (ov::supported_properties == name) {
    auto props = gfx_compiled_model_supported_properties();
    return decltype(ov::supported_properties)::value_type(props.begin(),
                                                          props.end());
  }

  if (name == kGfxBackendProperty) {
    return m_backend_name;
  }
  if (name == "PERF_COUNT") {
    return m_enable_profiling;
  }
  if (name == ov::cache_dir.name()) {
    if (auto it = m_config.find(name); it != m_config.end()) {
      return it->second;
    }
    return std::string{};
  }

  if (auto it = m_config.find(name); it != m_config.end()) {
    return it->second;
  }

  if (name == kGfxMemStatsProperty) {
    return m_backend_state ? m_backend_state->get_mem_stats() : ov::Any{};
  }
  if (name == kGfxProfilingReportProperty) {
    return last_profiling_report_json();
  }
  if (name == kGfxProfilingLevelProperty) {
    return static_cast<int>(profiling_level());
  }

  if (name == ov::hint::inference_precision.name()) {
    return m_inference_precision;
  }
  if (name == ov::enable_profiling.name()) {
    return m_enable_profiling;
  }

  OPENVINO_THROW("CompiledModel unsupported property: ", name);
}

void CompiledModel::update_last_profiling_report_json(
    std::string report_json) const {
  std::lock_guard<std::mutex> lock(m_report_mutex);
  m_last_report_json = std::move(report_json);
}

std::string CompiledModel::last_profiling_report_json() const {
  std::lock_guard<std::mutex> lock(m_report_mutex);
  return m_last_report_json;
}

void CompiledModel::update_compile_profiling_report_json(
    std::string report_json) const {
  std::lock_guard<std::mutex> lock(m_report_mutex);
  m_compile_report_json = std::move(report_json);
}

std::string CompiledModel::compile_profiling_report_json() const {
  std::lock_guard<std::mutex> lock(m_report_mutex);
  return m_compile_report_json;
}

void CompiledModel::build_op_pipeline(GfxProfilingTrace *compile_trace) {
  if (!m_runtime_model) {
    gfx_log_warn("OpFactory") << "Cannot build pipeline: runtime model is null";
    return;
  }

  if (!backend_has_const_manager(backend_state())) {
    gfx_log_warn("OpFactory") << "Cannot build pipeline: const manager is null";
    return;
  }

  const auto resources = get_backend_resources(backend_state());
  auto *backend_state = m_backend_state.get();
  OPENVINO_ASSERT(backend_state, "GFX: backend state is not initialized");
  GpuStageRuntimeOptions stage_runtime_options{};
  stage_runtime_options.diagnostic_f32_vendor_image = m_diagnostic_f32_vendor_image;
  auto configure_stage_runtime_options =
      [&](const std::unique_ptr<GpuStage> &stage) {
        if (stage) {
          stage->set_runtime_options(stage_runtime_options);
        }
      };
  gfx_log_info("StageFactory")
      << "Building pipeline for backend=" << m_backend_name
      << " ops=" << m_runtime_model->get_ops().size();
  const auto build_start = compile_trace
                               ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
  if (compile_trace) {
    compile_trace->set_counter(
        "runtime_model_op_count",
        static_cast<uint64_t>(m_runtime_model->get_ops().size()));
  }
  // Map Parameter nodes to input indices.
  for (size_t i = 0; i < m_runtime_model->inputs().size(); ++i) {
    m_param_index[m_runtime_model->inputs()[i].get_node()] = i;
  }

  // Track model outputs for bookkeeping (outputs remain device-only).
  std::unordered_map<const ov::Node *, std::vector<bool>> model_outputs;
  for (const auto &result : m_runtime_model->get_results()) {
    auto src = result->input_value(0).get_node_shared_ptr();
    const size_t port = result->input_value(0).get_index();
    auto &flags = model_outputs[src.get()];
    if (flags.empty()) {
      flags.resize(src->get_output_size(), false);
    }
    if (port < flags.size()) {
      flags[port] = true;
    }
  }

  const auto ordered_ops = m_runtime_model->get_ordered_ops();
  gfx_log_info("StageFactory") << "Ordered ops count=" << ordered_ops.size();
  m_pipeline.reserve(ordered_ops.size());

  const bool has_unobserved_stage_edges = [&]() {
    for (const auto &node : ordered_ops) {
      if (!is_executable_stage_node(node)) {
        continue;
      }
      for (size_t port = 0; port < node->get_output_size(); ++port) {
        if (!is_model_output_port(model_outputs, node.get(), port) &&
            !node->output(port).get_target_inputs().empty()) {
          return true;
        }
      }
    }
    return false;
  }();

  FusionPlan fusion_plan;
  std::unordered_map<size_t, const FusionGroup *> fusion_primary;
  std::unordered_set<size_t> planned_fused_indices;
  std::unordered_set<const ov::Node *> planned_fused_nodes;
  std::unordered_set<const ov::Node *> fused_nodes;
  auto fusion_group_has_fp32_precision = [&](const FusionGroup *group) {
    if (!group) {
      return false;
    }
    for (const auto node_idx : group->node_indices) {
      if (node_idx < ordered_ops.size() &&
          ov::fp16_compression_is_disabled(ordered_ops[node_idx])) {
        return true;
      }
    }
    return false;
  };
  auto is_precision_sensitive_arithmetic_fusion_group =
      [](const FusionGroup &group) {
    return group.kind == "ConvActivation" ||
           group.kind == "ConvBiasActivation" ||
           group.kind == "ConvBatchNormAct" ||
           group.kind == "ConvBias" ||
           group.kind == "ConvBatchNorm" ||
           group.kind == "ConvScale" ||
           group.kind == "ConvScaleActivation" ||
           group.kind == "MatMulActivation" ||
           group.kind == "MatMulBiasActivation" ||
           group.kind == "MatMulBias" ||
           group.kind == "EltwiseActivation" ||
           group.kind == "EltwiseBiasActivation" ||
           group.kind == "EltwiseBias" ||
           group.kind == "EltwiseInputActivation";
  };
  auto is_apple_mpsrt_precision_sensitive_arithmetic_fusion_group =
      [&](const FusionGroup &group) {
        if (backend_state->backend() != GpuBackend::Metal ||
            group.node_indices.empty() || group.input_activation.has_value() ||
            group.batchnorm.has_value()) {
          return false;
        }
        const auto primary_idx = group.node_indices.front();
        if (primary_idx >= ordered_ops.size() || !ordered_ops[primary_idx]) {
          return false;
        }
        const auto &primary = ordered_ops[primary_idx];
        const std::string stage_type = primary->get_type_name();
        if (stage_type != "Convolution" && stage_type != "GroupConvolution") {
          return false;
        }

        if (group.kind == "ConvBias") {
          if (!group.bias.has_value() || group.activation.has_value()) {
            return false;
          }
        } else if (group.kind == "ConvActivation" ||
                   group.kind == "ConvBiasActivation") {
          if (!group.activation.has_value() ||
              (group.kind == "ConvBiasActivation" && !group.bias.has_value()) ||
              !allow_stage_activation_fusion(GpuBackend::Metal, stage_type,
                                             *group.activation)) {
            return false;
          }
        } else {
          return false;
        }

        const auto plan = select_stage_optimization_plan(
            resources.const_manager, GpuBackend::Metal, stage_type, primary,
            primary->get_output_element_type(0), group.bias.has_value(),
            group.activation.has_value(),
            /*has_batchnorm=*/false, {});
        return plan.placement.domain == GfxStageBackendDomain::AppleMps &&
               plan.placement.storage == GfxStageStorageKind::Image &&
               plan.placement.uses_vendor_primitive &&
               !plan.placement.uses_custom_kernel;
      };
  if (m_enable_fusion && has_unobserved_stage_edges) {
    FusionConfig fusion_cfg;
    fusion_cfg.enable_fusion = true;
    fusion_cfg.debug_dump_ir = gfx_log_debug_enabled();
    fusion_cfg.enable_attention_fusion =
        backend_state->enable_generic_attention_fusion();
    fusion_cfg.enable_vendor_attention_fusion =
        backend_state->supports_vendor_attention_stage();
    fusion_cfg.enable_conv_activation_fusion =
        backend_state->enable_conv_activation_fusion();
    fusion_cfg.enable_conv_swish_fusion = true;
    fusion_plan = build_fusion_plan(m_runtime_model, fusion_cfg);
    if (compile_trace) {
      compile_trace->set_counter(
          "fusion_group_count",
          static_cast<uint64_t>(fusion_plan.groups.size()));
    }
    fusion_primary.reserve(fusion_plan.groups.size());
    for (const auto &group : fusion_plan.groups) {
      if (!backend_state->enable_precision_sensitive_arithmetic_fusion() &&
          is_precision_sensitive_arithmetic_fusion_group(group) &&
          fusion_group_has_fp32_precision(&group)) {
        if (is_apple_mpsrt_precision_sensitive_arithmetic_fusion_group(
                group)) {
          if (compile_trace) {
            compile_trace->increment_counter(
                "fusion_precision_sensitive_mpsrt_allow_count");
          }
        } else {
          if (compile_trace) {
            compile_trace->increment_counter(
                "fusion_precision_sensitive_arithmetic_skip_count");
          }
          if (gfx_log_debug_enabled()) {
            gfx_log_debug("Fusion")
                << "Skipped backend precision-sensitive arithmetic fusion kind="
                << group.kind;
          }
          continue;
        }
      }
      if (group.node_indices.size() < 2) {
        continue;
      }
      if (gfx_log_debug_enabled()) {
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
      const bool attention_group =
          group.kind == "Attention" || group.kind == "AttentionScale" ||
          group.kind == "AttentionScaleMask" || group.kind == "NativeSDPA" ||
          group.kind == "VendorAttention";
      if (group.kind == "VendorAttention") {
        auto plan = make_vendor_attention_subgraph_plan(group, ordered_ops);
        if (!plan || !backend_state->create_vendor_attention_stage(plan->spec)) {
          continue;
        }
      }
      const size_t primary_idx = attention_group ? group.node_indices.back()
                                                 : group.node_indices.front();
      fusion_primary[primary_idx] = &group;
      auto input_activation_has_exclusive_consumer = [&]() {
        if (group.kind != "EltwiseInputActivation" ||
            group.node_indices.size() < 2 ||
            primary_idx >= ordered_ops.size()) {
          return false;
        }
        const size_t act_idx = group.node_indices[1];
        if (act_idx >= ordered_ops.size()) {
          return false;
        }
        const auto &act_node = ordered_ops[act_idx];
        if (!act_node || act_node->get_output_size() != 1 ||
            model_outputs.count(act_node.get()) != 0) {
          return false;
        }
        const auto &targets = act_node->output(0).get_target_inputs();
        if (targets.size() != 1) {
          return false;
        }
        const auto &target = *targets.begin();
        return target.get_node() == ordered_ops[primary_idx].get() &&
               target.get_index() == group.input_activation_input;
      };
      const bool pre_fusion_supported = [&]() {
        if (group.kind != "EltwiseInputActivation" ||
            !group.input_activation.has_value() ||
            primary_idx >= ordered_ops.size() ||
            !input_activation_has_exclusive_consumer()) {
          return false;
        }
        auto probe_stage =
            backend_state->create_stage(ordered_ops[primary_idx]);
        configure_stage_runtime_options(probe_stage);
        return probe_stage &&
               probe_stage->fuse_input_activation(group.input_activation_input,
                                                  *group.input_activation,
                                                  group.input_activation_alpha);
      }();
      for (const auto node_idx : group.node_indices) {
        if (node_idx < ordered_ops.size()) {
          fused_nodes.insert(ordered_ops[node_idx].get());
          if ((attention_group || pre_fusion_supported) &&
              node_idx != primary_idx) {
            planned_fused_indices.insert(node_idx);
            planned_fused_nodes.insert(ordered_ops[node_idx].get());
          }
        }
      }
    }
    if (gfx_log_debug_enabled()) {
      gfx_log_debug("Fusion")
          << "Fusion enabled: groups=" << fusion_plan.groups.size();
    }
  } else if (gfx_log_debug_enabled()) {
    gfx_log_debug("Fusion")
        << (m_enable_fusion ? "Fusion skipped: all stage edges are observable"
                            : "Fusion disabled via GFX_ENABLE_FUSION");
  }

  std::unordered_map<const ov::Node *, std::shared_ptr<const ov::op::v1::Add>>
      rms_residual_adds;
  std::unordered_set<const ov::Node *> rms_residual_add_nodes;
  for (const auto &node : ordered_ops) {
    auto add = residual_add_for_rms(node, model_outputs, planned_fused_nodes);
    if (!add) {
      continue;
    }
    auto probe_stage = backend_state->create_stage(node);
    configure_stage_runtime_options(probe_stage);
    if (!probe_stage || !probe_stage->fuse_residual_add()) {
      continue;
    }
    rms_residual_adds.emplace(node.get(), add);
    rms_residual_add_nodes.insert(add.get());
  }
  if (compile_trace && !rms_residual_adds.empty()) {
    compile_trace->set_counter("rms_residual_add_fusion_count",
                               static_cast<uint64_t>(rms_residual_adds.size()));
  }
  if (gfx_log_debug_enabled() && !rms_residual_adds.empty()) {
    gfx_log_debug("Fusion")
        << "RMS residual Add fusion candidates=" << rms_residual_adds.size();
  }

  std::unordered_set<size_t> fused_indices;
  fused_indices.reserve(ordered_ops.size());
  struct NodePortKey {
    const ov::Node *node = nullptr;
    size_t port = 0;
    bool operator==(const NodePortKey &other) const {
      return node == other.node && port == other.port;
    }
  };
  struct NodePortKeyHash {
    size_t operator()(const NodePortKey &key) const {
      size_t h1 = std::hash<const ov::Node *>()(key.node);
      size_t h2 = std::hash<size_t>()(key.port);
      return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
  };
  std::unordered_map<NodePortKey, size_t, NodePortKeyHash>
      fused_output_port_aliases;
  auto remap_input_link = [&](std::shared_ptr<const ov::Node> linked_node,
                              size_t linked_port) {
    auto it = fused_output_port_aliases.find({linked_node.get(), linked_port});
    if (it != fused_output_port_aliases.end()) {
      linked_port = it->second;
    }
    return PipelineStageDesc::InputLink{std::move(linked_node), linked_port};
  };
  auto merge_model_outputs = [&](PipelineStageDesc &stage_desc,
                                 const ov::Node *node) {
    auto it = model_outputs.find(node);
    if (it == model_outputs.end()) {
      return;
    }
    const auto &flags = it->second;
    for (size_t oi = 0; oi < stage_desc.outputs.size() && oi < flags.size();
         ++oi) {
      stage_desc.outputs[oi].is_model_output =
          stage_desc.outputs[oi].is_model_output || flags[oi];
    }
  };
  auto append_output_alias =
      [](PipelineStageDesc &stage_desc,
         const std::shared_ptr<const ov::Node> &source_node, size_t source_port,
         size_t output_port) {
        if (!source_node || output_port >= stage_desc.outputs.size()) {
          return;
        }
        const auto duplicate = std::any_of(
            stage_desc.output_aliases.begin(), stage_desc.output_aliases.end(),
            [&](const PipelineStageDesc::OutputAlias &alias) {
              return alias.node.get() == source_node.get() &&
                     alias.source_port == source_port &&
                     alias.output_port == output_port;
            });
        if (!duplicate) {
          stage_desc.output_aliases.push_back(
              {source_node, source_port, output_port});
        }
      };
  auto fusion_requires_bias_payload = [](const std::string &kind) {
    return kind == "ConvBias" || kind == "ConvBiasActivation" ||
           kind == "EltwiseBias" || kind == "EltwiseBiasActivation" ||
           kind == "MatMulBias" || kind == "MatMulBiasActivation";
  };
  auto fusion_requires_batchnorm_payload = [](const std::string &kind) {
    return kind == "ConvBatchNorm" || kind == "ConvBatchNormAct" ||
           kind == "ConvScale" || kind == "ConvScaleActivation";
  };

  std::unordered_map<const ov::Node *,
                     std::unordered_map<size_t, GfxInputTransform>>
      absorbed_input_transforms;
  std::unordered_set<const ov::Node *> absorbed_transpose_nodes;
  for (const auto &node : ordered_ops) {
    if (!is_absorbable_transpose_candidate(node, model_outputs, fused_nodes)) {
      continue;
    }
    OPENVINO_ASSERT(
        node->get_output_size() == 1,
        "GFX: transpose absorption expects single-output transpose");
    const auto &consumers = node->output(0).get_target_inputs();
    if (consumers.size() != 1) {
      continue;
    }
    const auto &consumer_input = *consumers.begin();
    auto consumer = consumer_input.get_node()->shared_from_this();
    if (!consumer || !is_supported_absorbing_consumer(consumer) ||
        !is_supported_absorbing_input(consumer, consumer_input.get_index()) ||
        fused_nodes.count(consumer.get()) != 0) {
      continue;
    }
    auto source = node->input_value(0).get_node_shared_ptr();
    if (!source || !node->input_value(0).get_partial_shape().is_static()) {
      continue;
    }
    auto permutation = evaluate_constant_i64(node->input_value(1));
    if (permutation.size() != node->get_input_shape(0).size()) {
      continue;
    }
    GfxInputTransform transform;
    transform.source_shape = node->get_input_shape(0);
    transform.transpose_permutation = std::move(permutation);
    absorbed_input_transforms[consumer.get()][consumer_input.get_index()] =
        std::move(transform);
    absorbed_transpose_nodes.insert(node.get());
  }

  for (size_t op_index = 0; op_index < ordered_ops.size(); ++op_index) {
    const auto &node = ordered_ops[op_index];
    if (!is_executable_stage_node(node)) {
      continue;
    }

    if (absorbed_transpose_nodes.count(node.get()) != 0) {
      continue;
    }

    if (rms_residual_add_nodes.count(node.get()) != 0) {
      continue;
    }

    if (fused_indices.count(op_index)) {
      continue;
    }
    if (planned_fused_indices.count(op_index) &&
        fusion_primary.find(op_index) == fusion_primary.end()) {
      continue;
    }

    auto f_it = fusion_primary.find(op_index);
    if (f_it != fusion_primary.end() && f_it->second) {
      const auto *group = f_it->second;
      if (group->kind == "VendorAttention") {
        auto plan = make_vendor_attention_subgraph_plan(*group, ordered_ops);
        auto stage = plan ? backend_state->create_vendor_attention_stage(plan->spec)
                          : nullptr;
        configure_stage_runtime_options(stage);
        if (compile_trace) {
          compile_trace->increment_counter("stage_create_count");
        }
        if (stage && !group->node_indices.empty()) {
          const auto &final_node = ordered_ops[group->node_indices.back()];
          PipelineStageDesc stage_desc;
          stage_desc.node = final_node;
          stage_desc.stage = std::move(stage);
          stage_desc.inputs.push_back(remap_input_link(
              plan->query.get_node_shared_ptr(), plan->query.get_index()));
          stage_desc.inputs.push_back(remap_input_link(
              plan->key.get_node_shared_ptr(), plan->key.get_index()));
          stage_desc.inputs.push_back(remap_input_link(
              plan->value.get_node_shared_ptr(), plan->value.get_index()));
          stage_desc.outputs.resize(1);
          stage_desc.outputs[0].shape = plan->spec.output_shape;
          stage_desc.outputs[0].type = plan->spec.element_type;
          stage_desc.outputs[0].source_node = final_node;
          stage_desc.outputs[0].source_port = 0;
          merge_model_outputs(stage_desc, final_node.get());
          append_output_alias(stage_desc, final_node, 0, 0);
          fused_output_port_aliases[{final_node.get(), 0}] = 0;

          if (compile_trace) {
            compile_trace->increment_counter("fused_stage_count");
            compile_trace->increment_counter(
                "fused_node_count",
                static_cast<uint64_t>(group->node_indices.size()));
            compile_trace->increment_counter("vendor_attention_stage_count");
          }

          const size_t idx = m_pipeline.size();
          m_pipeline.emplace_back(std::move(stage_desc));
          for (const auto node_idx : group->node_indices) {
            if (node_idx < ordered_ops.size()) {
              fused_indices.insert(node_idx);
              const auto &fused_node = ordered_ops[node_idx];
              m_node_to_stage[fused_node.get()] = idx;
            }
          }
          continue;
        }
      }
      if (group->kind == "Attention" || group->kind == "AttentionScale" ||
          group->kind == "AttentionScaleMask" || group->kind == "NativeSDPA") {
        const size_t stage_count = group->node_indices.size();
        if (stage_count >= 3) {
          struct InputKey {
            const ov::Node *node = nullptr;
            size_t port = 0;
            bool operator==(const InputKey &other) const {
              return node == other.node && port == other.port;
            }
          };
          struct InputKeyHash {
            size_t operator()(const InputKey &key) const {
              size_t h1 = std::hash<const ov::Node *>()(key.node);
              size_t h2 = std::hash<size_t>()(key.port);
              return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
            }
          };

          std::unordered_map<const ov::Node *, size_t> stage_index;
          stage_index.reserve(stage_count);
          for (size_t i = 0; i < stage_count; ++i) {
            const auto idx = group->node_indices[i];
            if (idx < ordered_ops.size()) {
              stage_index[ordered_ops[idx].get()] = i;
            }
          }

          bool can_fuse = true;
          std::vector<std::vector<size_t>> stage_output_slots(stage_count);
          size_t fused_output_count = 1;
          for (size_t i = 0; i < stage_count; ++i) {
            const size_t idx = group->node_indices[i];
            if (idx >= ordered_ops.size()) {
              can_fuse = false;
              break;
            }
            const auto &stage_node = ordered_ops[idx];
            stage_output_slots[i].reserve(stage_node->get_output_size());
            for (size_t port = 0; port < stage_node->get_output_size();
                 ++port) {
              if (i + 1 == stage_count && port == 0) {
                stage_output_slots[i].push_back(0);
              } else {
                stage_output_slots[i].push_back(fused_output_count++);
              }
            }
          }
          if (!can_fuse) {
            continue;
          }

          std::unordered_map<InputKey, size_t, InputKeyHash> external_map;
          std::vector<PipelineStageDesc::InputLink> fused_inputs;
          std::vector<FusedStageInfo> fused_stages;
          fused_stages.reserve(stage_count);

          for (size_t i = 0; i < stage_count; ++i) {
            const size_t idx = group->node_indices[i];
            if (idx >= ordered_ops.size()) {
              can_fuse = false;
              break;
            }
            const auto &stage_node = ordered_ops[idx];
            auto stage = backend_state->create_stage(stage_node);
            configure_stage_runtime_options(stage);
            if (compile_trace) {
              compile_trace->increment_counter("stage_create_count");
            }
            if (!stage) {
              can_fuse = false;
              break;
            }

            FusedStageInfo info;
            info.stage = std::move(stage);
            info.output_indices = stage_output_slots[i];
            info.inputs.reserve(stage_node->get_input_size());
            for (const auto &iv : stage_node->input_values()) {
              auto src_node = iv.get_node();
              const auto it_stage = stage_index.find(src_node);
              if (it_stage != stage_index.end()) {
                const size_t src_stage = it_stage->second;
                if (iv.get_index() >= stage_output_slots[src_stage].size()) {
                  can_fuse = false;
                  break;
                }
                info.inputs.push_back(
                    {FusedInputKind::Output,
                     stage_output_slots[src_stage][iv.get_index()]});
                continue;
              }
              if (ov::as_type_ptr<const ov::op::v0::Constant>(
                      iv.get_node_shared_ptr())) {
                info.inputs.push_back({FusedInputKind::None, 0});
                continue;
              }
              size_t linked_port = iv.get_index();
              if (auto alias_it =
                      fused_output_port_aliases.find({src_node, linked_port});
                  alias_it != fused_output_port_aliases.end()) {
                linked_port = alias_it->second;
              }
              InputKey key{src_node, linked_port};
              auto it_ext = external_map.find(key);
              size_t ext_idx = 0;
              if (it_ext == external_map.end()) {
                ext_idx = fused_inputs.size();
                external_map.emplace(key, ext_idx);
                fused_inputs.push_back({iv.get_node_shared_ptr(), linked_port});
              } else {
                ext_idx = it_ext->second;
              }
              info.inputs.push_back({FusedInputKind::External, ext_idx});
            }
            if (!can_fuse) {
              break;
            }
            fused_stages.emplace_back(std::move(info));
          }

          if (can_fuse && fused_stages.size() == stage_count) {
            PipelineStageDesc stage_desc;
            const auto &final_node = ordered_ops[group->node_indices.back()];
            if (compile_trace) {
              compile_trace->increment_counter("fused_stage_count");
              compile_trace->increment_counter(
                  "fused_node_count", static_cast<uint64_t>(stage_count));
            }
            stage_desc.node = final_node;
            stage_desc.stage = std::make_unique<FusedSequenceStage>(
                std::move(fused_stages),
                final_node ? final_node->get_friendly_name()
                           : std::string("fused_attention"),
                "FusedAttention");
            stage_desc.inputs = std::move(fused_inputs);
            stage_desc.outputs.resize(fused_output_count);

            auto is_model_output = [&](const ov::Node *n, size_t port) -> bool {
              auto it = model_outputs.find(n);
              if (it == model_outputs.end() || it->second.empty()) {
                return false;
              }
              return port < it->second.size() ? it->second[port] : false;
            };

            for (size_t stage_idx = 0; stage_idx < stage_count; ++stage_idx) {
              const size_t node_idx = group->node_indices[stage_idx];
              if (node_idx >= ordered_ops.size()) {
                continue;
              }
              const auto &out_node = ordered_ops[node_idx];
              for (size_t port = 0; port < out_node->get_output_size();
                   ++port) {
                const size_t slot = stage_output_slots[stage_idx][port];
                auto &out_desc = stage_desc.outputs[slot];
                if (out_node->get_output_partial_shape(port).is_static()) {
                  out_desc.shape = out_node->get_output_shape(port);
                }
                out_desc.type = out_node->get_output_element_type(port);
                out_desc.is_model_output =
                    is_model_output(out_node.get(), port);
                out_desc.source_node = out_node;
                out_desc.source_port = port;
                fused_output_port_aliases[{out_node.get(), port}] = slot;
              }
            }

            const size_t idx = m_pipeline.size();
            m_pipeline.emplace_back(std::move(stage_desc));
            for (const auto node_idx : group->node_indices) {
              if (node_idx < ordered_ops.size()) {
                fused_indices.insert(node_idx);
                const auto &fused_node = ordered_ops[node_idx];
                m_node_to_stage[fused_node.get()] = idx;
              }
            }
            continue;
          }
        }
      }
    }

    if (gfx_log_debug_enabled()) {
      gfx_log_debug("StageFactory")
          << "Preparing stage for " << node->get_type_name()
          << " name=" << node->get_friendly_name();
    }
    if (auto f_it_primary = fusion_primary.find(op_index);
        f_it_primary != fusion_primary.end() &&
        fusion_group_has_fp32_precision(f_it_primary->second) &&
        !ov::fp16_compression_is_disabled(node)) {
      ov::disable_fp16_compression(node);
    }
    auto gpu_stage = backend_state->create_stage(node);
    configure_stage_runtime_options(gpu_stage);
    if (compile_trace) {
      compile_trace->increment_counter("stage_create_count");
    }
    OPENVINO_ASSERT(gpu_stage, "GFX: unsupported op in MLIR pipeline: ",
                    node->get_friendly_name(), " (", node->get_type_name(),
                    ")");

    PipelineStageDesc stage_desc;
    stage_desc.node = node;
    stage_desc.stage = std::move(gpu_stage);
    const size_t out_count = node->get_output_size();
    stage_desc.outputs.reserve(out_count);
    for (size_t oi = 0; oi < out_count; ++oi) {
      OutputDesc out_desc{};
      if (node->get_output_partial_shape(oi).is_static()) {
        out_desc.shape = node->get_output_shape(oi);
      }
      out_desc.type = node->get_output_element_type(oi);
      out_desc.source_node = node;
      out_desc.source_port = oi;
      stage_desc.outputs.emplace_back(std::move(out_desc));
    }
    merge_model_outputs(stage_desc, node.get());
    const auto residual_it = rms_residual_adds.find(node.get());
    if (residual_it != rms_residual_adds.end() && residual_it->second &&
        stage_desc.stage->fuse_residual_add()) {
      const auto &add = residual_it->second;
      stage_desc.inputs.push_back(
          remap_input_link(add->input_value(0).get_node_shared_ptr(),
                           add->input_value(0).get_index()));
      stage_desc.inputs.push_back(
          remap_input_link(node->input_value(1).get_node_shared_ptr(),
                           node->input_value(1).get_index()));
      stage_desc.inputs.push_back(
          remap_input_link(add->input_value(1).get_node_shared_ptr(),
                           add->input_value(1).get_index()));
    } else {
      const auto absorbed_it = absorbed_input_transforms.find(node.get());
      for (size_t input_idx = 0; input_idx < node->get_input_size();
           ++input_idx) {
        const auto &iv = node->input_value(input_idx);
        auto linked_node = iv.get_node_shared_ptr();
        size_t linked_port = iv.get_index();
        if (absorbed_it != absorbed_input_transforms.end()) {
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
            stage_desc.stage->set_input_transform(input_idx,
                                                  transform_it->second);
          }
        }
        stage_desc.inputs.push_back(remap_input_link(linked_node, linked_port));
      }
    }
    if (gfx_log_debug_enabled()) {
      gfx_log_debug("StageFactory")
          << "Created GpuStage for " << node->get_type_name()
          << " name=" << node->get_friendly_name();
    }

    const size_t idx = m_pipeline.size();
    m_node_to_stage[node.get()] = idx;
    if (residual_it != rms_residual_adds.end() && residual_it->second) {
      m_node_to_stage[residual_it->second.get()] = idx;
    }
    m_pipeline.emplace_back(std::move(stage_desc));

    auto f_it2 = fusion_primary.find(op_index);
    if (f_it2 != fusion_primary.end() && f_it2->second) {
      const auto *group = f_it2->second;
      auto &stage = m_pipeline[idx];
      auto mark_fused = [&](size_t fused_idx,
                            bool aliases_stage_output) -> bool {
        if (fused_idx >= ordered_ops.size()) {
          return false;
        }
        fused_indices.insert(fused_idx);
        const auto &fused_node = ordered_ops[fused_idx];
        m_node_to_stage[fused_node.get()] = idx;
        if (aliases_stage_output && fused_node->get_output_size() == 1) {
          merge_model_outputs(stage, fused_node.get());
          append_output_alias(stage, fused_node, 0, 0);
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
        const bool input_activation_exclusive = [&]() {
          if (group->node_indices.size() <= 1) {
            return false;
          }
          const size_t act_idx = group->node_indices[1];
          if (act_idx >= ordered_ops.size()) {
            return false;
          }
          const auto &act_node = ordered_ops[act_idx];
          if (!act_node || act_node->get_output_size() != 1 ||
              model_outputs.count(act_node.get()) != 0) {
            return false;
          }
          const auto &targets = act_node->output(0).get_target_inputs();
          if (targets.size() != 1) {
            return false;
          }
          const auto &target = *targets.begin();
          return stage.node && target.get_node() == stage.node.get() &&
                 target.get_index() == input_idx;
        }();
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
      if (fusion_requires_batchnorm_payload(group->kind) &&
          !group->batchnorm.has_value()) {
        post_ops_ok = false;
      }
      if (fusion_requires_bias_payload(group->kind) &&
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

  for (auto &stage : m_pipeline) {
    std::vector<GpuTensor *> inputs;
    inputs.reserve(stage.node->get_input_size());
    for (const auto &link : stage.inputs) {
      (void)link;
      inputs.push_back(nullptr); // Parameter/Constant or previous stage; not
                                 // needed for compile
    }
    if (gfx_log_debug_enabled()) {
      gfx_log_debug("StageFactory")
          << "Compiling stage for " << stage.node->get_type_name()
          << " name=" << stage.node->get_friendly_name();
    }
    stage.stage->set_inputs(inputs);
    GpuBufferManager *buffer_manager = resources.const_manager;
    stage.stage->init(buffer_manager);
    const auto stage_compile_start =
        compile_trace ? std::chrono::steady_clock::now()
                      : std::chrono::steady_clock::time_point{};
    const auto compile_scope_name =
        stage.node ? stage.node->get_friendly_name() : stage.stage->name();
    ScopedCompileProfilingContext compile_scope(compile_trace,
                                                compile_scope_name);
    stage.stage->compile(buffer_manager);
    if (compile_trace) {
      compile_trace->increment_counter("stage_compile_count");
      compile_trace->add_segment(
          "compile",
          stage.node ? stage.node->get_friendly_name() : stage.stage->name(),
          static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - stage_compile_start)
                  .count()));
    }
  }

  m_pipeline_built = true;
  if (compile_trace) {
    compile_trace->set_counter("pipeline_stage_count",
                               static_cast<uint64_t>(m_pipeline.size()));
    compile_trace->add_segment(
        "compile", "build_op_pipeline",
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - build_start)
                .count()));
  }
  gfx_log_info("StageFactory")
      << "Built GFX " << m_backend_name << " pipeline with "
      << m_pipeline.size() << " stages";
}

} // namespace gfx_plugin
} // namespace ov
