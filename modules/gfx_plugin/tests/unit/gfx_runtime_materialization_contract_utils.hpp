// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "unit/gfx_backend_architecture_contract_common.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

class UnitVendorPayload final : public KernelArtifactPayload {
public:
  KernelArtifactPayloadKind payload_kind() const noexcept override {
    return KernelArtifactPayloadKind::VendorDescriptor;
  }

  std::string_view backend_domain() const noexcept override {
    return kBackendMetal;
  }

  std::string_view source_id() const noexcept override {
    return "unit_vendor_descriptor";
  }

  std::string_view entry_point() const noexcept override {
    return "unit_vendor_entry";
  }

  bool valid() const noexcept override { return true; }
};

class UnitNoopStage final : public GpuStage {
public:
  UnitNoopStage(std::string name, std::string type)
      : m_name(std::move(name)), m_type(std::move(type)) {}

  void init(GpuBufferManager *) override {}
  void prepare_runtime_handle(GpuBufferManager *) override {}
  void execute(GpuCommandBufferHandle) override {}
  void set_inputs(const std::vector<GpuTensor *> &inputs) override {
    m_inputs = inputs;
  }
  void set_output(GpuTensor *output) override { m_output = output; }
  const std::string &name() const override { return m_name; }
  const std::string &type() const override { return m_type; }
  std::unique_ptr<GpuStage> clone() const override {
    auto stage = std::make_unique<UnitNoopStage>(m_name, m_type);
    stage->m_inputs = m_inputs;
    stage->m_output = m_output;
    return stage;
  }

private:
  std::string m_name;
  std::string m_type;
  std::vector<GpuTensor *> m_inputs;
  GpuTensor *m_output = nullptr;
};

class UnitBackendStageFactory final : public BackendStageFactory {
public:
  GpuBackend backend() const override { return GpuBackend::Metal; }

  std::unique_ptr<GpuStage>
  create_stage(const RuntimeStageMaterializationContext &) const override {
    return nullptr;
  }
};

class CapturingBackendStageFactory final : public BackendStageFactory {
public:
  GpuBackend backend() const override { return GpuBackend::Metal; }

  std::unique_ptr<GpuStage> create_stage(
      const RuntimeStageMaterializationContext &context) const override {
    stage_names.push_back(context.op_friendly_name());

    const auto &descriptor = context.require_descriptor();
    if (auto stateful = create_stateful_stage(descriptor)) {
      return stateful;
    }
    if (auto view = create_view_only_stage(descriptor)) {
      return view;
    }
    return std::make_unique<UnitNoopStage>(context.op_friendly_name(),
                                           context.op_type_name());
  }

  mutable std::vector<std::string> stage_names;
};

RuntimeStageExecutableDescriptor
make_materializer_base_descriptor(const std::shared_ptr<ov::Node> &node) {
  RuntimeStageExecutableDescriptor descriptor;
  descriptor.stage_index = 0;
  descriptor.stage_record_key = 0x1234u;
  descriptor.artifact_descriptor_index = 0;
  descriptor.manifest_ref = "manifest://unit/relu";
  descriptor.abi_fingerprint = "abi://unit/relu";
  descriptor.artifact_key = "artifact://unit/relu";
  descriptor.backend_domain = kBackendMetal;
  descriptor.kernel_id = "unit/relu";
  descriptor.op_family = node->get_type_name();
  descriptor.stage_name = node->get_friendly_name();
  descriptor.origin = KernelArtifactOrigin::Generated;
  descriptor.payload_kind = KernelArtifactPayloadKind::MslSource;
  descriptor.entry_point = "unit_relu";
  descriptor.abi_arg_count = 1;
  descriptor.abi_output_arg_count = 1;
  descriptor.tensor_roles = {"TensorInput", "TensorOutput"};
  descriptor.input_bindings.push_back(make_runtime_binding(
      "parameter.output0", "compiler_input_region", "TensorInput"));
  descriptor.output_bindings.push_back(make_runtime_binding(
      "relu.output0", "compiler_output_region", "TensorOutput"));
  return descriptor;
}

PipelineStageMaterializationPlan
make_vendor_materialization_plan(const std::shared_ptr<ov::Node> &input,
                                 const std::shared_ptr<ov::Node> &node,
                                 RuntimeStageExecutableDescriptor descriptor) {
  (void)input;
  descriptor.payload_kind = KernelArtifactPayloadKind::VendorDescriptor;
  descriptor.payload = std::make_shared<UnitVendorPayload>();
  descriptor.artifact_key = "artifact://unit/vendor_attention";

  PipelineStageMaterializationPlan plan;
  plan.kind = PipelineStageMaterializationKind::VendorAttention;
  plan.io_plan.stage_name =
      node ? node->get_friendly_name() : "unit_vendor_attention";
  plan.io_plan.op_family = node ? node->get_type_name() : "VendorAttention";
  plan.io_plan.runtime_stage_index = 0;
  plan.descriptor_stage_index = descriptor.stage_index;
  PipelineStageInputLink input_link;
  input_link.port = 0;
  input_link.source_ref.kind = PipelineStageTensorRefKind::Parameter;
  input_link.source_ref.index = 0;
  input_link.source_ref.port = 0;
  plan.io_plan.inputs.push_back(input_link);

  PipelineStageOutputDesc output;
  output.shape = ov::Shape{1};
  output.type = ov::element::f32;
  output.source_port = 0;
  output.source_ref.kind = PipelineStageTensorRefKind::StageOutput;
  output.source_ref.index = 0;
  output.source_ref.port = 0;
  plan.io_plan.outputs.push_back(std::move(output));

  plan.vendor_attention.name = "unit_vendor_attention";
  plan.vendor_attention.descriptor = descriptor;
  plan.materialized_descriptor = std::move(descriptor);
  plan.materialized_descriptor_valid = true;
  return plan;
}

PipelineStageMaterializationPlan
make_single_materialization_plan(const std::shared_ptr<const ov::Node> &node,
                                 RuntimeStageExecutableDescriptor descriptor) {
  PipelineStageMaterializationPlan plan;
  plan.kind = PipelineStageMaterializationKind::SingleStage;
  plan.io_plan.stage_name =
      node ? node->get_friendly_name() : std::string("unit_single_stage");
  plan.io_plan.op_family =
      node ? node->get_type_name() : std::string("Unknown");
  plan.io_plan.runtime_stage_index = descriptor.stage_index;
  plan.descriptor_stage_index = descriptor.stage_index;

  PipelineStageOutputDesc output;
  output.shape = ov::Shape{1};
  output.type = ov::element::f32;
  output.source_port = 0;
  output.source_ref.kind = PipelineStageTensorRefKind::StageOutput;
  output.source_ref.index = descriptor.stage_index;
  output.source_ref.port = 0;
  plan.io_plan.outputs.push_back(std::move(output));

  plan.materialized_descriptor = std::move(descriptor);
  plan.materialized_descriptor_valid = true;
  return plan;
}

} // namespace
} // namespace gfx_plugin
} // namespace ov
