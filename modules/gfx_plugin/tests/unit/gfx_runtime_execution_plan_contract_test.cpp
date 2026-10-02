// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common/artifact_payload.hpp"
#include "common/gpu_backend.hpp"
#include "runtime/backend_stage_factory.hpp"
#include "runtime/gpu_stage.hpp"
#include "runtime/runtime_execution_plan.hpp"
#include "runtime/stage_materialization_context.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

class NoopPlanStage final : public GpuStage {
public:
  NoopPlanStage(std::string name, std::string type)
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
    auto stage = std::make_unique<NoopPlanStage>(m_name, m_type);
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

class PlanStageFactory final : public BackendStageFactory {
public:
  GpuBackend backend() const override { return GpuBackend::Metal; }

  std::unique_ptr<GpuStage>
  create_stage(const RuntimeStageMaterializationContext &context) const override {
    const auto &descriptor = context.require_descriptor();
    seen_kernel_ids.push_back(descriptor.kernel_id);
    return std::make_unique<NoopPlanStage>(context.op_friendly_name(),
                                           context.op_type_name());
  }

  mutable std::vector<std::string> seen_kernel_ids;
};

class TestKernelPayload final : public KernelArtifactPayload {
public:
  explicit TestKernelPayload(std::string source_id)
      : m_source_id(std::move(source_id)) {}

  KernelArtifactPayloadKind payload_kind() const noexcept override {
    return KernelArtifactPayloadKind::MslSource;
  }

  std::string_view backend_domain() const noexcept override {
    return kBackendMetal;
  }

  std::string_view source_id() const noexcept override {
    return m_source_id;
  }

  std::string_view entry_point() const noexcept override {
    return m_source_id;
  }

  bool valid() const noexcept override { return !m_source_id.empty(); }

private:
  std::string m_source_id;
};

RuntimeTensorBindingContract binding(std::string logical_name,
                                     std::string region_id,
                                     std::string role) {
  RuntimeTensorBindingContract result;
  result.logical_name = std::move(logical_name);
  result.memory_region_id = std::move(region_id);
  result.role = std::move(role);
  result.element_type = "f32";
  result.partial_shape = "{1}";
  result.layout = "logical";
  result.storage_kind = "device_buffer";
  result.lifetime_class = "contract";
  result.alias_group = result.memory_region_id;
  return result;
}

RuntimeStageExecutableDescriptor make_stage_descriptor() {
  RuntimeStageExecutableDescriptor descriptor;
  descriptor.stage_index = 0;
  descriptor.stage_record_key = 0x42u;
  descriptor.artifact_descriptor_index = 0;
  descriptor.manifest_ref = "manifest://execution-plan/stage0";
  descriptor.abi_fingerprint = "abi://execution-plan/stage0";
  descriptor.artifact_key = "artifact://execution-plan/stage0";
  descriptor.backend_domain = kBackendMetal;
  descriptor.kernel_id = "execution_plan/noop";
  descriptor.op_family = "Noop";
  descriptor.stage_name = "execution_plan_noop";
  descriptor.origin = KernelArtifactOrigin::Generated;
  descriptor.payload_kind = KernelArtifactPayloadKind::MslSource;
  descriptor.entry_point = "execution_plan_noop";
  descriptor.abi_arg_count = 1;
  descriptor.abi_output_arg_count = 1;
  descriptor.payload = std::make_shared<TestKernelPayload>(
      "execution_plan/noop/msl");
  descriptor.input_bindings.push_back(
      binding("input0", "stage0.input0", "TensorInput"));
  descriptor.output_bindings.push_back(
      binding("output0", "stage0.output0", "TensorOutput"));
  return descriptor;
}

PipelineStageMaterializationPlan make_plan(
    const RuntimeStageExecutableDescriptor &descriptor) {
  PipelineStageMaterializationPlan plan;
  plan.kind = PipelineStageMaterializationKind::SingleStage;
  plan.io_plan.stage_name = descriptor.stage_name;
  plan.io_plan.op_family = descriptor.op_family;
  plan.io_plan.runtime_stage_index = descriptor.stage_index;
  plan.descriptor_stage_index = descriptor.stage_index;

  PipelineStageInputLink input;
  input.port = 0;
  input.source_ref.kind = PipelineStageTensorRefKind::Parameter;
  input.source_ref.index = 0;
  input.source_ref.port = 0;
  plan.io_plan.inputs.push_back(input);

  PipelineStageOutputDesc output;
  output.shape = ov::Shape{1};
  output.type = ov::element::f32;
  output.source_port = 0;
  output.source_ref.kind = PipelineStageTensorRefKind::StageOutput;
  output.source_ref.index = descriptor.stage_index;
  output.source_ref.port = 0;
  plan.io_plan.outputs.push_back(output);

  plan.materialized_descriptor = descriptor;
  plan.materialized_descriptor_valid = true;
  return plan;
}

std::shared_ptr<RuntimeExecutableDescriptor> make_descriptor() {
  auto descriptor = std::make_shared<RuntimeExecutableDescriptor>();
  descriptor->target_fingerprint =
      "backend=metal;runtime=metal;family=apple;profile=unit";
  descriptor->materialization_finalized = true;
  auto stage = make_stage_descriptor();
  descriptor->stages.push_back(stage);
  descriptor->materialization_stages.push_back(make_plan(stage));
  RuntimePublicOutputDescriptor output;
  output.kind = RuntimePublicOutputSourceKind::StageOutput;
  output.index = 0;
  output.port = 0;
  output.static_shape = ov::Shape{1};
  output.static_type = ov::element::f32;
  descriptor->public_outputs.push_back(output);
  return descriptor;
}

} // namespace

TEST(RuntimeExecutionPlanContract, OwnsMaterializedStagesAndDescriptor) {
  PlanStageFactory factory;
  RuntimeExecutionPlanBuildRequest request;
  request.stage_factory = &factory;
  request.runtime_descriptor = make_descriptor();

  auto plan = RuntimeExecutionPlan::build(std::move(request));

  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->stage_count(), 1u);
  EXPECT_EQ(plan->descriptor().stages.size(), 1u);
  ASSERT_EQ(plan->stages().size(), 1u);
  EXPECT_EQ(plan->stages()[0].runtime_stage_index, 0u);
  ASSERT_TRUE(plan->stages()[0].stage);
  ASSERT_TRUE(plan->stages()[0].runtime_descriptor);
  EXPECT_EQ(plan->stages()[0].runtime_descriptor->kernel_id,
            "execution_plan/noop");
  EXPECT_EQ(factory.seen_kernel_ids,
            (std::vector<std::string>{"execution_plan/noop"}));
}

TEST(RuntimeExecutionPlanContract, RejectsUnfinalizedDescriptor) {
  PlanStageFactory factory;
  auto descriptor = make_descriptor();
  descriptor->materialization_finalized = false;

  RuntimeExecutionPlanBuildRequest request;
  request.stage_factory = &factory;
  request.runtime_descriptor = std::move(descriptor);

  EXPECT_ANY_THROW(RuntimeExecutionPlan::build(std::move(request)));
}

} // namespace gfx_plugin
} // namespace ov
