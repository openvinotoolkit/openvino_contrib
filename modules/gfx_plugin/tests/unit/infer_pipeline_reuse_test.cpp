// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <utility>

#include "plugin/infer_io_utils.hpp"
#include "runtime/fused_output_lifetime_plan.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

class FakeAllocator final : public IGpuAllocator {
public:
    GpuBackend backend() const override {
        return GpuBackend::Metal;
    }

    GpuBuffer allocate(const GpuBufferDesc& desc) override {
        GpuBuffer buf;
        buf.backend = backend();
        buf.buffer = reinterpret_cast<GpuBufferHandle>(++m_next_handle);
        buf.size = desc.bytes;
        buf.type = desc.type;
        buf.host_visible = desc.cpu_read || desc.cpu_write || !desc.prefer_device_local;
        buf.allocation_uid = allocate_gpu_buffer_uid();
        ++m_allocate_count;
        return buf;
    }

    GpuBuffer wrap_shared(void*, size_t, ov::element::Type) override {
        return {};
    }

    void release(GpuBuffer&&) override {}

    size_t allocate_count() const {
        return m_allocate_count;
    }

private:
    uintptr_t m_next_handle = 0x1000;
    size_t m_allocate_count = 0;
};

class CountingStage final : public GpuStage {
public:
    static void reset_counters() {
        s_clone_count = 0;
        s_init_count = 0;
        s_compile_count = 0;
    }

    static size_t clone_count() {
        return s_clone_count;
    }

    static size_t init_count() {
        return s_init_count;
    }

    static size_t compile_count() {
        return s_compile_count;
    }

    void init(GpuBufferManager*) override {
        ++s_init_count;
    }

    void compile(GpuBufferManager*) override {
        ++s_compile_count;
    }
    void execute(GpuCommandBufferHandle) override {}
    void set_inputs(const std::vector<GpuTensor*>&) override {}
    void set_output(GpuTensor*) override {}

    const std::string& name() const override {
        static const std::string kName = "CountingStage";
        return kName;
    }

    const std::string& type() const override {
        static const std::string kType = "Counting";
        return kType;
    }

    std::unique_ptr<GpuStage> clone() const override {
        ++s_clone_count;
        return std::make_unique<CountingStage>();
    }

private:
    static inline size_t s_clone_count = 0;
    static inline size_t s_init_count = 0;
    static inline size_t s_compile_count = 0;
};

class TrackingStage final : public GpuStage {
public:
    void init(GpuBufferManager*) override {}
    void compile(GpuBufferManager*) override {}
    void execute(GpuCommandBufferHandle) override {}
    void prewarm_runtime_state() override {
        ++prewarm_count;
        prewarm_inputs = last_inputs;
        prewarm_inputs_data = last_inputs_data;
    }

    void set_inputs(const std::vector<GpuTensor*>& inputs) override {
        last_inputs = inputs;
        last_inputs_data = inputs.data();
        ++set_inputs_count;
    }

    void set_output(GpuTensor*) override {}

    const std::string& name() const override {
        static const std::string kName = "TrackingStage";
        return kName;
    }

    const std::string& type() const override {
        static const std::string kType = "Tracking";
        return kType;
    }

    std::unique_ptr<GpuStage> clone() const override {
        return std::make_unique<TrackingStage>();
    }

    std::vector<GpuTensor*> last_inputs;
    const GpuTensor* const* last_inputs_data = nullptr;
    size_t set_inputs_count = 0;
    std::vector<GpuTensor*> prewarm_inputs;
    const GpuTensor* const* prewarm_inputs_data = nullptr;
    size_t prewarm_count = 0;
};

class DescriptorProbeStage final : public GpuStage {
public:
    explicit DescriptorProbeStage(std::string type_name)
        : m_type_name(std::move(type_name)) {}

    void init(GpuBufferManager*) override {}
    void compile(GpuBufferManager*) override {}
    void execute(GpuCommandBufferHandle) override {}
    void set_inputs(const std::vector<GpuTensor*>&) override {}
    void set_output(GpuTensor*) override {}

    const std::string& name() const override {
        static const std::string kName = "DescriptorProbeStage";
        return kName;
    }

    const std::string& type() const override {
        return m_type_name;
    }

    std::unique_ptr<GpuStage> clone() const override {
        return std::make_unique<DescriptorProbeStage>(m_type_name);
    }

private:
    std::string m_type_name;
};

std::shared_ptr<RuntimeExecutableDescriptor>
make_test_runtime_descriptor(size_t stage_count,
                             std::vector<size_t> output_last_stages = {}) {
    auto descriptor = std::make_shared<RuntimeExecutableDescriptor>();
    descriptor->target_fingerprint = "test/runtime-session";
    descriptor->stages.reserve(stage_count);
    RuntimeTransientArenaDescriptor arena;
    arena.arena_id = "test_transient_arena";
    arena.storage_kind = "device_buffer";
    for (size_t i = 0; i < stage_count; ++i) {
        RuntimeStageExecutableDescriptor stage;
        stage.stage_index = i;
        stage.stage_record_key = static_cast<uint64_t>(i + 1);
        stage.artifact_descriptor_index = i;
        stage.manifest_ref = "test/manifest/" + std::to_string(i);
        stage.abi_fingerprint = "test/abi/" + std::to_string(i);
        stage.artifact_key = "test/artifact/" + std::to_string(i);
        stage.backend_domain = "test";
        stage.kernel_id = "test/kernel/" + std::to_string(i);
        stage.op_family = "test";
        stage.origin = compiler::KernelArtifactOrigin::Metadata;
        stage.payload_kind = compiler::KernelArtifactPayloadKind::None;
        RuntimeTensorBindingContract output_binding;
        output_binding.logical_name = "test.stage." + std::to_string(i) + ".output0";
        output_binding.memory_region_id =
            "stage_" + std::to_string(i) + ".output_0";
        output_binding.role = "tensor_output";
        output_binding.element_type = "f32";
        output_binding.partial_shape = "{4}";
        output_binding.layout = "logical";
        output_binding.storage_kind = "device_buffer";
        output_binding.lifetime_class = "stage_output";
        output_binding.alias_group = "stage_" + std::to_string(i);
        stage.output_bindings.push_back(std::move(output_binding));

        RuntimeMemoryRegionDescriptor region;
        region.region_id = "stage_" + std::to_string(i) + ".output_0";
        region.logical_tensor_name =
            "test.stage." + std::to_string(i) + ".output0";
        region.kind = "transient_tensor";
        region.element_type = "f32";
        region.partial_shape = "{4}";
        region.layout = "logical";
        region.storage_kind = "device_buffer";
        region.alias_group = "stage_" + std::to_string(i);
        region.first_stage = i;
        region.last_stage =
            i < output_last_stages.size() ? output_last_stages[i] : i;
        descriptor->memory_plan.regions.push_back(std::move(region));

        RuntimeMemoryAliasGroupDescriptor alias_group;
        alias_group.group_id = "stage_" + std::to_string(i);
        alias_group.region_ids.push_back("stage_" + std::to_string(i) +
                                         ".output_0");
        descriptor->memory_plan.alias_groups.push_back(std::move(alias_group));
        arena.region_ids.push_back("stage_" + std::to_string(i) + ".output_0");
        descriptor->stages.push_back(std::move(stage));
    }
    if (!arena.region_ids.empty()) {
        descriptor->memory_plan.transient_arenas.push_back(std::move(arena));
    }
    return descriptor;
}

TEST(InferPipelineReuseTest, RuntimeAliasContractIgnoresLegacyStageTypeFallback) {
    InferStage split_stage;
    split_stage.stage = std::make_unique<DescriptorProbeStage>("Split");

    EXPECT_FALSE(is_view_op(split_stage));
    EXPECT_FALSE(stage_outputs_may_alias_inputs(split_stage));
}

TEST(InferPipelineReuseTest, RuntimeAliasContractUsesCompilerMemoryPlanOnly) {
    auto runtime_descriptor = make_test_runtime_descriptor(1);
    ASSERT_FALSE(runtime_descriptor->memory_plan.alias_groups.empty());
    runtime_descriptor->memory_plan.alias_groups.front().output_aliasing = true;

    InferStage stage;
    stage.stage = std::make_unique<DescriptorProbeStage>("Counting");
    stage.runtime_session = std::make_shared<RuntimeSession>(runtime_descriptor);
    stage.runtime_stage_index = 0;

    EXPECT_FALSE(is_view_op(stage));
    EXPECT_TRUE(stage_outputs_may_alias_inputs(stage));
}

TEST(InferPipelineReuseTest, RuntimeViewContractUsesCompilerDescriptorOnly) {
    auto runtime_descriptor = make_test_runtime_descriptor(1);
    runtime_descriptor->stages.front().tensor_view_only = true;

    InferStage stage;
    stage.stage = std::make_unique<DescriptorProbeStage>("Counting");
    stage.runtime_session = std::make_shared<RuntimeSession>(runtime_descriptor);
    stage.runtime_stage_index = 0;

    EXPECT_TRUE(is_view_op(stage));
    EXPECT_TRUE(stage_outputs_may_alias_inputs(stage));
}

TEST(InferPipelineReuseTest,
     SharedFusedOutputLifetimePlannerUsesRuntimeMemoryContracts) {
    auto runtime_descriptor = make_test_runtime_descriptor(3);
    ASSERT_EQ(runtime_descriptor->stages.size(), 3u);

    auto& view_stage = runtime_descriptor->stages[1];
    view_stage.tensor_view_only = true;
    RuntimeTensorBindingContract view_input;
    view_input.alias_group = "stage_0";
    view_stage.input_bindings.push_back(std::move(view_input));
    ASSERT_FALSE(view_stage.output_bindings.empty());
    view_stage.output_bindings.front().alias_group = "stage_0";

    std::vector<FusedOutputLifetimeStage> stages(3);
    stages[0].descriptor = &runtime_descriptor->stages[0];
    stages[0].output_indices = {0};
    stages[1].descriptor = &runtime_descriptor->stages[1];
    stages[1].inputs = {
        {FusedOutputLifetimeInputRef::Kind::Output, 0}};
    stages[1].output_indices = {1};
    stages[2].descriptor = &runtime_descriptor->stages[2];
    stages[2].inputs = {
        {FusedOutputLifetimeInputRef::Kind::Output, 1}};
    stages[2].output_indices = {2};

    const auto lifetimes = build_fused_output_lifetime_plan(
        stages, runtime_descriptor->memory_plan, /*output_count=*/3);

    ASSERT_EQ(lifetimes.size(), 3u);
    EXPECT_EQ(lifetimes[0].produced_at, 0u);
    EXPECT_EQ(lifetimes[0].last_used_at, 2u);
    EXPECT_TRUE(lifetimes[0].requires_buffer);
    EXPECT_EQ(lifetimes[1].produced_at, 1u);
    EXPECT_EQ(lifetimes[1].last_used_at, 2u);
    EXPECT_FALSE(lifetimes[1].requires_buffer);
    EXPECT_EQ(lifetimes[1].storage_source_output, 0u);
    EXPECT_EQ(lifetimes[2].produced_at, 2u);
    EXPECT_EQ(lifetimes[2].last_used_at, 2u);
}

TEST(InferPipelineReuseTest, BuildPipelineCarriesCompilerOwnedOutputLifetimes) {
    PipelineStageDesc desc;
    desc.stage = std::make_unique<CountingStage>();
    desc.outputs.push_back(OutputDesc{{4}, ov::element::f32, false});
    PipelineStageDesc::OutputLifetime first_lifetime;
    first_lifetime.produced_at = 0;
    first_lifetime.last_used_at = 2;
    first_lifetime.requires_buffer = false;
    desc.output_lifetimes = {first_lifetime};

    std::vector<PipelineStageDesc> descs;
    descs.push_back(std::move(desc));

    auto pipeline = build_infer_pipeline(descs,
                                         nullptr,
                                         nullptr,
                                         false,
                                         make_test_runtime_descriptor(1));

    ASSERT_EQ(pipeline.size(), 1u);
    ASSERT_EQ(pipeline.front().output_lifetimes.size(), 1u);
    EXPECT_EQ(pipeline.front().output_lifetimes[0].produced_at, 0u);
    EXPECT_EQ(pipeline.front().output_lifetimes[0].last_used_at, 2u);
    EXPECT_FALSE(pipeline.front().output_lifetimes[0].requires_buffer);
}

TEST(InferPipelineReuseTest, ReusesClonedPipelineAndOutputHandlesAcrossPreparations) {
    CountingStage::reset_counters();

    PipelineStageDesc desc;
    desc.stage = std::make_unique<CountingStage>();
    desc.runtime_stage_index = 2;
    desc.outputs.push_back(OutputDesc{{4}, ov::element::f32, false});
    std::vector<PipelineStageDesc> descs;
    descs.push_back(std::move(desc));

    FakeAllocator allocator;
    GpuBufferPool pool(allocator);
    std::vector<InferStage> reusable_pipeline;
    std::vector<std::vector<BufferHandle>> stage_handles;
    StageOutputBufferWorkspace stage_workspace;
    std::vector<std::shared_ptr<GfxRemoteTensor>> remote_outputs;
    const std::vector<std::shared_ptr<GfxRemoteTensor>> remote_inputs;
    const std::vector<ov::Output<const ov::Node>> outputs;
    const std::unordered_map<const ov::Node*, size_t> node_map;
    const std::unordered_map<const ov::Node*, size_t> param_map;
    auto runtime_descriptor = make_test_runtime_descriptor(3);

    auto describe_output = [](InferStage& stage,
                              size_t oi,
                              GpuTensor& out_ref,
                              GpuBufferDesc& desc,
                              const char* error_prefix) {
        return init_stage_output_desc(GpuBackend::Metal,
                                      stage,
                                      oi,
                                      out_ref,
                                      desc,
                                      /*is_model_output=*/false,
                                      /*skip_view_ops=*/true,
                                      error_prefix);
    };

    auto& first = prepare_reusable_pipeline_with_outputs(reusable_pipeline,
                                                         descs,
                                                         nullptr,
                                                         nullptr,
                                                         false,
                                                         nullptr,
                                                         outputs,
                                                         node_map,
                                                         param_map,
                                                         remote_outputs,
                                                         remote_inputs,
                                                         GpuBackend::Metal,
                                                         runtime_descriptor,
                                                         pool,
                                                         stage_handles,
                                                         &stage_workspace,
                                                         [](std::vector<InferStage>&) {},
                                                         describe_output,
                                                         "test");
    ASSERT_EQ(first.size(), 1u);
    ASSERT_EQ(first.front().outputs.size(), 1u);
    ASSERT_TRUE(first.front().outputs.front()->buf.valid());
    const auto first_uid = first.front().outputs.front()->buf.allocation_uid;

    EXPECT_EQ(CountingStage::clone_count(), 1u);
    EXPECT_EQ(CountingStage::init_count(), 1u);
    EXPECT_EQ(CountingStage::compile_count(), 1u);
    EXPECT_EQ(allocator.allocate_count(), 1u);

    auto& second = prepare_reusable_pipeline_with_outputs(reusable_pipeline,
                                                          descs,
                                                          nullptr,
                                                          nullptr,
                                                          false,
                                                          nullptr,
                                                          outputs,
                                                          node_map,
                                                          param_map,
                                                          remote_outputs,
                                                          remote_inputs,
                                                          GpuBackend::Metal,
                                                          runtime_descriptor,
                                                          pool,
                                                          stage_handles,
                                                          &stage_workspace,
                                                          [](std::vector<InferStage>&) {},
                                                          describe_output,
                                                          "test");
    ASSERT_EQ(second.size(), 1u);
    ASSERT_EQ(second.front().outputs.size(), 1u);
    ASSERT_TRUE(second.front().outputs.front()->buf.valid());

    EXPECT_EQ(CountingStage::clone_count(), 1u);
    EXPECT_EQ(CountingStage::init_count(), 1u);
    EXPECT_EQ(CountingStage::compile_count(), 1u);
    EXPECT_EQ(allocator.allocate_count(), 1u);
    EXPECT_EQ(second.front().outputs.front()->buf.allocation_uid, first_uid);
}

TEST(InferPipelineReuseTest, RuntimeMemoryPlanExtendsWorkspaceOutputLifetime) {
    CountingStage::reset_counters();

    std::vector<PipelineStageDesc> descs;
    for (size_t i = 0; i < 2; ++i) {
        PipelineStageDesc desc;
        desc.stage = std::make_unique<CountingStage>();
        desc.runtime_stage_index = i;
        desc.outputs.push_back(OutputDesc{{4}, ov::element::f32, false});
        descs.push_back(std::move(desc));
    }

    FakeAllocator allocator;
    GpuBufferPool pool(allocator);
    std::vector<InferStage> reusable_pipeline;
    std::vector<std::vector<BufferHandle>> stage_handles;
    StageOutputBufferWorkspace stage_workspace;
    std::vector<std::shared_ptr<GfxRemoteTensor>> remote_outputs;
    const std::vector<std::shared_ptr<GfxRemoteTensor>> remote_inputs;
    const std::vector<ov::Output<const ov::Node>> outputs;
    const std::unordered_map<const ov::Node*, size_t> node_map;
    const std::unordered_map<const ov::Node*, size_t> param_map;
    auto runtime_descriptor =
        make_test_runtime_descriptor(2, /*output_last_stages=*/{1, 1});

    auto describe_output = [](InferStage& stage,
                              size_t oi,
                              GpuTensor& out_ref,
                              GpuBufferDesc& desc,
                              const char* error_prefix) {
        return init_stage_output_desc(GpuBackend::Metal,
                                      stage,
                                      oi,
                                      out_ref,
                                      desc,
                                      /*is_model_output=*/false,
                                      /*skip_view_ops=*/true,
                                      error_prefix);
    };

    auto& pipeline = prepare_reusable_pipeline_with_outputs(
        reusable_pipeline,
        descs,
        nullptr,
        nullptr,
        false,
        nullptr,
        outputs,
        node_map,
        param_map,
        remote_outputs,
        remote_inputs,
        GpuBackend::Metal,
        runtime_descriptor,
        pool,
        stage_handles,
        &stage_workspace,
        [](std::vector<InferStage>&) {},
        describe_output,
        "test");

    ASSERT_EQ(pipeline.size(), 2u);
    ASSERT_TRUE(pipeline[0].outputs.front()->buf.valid());
    ASSERT_TRUE(pipeline[1].outputs.front()->buf.valid());
    EXPECT_NE(pipeline[0].outputs.front()->buf.allocation_uid,
              pipeline[1].outputs.front()->buf.allocation_uid);
    EXPECT_EQ(allocator.allocate_count(), 2u);
    EXPECT_EQ(stage_workspace.last_slots_used, 2u);
    EXPECT_EQ(stage_workspace.last_peak_live_slots, 2u);
}

TEST(InferPipelineReuseTest, ReusesPreparedExecutionInputsAcrossInferences) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto relu = std::make_shared<ov::op::v0::Relu>(param0);
    auto add = std::make_shared<ov::op::v1::Add>(relu, param1);

    std::vector<InferStage> pipeline(2);
    pipeline[0].node = relu;
    pipeline[0].stage = std::make_unique<TrackingStage>();
    pipeline[0].outputs.push_back(std::make_unique<GpuTensor>());
    pipeline[0].outputs[0]->shape = {1, 4};
    pipeline[0].outputs[0]->expected_type = ov::element::f16;

    pipeline[1].node = add;
    pipeline[1].stage = std::make_unique<TrackingStage>();
    pipeline[1].outputs.push_back(std::make_unique<GpuTensor>());
    pipeline[1].outputs[0]->shape = {1, 4};
    pipeline[1].outputs[0]->expected_type = ov::element::f16;
    pipeline[1].inputs.push_back({relu, 0});
    pipeline[1].inputs.push_back({param1, 0});

    const std::unordered_map<const ov::Node*, size_t> node_map = {
        {relu.get(), 0},
        {add.get(), 1},
    };
    const std::unordered_map<const ov::Node*, size_t> param_map = {
        {param1.get(), 0},
    };

    PreparedInferExecutionPlan plan;
    prepare_reusable_execution_plan(plan, pipeline, node_map, param_map);
    ASSERT_EQ(plan.stages.size(), pipeline.size());
    ASSERT_EQ(plan.stages[1].resolved_inputs.size(), 2u);
    EXPECT_EQ(plan.stages[1].resolved_inputs[0], pipeline[0].outputs[0].get());
    EXPECT_EQ(plan.stages[1].resolved_inputs[1], nullptr);

    GpuTensor external_a;
    external_a.shape = {1, 4};
    external_a.expected_type = ov::element::f16;
    execute_pipeline(
        pipeline,
        node_map,
        param_map,
        [&](size_t idx) -> GpuTensor* {
            return idx == 0 ? &external_a : nullptr;
        },
        [](InferStage&, const std::vector<GpuTensor*>&) {},
        &plan);

    auto* tracking = static_cast<TrackingStage*>(pipeline[1].stage.get());
    ASSERT_EQ(tracking->last_inputs.size(), 2u);
    EXPECT_EQ(tracking->last_inputs[0], pipeline[0].outputs[0].get());
    EXPECT_EQ(tracking->last_inputs[1], &external_a);
    const auto* first_inputs_data = tracking->last_inputs_data;

    GpuTensor external_b;
    external_b.shape = {1, 4};
    external_b.expected_type = ov::element::f16;
    execute_pipeline(
        pipeline,
        node_map,
        param_map,
        [&](size_t idx) -> GpuTensor* {
            return idx == 0 ? &external_b : nullptr;
        },
        [](InferStage&, const std::vector<GpuTensor*>&) {},
        &plan);

    EXPECT_EQ(tracking->set_inputs_count, 2u);
    EXPECT_EQ(tracking->last_inputs[0], pipeline[0].outputs[0].get());
    EXPECT_EQ(tracking->last_inputs[1], &external_b);
    EXPECT_EQ(tracking->last_inputs_data, first_inputs_data);
}

TEST(InferPipelineReuseTest, PrewarmPipelineUsesPreparedExecutionInputs) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto relu = std::make_shared<ov::op::v0::Relu>(param);

    std::vector<InferStage> pipeline(1);
    pipeline[0].node = relu;
    pipeline[0].stage = std::make_unique<TrackingStage>();
    pipeline[0].inputs.push_back({param, 0});
    pipeline[0].outputs.push_back(std::make_unique<GpuTensor>());
    pipeline[0].outputs[0]->shape = {1, 4};
    pipeline[0].outputs[0]->expected_type = ov::element::f16;

    const std::unordered_map<const ov::Node*, size_t> node_map = {
        {relu.get(), 0},
    };
    const std::unordered_map<const ov::Node*, size_t> param_map = {
        {param.get(), 0},
    };

    PreparedInferExecutionPlan plan;
    prepare_reusable_execution_plan(plan, pipeline, node_map, param_map);

    GpuTensor external;
    external.shape = {1, 4};
    external.expected_type = ov::element::f16;
    prewarm_pipeline_runtime_state(
        pipeline,
        node_map,
        param_map,
        [&](size_t idx) -> GpuTensor* {
            return idx == 0 ? &external : nullptr;
        },
        &plan);

    auto* tracking = static_cast<TrackingStage*>(pipeline[0].stage.get());
    ASSERT_EQ(tracking->prewarm_count, 1u);
    ASSERT_EQ(tracking->prewarm_inputs.size(), 1u);
    EXPECT_EQ(tracking->prewarm_inputs[0], &external);
    EXPECT_EQ(tracking->prewarm_inputs_data, plan.stages[0].resolved_inputs.data());
}

TEST(InferPipelineReuseTest, ReusesPreparedOutputResolutionAcrossInferences) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    auto result = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    std::vector<InferStage> pipeline(1);
    pipeline[0].node = relu;
    pipeline[0].stage = std::make_unique<TrackingStage>();
    pipeline[0].outputs.push_back(std::make_unique<GpuTensor>());
    pipeline[0].outputs[0]->shape = {1, 4};
    pipeline[0].outputs[0]->expected_type = ov::element::f16;
    pipeline[0].outputs[0]->buf.buffer = reinterpret_cast<GpuBufferHandle>(0x1234);
    pipeline[0].outputs[0]->buf.size = 8;
    pipeline[0].outputs[0]->buf.type = ov::element::f16;

    const std::unordered_map<const ov::Node*, size_t> node_map = {
        {relu.get(), 0},
    };
    const std::unordered_map<const ov::Node*, size_t> param_map;
    const auto model_outputs = model->outputs();
    const std::vector<ov::Output<const ov::Node>> public_outputs(model_outputs.begin(), model_outputs.end());

    PreparedInferOutputPlan output_plan;
    prepare_reusable_output_plan(output_plan,
                                 public_outputs,
                                 model,
                                 pipeline,
                                 node_map,
                                 param_map,
                                 "test");
    ASSERT_EQ(output_plan.outputs.size(), 1u);
    EXPECT_EQ(output_plan.outputs[0].kind, PreparedOutputSourceKind::StageOutput);
    EXPECT_EQ(output_plan.outputs[0].index, 0u);
    EXPECT_EQ(output_plan.outputs[0].port, 0u);
    EXPECT_EQ(output_plan.outputs[0].static_shape, (ov::Shape{1, 4}));
    EXPECT_EQ(output_plan.outputs[0].static_type, ov::element::f16);

    std::vector<std::shared_ptr<GfxRemoteTensor>> remote_outputs;
    GpuTensor* resolved_output = nullptr;
    OutputViewInfo resolved_info;
    size_t host_override_calls = 0;

    bind_outputs_common(
        public_outputs,
        model,
        node_map,
        param_map,
        pipeline,
        [](size_t) -> GpuTensor* {
            return nullptr;
        },
        remote_outputs,
        [&](size_t idx, const ov::element::Type& type, const ov::Shape& shape, const char*) -> const ov::Tensor* {
            ++host_override_calls;
            EXPECT_EQ(idx, 0u);
            EXPECT_EQ(type, ov::element::f16);
            EXPECT_EQ(shape, (ov::Shape{1, 4}));
            return nullptr;
        },
        [](size_t, const std::shared_ptr<GfxRemoteTensor>&) {
            FAIL() << "unexpected remote output binding";
        },
        [&](size_t idx, GpuTensor& dev, const OutputViewInfo& info, const ov::Tensor*) {
            EXPECT_EQ(idx, 0u);
            resolved_output = &dev;
            resolved_info = info;
        },
        &output_plan,
        /*allow_missing=*/false,
        "test");

    EXPECT_EQ(host_override_calls, 1u);
    EXPECT_EQ(resolved_output, pipeline[0].outputs[0].get());
    EXPECT_EQ(resolved_info.shape, (ov::Shape{1, 4}));
    EXPECT_EQ(resolved_info.type, ov::element::f16);
    ASSERT_TRUE(resolved_info.source.node);
    EXPECT_EQ(resolved_info.source.node.get(), relu.get());
    EXPECT_EQ(resolved_info.source.port, 0u);
}

TEST(InferPipelineReuseTest, ReusesPreparedHostOutputsAcrossInferences) {
    PreparedInferOutputPlan output_plan;
    output_plan.outputs.resize(1);
    output_plan.outputs[0].static_shape = {1, 4};
    output_plan.outputs[0].static_type = ov::element::f16;

    PreparedInferHostOutputPlan host_plan;
    std::vector<ov::Tensor> bound_output_hosts(1);

    prepare_reusable_host_output_plan(host_plan, output_plan, bound_output_hosts);
    ASSERT_EQ(host_plan.outputs.size(), 1u);
    ASSERT_TRUE(host_plan.outputs[0].host);
    EXPECT_EQ(host_plan.outputs[0].host.get_shape(), (ov::Shape{1, 4}));
    EXPECT_EQ(host_plan.outputs[0].host.get_element_type(), ov::element::f16);
    auto* first_ptr = host_plan.outputs[0].host.data();
    ASSERT_NE(first_ptr, nullptr);

    prepare_reusable_host_output_plan(host_plan, output_plan, bound_output_hosts);
    ASSERT_TRUE(host_plan.outputs[0].host);
    EXPECT_EQ(host_plan.outputs[0].host.data(), first_ptr);

    bound_output_hosts[0] = ov::Tensor(ov::element::f16, ov::Shape{1, 4});
    prepare_reusable_host_output_plan(host_plan, output_plan, bound_output_hosts);
    EXPECT_FALSE(host_plan.outputs[0].host);
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
