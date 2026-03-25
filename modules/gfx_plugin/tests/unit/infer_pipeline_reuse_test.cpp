// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "plugin/infer_io_utils.hpp"
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
    }

    static size_t clone_count() {
        return s_clone_count;
    }

    static size_t init_count() {
        return s_init_count;
    }

    void init(GpuBufferManager*) override {
        ++s_init_count;
    }

    void compile(GpuBufferManager*) override {}
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
};

class TrackingStage final : public GpuStage {
public:
    void init(GpuBufferManager*) override {}
    void compile(GpuBufferManager*) override {}
    void execute(GpuCommandBufferHandle) override {}

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
};

TEST(InferPipelineReuseTest, ReusesClonedPipelineAndOutputHandlesAcrossPreparations) {
    CountingStage::reset_counters();

    PipelineStageDesc desc;
    desc.stage = std::make_unique<CountingStage>();
    desc.outputs.push_back(OutputDesc{{4}, ov::element::f32, false});
    std::vector<PipelineStageDesc> descs;
    descs.push_back(std::move(desc));

    FakeAllocator allocator;
    GpuBufferPool pool(allocator);
    std::vector<InferStage> reusable_pipeline;
    std::vector<std::vector<BufferHandle>> stage_handles;
    std::vector<std::shared_ptr<GfxRemoteTensor>> remote_outputs;
    const std::vector<std::shared_ptr<GfxRemoteTensor>> remote_inputs;
    const std::vector<ov::Output<const ov::Node>> outputs;
    const std::unordered_map<const ov::Node*, size_t> node_map;
    const std::unordered_map<const ov::Node*, size_t> param_map;

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
                                                         pool,
                                                         stage_handles,
                                                         [](std::vector<InferStage>&) {},
                                                         describe_output,
                                                         "test");
    ASSERT_EQ(first.size(), 1u);
    ASSERT_EQ(first.front().outputs.size(), 1u);
    ASSERT_TRUE(first.front().outputs.front()->buf.valid());
    const auto first_uid = first.front().outputs.front()->buf.allocation_uid;

    EXPECT_EQ(CountingStage::clone_count(), 1u);
    EXPECT_EQ(CountingStage::init_count(), 1u);
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
                                                          pool,
                                                          stage_handles,
                                                          [](std::vector<InferStage>&) {},
                                                          describe_output,
                                                          "test");
    ASSERT_EQ(second.size(), 1u);
    ASSERT_EQ(second.front().outputs.size(), 1u);
    ASSERT_TRUE(second.front().outputs.front()->buf.valid());

    EXPECT_EQ(CountingStage::clone_count(), 1u);
    EXPECT_EQ(CountingStage::init_count(), 1u);
    EXPECT_EQ(allocator.allocate_count(), 1u);
    EXPECT_EQ(second.front().outputs.front()->buf.allocation_uid, first_uid);
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
