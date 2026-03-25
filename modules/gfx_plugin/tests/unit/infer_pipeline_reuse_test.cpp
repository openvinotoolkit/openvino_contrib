// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "plugin/infer_io_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"

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

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
