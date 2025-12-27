// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "runtime/gpu_stage.hpp"

using namespace ov::gfx_plugin;

namespace {

class DummyStage : public GpuStage {
public:
    DummyStage() = default;

    void init(GpuBufferManager* /*buffer_manager*/) override {
        init_called = true;
    }

    void compile(GpuBufferManager* /*buffer_manager*/) override {}

    void execute(GpuCommandBufferHandle /*cb*/) override {
        executed = true;
    }

    void set_inputs(const std::vector<GpuTensor*>& inputs) override {
        inputs_ = inputs;
    }

    void set_output(GpuTensor* output) override {
        output_ = output;
    }

    const std::string& name() const override { return name_; }
    const std::string& type() const override { return type_; }

    std::unique_ptr<GpuStage> clone() const override {
        return std::make_unique<DummyStage>(*this);
    }

    const std::vector<GpuTensor*>& inputs() const { return inputs_; }
    GpuTensor* output() const { return output_; }

    bool init_called = false;
    bool executed = false;

private:
    std::string name_{"dummy_stage"};
    std::string type_{"DummyType"};
    std::vector<GpuTensor*> inputs_;
    GpuTensor* output_ = nullptr;
};

class IncrementStage : public GpuStage {
public:
    void init(GpuBufferManager* /*buffer_manager*/) override {}
    void compile(GpuBufferManager* /*buffer_manager*/) override {}

    void execute(GpuCommandBufferHandle /*cb*/) override {
        if (!output_ || !output_->buf.buffer) {
            return;
        }
        auto* value = static_cast<int*>(output_->buf.buffer);
        *value += 1;
    }

    void set_inputs(const std::vector<GpuTensor*>& inputs) override { inputs_ = inputs; }
    void set_output(GpuTensor* output) override { output_ = output; }

    const std::string& name() const override { return name_; }
    const std::string& type() const override { return type_; }

    std::unique_ptr<GpuStage> clone() const override {
        return std::make_unique<IncrementStage>(*this);
    }

private:
    std::string name_{"increment_stage"};
    std::string type_{"Mock"}; 
    std::vector<GpuTensor*> inputs_;
    GpuTensor* output_ = nullptr;
};

}  // namespace

TEST(GpuStageTest, StoresMetadataAndBindings) {
    DummyStage stage;
    GpuTensor in0;
    GpuTensor in1;
    GpuTensor out;

    stage.set_inputs({&in0, &in1});
    stage.set_output(&out);

    EXPECT_EQ(stage.name(), "dummy_stage");
    EXPECT_EQ(stage.type(), "DummyType");
    ASSERT_EQ(stage.inputs().size(), 2u);
    EXPECT_EQ(stage.inputs()[0], &in0);
    EXPECT_EQ(stage.inputs()[1], &in1);
    EXPECT_EQ(stage.output(), &out);
}

TEST(GpuStageTest, VirtualDispatchExecute) {
    DummyStage derived;
    GpuStage* base = &derived;

    base->init(nullptr);
    base->execute(nullptr);

    EXPECT_TRUE(derived.init_called);
    EXPECT_TRUE(derived.executed);
}

TEST(GpuStageTest, MockStageIncrementsBuffer) {
    IncrementStage stage;
    int value = 41;
    GpuTensor out;
    out.buf.buffer = &value;
    out.buf.size = sizeof(value);

    stage.set_output(&out);
    stage.execute(nullptr);

    EXPECT_EQ(value, 42);
}
