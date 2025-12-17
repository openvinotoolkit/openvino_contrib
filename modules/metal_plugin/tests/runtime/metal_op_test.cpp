// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "runtime/metal_op.hpp"

using namespace ov::metal_plugin;

namespace {

class DummyOp : public MetalOp {
public:
    DummyOp()
        : MetalOp("dummy_op",
                  "DummyType",
                  ov::Shape{1, 3, 4, 4},
                  /*device=*/nullptr,
                  /*queue=*/nullptr) {}

    void init(MetalBufferManager* buffer_manager) override {
        init_called = true;
        MetalOp::init(buffer_manager);
    }

    void execute() override {
        start_profiling();
        executed = true;
        stop_profiling_ms();
    }

    bool init_called = false;
    bool executed = false;
};

}  // namespace

TEST(MetalOpTest, StoresMetadataAndBindings) {
    DummyOp op;
    MetalTensor in0;
    MetalTensor in1;
    MetalTensor out;

    op.set_inputs({&in0, &in1});
    op.set_output(&out);

    EXPECT_EQ(op.name(), "dummy_op");
    EXPECT_EQ(op.type(), "DummyType");
    EXPECT_EQ(op.output_shape(), ov::Shape({1, 3, 4, 4}));
    ASSERT_EQ(op.inputs().size(), 2u);
    EXPECT_EQ(op.inputs()[0], &in0);
    EXPECT_EQ(op.inputs()[1], &in1);
    EXPECT_EQ(op.output(), &out);
}

TEST(MetalOpTest, VirtualDispatchExecute) {
    DummyOp derived;
    MetalOp* base = &derived;

    base->enable_profiling(true);
    base->init(nullptr);
    base->execute();

    EXPECT_TRUE(derived.init_called);
    EXPECT_TRUE(derived.executed);
    EXPECT_GE(derived.last_exec_duration_ms(), 0.0);
}

