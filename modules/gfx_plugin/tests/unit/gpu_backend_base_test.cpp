// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "runtime/gpu_backend_base.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

struct FakeBackendState {
    explicit FakeBackendState(int value) : value(value) {}
    int value = 0;
};

class FakeCompiledKernel final : public CompiledKernelBase {
public:
    explicit FakeCompiledKernel(uint32_t arg_count = 0) : CompiledKernelBase(arg_count) {}

    explicit FakeCompiledKernel(std::shared_ptr<const KernelBindingPlan> binding_plan)
        : CompiledKernelBase(std::move(binding_plan)) {}

    FakeCompiledKernel(std::shared_ptr<const KernelBindingPlan> binding_plan, std::shared_ptr<void> prepared_binding_cache)
        : CompiledKernelBase(std::move(binding_plan), std::move(prepared_binding_cache)) {}

    size_t clamp_threadgroup_size(size_t desired) const override {
        return desired;
    }

    std::shared_ptr<ICompiledKernel> fork() const override {
        return std::make_shared<FakeCompiledKernel>(binding_plan(), prepared_binding_cache());
    }

    void execute(GpuCommandBufferHandle,
                 const KernelDispatch&,
                 const std::vector<KernelArg>&,
                 const KernelExecutionHooks*) override {}

    uint32_t resolve(const std::vector<KernelArg>& args, const char* label) const {
        return resolve_runtime_arg_count(args, label);
    }

    KernelBindingTable bindings(const std::vector<KernelArg>& args, const char* label) const {
        return materialize_runtime_bindings(args, label);
    }

    std::shared_ptr<const PreparedKernelBindings> prepared(const std::vector<KernelArg>& args, const char* label) const {
        return get_or_create_prepared_bindings(args, label);
    }

    mutable size_t prepared_binding_create_count = 0;

private:
    std::shared_ptr<const PreparedKernelBindings> create_prepared_bindings(
        const KernelBindingTable& bindings) const override {
        ++prepared_binding_create_count;
        return CompiledKernelBase::create_prepared_bindings(bindings);
    }
};

TEST(GpuBackendBaseTest, KernelBindingPlanValidatesDenseArgs) {
    FakeCompiledKernel kernel(/*arg_count=*/2);

    GpuBuffer a{};
    a.buffer = reinterpret_cast<GpuBufferHandle>(0x1);
    a.size = 16;
    GpuBuffer b{};
    b.buffer = reinterpret_cast<GpuBufferHandle>(0x2);
    b.size = 16;

    std::vector<KernelArg> args = {make_buffer_arg(0, a), make_buffer_arg(1, b)};
    EXPECT_EQ(kernel.resolve(args, "FakeKernel"), 2u);
}

TEST(GpuBackendBaseTest, KernelBindingPlanRejectsMismatchedArgCount) {
    FakeCompiledKernel kernel(/*arg_count=*/2);

    GpuBuffer a{};
    a.buffer = reinterpret_cast<GpuBufferHandle>(0x1);
    a.size = 16;

    std::vector<KernelArg> args = {make_buffer_arg(0, a)};
    EXPECT_THROW(kernel.resolve(args, "FakeKernel"), ov::Exception);
}

TEST(GpuBackendBaseTest, ForkReusesResolvedBindingPlan) {
    FakeCompiledKernel kernel;
    kernel.set_args_count(3);

    auto forked = kernel.fork();
    ASSERT_TRUE(forked);
    EXPECT_EQ(kernel.args_count(), 3u);
    EXPECT_EQ(forked->args_count(), 3u);
}

TEST(GpuBackendBaseTest, MaterializeKernelBindingTableProducesDenseOrderedBindings) {
    GpuBuffer a{};
    a.buffer = reinterpret_cast<GpuBufferHandle>(0x10);
    a.size = 64;
    GpuBuffer b{};
    b.buffer = reinterpret_cast<GpuBufferHandle>(0x20);
    b.size = 128;

    std::vector<KernelArg> args;
    args.push_back(make_buffer_arg(1, b, 8));
    args.push_back(make_buffer_arg(0, a, 4));

    const auto table = materialize_kernel_binding_table(args, "FakeKernel");
    ASSERT_EQ(table.buffers.size(), 2u);
    EXPECT_EQ(table.buffers[0].buffer.buffer, a.buffer);
    EXPECT_EQ(table.buffers[0].offset, 4u);
    EXPECT_EQ(table.buffers[1].buffer.buffer, b.buffer);
    EXPECT_EQ(table.buffers[1].offset, 8u);
}

TEST(GpuBackendBaseTest, KernelBindingPlanMaterializesFixedAbiBindings) {
    FakeCompiledKernel kernel(/*arg_count=*/2);

    GpuBuffer a{};
    a.buffer = reinterpret_cast<GpuBufferHandle>(0x10);
    a.size = 64;
    GpuBuffer b{};
    b.buffer = reinterpret_cast<GpuBufferHandle>(0x20);
    b.size = 128;

    std::vector<KernelArg> args = {make_buffer_arg(0, a, 4), make_buffer_arg(1, b, 8)};
    const auto table = kernel.bindings(args, "FakeKernel");
    ASSERT_EQ(table.buffers.size(), 2u);
    EXPECT_EQ(table.buffers[0].buffer.buffer, a.buffer);
    EXPECT_EQ(table.buffers[0].offset, 4u);
    EXPECT_EQ(table.buffers[1].buffer.buffer, b.buffer);
    EXPECT_EQ(table.buffers[1].offset, 8u);
}

TEST(GpuBackendBaseTest, KernelBindingTableUsesStableAllocationIdentityWhenAvailable) {
    KernelBindingTable first;
    KernelBindingTable second;

    first.buffers.resize(1);
    second.buffers.resize(1);
    first.buffers[0].buffer.buffer = reinterpret_cast<GpuBufferHandle>(0x10);
    second.buffers[0].buffer.buffer = reinterpret_cast<GpuBufferHandle>(0x10);
    first.buffers[0].buffer.size = 64;
    second.buffers[0].buffer.size = 64;
    first.buffers[0].buffer.allocation_uid = 1;
    second.buffers[0].buffer.allocation_uid = 2;

    EXPECT_FALSE(first == second);
    EXPECT_NE(KernelBindingTableHash{}(first), KernelBindingTableHash{}(second));
}

TEST(GpuBackendBaseTest, PreparedBindingsAreReusedAcrossForkedKernels) {
    FakeCompiledKernel kernel(/*arg_count=*/1);

    GpuBuffer a{};
    a.buffer = reinterpret_cast<GpuBufferHandle>(0x10);
    a.size = 64;
    a.allocation_uid = 7;

    std::vector<KernelArg> args = {make_buffer_arg(0, a, 4)};
    auto first = kernel.prepared(args, "FakeKernel");
    ASSERT_TRUE(first);
    EXPECT_EQ(kernel.prepared_binding_create_count, 1u);

    auto forked = kernel.fork();
    ASSERT_TRUE(forked);
    auto* forked_kernel = dynamic_cast<FakeCompiledKernel*>(forked.get());
    ASSERT_NE(forked_kernel, nullptr);

    auto second = forked_kernel->prepared(args, "FakeKernel");
    ASSERT_TRUE(second);
    EXPECT_EQ(second.get(), first.get());
    EXPECT_EQ(kernel.prepared_binding_create_count, 1u);
    EXPECT_EQ(forked_kernel->prepared_binding_create_count, 0u);
}

TEST(GpuBackendBaseTest, PreparedBindingsAreReusedAcrossDistinctKernelsSharingRegistryCache) {
    auto binding_plan = std::make_shared<KernelBindingPlan>(1);
    auto shared_cache = acquire_shared_prepared_binding_cache(GpuBackend::Metal,
                                                              /*device=*/0x1234,
                                                              /*arg_count=*/1);

    FakeCompiledKernel first(binding_plan, shared_cache);
    FakeCompiledKernel second(binding_plan, shared_cache);

    GpuBuffer a{};
    a.buffer = reinterpret_cast<GpuBufferHandle>(0x10);
    a.size = 64;
    a.allocation_uid = 7;

    std::vector<KernelArg> args = {make_buffer_arg(0, a, 4)};
    auto first_prepared = first.prepared(args, "FakeKernel");
    auto second_prepared = second.prepared(args, "FakeKernel");

    ASSERT_TRUE(first_prepared);
    ASSERT_TRUE(second_prepared);
    EXPECT_EQ(first_prepared.get(), second_prepared.get());
    EXPECT_EQ(first.prepared_binding_create_count, 1u);
    EXPECT_EQ(second.prepared_binding_create_count, 0u);
}

TEST(GpuBackendBaseTest, PreparedBindingsStayAliveInSharedCacheAcrossLookups) {
    FakeCompiledKernel kernel(/*arg_count=*/1);

    GpuBuffer a{};
    a.buffer = reinterpret_cast<GpuBufferHandle>(0x10);
    a.size = 64;
    a.allocation_uid = 7;

    std::vector<KernelArg> args = {make_buffer_arg(0, a, 4)};
    auto first = kernel.prepared(args, "FakeKernel");
    ASSERT_TRUE(first);
    std::weak_ptr<const PreparedKernelBindings> weak = first;
    first.reset();

    EXPECT_FALSE(weak.expired());

    auto second = kernel.prepared(args, "FakeKernel");
    ASSERT_TRUE(second);
    ASSERT_FALSE(weak.expired());
    EXPECT_EQ(second.get(), weak.lock().get());
    EXPECT_EQ(kernel.prepared_binding_create_count, 1u);
}

TEST(GpuBackendBaseTest, PreparedBindingsReuseBackendStateForSameSchemaKey) {
    FakeCompiledKernel kernel(/*arg_count=*/1);

    GpuBuffer a{};
    a.buffer = reinterpret_cast<GpuBufferHandle>(0x10);
    a.size = 64;
    a.allocation_uid = 7;

    std::vector<KernelArg> args = {make_buffer_arg(0, a, 4)};
    auto prepared = kernel.prepared(args, "FakeKernel");
    ASSERT_TRUE(prepared);

    size_t create_count = 0;
    auto first = prepared->get_or_create_backend_state<FakeBackendState>(
        /*state_key=*/0x100,
        [&]() {
            ++create_count;
            return std::make_shared<FakeBackendState>(17);
        });
    auto second = prepared->get_or_create_backend_state<FakeBackendState>(
        /*state_key=*/0x100,
        [&]() {
            ++create_count;
            return std::make_shared<FakeBackendState>(19);
        });

    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_EQ(first.get(), second.get());
    EXPECT_EQ(first->value, 17);
    EXPECT_EQ(create_count, 1u);
}

TEST(GpuBackendBaseTest, PreparedBindingsKeepBackendStateAliveAcrossTransientHandles) {
    FakeCompiledKernel kernel(/*arg_count=*/1);

    GpuBuffer a{};
    a.buffer = reinterpret_cast<GpuBufferHandle>(0x10);
    a.size = 64;
    a.allocation_uid = 7;

    std::vector<KernelArg> args = {make_buffer_arg(0, a, 4)};
    auto prepared = kernel.prepared(args, "FakeKernel");
    ASSERT_TRUE(prepared);

    auto state = prepared->get_or_create_backend_state<FakeBackendState>(
        /*state_key=*/0x100,
        [&]() { return std::make_shared<FakeBackendState>(17); });
    ASSERT_TRUE(state);
    std::weak_ptr<FakeBackendState> weak_state = state;
    state.reset();
    prepared.reset();

    EXPECT_FALSE(weak_state.expired());

    auto prepared_again = kernel.prepared(args, "FakeKernel");
    ASSERT_TRUE(prepared_again);
    auto state_again = prepared_again->get_or_create_backend_state<FakeBackendState>(
        /*state_key=*/0x100,
        [&]() { return std::make_shared<FakeBackendState>(23); });

    ASSERT_TRUE(state_again);
    ASSERT_FALSE(weak_state.expired());
    EXPECT_EQ(state_again.get(), weak_state.lock().get());
    EXPECT_EQ(state_again->value, 17);
}

TEST(GpuBackendBaseTest, PreparedBindingsKeepBackendStateSeparatedPerSchemaKey) {
    FakeCompiledKernel kernel(/*arg_count=*/1);

    GpuBuffer a{};
    a.buffer = reinterpret_cast<GpuBufferHandle>(0x10);
    a.size = 64;
    a.allocation_uid = 7;

    std::vector<KernelArg> args = {make_buffer_arg(0, a, 4)};
    auto prepared = kernel.prepared(args, "FakeKernel");
    ASSERT_TRUE(prepared);

    auto first = prepared->get_or_create_backend_state<FakeBackendState>(
        /*state_key=*/0x100,
        [&]() { return std::make_shared<FakeBackendState>(17); });
    auto second = prepared->get_or_create_backend_state<FakeBackendState>(
        /*state_key=*/0x200,
        [&]() { return std::make_shared<FakeBackendState>(23); });

    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_NE(first.get(), second.get());
    EXPECT_EQ(first->value, 17);
    EXPECT_EQ(second->value, 23);
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
