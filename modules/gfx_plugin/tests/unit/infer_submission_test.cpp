// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <limits>

#include "plugin/infer_submission.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

class FakeStage final : public GpuStage {
public:
    explicit FakeStage(GpuStageSubmitPolicy policy = {}) : m_policy(policy) {}

    void init(GpuBufferManager*) override {}
    void compile(GpuBufferManager*) override {}

    void execute(GpuCommandBufferHandle) override {
        ++execute_count;
    }

    void set_inputs(const std::vector<GpuTensor*>& inputs) override {
        set_inputs_count = inputs.size();
    }

    void set_output(GpuTensor*) override {}

    void on_command_buffer_complete() override {
        ++completion_count;
    }

    const std::string& name() const override {
        static const std::string kName = "FakeStage";
        return kName;
    }

    const std::string& type() const override {
        static const std::string kType = "Fake";
        return kType;
    }

    GpuStageSubmitPolicy submit_policy() const override {
        return m_policy;
    }

    std::unique_ptr<GpuStage> clone() const override {
        return std::make_unique<FakeStage>(m_policy);
    }

    size_t execute_count = 0;
    size_t completion_count = 0;
    size_t set_inputs_count = 0;

private:
    GpuStageSubmitPolicy m_policy;
};

class FakeSubmissionSession final : public TrackedInferSubmissionSession {
public:
    explicit FakeSubmissionSession(bool incremental_submit) : m_incremental_submit(incremental_submit) {}

    bool supports_incremental_submit() const override {
        return m_incremental_submit;
    }

    size_t submit_count() const {
        return m_submit_count;
    }

    size_t finish_count() const {
        return m_finish_count;
    }

    const std::vector<bool>& continue_recording_flags() const {
        return m_continue_recording_flags;
    }

protected:
    GpuCommandBufferHandle begin_recording_impl() override {
        ++m_begin_count;
        return reinterpret_cast<GpuCommandBufferHandle>(m_begin_count);
    }

    void submit_recorded_impl(GpuCommandBufferHandle, bool continue_recording) override {
        ++m_submit_count;
        m_continue_recording_flags.push_back(continue_recording);
    }

    void finish_impl() override {
        ++m_finish_count;
    }

private:
    bool m_incremental_submit = true;
    size_t m_begin_count = 0;
    size_t m_submit_count = 0;
    size_t m_finish_count = 0;
    std::vector<bool> m_continue_recording_flags;
};

class FakeSingleFlightSubmissionSession final : public SingleFlightInferSubmissionSession {
public:
    size_t prepare_count() const {
        return m_prepare_count;
    }

    size_t submit_count() const {
        return m_submit_count;
    }

    size_t finish_count() const {
        return m_finish_count;
    }

    const std::vector<bool>& continue_recording_flags() const {
        return m_continue_recording_flags;
    }

protected:
    void prepare_submission_slot() override {
        ++m_prepare_count;
    }

    GpuCommandBufferHandle begin_recording_on_slot() override {
        ++m_begin_count;
        return reinterpret_cast<GpuCommandBufferHandle>(m_begin_count);
    }

    void submit_recorded_on_slot(GpuCommandBufferHandle, bool continue_recording) override {
        ++m_submit_count;
        m_continue_recording_flags.push_back(continue_recording);
    }

    void finish_submission_slot() override {
        ++m_finish_count;
    }

private:
    size_t m_prepare_count = 0;
    size_t m_begin_count = 0;
    size_t m_submit_count = 0;
    size_t m_finish_count = 0;
    std::vector<bool> m_continue_recording_flags;
};

class FakeRotatingSubmissionSession final : public RotatingSlotInferSubmissionSession {
public:
    explicit FakeRotatingSubmissionSession(size_t slot_count)
        : RotatingSlotInferSubmissionSession(slot_count) {}

    const std::vector<size_t>& begin_slots() const {
        return m_begin_slots;
    }

    const std::vector<size_t>& submit_slots() const {
        return m_submit_slots;
    }

    const std::vector<bool>& continue_recording_flags() const {
        return m_continue_recording_flags;
    }

    size_t finish_count() const {
        return m_finish_count;
    }

protected:
    GpuCommandBufferHandle begin_recording_on_slot(size_t slot_index) override {
        m_begin_slots.push_back(slot_index);
        return reinterpret_cast<GpuCommandBufferHandle>(slot_index + 1);
    }

    void submit_recorded_on_slot(size_t slot_index,
                                 GpuCommandBufferHandle,
                                 bool continue_recording) override {
        m_submit_slots.push_back(slot_index);
        m_continue_recording_flags.push_back(continue_recording);
    }

    void finish_submission_slots() override {
        ++m_finish_count;
    }

private:
    std::vector<size_t> m_begin_slots;
    std::vector<size_t> m_submit_slots;
    std::vector<bool> m_continue_recording_flags;
    size_t m_finish_count = 0;
};

TEST(InferSubmissionTest, IncrementalSubmitNotifiesOnlySubmittedStages) {
    std::vector<InferStage> pipeline;
    pipeline.emplace_back();
    pipeline.emplace_back();
    pipeline.emplace_back();
    auto* stage0 = new FakeStage();
    auto* stage1 = new FakeStage(GpuStageSubmitPolicy{.weight = 1, .isolate = true});
    auto* stage2 = new FakeStage();
    pipeline[0].stage.reset(stage0);
    pipeline[1].stage.reset(stage1);
    pipeline[2].stage.reset(stage2);

    FakeSubmissionSession submission(/*incremental_submit=*/true);
    InferSubmissionConfig config;
    config.max_stages_per_submit = std::numeric_limits<size_t>::max();
    config.max_output_bytes_per_submit = std::numeric_limits<size_t>::max();

    execute_pipeline_with_submission(
        pipeline,
        {},
        {},
        [](size_t) -> GpuTensor* {
            return nullptr;
        },
        submission,
        config);

    EXPECT_EQ(submission.submit_count(), 2u);
    ASSERT_EQ(submission.continue_recording_flags().size(), 2u);
    EXPECT_TRUE(submission.continue_recording_flags()[0]);
    EXPECT_FALSE(submission.continue_recording_flags()[1]);
    EXPECT_EQ(submission.finish_count(), 1u);
    EXPECT_EQ(stage0->execute_count, 1u);
    EXPECT_EQ(stage1->execute_count, 1u);
    EXPECT_EQ(stage2->execute_count, 1u);
    EXPECT_EQ(stage0->completion_count, 1u);
    EXPECT_EQ(stage1->completion_count, 1u);
    EXPECT_EQ(stage2->completion_count, 1u);
}

TEST(InferSubmissionTest, NonIncrementalSubmitDefersFlushUntilFinish) {
    std::vector<InferStage> pipeline;
    pipeline.emplace_back();
    pipeline.emplace_back();
    auto* stage0 = new FakeStage();
    auto* stage1 = new FakeStage();
    pipeline[0].stage.reset(stage0);
    pipeline[1].stage.reset(stage1);

    FakeSubmissionSession submission(/*incremental_submit=*/false);
    InferSubmissionConfig config;
    config.max_stages_per_submit = 1;
    config.max_output_bytes_per_submit = std::numeric_limits<size_t>::max();

    execute_pipeline_with_submission(
        pipeline,
        {},
        {},
        [](size_t) -> GpuTensor* {
            return nullptr;
        },
        submission,
        config);

    EXPECT_EQ(submission.submit_count(), 1u);
    ASSERT_EQ(submission.continue_recording_flags().size(), 1u);
    EXPECT_FALSE(submission.continue_recording_flags()[0]);
    EXPECT_EQ(submission.finish_count(), 1u);
    EXPECT_EQ(stage0->execute_count, 1u);
    EXPECT_EQ(stage1->execute_count, 1u);
    EXPECT_EQ(stage0->completion_count, 1u);
    EXPECT_EQ(stage1->completion_count, 1u);
}

TEST(InferSubmissionTest, SingleFlightSessionReusesPreparedSlotAcrossRecordingWindows) {
    std::vector<InferStage> pipeline;
    pipeline.emplace_back();
    pipeline.emplace_back();
    auto* stage0 = new FakeStage(GpuStageSubmitPolicy{.weight = 1, .isolate = true});
    auto* stage1 = new FakeStage();
    pipeline[0].stage.reset(stage0);
    pipeline[1].stage.reset(stage1);

    FakeSingleFlightSubmissionSession submission;
    InferSubmissionConfig config;
    config.max_stages_per_submit = std::numeric_limits<size_t>::max();
    config.max_output_bytes_per_submit = std::numeric_limits<size_t>::max();

    execute_pipeline_with_submission(
        pipeline,
        {},
        {},
        [](size_t) -> GpuTensor* {
            return nullptr;
        },
        submission,
        config);

    EXPECT_EQ(submission.prepare_count(), 1u);
    EXPECT_EQ(submission.submit_count(), 1u);
    EXPECT_EQ(submission.finish_count(), 1u);
    ASSERT_EQ(submission.continue_recording_flags().size(), 1u);
    EXPECT_FALSE(submission.continue_recording_flags()[0]);
}

TEST(InferSubmissionTest, RotatingSessionAdvancesAcrossSubmissionWindowsWithoutReusingHotSlot) {
    std::vector<InferStage> pipeline;
    pipeline.emplace_back();
    pipeline.emplace_back();
    pipeline.emplace_back();
    auto* stage0 = new FakeStage(GpuStageSubmitPolicy{.weight = 1, .isolate = true});
    auto* stage1 = new FakeStage(GpuStageSubmitPolicy{.weight = 1, .isolate = true});
    auto* stage2 = new FakeStage();
    pipeline[0].stage.reset(stage0);
    pipeline[1].stage.reset(stage1);
    pipeline[2].stage.reset(stage2);

    FakeRotatingSubmissionSession submission(/*slot_count=*/3);
    InferSubmissionConfig config;
    config.max_stages_per_submit = std::numeric_limits<size_t>::max();
    config.max_output_bytes_per_submit = std::numeric_limits<size_t>::max();

    execute_pipeline_with_submission(
        pipeline,
        {},
        {},
        [](size_t) -> GpuTensor* {
            return nullptr;
        },
        submission,
        config);

    EXPECT_EQ(submission.begin_slots(), (std::vector<size_t>{0u, 1u}));
    EXPECT_EQ(submission.submit_slots(), (std::vector<size_t>{0u, 1u}));
    ASSERT_EQ(submission.continue_recording_flags().size(), 2u);
    EXPECT_TRUE(submission.continue_recording_flags()[0]);
    EXPECT_FALSE(submission.continue_recording_flags()[1]);
    EXPECT_EQ(submission.finish_count(), 1u);
}

TEST(InferSubmissionTest, FinalRecordHookCanAppendWorkBeforeLastSubmit) {
    std::vector<InferStage> pipeline;
    FakeSubmissionSession submission(/*incremental_submit=*/true);
    InferSubmissionConfig config;
    config.max_stages_per_submit = std::numeric_limits<size_t>::max();
    config.max_output_bytes_per_submit = std::numeric_limits<size_t>::max();

    size_t final_hook_calls = 0;
    GpuCommandBufferHandle final_command_buffer = nullptr;

    execute_pipeline_with_submission(
        pipeline,
        {},
        {},
        [](size_t) -> GpuTensor* {
            return nullptr;
        },
        submission,
        config,
        nullptr,
        nullptr,
        {},
        [&](GpuCommandBufferHandle command_buffer) {
            ++final_hook_calls;
            final_command_buffer = command_buffer;
            InferSubmissionExtraWork extra_work{};
            extra_work.weight = 1;
            extra_work.output_bytes = 128u;
            return extra_work;
        });

    EXPECT_EQ(final_hook_calls, 1u);
    EXPECT_NE(final_command_buffer, nullptr);
    EXPECT_EQ(submission.submit_count(), 1u);
    ASSERT_EQ(submission.continue_recording_flags().size(), 1u);
    EXPECT_FALSE(submission.continue_recording_flags()[0]);
    EXPECT_EQ(submission.finish_count(), 1u);
}

TEST(InferSubmissionTest, DisabledIncrementalSubmitIgnoresIsolateFlushes) {
    std::vector<InferStage> pipeline;
    pipeline.emplace_back();
    pipeline.emplace_back();
    auto* stage0 = new FakeStage(GpuStageSubmitPolicy{.weight = 1, .isolate = true});
    auto* stage1 = new FakeStage();
    pipeline[0].stage.reset(stage0);
    pipeline[1].stage.reset(stage1);

    FakeSubmissionSession submission(/*incremental_submit=*/true);
    InferSubmissionConfig config;
    config.max_stages_per_submit = std::numeric_limits<size_t>::max();
    config.max_output_bytes_per_submit = std::numeric_limits<size_t>::max();
    config.allow_incremental_submit = false;

    execute_pipeline_with_submission(
        pipeline,
        {},
        {},
        [](size_t) -> GpuTensor* {
            return nullptr;
        },
        submission,
        config);

    EXPECT_EQ(submission.submit_count(), 1u);
    ASSERT_EQ(submission.continue_recording_flags().size(), 1u);
    EXPECT_FALSE(submission.continue_recording_flags()[0]);
    EXPECT_EQ(stage0->completion_count, 1u);
    EXPECT_EQ(stage1->completion_count, 1u);
}

TEST(InferSubmissionTest, SubmissionTuningUsesMultipleSlotsForDeepIncrementalPipelines) {
    InferSubmissionTuningCaps caps{};
    caps.backend = GpuBackend::Vulkan;
    caps.preferred_simd_width = 64u;
    caps.subgroup_size = 64u;
    caps.max_total_threads_per_group = 1024u;
    caps.supports_incremental_submit = true;

    const auto tuning = select_infer_submission_tuning(caps, /*stage_count=*/192u);

    EXPECT_GE(tuning.slot_count, 2u);
    EXPECT_GE(tuning.config.max_stages_per_submit, 1u);
}

TEST(InferSubmissionTest, SubmissionTuningWidensWindowBudgetWhenMultipleSlotsAreAvailable) {
    InferSubmissionTuningCaps caps{};
    caps.backend = GpuBackend::Vulkan;
    caps.preferred_simd_width = 32u;
    caps.subgroup_size = 32u;
    caps.max_total_threads_per_group = 256u;
    caps.supports_incremental_submit = true;

    const auto tuning = select_infer_submission_tuning(caps, /*stage_count=*/192u);

    EXPECT_GT(tuning.slot_count, 1u);
    EXPECT_GT(tuning.config.max_stages_per_submit, 12u);
    EXPECT_GT(tuning.config.max_output_bytes_per_submit, 32u * 1024u * 1024u);
}

TEST(InferSubmissionTest, SubmissionTuningPrefersMonolithicSubmitForExtremelyDeepVulkanPipelines) {
    InferSubmissionTuningCaps caps{};
    caps.backend = GpuBackend::Vulkan;
    caps.preferred_simd_width = 64u;
    caps.subgroup_size = 64u;
    caps.max_total_threads_per_group = 1024u;
    caps.supports_incremental_submit = true;

    const auto tuning = select_infer_submission_tuning(caps, /*stage_count=*/390u);

    EXPECT_EQ(tuning.slot_count, 1u);
    EXPECT_EQ(tuning.config.max_stages_per_submit, 390u);
    EXPECT_FALSE(tuning.config.allow_incremental_submit);
}

TEST(InferSubmissionTest, SubmissionTuningKeepsSingleSlotWithoutIncrementalSubmit) {
    InferSubmissionTuningCaps caps{};
    caps.backend = GpuBackend::Metal;
    caps.preferred_simd_width = 32u;
    caps.subgroup_size = 32u;
    caps.max_total_threads_per_group = 512u;
    caps.supports_incremental_submit = false;

    const auto tuning = select_infer_submission_tuning(caps, /*stage_count=*/390u);

    EXPECT_EQ(tuning.slot_count, 1u);
    EXPECT_EQ(tuning.config.max_stages_per_submit, 390u);
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
