// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/gfx_plugin/infer_request.hpp"

#include <cstring>
#include <chrono>
#include <optional>
#include <string>
#include <vector>

#include "openvino/gfx_plugin/compiled_model.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "runtime/memory_manager.hpp"
#include "runtime/gpu_buffer_pool.hpp"
#include "runtime/gpu_types.hpp"
#include "runtime/gfx_backend_utils.hpp"
#include "runtime/gfx_profiler.hpp"
#include "backends/vulkan/runtime/gpu_memory.hpp"
#include "backends/vulkan/codegen/vulkan_codegen_backend.hpp"
#include "backends/vulkan/runtime/vulkan_backend.hpp"
#include "backends/vulkan/runtime/vulkan_memory.hpp"
#include "backends/vulkan/plugin/infer_io_vulkan.hpp"
#include "plugin/gfx_profiling_utils.hpp"
#include "plugin/infer_profiling_utils.hpp"
#include "plugin/infer_request_state.hpp"
#include "plugin/compiled_model_backend_resources.hpp"
#include "plugin/infer_submission.hpp"
#include "plugin/stateful_execution.hpp"
#include "runtime/gfx_remote_context.hpp"
#include "plugin/infer_pipeline.hpp"
#include "plugin/infer_io_utils.hpp"
#include "backends/vulkan/plugin/compiled_model_state.hpp"
#include "runtime/gfx_logger.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

std::string vk_result_to_string(VkResult res) {
    switch (res) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    default:
        break;
    }
    return "VK_ERROR_UNKNOWN";
}

inline void release_buffer_handles(std::vector<BufferHandle>& handles, VulkanGpuAllocator& allocator) {
    for (auto& handle : handles) {
        if (handle.valid()) {
            allocator.release(std::move(handle.buf));
        }
    }
}

struct VulkanCommandSubmission {
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::vector<VkCommandPool> pools;
    std::vector<VkCommandBuffer> commands;
    std::vector<VkFence> fences;
    std::vector<bool> fence_pending;

    ~VulkanCommandSubmission() {
        for (auto& fence : fences) {
            if (fence && device) {
                vkDestroyFence(device, fence, nullptr);
            }
        }
        for (auto& pool : pools) {
            if (pool && device) {
                vkDestroyCommandPool(device, pool, nullptr);
            }
        }
    }
};

struct VulkanInferState final : BackendInferState {
    ~VulkanInferState() override {
        VulkanGpuAllocator allocator;
        release_buffer_handles(input_handles, allocator);
        release_buffer_handles(input_staging_handles, allocator);
        for (auto& stage_handles : stage_output_handles) {
            release_buffer_handles(stage_handles, allocator);
        }
        release_buffer_handles(output_staging_handles, allocator);
    }

    std::unique_ptr<VulkanCommandSubmission> submission;
};

struct VulkanBufferRange {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
};

bool operator==(const VulkanBufferRange& lhs, const VulkanBufferRange& rhs) {
    return lhs.buffer == rhs.buffer && lhs.offset == rhs.offset && lhs.size == rhs.size;
}

std::optional<VulkanBufferRange> make_vulkan_buffer_range(const GpuBuffer& buffer) {
    if (!buffer.valid() || buffer.backend != GpuBackend::Vulkan) {
        return std::nullopt;
    }
    VkBuffer vk_buffer = vk_buffer_from_gpu(buffer);
    if (vk_buffer == VK_NULL_HANDLE) {
        return std::nullopt;
    }
    VulkanBufferRange range{};
    range.buffer = vk_buffer;
    range.offset = static_cast<VkDeviceSize>(buffer.offset);
    range.size = buffer.size ? static_cast<VkDeviceSize>(buffer.size) : VK_WHOLE_SIZE;
    return range;
}

bool ranges_overlap(const VulkanBufferRange& lhs, const VulkanBufferRange& rhs) {
    if (lhs.buffer != rhs.buffer) {
        return false;
    }
    if (lhs.size == VK_WHOLE_SIZE || rhs.size == VK_WHOLE_SIZE) {
        return true;
    }
    const VkDeviceSize lhs_end = lhs.offset + lhs.size;
    const VkDeviceSize rhs_end = rhs.offset + rhs.size;
    return lhs.offset < rhs_end && rhs.offset < lhs_end;
}

bool append_unique_vulkan_buffer_range(std::vector<VulkanBufferRange>& ranges, const GpuBuffer& buffer) {
    auto range = make_vulkan_buffer_range(buffer);
    if (!range.has_value()) {
        return false;
    }
    if (std::find(ranges.begin(), ranges.end(), *range) != ranges.end()) {
        return false;
    }
    ranges.push_back(*range);
    return true;
}

class VulkanCrossSubmitBarrierTracker final {
public:
    void record_buffer_write(const GpuBuffer& buffer) {
        append_unique_vulkan_buffer_range(m_recording_window_writes, buffer);
    }

    void record_stage_outputs(const InferStage& stage) {
        for (const auto& output : stage.outputs) {
            if (!output) {
                continue;
            }
            record_buffer_write(output->buf);
        }
    }

    void commit_submitted_window() {
        for (const auto& range : m_recording_window_writes) {
            if (std::find(m_pending_writes.begin(), m_pending_writes.end(), range) == m_pending_writes.end()) {
                m_pending_writes.push_back(range);
            }
        }
        m_recording_window_writes.clear();
    }

    void reset() {
        m_recording_window_writes.clear();
        m_pending_writes.clear();
    }

    void emit_stage_barriers(VkCommandBuffer command_buffer,
                             const InferStage& stage,
                             const std::vector<GpuTensor*>& resolved_inputs,
                             GfxProfiler* profiler) {
        std::vector<GpuBuffer> required_buffers;
        required_buffers.reserve(resolved_inputs.size());
        for (const auto* input : resolved_inputs) {
            if (input) {
                required_buffers.push_back(input->buf);
            }
        }
        emit_buffer_barriers(command_buffer,
                             required_buffers,
                             profiler,
                             "cross_submit_memory_barrier");
    }

    void emit_buffer_barriers(VkCommandBuffer command_buffer,
                              const std::vector<GpuBuffer>& buffers,
                              GfxProfiler* profiler,
                              const char* barrier_name) {
        if (command_buffer == VK_NULL_HANDLE) {
            return;
        }

        std::vector<VulkanBufferRange> required_ranges;
        required_ranges.reserve(buffers.size());
        for (const auto& buffer : buffers) {
            auto input_range = make_vulkan_buffer_range(buffer);
            if (!input_range.has_value()) {
                continue;
            }
            const bool needs_barrier =
                std::any_of(m_pending_writes.begin(),
                            m_pending_writes.end(),
                            [&](const VulkanBufferRange& pending) { return ranges_overlap(*input_range, pending); });
            if (needs_barrier && std::find(required_ranges.begin(), required_ranges.end(), *input_range) == required_ranges.end()) {
                required_ranges.push_back(*input_range);
            }
        }
        if (required_ranges.empty()) {
            return;
        }

        std::vector<VkBufferMemoryBarrier> barriers;
        barriers.reserve(required_ranges.size());
        for (const auto& required : required_ranges) {
            VkBufferMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                    VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = required.buffer;
            barrier.offset = required.offset;
            barrier.size = required.size;
            const bool duplicate = std::any_of(barriers.begin(), barriers.end(), [&](const VkBufferMemoryBarrier& existing) {
                return existing.buffer == barrier.buffer && existing.offset == barrier.offset && existing.size == barrier.size;
            });
            if (!duplicate) {
                barriers.push_back(barrier);
            }
        }
        if (barriers.empty()) {
            return;
        }

        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0,
                             0,
                             nullptr,
                             static_cast<uint32_t>(barriers.size()),
                             barriers.data(),
                             0,
                             nullptr);
        if (profiler) {
            profiler->record_segment("barrier",
                                     barrier_name,
                                     std::chrono::microseconds{0},
                                     0,
                                     0,
                                     0,
                                     0,
                                     0,
                                     0,
                                     -1,
                                     0,
                                     reinterpret_cast<uint64_t>(command_buffer));
            profiler->increment_counter("cross_submit_barrier_count");
            profiler->increment_counter("cross_submit_barrier_buffer_count",
                                        static_cast<uint64_t>(barriers.size()));
        }
    }

private:
    std::vector<VulkanBufferRange> m_recording_window_writes;
    std::vector<VulkanBufferRange> m_pending_writes;
};

struct PreparedVulkanOutputReadback {
    std::shared_ptr<GfxRemoteTensor> remote_tensor;
    VulkanOutputBindingResult local_binding{};
};

class ScopedConstUploadBatch final {
public:
    explicit ScopedConstUploadBatch(GpuBufferManager* manager) : m_manager(manager) {
        if (m_manager) {
            m_manager->begin_const_upload_batch();
        }
    }

    ~ScopedConstUploadBatch() {
        if (m_manager) {
            m_manager->end_const_upload_batch();
        }
    }

    void flush(GpuCommandBufferHandle command_buffer, GfxProfiler* profiler) {
        if (m_manager) {
            m_manager->flush_const_upload_batch(command_buffer, profiler);
        }
    }

    ScopedConstUploadBatch(const ScopedConstUploadBatch&) = delete;
    ScopedConstUploadBatch& operator=(const ScopedConstUploadBatch&) = delete;

private:
    GpuBufferManager* m_manager = nullptr;
};

VulkanInferState* get_vulkan_state(InferRequestState& state) {
    return dynamic_cast<VulkanInferState*>(state.backend.get());
}

const VulkanInferState* get_vulkan_state(const InferRequestState& state) {
    return dynamic_cast<const VulkanInferState*>(state.backend.get());
}

void ensure_vulkan_submission_slot(VulkanCommandSubmission& submission, size_t slot_index) {
    OPENVINO_ASSERT(slot_index < submission.pools.size(), "GFX Vulkan: invalid submission slot index");
    if (submission.pools[slot_index] && submission.commands[slot_index] && submission.fences[slot_index]) {
        return;
    }

    auto& ctx = VulkanContext::instance();

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = ctx.queue_family_index();
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkResult res = vkCreateCommandPool(submission.device, &pool_info, nullptr, &submission.pools[slot_index]);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkCreateCommandPool failed for infer command buffer");
    }

    VkCommandBufferAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = submission.pools[slot_index];
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    res = vkAllocateCommandBuffers(submission.device, &alloc, &submission.commands[slot_index]);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkAllocateCommandBuffers failed for infer command buffer");
    }

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    res = vkCreateFence(submission.device, &fence_info, nullptr, &submission.fences[slot_index]);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkCreateFence failed for infer command buffer: ",
                       vk_result_to_string(res));
    }
}

std::unique_ptr<VulkanCommandSubmission> create_vulkan_infer_submission(size_t slot_count) {
    auto submission = std::make_unique<VulkanCommandSubmission>();
    auto& ctx = VulkanContext::instance();
    submission->device = ctx.device();
    submission->queue = ctx.queue();
    submission->pools.resize(slot_count);
    submission->commands.resize(slot_count);
    submission->fences.resize(slot_count);
    submission->fence_pending.assign(slot_count, false);
    return submission;
}

void begin_vulkan_infer_commands(VulkanCommandSubmission& submission, size_t slot_index, GfxProfiler* profiler) {
    OPENVINO_ASSERT(submission.device,
                    "GFX Vulkan: infer command submission is not initialized");
    ensure_vulkan_submission_slot(submission, slot_index);
    OPENVINO_ASSERT(submission.commands[slot_index], "GFX Vulkan: infer command buffer is not allocated");
    OPENVINO_ASSERT(submission.fences[slot_index], "GFX Vulkan: infer fence is not initialized");

    if (submission.fence_pending[slot_index]) {
        const bool profiling = (profiler != nullptr);
        const auto wait_start = profiling ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
        VkResult wait_res = vkWaitForFences(submission.device,
                                            1,
                                            &submission.fences[slot_index],
                                            VK_TRUE,
                                            UINT64_MAX);
        if (wait_res != VK_SUCCESS) {
            OPENVINO_THROW("GFX Vulkan: vkWaitForFences failed for infer command buffer: ",
                           vk_result_to_string(wait_res));
        }
        if (profiling) {
            const auto wait_cpu_us =
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - wait_start);
            profiler->record_segment("wait", "slot_reuse_wait", wait_cpu_us, 0, 0, 0, 0, 0, 0,
                                     static_cast<int64_t>(slot_index));
            profiler->increment_counter("fence_wait_count");
        }
        submission.fence_pending[slot_index] = false;
    }

    const bool profiling = (profiler != nullptr);

    const auto reset_pool_start = profiling ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
    VkResult res = vkResetCommandPool(submission.device, submission.pools[slot_index], 0);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkResetCommandPool failed for infer command buffer: ",
                       vk_result_to_string(res));
    }
    if (profiling) {
        const auto reset_pool_cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - reset_pool_start);
        profiler->record_segment("submit",
                                 "vkResetCommandPool",
                                 reset_pool_cpu_us,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 static_cast<int64_t>(slot_index));
    }

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    const auto begin_cb_start = profiling ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
    res = vkBeginCommandBuffer(submission.commands[slot_index], &begin);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkBeginCommandBuffer failed for infer command buffer");
    }
    vulkan_reset_command_buffer_access_tracker(submission.commands[slot_index]);
    if (profiling) {
        const auto begin_cb_cpu_us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin_cb_start);
        profiler->record_segment("submit",
                                 "vkBeginCommandBuffer",
                                 begin_cb_cpu_us,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 static_cast<int64_t>(slot_index));
    }
}

void submit_vulkan_infer_commands(VulkanCommandSubmission& submission, size_t slot_index, GfxProfiler* profiler) {
    OPENVINO_ASSERT(submission.device && submission.queue && slot_index < submission.commands.size(),
                    "GFX Vulkan: infer command submission is not initialized");
    VkCommandBuffer& command_buffer = submission.commands[slot_index];
    VkFence& fence = submission.fences[slot_index];
    const bool profiling = (profiler != nullptr);

    const auto end_cb_start = profiling ? std::chrono::steady_clock::now()
                                        : std::chrono::steady_clock::time_point{};
    VkResult res = vkEndCommandBuffer(command_buffer);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkEndCommandBuffer failed for infer command buffer");
    }
    if (profiling) {
        const auto end_cb_cpu_us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - end_cb_start);
        profiler->record_segment("submit",
                                 "vkEndCommandBuffer",
                                 end_cb_cpu_us,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 static_cast<int64_t>(slot_index),
                                 0,
                                 reinterpret_cast<uint64_t>(command_buffer));
    }

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_buffer;
    const auto reset_fence_start = profiling ? std::chrono::steady_clock::now()
                                             : std::chrono::steady_clock::time_point{};
    res = vkResetFences(submission.device, 1, &fence);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkResetFences failed for infer command buffer: ",
                       vk_result_to_string(res));
    }
    if (profiling) {
        const auto reset_fence_cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - reset_fence_start);
        profiler->record_segment("submit",
                                 "vkResetFences",
                                 reset_fence_cpu_us,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 static_cast<int64_t>(slot_index));
    }
    const auto submit_start = profiling ? std::chrono::steady_clock::now()
                                        : std::chrono::steady_clock::time_point{};
    res = vkQueueSubmit(submission.queue, 1, &submit, fence);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkQueueSubmit failed for infer command buffer: ",
                       vk_result_to_string(res));
    }
    if (profiling) {
        const auto submit_cpu_us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - submit_start);
        profiler->record_segment("submit",
                                 "vkQueueSubmit",
                                 submit_cpu_us,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 static_cast<int64_t>(slot_index),
                                 0,
                                 reinterpret_cast<uint64_t>(command_buffer));
        profiler->increment_counter("vkQueueSubmit_count");
    }
    submission.fence_pending[slot_index] = true;
}

void wait_vulkan_infer_commands(VulkanCommandSubmission& submission, size_t slot_index, GfxProfiler* profiler) {
    if (slot_index >= submission.fence_pending.size() || !submission.fence_pending[slot_index]) {
        return;
    }
    const bool profiling = (profiler != nullptr);
    const auto wait_start = profiling ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
    VkResult res = vkWaitForFences(submission.device,
                                   1,
                                   &submission.fences[slot_index],
                                   VK_TRUE,
                                   UINT64_MAX);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkWaitForFences failed for infer command buffer: ",
                       vk_result_to_string(res));
    }
    if (profiling) {
        const auto wait_cpu_us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - wait_start);
        profiler->record_segment("wait", "final_fence_wait", wait_cpu_us, 0, 0, 0, 0, 0, 0,
                                 static_cast<int64_t>(slot_index));
        profiler->increment_counter("fence_wait_count");
    }
    submission.fence_pending[slot_index] = false;
}

class VulkanInferSubmissionSession final : public RotatingSlotInferSubmissionSession {
public:
    explicit VulkanInferSubmissionSession(VulkanInferState& state, size_t slot_count, GfxProfiler* profiler)
        : RotatingSlotInferSubmissionSession(slot_count),
          m_state(state),
          m_slot_count(std::max<size_t>(slot_count, 1u)),
          m_profiler(profiler) {}

    void record_buffer_write(const GpuBuffer& buffer) {
        m_cross_submit_barriers.record_buffer_write(buffer);
    }

    void prepare_stage_submission(const InferStage& stage,
                                  const std::vector<GpuTensor*>& resolved_inputs,
                                  GpuCommandBufferHandle command_buffer) {
        m_cross_submit_barriers.emit_stage_barriers(reinterpret_cast<VkCommandBuffer>(command_buffer),
                                                    stage,
                                                    resolved_inputs,
                                                    m_profiler);
    }

    void prepare_buffer_read(const GpuBuffer& buffer, GpuCommandBufferHandle command_buffer) {
        std::vector<GpuBuffer> buffers;
        buffers.push_back(buffer);
        m_cross_submit_barriers.emit_buffer_barriers(reinterpret_cast<VkCommandBuffer>(command_buffer),
                                                     buffers,
                                                     m_profiler,
                                                     "cross_submit_readback_barrier");
    }

    void finish_stage_submission(const InferStage& stage) {
        m_cross_submit_barriers.record_stage_outputs(stage);
    }

protected:
    void prepare_submission_slot(size_t slot_index) override {
        if (!m_state.submission) {
            m_state.submission = create_vulkan_infer_submission(m_slot_count);
        }
        ensure_vulkan_submission_slot(*m_state.submission, slot_index);
    }

    GpuCommandBufferHandle begin_recording_on_slot(size_t slot_index) override {
        OPENVINO_ASSERT(m_state.submission, "GFX Vulkan: infer submission is not initialized");
        begin_vulkan_infer_commands(*m_state.submission, slot_index, m_profiler);
        return reinterpret_cast<GpuCommandBufferHandle>(m_state.submission->commands[slot_index]);
    }

    void submit_recorded_on_slot(size_t slot_index,
                                 GpuCommandBufferHandle /*command_buffer*/,
                                 bool /*continue_recording*/) override {
        OPENVINO_ASSERT(m_state.submission, "GFX Vulkan: infer submission is not initialized");
        submit_vulkan_infer_commands(*m_state.submission, slot_index, m_profiler);
        m_cross_submit_barriers.commit_submitted_window();
    }

    void finish_submission_slots() override {
        OPENVINO_ASSERT(m_state.submission, "GFX Vulkan: infer submission is not initialized");
        for (size_t slot_index = 0; slot_index < m_state.submission->commands.size(); ++slot_index) {
            wait_vulkan_infer_commands(*m_state.submission, slot_index, m_profiler);
        }
        m_cross_submit_barriers.reset();
    }

private:
    VulkanInferState& m_state;
    size_t m_slot_count = 1;
    GfxProfiler* m_profiler = nullptr;
    VulkanCrossSubmitBarrierTracker m_cross_submit_barriers;
};

}  // namespace

void VulkanBackendState::init_infer_state(InferRequestState& state) const {
    state.backend = std::make_unique<VulkanInferState>();
}

void InferRequest::infer_vulkan_impl(const std::shared_ptr<const CompiledModel>& cm) {
    OPENVINO_ASSERT(cm, "CompiledModel is null");
    if (cm->backend() != GpuBackend::Vulkan) {
        OPENVINO_THROW("GFX backend '",
                       backend_to_string(cm->backend()),
                       "' is not supported by Vulkan infer path");
    }
    auto& state = *m_state;
    auto* vk_state = get_vulkan_state(state);
    OPENVINO_ASSERT(vk_state, "GFX: Vulkan infer state is not initialized");
    OPENVINO_ASSERT(cm->op_pipeline_built(),
                    "GFX: op pipeline is not built");
    const auto resources = get_backend_resources(cm->backend_state());
    OPENVINO_ASSERT(resources.const_manager,
                    "GFX Vulkan: const buffer manager is required for infer pipeline");

    const auto& descs = cm->pipeline_desc();
    const auto& node_map = cm->node_to_stage();
    const auto& param_map = cm->parameter_index();

    GfxProfiler* profiler = prepare_infer_profiler(*cm, state, "GFX Vulkan");
    if (profiler) {
        profiler->begin_infer(descs.size());
    }
    const bool profiling = (profiler != nullptr);
    const bool profiling_enabled = (profiler != nullptr);
    void* stage_profiler = profiler ? profiler->native_handle() : nullptr;
    const auto pipeline_prepare_start = profiling ? std::chrono::steady_clock::now()
                                                  : std::chrono::steady_clock::time_point{};
    VulkanGpuAllocator allocator;
    GpuBufferPool pool(allocator);
    ScopedConstUploadBatch const_upload_batch(resources.const_manager);
    auto& pipeline = prepare_reusable_pipeline_with_outputs(
        vk_state->reusable_pipeline,
        descs,
        resources.const_manager,
        stage_profiler,
        profiling_enabled,
        cm->get_runtime_model(),
        get_outputs(),
        node_map,
        param_map,
        state.bound_remote_outputs,
        state.bound_remote_inputs,
        GpuBackend::Vulkan,
        pool,
        vk_state->stage_output_handles,
        [&](std::vector<InferStage>& stages) {
            for (auto& stage : stages) {
                for (auto& out : stage.outputs) {
                    if (out) {
                        out->prefer_private = true;
                    }
                }
            }
        },
        [&](InferStage& stage,
            size_t oi,
            GpuTensor& out_ref,
            GpuBufferDesc& desc,
            const char* error_prefix) {
            const bool is_model_output =
                (oi < stage.output_is_model_output.size()) && stage.output_is_model_output[oi];
            return init_stage_output_desc(GpuBackend::Vulkan,
                                          stage,
                                          oi,
                                          out_ref,
                                          desc,
                                          is_model_output,
                                          /*skip_view_ops=*/true,
                                          error_prefix);
        },
        "GFX Vulkan");
    prepare_reusable_execution_plan(vk_state->reusable_execution_plan, pipeline, node_map, param_map);
    if (profiling) {
        profiler->record_segment("compile",
                                 "prepare_reusable_pipeline",
                                 std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now() - pipeline_prepare_start));
    }

    const auto& vk_ctx = VulkanContext::instance();
    InferSubmissionTuningCaps submission_caps{};
    submission_caps.backend = GpuBackend::Vulkan;
    submission_caps.preferred_simd_width = std::max<uint32_t>(vk_ctx.subgroup_size(), 1u);
    submission_caps.subgroup_size = std::max<uint32_t>(vk_ctx.subgroup_size(), 1u);
    submission_caps.max_total_threads_per_group =
        std::max<uint32_t>(vk_ctx.max_compute_workgroup_invocations(), 1u);
    submission_caps.supports_incremental_submit = true;
    const auto submission_tuning = select_infer_submission_tuning(submission_caps, pipeline.size());
    record_infer_submission_tuning_counters(submission_tuning, submission_caps, profiler);
    if (gfx_log_debug_enabled()) {
        gfx_log_debug("InferSubmit") << "Vulkan submission tuning: slots=" << submission_tuning.slot_count
                                     << " max_stages=" << submission_tuning.config.max_stages_per_submit
                                     << " max_output_bytes=" << submission_tuning.config.max_output_bytes_per_submit
                                     << " simd=" << submission_caps.preferred_simd_width
                                     << " max_threads=" << submission_caps.max_total_threads_per_group
                                     << " pipeline_stages=" << pipeline.size();
    }
    VulkanInferSubmissionSession submission(*vk_state, submission_tuning.slot_count, profiler);
    submission.begin_recording();
    const_upload_batch.flush(submission.current_command_buffer(), profiler);

    // Create device buffers for inputs (or use remote buffers when provided).
    std::vector<GpuTensor> input_tensors(get_inputs().size());
    ensure_input_handles(get_inputs().size(), /*with_staging=*/true, "GFX Vulkan");
    auto& input_handles = vk_state->input_handles;
    auto& input_staging_handles = vk_state->input_staging_handles;
    const auto bind_inputs_start = profiling ? std::chrono::steady_clock::now()
                                             : std::chrono::steady_clock::time_point{};
    bind_inputs_for_infer(
        GpuBackend::Vulkan,
        [&](size_t idx, const GpuTensor& dev) {
            input_tensors[idx] = dev;
        },
        [&](size_t idx, const ov::Tensor& host) {
            input_tensors[idx] = bind_host_input_vulkan(host,
                                                        &pool,
                                                        &input_handles[idx],
                                                        &input_staging_handles[idx],
                                                        submission.current_command_buffer(),
                                                        profiler,
                                                        "GFX Vulkan");
            submission.record_buffer_write(input_tensors[idx].buf);
        },
        "GFX Vulkan");
    if (profiling) {
        profiler->record_segment("upload",
                                 "bind_inputs_for_infer",
                                 std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now() - bind_inputs_start));
    }
    if (!vk_state->reusable_pipeline_runtime_prewarmed) {
        const auto prewarm_start = profiling ? std::chrono::steady_clock::now()
                                             : std::chrono::steady_clock::time_point{};
        prewarm_pipeline_runtime_state(
            pipeline,
            node_map,
            param_map,
            [&](size_t input_idx) -> GpuTensor* {
                if (input_idx < input_tensors.size()) {
                    return &input_tensors[input_idx];
                }
                return nullptr;
            },
            &vk_state->reusable_execution_plan);
        vk_state->reusable_pipeline_runtime_prewarmed = true;
        if (profiling) {
            profiler->record_segment("compile",
                                     "prewarm_reusable_pipeline",
                                     std::chrono::duration_cast<std::chrono::microseconds>(
                                         std::chrono::steady_clock::now() - prewarm_start));
        }
    }

    prepare_reusable_output_plan(vk_state->reusable_output_plan,
                                 get_outputs(),
                                 cm->get_runtime_model(),
                                 pipeline,
                                 node_map,
                                 param_map,
                                 "GFX Vulkan");
    prepare_reusable_host_output_plan(vk_state->reusable_host_output_plan,
                                      vk_state->reusable_output_plan,
                                      state.bound_output_hosts);
    ensure_output_staging_handles(get_outputs().size(), "GFX Vulkan");
    auto& output_handles = vk_state->output_staging_handles;
    std::vector<PreparedVulkanOutputReadback> prepared_output_readbacks(get_outputs().size());
    auto output_input_lookup = [&](size_t input_idx) -> GpuTensor* {
        if (input_idx < input_tensors.size()) {
            return &input_tensors[input_idx];
        }
        return nullptr;
    };
    auto prepare_output_readbacks = [&](GpuCommandBufferHandle command_buffer) -> InferSubmissionExtraWork {
        InferSubmissionExtraWork extra_work{};
        for (auto& prepared : prepared_output_readbacks) {
            prepared = {};
        }
        for_each_output_tensor(
            get_outputs(),
            cm->get_runtime_model(),
            node_map,
            param_map,
            pipeline,
            output_input_lookup,
            state.bound_remote_outputs,
            [&](size_t idx, const ov::element::Type& type, const ov::Shape& shape, const char* err) {
                return get_host_output_override(idx, type, shape, err);
            },
            [&](size_t idx, const std::shared_ptr<GfxRemoteTensor>& remote) {
                prepared_output_readbacks[idx].remote_tensor = remote;
            },
            [&](size_t idx, GpuTensor& dev, const OutputViewInfo& info, const ov::Tensor* host_override) {
                ov::Tensor* reusable_host = nullptr;
                if (idx < vk_state->reusable_host_output_plan.outputs.size()) {
                    auto& prepared = vk_state->reusable_host_output_plan.outputs[idx];
                    if (prepared.host) {
                        reusable_host = &prepared.host;
                    }
                }
                if (!dev.buf.host_visible && dev.buf.valid()) {
                    submission.prepare_buffer_read(dev.buf, command_buffer);
                }
                prepared_output_readbacks[idx].local_binding = prepare_host_output_vulkan(dev,
                                                                                          info,
                                                                                          host_override,
                                                                                          reusable_host,
                                                                                          &pool,
                                                                                          &output_handles[idx],
                                                                                          command_buffer,
                                                                                          profiler,
                                                                                          "GFX Vulkan");
                if (prepared_output_readbacks[idx].local_binding.readback_bytes > 0) {
                    extra_work.weight = 1;
                    extra_work.output_bytes += prepared_output_readbacks[idx].local_binding.readback_bytes;
                }
            },
            &vk_state->reusable_output_plan,
            /*allow_missing=*/false,
            "GFX Vulkan");
        return extra_work;
    };

    const InferSubmissionConfig& submission_cfg = submission_tuning.config;
    const auto infer_start = profiling ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};
    execute_pipeline_with_submission(
        pipeline,
        node_map,
        param_map,
        [&](size_t input_idx) -> GpuTensor* {
            if (input_idx < input_tensors.size()) {
                return &input_tensors[input_idx];
            }
            return nullptr;
        },
        submission,
        submission_cfg,
        profiler,
        &vk_state->reusable_execution_plan,
        [&](InferStage& stage, const std::vector<GpuTensor*>& resolved_inputs, GpuCommandBufferHandle command_buffer) {
            if (execute_stateful_stage(state, stage, resolved_inputs, pool, command_buffer)) {
                return;
            }
            submission.prepare_stage_submission(stage, resolved_inputs, command_buffer);
            stage.stage->execute(command_buffer);
            submission.finish_stage_submission(stage);
        },
        prepare_output_readbacks,
        true);
    if (profiling) {
        profiler->record_segment("infer",
                                 "execute_pipeline_with_submission",
                                 std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now() - infer_start));
    }

    const auto bind_outputs_start = profiling ? std::chrono::steady_clock::now()
                                              : std::chrono::steady_clock::time_point{};
    for (size_t idx = 0; idx < prepared_output_readbacks.size(); ++idx) {
        auto& prepared = prepared_output_readbacks[idx];
        if (prepared.remote_tensor) {
            ov::ISyncInferRequest::set_tensor(get_outputs().at(idx),
                                              ov::SoPtr<ov::ITensor>{prepared.remote_tensor, nullptr});
            continue;
        }
        finalize_host_output_vulkan(prepared.local_binding, profiler, "GFX Vulkan");
        ov::ISyncInferRequest::set_tensor(get_outputs().at(idx),
                                          ov::get_tensor_impl(prepared.local_binding.binding.host_tensor));
    }
    if (profiling) {
        profiler->record_segment("download",
                                 "bind_outputs_for_infer",
                                 std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now() - bind_outputs_start));
    }
    finalize_infer_profiling("vulkan", cm, state, profiler, nullptr);

}

}  // namespace gfx_plugin
}  // namespace ov
