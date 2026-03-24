// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/gfx_plugin/infer_request.hpp"

#include <cstring>
#include <string>

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
#include "backends/vulkan/runtime/vulkan_backend.hpp"
#include "backends/vulkan/runtime/vulkan_memory.hpp"
#include "backends/vulkan/plugin/infer_io_vulkan.hpp"
#include "plugin/gfx_profiling_utils.hpp"
#include "plugin/infer_profiling_utils.hpp"
#include "plugin/infer_request_state.hpp"
#include "plugin/compiled_model_backend_resources.hpp"
#include "runtime/gfx_remote_context.hpp"
#include "plugin/infer_pipeline.hpp"
#include "plugin/infer_io_utils.hpp"
#include "backends/vulkan/plugin/compiled_model_state.hpp"

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
};

VulkanInferState* get_vulkan_state(InferRequestState& state) {
    return dynamic_cast<VulkanInferState*>(state.backend.get());
}

const VulkanInferState* get_vulkan_state(const InferRequestState& state) {
    return dynamic_cast<const VulkanInferState*>(state.backend.get());
}

struct VulkanCommandSubmission {
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;

    ~VulkanCommandSubmission() {
        if (cmd && pool && device) {
            vkFreeCommandBuffers(device, pool, 1, &cmd);
        }
        if (pool && device) {
            vkDestroyCommandPool(device, pool, nullptr);
        }
    }
};

VulkanCommandSubmission begin_vulkan_infer_commands() {
    VulkanCommandSubmission submission;
    auto& ctx = VulkanContext::instance();
    submission.device = ctx.device();
    submission.queue = ctx.queue();

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = ctx.queue_family_index();
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    VkResult res = vkCreateCommandPool(submission.device, &pool_info, nullptr, &submission.pool);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkCreateCommandPool failed for infer command buffer");
    }

    VkCommandBufferAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = submission.pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    res = vkAllocateCommandBuffers(submission.device, &alloc, &submission.cmd);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkAllocateCommandBuffers failed for infer command buffer");
    }

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    res = vkBeginCommandBuffer(submission.cmd, &begin);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkBeginCommandBuffer failed for infer command buffer");
    }

    return submission;
}

void submit_vulkan_infer_commands(VulkanCommandSubmission& submission) {
    OPENVINO_ASSERT(submission.device && submission.queue && submission.cmd,
                    "GFX Vulkan: infer command submission is not initialized");
    VkResult res = vkEndCommandBuffer(submission.cmd);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkEndCommandBuffer failed for infer command buffer");
    }

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &submission.cmd;
    res = vkQueueSubmit(submission.queue, 1, &submit, VK_NULL_HANDLE);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkQueueSubmit failed for infer command buffer: ",
                       vk_result_to_string(res));
    }
    res = vkQueueWaitIdle(submission.queue);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkQueueWaitIdle failed for infer command buffer: ",
                       vk_result_to_string(res));
    }
}

void notify_pipeline_submission_complete(std::vector<InferStage>& pipeline) {
    for (auto& stage : pipeline) {
        if (stage.stage) {
            stage.stage->on_command_buffer_complete();
        }
    }
}

void restart_vulkan_infer_commands(VulkanCommandSubmission& submission) {
    OPENVINO_ASSERT(submission.device && submission.pool && submission.cmd,
                    "GFX Vulkan: infer command submission is not initialized");
    VkResult res = vkResetCommandPool(submission.device, submission.pool, 0);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkResetCommandPool failed for infer command buffer: ",
                       vk_result_to_string(res));
    }
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    res = vkBeginCommandBuffer(submission.cmd, &begin);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkBeginCommandBuffer failed for infer command buffer restart: ",
                       vk_result_to_string(res));
    }
}

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
    const bool profiling_enabled = (profiler != nullptr);
    void* stage_profiler = profiler ? profiler->native_handle() : nullptr;
    VulkanGpuAllocator allocator;
    GpuBufferPool pool(allocator);
    auto pipeline = build_pipeline_with_outputs(
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

    // Create device buffers for inputs (or use remote buffers when provided).
    std::vector<GpuTensor> input_tensors(get_inputs().size());
    ensure_input_handles(get_inputs().size(), /*with_staging=*/true, "GFX Vulkan");
    auto& input_handles = vk_state->input_handles;
    auto& input_staging_handles = vk_state->input_staging_handles;
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
                                                        "GFX Vulkan");
        },
        "GFX Vulkan");

    // Execute pipeline.
    constexpr size_t kStagesPerSubmit = 16;
    constexpr size_t kMaxOutputBytesPerSubmit = 16u * 1024u * 1024u;
    auto submission = begin_vulkan_infer_commands();
    size_t recorded_stage_count = 0;
    size_t recorded_output_bytes = 0;
    auto flush_submission = [&]() {
        if (recorded_stage_count == 0) {
            return;
        }
        submit_vulkan_infer_commands(submission);
        notify_pipeline_submission_complete(pipeline);
        restart_vulkan_infer_commands(submission);
        recorded_stage_count = 0;
        recorded_output_bytes = 0;
    };
    execute_pipeline(
        pipeline,
        node_map,
        param_map,
        [&](size_t input_idx) -> GpuTensor* {
            if (input_idx < input_tensors.size()) {
                return &input_tensors[input_idx];
            }
            return nullptr;
        },
        [&](InferStage& stage, const std::vector<GpuTensor*>& /*resolved*/) {
            const auto policy = stage.stage->submit_policy();
            if (policy.isolate && recorded_stage_count > 0) {
                flush_submission();
            }
            stage.stage->execute(reinterpret_cast<GpuCommandBufferHandle>(submission.cmd));
            recorded_stage_count += std::max<size_t>(policy.weight, 1);
            for (const auto& out : stage.outputs) {
                if (out && out->buf.valid()) {
                    recorded_output_bytes += out->buf.size;
                }
            }
            if (policy.isolate ||
                recorded_stage_count >= kStagesPerSubmit ||
                recorded_output_bytes >= kMaxOutputBytesPerSubmit) {
                flush_submission();
            }
        });
    if (recorded_stage_count > 0) {
        submit_vulkan_infer_commands(submission);
        notify_pipeline_submission_complete(pipeline);
    }

    finalize_infer_profiling("vulkan",
                             cm,
                             state,
                             profiler,
                             reinterpret_cast<GpuCommandBufferHandle>(submission.cmd));

    // Copy outputs back to host tensors via host-visible staging buffers.
    ensure_output_staging_handles(get_outputs().size(), "GFX Vulkan");
    auto& output_handles = vk_state->output_staging_handles;
    auto output_input_lookup = [&](size_t input_idx) -> GpuTensor* {
        if (input_idx < input_tensors.size()) {
            return &input_tensors[input_idx];
        }
        return nullptr;
    };
    bind_outputs_for_infer(
        cm,
        pipeline,
        node_map,
        param_map,
        output_input_lookup,
        [&](size_t idx, const std::shared_ptr<GfxRemoteTensor>& remote) {
            ov::ISyncInferRequest::set_tensor(get_outputs().at(idx),
                                              ov::SoPtr<ov::ITensor>{remote, nullptr});
        },
        [&](size_t idx, GpuTensor& dev, const OutputViewInfo& info, const ov::Tensor* host_override) {
            auto bound = bind_host_output_vulkan(dev,
                                                 info,
                                                 host_override,
                                                 &pool,
                                                 &output_handles[idx],
                                                 "GFX Vulkan");
            ov::ISyncInferRequest::set_tensor(get_outputs().at(idx),
                                              ov::get_tensor_impl(bound.host_tensor));
        },
        /*allow_missing=*/false,
        "GFX Vulkan");

}

}  // namespace gfx_plugin
}  // namespace ov
