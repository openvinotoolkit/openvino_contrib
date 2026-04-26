// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/gfx_plugin/infer_request.hpp"

#include <chrono>
#include <iostream>
#include <algorithm>
#include <sstream>

#import <Metal/Metal.h>
#ifdef NO
#undef NO
#endif
#ifdef YES
#undef YES
#endif

#include "openvino/gfx_plugin/compiled_model.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_profiler.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "runtime/gpu_buffer_pool.hpp"
#include "runtime/memory_manager.hpp"
#include "backends/metal/plugin/compiled_model_state.hpp"
#include "backends/metal/runtime/gpu_memory.hpp"
#include "backends/metal/runtime/metal_memory.hpp"
#include "backends/metal/runtime/profiling/profiler.hpp"
#include "backends/metal/plugin/infer_io_metal.hpp"
#include "plugin/infer_request_state.hpp"
#include "plugin/gfx_profiling_utils.hpp"
#include "plugin/infer_profiling_utils.hpp"
#include "plugin/infer_pipeline.hpp"
#include "plugin/infer_io_utils.hpp"
#include "plugin/infer_submission.hpp"
#include "plugin/stateful_execution.hpp"

namespace ov {
namespace gfx_plugin {

struct MetalInferState final : BackendInferState {
    MetalAllocatorCore* alloc_core = nullptr;
    MetalDeviceCaps caps{};
    MetalTensorMap tensor_map;
    MetalHeapPool heaps;
    MetalFreeList freelist;
    MetalStagingPool staging;
    MetalAllocator allocator;

    MetalInferState(MetalAllocatorCore& core, const MetalDeviceCaps& caps_in)
        : alloc_core(&core),
          caps(caps_in),
          heaps(core),
          freelist(),
          staging(core),
          allocator(core, heaps, freelist, staging, caps_in) {}
};

namespace {

MetalInferState* get_metal_state(InferRequestState& state) {
    return dynamic_cast<MetalInferState*>(state.backend.get());
}

const MetalInferState* get_metal_state(const InferRequestState& state) {
    return dynamic_cast<const MetalInferState*>(state.backend.get());
}

class MetalInferSubmissionSession final : public SingleFlightInferSubmissionSession {
public:
    explicit MetalInferSubmissionSession(id<MTLCommandQueue> command_queue, GfxProfiler* profiler)
        : m_command_queue(command_queue),
          m_profiler(profiler) {}

    bool supports_incremental_submit() const override {
        return false;
    }

protected:
    GpuCommandBufferHandle begin_recording_on_slot() override {
        OPENVINO_ASSERT(m_command_queue, "GFX: command queue is null");
        m_command_buffer = [m_command_queue commandBuffer];
        return reinterpret_cast<GpuCommandBufferHandle>(m_command_buffer);
    }

    void submit_recorded_on_slot(GpuCommandBufferHandle /*command_buffer*/, bool /*continue_recording*/) override {
        // Metal path records the whole inference into a single command buffer.
    }

    void finish_submission_slot() override {
        if (!m_command_buffer) {
            return;
        }
        const bool profiling = (m_profiler != nullptr);
        const auto commit_start = profiling ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
        [m_command_buffer commit];
        if (profiling) {
            const auto commit_cpu_us =
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - commit_start);
            m_profiler->record_segment("submit",
                                       "metal_commit",
                                       commit_cpu_us,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       -1,
                                       0,
                                       reinterpret_cast<uint64_t>(m_command_buffer));
            m_profiler->increment_counter("metal_commit_count");
        }
        const auto wait_start = profiling ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
        [m_command_buffer waitUntilCompleted];
        if (profiling) {
            const auto wait_cpu_us =
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - wait_start);
            m_profiler->record_segment("wait",
                                       "metal_wait_until_completed",
                                       wait_cpu_us,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       -1,
                                       0,
                                       reinterpret_cast<uint64_t>(m_command_buffer));
            m_profiler->increment_counter("metal_wait_count");
        }
        set_completed_command_buffer(reinterpret_cast<GpuCommandBufferHandle>(m_command_buffer));
        m_command_buffer = nil;
    }

private:
    id<MTLCommandQueue> m_command_queue = nil;
    id<MTLCommandBuffer> m_command_buffer = nil;
    GfxProfiler* m_profiler = nullptr;
};

}  // namespace

void MetalBackendState::init_infer_state(InferRequestState& state) const {
    OPENVINO_ASSERT(alloc_core, "GFX: Metal allocator core is not initialized");
    state.backend = std::make_unique<MetalInferState>(*alloc_core, caps);
}

ov::SoPtr<ov::ITensor> MetalBackendState::get_tensor_override(
    const InferRequestState& state,
    size_t idx,
    const std::vector<ov::Output<const ov::Node>>& outputs) const {
    const auto* metal_state = get_metal_state(state);
    if (!metal_state || !metal_state->tensor_map.has_output_device(idx)) {
        return {};
    }
    const auto& dev = metal_state->tensor_map.get_output_device(idx);
    if (dev.buf.storage_mode != static_cast<uint32_t>(MTLStorageModeShared)) {
        OPENVINO_THROW("GFX: output buffer is not host-visible");
    }
    id<MTLBuffer> buf = static_cast<id<MTLBuffer>>(dev.buf.buffer);
    void* ptr = buf ? [buf contents] : nullptr;
    if (!ptr) {
        OPENVINO_THROW("GFX: shared output buffer has no CPU pointer");
    }
    ov::Shape shape = dev.shape;
    if (shape.empty()) {
        const auto& out = outputs.at(idx);
        if (out.get_partial_shape().is_static())
            shape = out.get_shape();
        else
            shape = ov::Shape{1};
    }
    ov::element::Type logical = dev.expected_type == ov::element::dynamic ? dev.buf.type : dev.expected_type;
    ov::Tensor view{logical, shape, ptr};
    return ov::get_tensor_impl(view);
}

void InferRequest::infer_metal_impl(const std::shared_ptr<const CompiledModel>& cm) {
    @autoreleasepool {
        OPENVINO_ASSERT(cm, "CompiledModel is null");
        OPENVINO_ASSERT(cm->backend() == GpuBackend::Metal, "GFX: Metal infer called for non-metal backend");

        auto& state = *m_state;
        auto* metal_state = get_metal_state(state);
        OPENVINO_ASSERT(metal_state && metal_state->alloc_core, "MetalAllocator is not initialized");
        auto* alloc_core = metal_state->alloc_core;
        auto& allocator = metal_state->allocator;
        auto& tensor_map = metal_state->tensor_map;
        auto& caps = metal_state->caps;

        if (!state.debug_buffers.empty()) {
            for (auto& buf : state.debug_buffers) {
                allocator.release(std::move(buf));
            }
            state.debug_buffers.clear();
            state.debug_tensors.clear();
        }

        tensor_map.reset_inference(alloc_core);
        allocator.reset_stats();

        MetalMemorySession session(allocator, nullptr, nullptr);
        MetalBufferManager::set_current_session(&session);
        struct SessionGuard {
            MetalMemorySession& session;
            ~SessionGuard() {
                session.end();
                MetalBufferManager::set_current_session(nullptr);
            }
        } session_guard{session};

        MetalGpuAllocator gpu_alloc(allocator, *alloc_core, caps);

        GfxProfiler* profiler = prepare_infer_profiler(*cm, state, "GFX");
        MetalProfiler* metal_profiler = profiler
                                            ? static_cast<MetalProfiler*>(profiler->native_handle())
                                            : nullptr;
        const bool detailed = (state.profiler_cfg.level == ProfilingLevel::Detailed);
        allocator.set_profiler(metal_profiler, detailed);
        const bool profiling = (profiler != nullptr);
        if (profiler) {
            profiler->begin_infer(cm->pipeline_desc().size());
        }
        const auto bind_inputs_start = profiling ? std::chrono::steady_clock::now()
                                                 : std::chrono::steady_clock::time_point{};

        bind_inputs_for_infer(
            GpuBackend::Metal,
            [&](size_t idx, const GpuTensor& dev) {
                tensor_map.bind_input_device(idx, dev);
            },
            [&](size_t idx, const ov::Tensor& host) {
                const ov::Tensor src = host;
                if (gfx_log_debug_enabled()) {
                    gfx_log_debug("InferIO")
                        << "input[" << idx << "] host_et=" << src.get_element_type().get_type_name()
                        << " bytes=" << src.get_byte_size()
                        << " ptr=" << src.data()
                        << " align4=" << (reinterpret_cast<uintptr_t>(src.data()) & 3u);
                    if (src && src.get_size() > 0 && src.get_element_type() == ov::element::f32) {
                        const float* p = src.data<const float>();
                        const size_t n = std::min<size_t>(4, src.get_size());
                        std::ostringstream vals;
                        for (size_t i = 0; i < n; ++i) {
                            if (i)
                                vals << " ";
                            vals << p[i];
                        }
                        gfx_log_debug("InferIO") << "input[" << idx << "] first=" << vals.str();
                    }
                }
                // Bind host -> device using zero-copy shared buffer (no memcpy).
                const auto dev = bind_host_input_metal(src,
                                                       &gpu_alloc,
                                                       profiler,
                                                       "GFX");
                tensor_map.bind_input_device(idx, dev);
            },
            "GFX");
        if (profiling) {
            profiler->record_segment("upload",
                                     "bind_inputs_for_infer",
                                     std::chrono::duration_cast<std::chrono::microseconds>(
                                         std::chrono::steady_clock::now() - bind_inputs_start));
        }

    auto run_pipeline = [&]() {
        auto shape_to_string = [](const ov::Shape& shape) {
            std::ostringstream ss;
            ss << "[";
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i) ss << ",";
                ss << shape[i];
            }
            ss << "]";
            return ss.str();
        };
        auto tensor_type = [](const MetalTensor& t) {
            return t.expected_type == ov::element::dynamic ? t.buf.type : t.expected_type;
        };
        auto safe_check = [&](const char* label,
                              const MetalTensor* t,
                              const ov::Shape& shape,
                              bool require_valid) {
            if (!t || !t->buf.valid()) {
                if (require_valid) {
                    OPENVINO_ASSERT(false, "GFX: ", label, " buffer is null");
                }
                return;
            }
            if (shape.empty())
                return;
            const size_t need = tensor_byte_size(shape, tensor_type(*t));
            OPENVINO_ASSERT(need <= t->buf.size,
                            "GFX: ", label, " buffer overflow (need ", need,
                            ", have ", t->buf.size, ", shape=", shape_to_string(shape), ")");
        };

        OPENVINO_ASSERT(cm->op_pipeline_built(),
                        "GFX: op pipeline is not built");
        const auto& descs = cm->pipeline_desc();
        const auto& node_map = cm->node_to_stage();
        const auto& param_map = cm->parameter_index();

        const bool profiling_enabled = (profiler != nullptr);
        void* stage_profiler = profiler ? profiler->native_handle() : nullptr;
        auto* metal = dynamic_cast<const MetalBackendState*>(cm->backend_state());
        OPENVINO_ASSERT(metal && metal->const_manager, "GFX: Metal buffer manager is not initialized");
        GpuBufferPool pool(gpu_alloc);
        const auto pipeline_prepare_start = profiling ? std::chrono::steady_clock::now()
                                                      : std::chrono::steady_clock::time_point{};
        auto& pipeline = prepare_reusable_pipeline_with_outputs(metal_state->reusable_pipeline,
                                                                descs,
                                                                metal->const_manager.get(),
                                                                stage_profiler,
                                                                profiling_enabled,
                                                                cm->get_runtime_model(),
                                                                get_outputs(),
                                                                node_map,
                                                                param_map,
                                                                state.bound_remote_outputs,
                                                                state.bound_remote_inputs,
                                                                GpuBackend::Metal,
                                                                pool,
                                                                metal_state->stage_output_handles,
                                                                [](std::vector<InferStage>&) {},
                                                                [&](InferStage& stage,
                                                                    size_t oi,
                                                                    GpuTensor& out_ref,
                                                                    GpuBufferDesc& desc,
                                                                    const char* error_prefix) {
                                                                    const bool is_model_output =
                                                                        (oi < stage.output_is_model_output.size()) &&
                                                                        stage.output_is_model_output[oi];
                                                                    return init_stage_output_desc(GpuBackend::Metal,
                                                                                                  stage,
                                                                                                  oi,
                                                                                                  out_ref,
                                                                                                  desc,
                                                                                                  is_model_output,
                                                                                                  /*skip_view_ops=*/true,
                                                                                                  error_prefix);
                                                                },
                                                                "GFX");
        prepare_reusable_execution_plan(metal_state->reusable_execution_plan, pipeline, node_map, param_map);
        if (!metal_state->reusable_pipeline_runtime_prewarmed) {
            const auto prewarm_start = profiling ? std::chrono::steady_clock::now()
                                                 : std::chrono::steady_clock::time_point{};
            prewarm_pipeline_runtime_state(
                pipeline,
                node_map,
                param_map,
                [&](size_t input_idx) -> GpuTensor* {
                    return &tensor_map.get_input_device(input_idx);
                },
                &metal_state->reusable_execution_plan);
            metal_state->reusable_pipeline_runtime_prewarmed = true;
            if (profiling) {
                profiler->record_segment("compile",
                                         "prewarm_reusable_pipeline",
                                         std::chrono::duration_cast<std::chrono::microseconds>(
                                             std::chrono::steady_clock::now() - prewarm_start));
            }
        }
        if (profiling) {
            profiler->record_segment("compile",
                                     "prepare_reusable_pipeline",
                                     std::chrono::duration_cast<std::chrono::microseconds>(
                                         std::chrono::steady_clock::now() - pipeline_prepare_start));
        }

        OPENVINO_ASSERT(metal && metal->command_queue, "GFX: command queue is null");
        id<MTLCommandQueue> cq = static_cast<id<MTLCommandQueue>>(metal->command_queue);
        OPENVINO_ASSERT(cq, "GFX: command queue is null");
        MetalInferSubmissionSession submission(cq, profiler);
        InferSubmissionTuningCaps submission_caps{};
        submission_caps.backend = GpuBackend::Metal;
        submission_caps.preferred_simd_width = std::max<uint32_t>(caps.preferred_simd_width, 1u);
        submission_caps.subgroup_size = std::max<uint32_t>(caps.preferred_simd_width, 1u);
        submission_caps.max_total_threads_per_group =
            std::max<uint32_t>(caps.max_total_threads_per_threadgroup, 1u);
        submission_caps.supports_incremental_submit = submission.supports_incremental_submit();
        const auto submission_tuning = select_infer_submission_tuning(submission_caps, pipeline.size());
        record_infer_submission_tuning_counters(submission_tuning, submission_caps, profiler);
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("InferSubmit") << "Metal submission tuning: slots=" << submission_tuning.slot_count
                                         << " max_stages=" << submission_tuning.config.max_stages_per_submit
                                         << " max_output_bytes="
                                         << submission_tuning.config.max_output_bytes_per_submit
                                         << " simd=" << submission_caps.preferred_simd_width
                                         << " max_threads=" << submission_caps.max_total_threads_per_group
                                         << " incremental="
                                         << (submission_caps.supports_incremental_submit ? "yes" : "no")
                                         << " pipeline_stages=" << pipeline.size();
        }
        const auto infer_start = profiling ? std::chrono::steady_clock::now()
                                           : std::chrono::steady_clock::time_point{};
        execute_pipeline_with_submission(
            pipeline,
            node_map,
            param_map,
            [&](size_t input_idx) -> GpuTensor* {
                return &tensor_map.get_input_device(input_idx);
            },
            submission,
            submission_tuning.config,
            profiler,
            &metal_state->reusable_execution_plan,
            [&](InferStage& stage, const std::vector<GpuTensor*>& resolved, GpuCommandBufferHandle command_buffer) {
                if (execute_stateful_stage(state, stage, resolved, pool, command_buffer)) {
                    return;
                }
                if (gfx_log_debug_enabled() || metal_safe_debug_enabled()) {
                    const std::string node_name =
                        stage.node ? stage.node->get_friendly_name() : stage.stage->name();
                    const std::string node_type =
                        stage.node ? stage.node->get_type_name() : stage.stage->type();
                    if (gfx_log_debug_enabled()) {
                        std::ostringstream oss;
                        oss << "Op=" << node_type << " name=" << node_name;
                        for (size_t i = 0; i < resolved.size(); ++i) {
                            if (!resolved[i])
                                continue;
                            const auto& t = *resolved[i];
                            oss << " in" << i << "_shape=" << shape_to_string(t.shape)
                                << " in" << i << "_bytes=" << t.buf.size;
                            if (stage.stage && stage.stage->has_internal_input_bindings() &&
                                i < stage.inputs.size() && stage.inputs[i].node) {
                                oss << " in" << i << "_src="
                                    << stage.inputs[i].node->get_friendly_name()
                                    << ":" << stage.inputs[i].port;
                            }
                        }
                        for (size_t i = 0; i < stage.outputs.size(); ++i) {
                            const auto& t = stage.outputs[i];
                            if (!t)
                                continue;
                            oss << " out" << i << "_shape=" << shape_to_string(t->shape)
                                << " out" << i << "_bytes=" << t->buf.size;
                        }
                        gfx_log_debug("PIPELINE") << oss.str();
                    }
                    const bool internal_input_bindings =
                        stage.stage && stage.stage->has_internal_input_bindings();
                    for (size_t i = 0; i < resolved.size(); ++i) {
                        if (!resolved[i])
                            continue;
                        ov::Shape in_shape = resolved[i]->shape;
                        if (!internal_input_bindings &&
                            in_shape.empty() && stage.node &&
                            i < stage.node->get_input_size() &&
                            stage.node->get_input_partial_shape(i).is_static()) {
                            in_shape = stage.node->get_input_shape(i);
                        }
                        safe_check(("input" + std::to_string(i)).c_str(),
                                   resolved[i],
                                   in_shape,
                                   !internal_input_bindings);
                    }
                    for (size_t i = 0; i < stage.outputs.size(); ++i) {
                        if (!stage.outputs[i])
                            continue;
                        ov::Shape out_shape = stage.outputs[i]->shape;
                        if (out_shape.empty() && stage.node &&
                            i < stage.node->get_output_size() &&
                            stage.node->get_output_partial_shape(i).is_static()) {
                            out_shape = stage.node->get_output_shape(i);
                        }
                        safe_check(("output" + std::to_string(i)).c_str(),
                                   stage.outputs[i].get(),
                                   out_shape,
                                   false);
                    }
                }
                stage.stage->execute(command_buffer);
            });
        if (profiling) {
            profiler->record_segment("infer",
                                     "execute_pipeline_with_submission",
                                     std::chrono::duration_cast<std::chrono::microseconds>(
                                         std::chrono::steady_clock::now() - infer_start));
        }
        return submission.completed_command_buffer();
    };

    auto completed_command_buffer = run_pipeline();
    auto& pipeline = metal_state->reusable_pipeline;

    ensure_output_staging_handles(get_outputs().size(), "GFX");
    auto& output_handles = metal_state->output_staging_handles;
    GpuBufferPool output_pool(gpu_alloc);

    auto output_input_lookup = [&](size_t input_idx) -> GpuTensor* {
        if (!tensor_map.has_input_device(input_idx)) {
            return nullptr;
        }
        return &tensor_map.get_input_device(input_idx);
    };
    const auto resources = cm->backend_state()->resources();
    OPENVINO_ASSERT(resources.queue, "GFX: command queue is null");
    const auto bind_outputs_start = profiling ? std::chrono::steady_clock::now()
                                              : std::chrono::steady_clock::time_point{};
    bind_outputs_for_infer(
        cm,
        pipeline,
        cm->node_to_stage(),
        cm->parameter_index(),
        output_input_lookup,
        [&](size_t idx, const std::shared_ptr<GfxRemoteTensor>& remote) {
            ov::ISyncInferRequest::set_tensor(get_outputs()[idx],
                                              ov::SoPtr<ov::ITensor>{remote, nullptr});
        },
        [&](size_t idx, GpuTensor& dev, const OutputViewInfo& info, const ov::Tensor* host_override) {
            if (gfx_log_debug_enabled()) {
                gfx_log_debug("InferIO") << "output[" << idx << "] dev_buf=" << dev.buf.buffer
                                        << " et=" << (dev.expected_type == ov::element::dynamic
                                                          ? dev.buf.type.get_type_name()
                                                          : dev.expected_type.get_type_name());
            }
            ov::Tensor* reusable_host = nullptr;
            if (idx < metal_state->reusable_host_output_plan.outputs.size()) {
                auto& prepared = metal_state->reusable_host_output_plan.outputs[idx];
                if (prepared.host) {
                    reusable_host = &prepared.host;
                }
            }
            auto bound = bind_host_output_metal(dev,
                                                info,
                                                host_override,
                                                reusable_host,
                                                &gpu_alloc,
                                                &output_pool,
                                                &output_handles[idx],
                                                resources.queue,
                                                profiler,
                                                "GFX");
            tensor_map.bind_output_device(idx, bound.device_tensor);
            ov::ISyncInferRequest::set_tensor(get_outputs()[idx], ov::get_tensor_impl(bound.host_tensor));
        },
        /*allow_missing=*/false,
        "GFX");
    if (profiling) {
        profiler->record_segment("download",
                                 "bind_outputs_for_infer",
                                 std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now() - bind_outputs_start));
    }

    finalize_infer_profiling("metal",
                             cm,
                             state,
                             profiler,
                             completed_command_buffer,
                             [&]() {
                                 if (metal_profiler) {
                                     const auto stats = allocator.stats();
                                     metal_profiler->set_memory_stats(stats);
                                 }
                             });

    if (cm && cm->backend_state()) {
        cm->backend_state()->set_mem_stats(ov::Any{allocator.stats()});
    }
    }
}

}  // namespace gfx_plugin
}  // namespace ov
