// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/gfx_plugin/infer_request.hpp"

#include <chrono>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <utility>

#include "openvino/gfx_plugin/compiled_model.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_profiler.hpp"
#include "runtime/gfx_target_profile.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "runtime/gpu_buffer_pool.hpp"
#include "runtime/memory_manager.hpp"
#include "backends/metal/plugin/compiled_model_state.hpp"
#include "backends/metal/runtime/gpu_memory.hpp"
#include "backends/metal/runtime/metal_memory.hpp"
#include "backends/metal/runtime/profiling/profiler.hpp"
#include "backends/metal/plugin/infer_io_metal.hpp"
#include "plugin/infer_request_backend_access.hpp"
#include "plugin/infer_request_state.hpp"
#include "plugin/gfx_profiling_utils.hpp"
#include "plugin/infer_profiling_utils.hpp"
#include "plugin/infer_io_utils.hpp"
#include "runtime/infer_executor.hpp"
#include "runtime/infer_pipeline.hpp"
#include "runtime/infer_submission.hpp"
#include "runtime/stateful_execution.hpp"
#include "runtime/tensor_binding_contract.hpp"
#include "backends/metal/runtime/metal_command_encoder.hpp"

#import <Metal/Metal.h>

namespace ov {
namespace gfx_plugin {

struct MetalInferState final : BackendInferState {
    MetalAllocatorCore* alloc_core = nullptr;
    MetalDeviceCaps caps{};
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

MetalInferState* get_metal_state(BackendRequestState& state) {
    return dynamic_cast<MetalInferState*>(state.backend.get());
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
        metal_end_compute_encoder(reinterpret_cast<GpuCommandBufferHandle>(m_command_buffer));
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

void MetalBackendState::init_infer_state(BackendRequestState& state) const {
    OPENVINO_ASSERT(alloc_core, "GFX: Metal allocator core is not initialized");
    state.backend = std::make_unique<MetalInferState>(*alloc_core, caps);
}

ov::SoPtr<ov::ITensor> MetalBackendState::get_tensor_override(
    const BackendRequestState& state,
    size_t idx,
    const std::vector<ov::Output<const ov::Node>>& outputs) const {
    if (!state.backend || idx >= state.backend->output_tensors.size()) {
        return {};
    }
    const auto& dev = state.backend->output_tensors[idx];
    if (!dev.buf.valid()) {
        return {};
    }
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

void execute_metal_infer_request(InferRequest& request,
                                 const std::shared_ptr<const CompiledModel>& cm) {
    @autoreleasepool {
        OPENVINO_ASSERT(cm, "CompiledModel is null");
        OPENVINO_ASSERT(cm->backend() == GpuBackend::Metal, "GFX: Metal infer called for non-metal backend");

        auto& state = InferRequestBackendAccess::state(request);
        auto* metal_state = get_metal_state(state.runtime);
        OPENVINO_ASSERT(metal_state && metal_state->alloc_core, "MetalAllocator is not initialized");
        auto* alloc_core = metal_state->alloc_core;
        auto& allocator = metal_state->allocator;
        auto& caps = metal_state->caps;

        if (!state.debug_buffers.empty()) {
            for (auto& buf : state.debug_buffers) {
                allocator.release(std::move(buf));
            }
            state.debug_buffers.clear();
            state.debug_tensors.clear();
        }

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
            auto* metal_backend = dynamic_cast<const MetalBackendState*>(cm->backend_state());
            if (metal_backend && metal_backend->const_manager) {
                if (auto info = metal_backend->const_manager->query_execution_device_info()) {
                    record_gfx_target_profile(make_gfx_target_profile(*info), profiler);
                }
            }
        }
        std::vector<GpuTensor> runtime_input_tensors;
        InferRequestBackendAccess::bind_inputs_before_infer(
            request,
            cm->target(),
            runtime_input_tensors,
            [&](size_t idx, const ov::Tensor& host, BufferHandle*) {
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
                return bind_host_input_metal(src,
                                             &gpu_alloc,
                                             profiler,
                                             "GFX");
            },
            {},
            profiler,
            profiling,
            /*with_staging=*/false,
            "GFX");

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
        auto tensor_type = [](const GpuTensor& t) {
            return t.expected_type == ov::element::dynamic ? t.buf.type : t.expected_type;
        };
        auto safe_check = [&](const char* label,
                              const GpuTensor* t,
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

        const bool profiling_enabled = (profiler != nullptr);
        void* stage_profiler = profiler ? profiler->native_handle() : nullptr;
        auto* metal = dynamic_cast<const MetalBackendState*>(cm->backend_state());
        OPENVINO_ASSERT(metal && metal->const_manager, "GFX: Metal buffer manager is not initialized");
        GpuBufferPool pool(gpu_alloc);
        auto input_lookup = [&](size_t input_idx) -> GpuTensor* {
            return lookup_runtime_input_tensor(runtime_input_tensors, input_idx);
        };
        OPENVINO_ASSERT(metal && metal->command_queue, "GFX: command queue is null");
        id<MTLCommandQueue> cq = static_cast<id<MTLCommandQueue>>(metal->command_queue);
        OPENVINO_ASSERT(cq, "GFX: command queue is null");
        MetalInferSubmissionSession submission(cq, profiler);
        InferSubmissionTuningCaps submission_caps{};
        submission_caps.preferred_simd_width = std::max<uint32_t>(caps.preferred_simd_width, 1u);
        submission_caps.subgroup_size = std::max<uint32_t>(caps.preferred_simd_width, 1u);
        submission_caps.max_total_threads_per_group =
            std::max<uint32_t>(caps.max_total_threads_per_threadgroup, 1u);
        submission_caps.mac_budget_scale_num = 3u;
        submission_caps.mac_budget_scale_den = 2u;
        InferRuntimeExecutionConfig execution_config{};
        execution_config.state = metal_state;
        execution_config.descs = &descs;
        execution_config.buffer_manager = metal->const_manager.get();
        execution_config.stage_profiler = stage_profiler;
        execution_config.profiling_enabled = profiling_enabled;
        execution_config.remote_outputs = &state.bound_remote_outputs;
        execution_config.remote_inputs = &state.bound_remote_inputs;
        execution_config.expected_target = &cm->target();
        execution_config.runtime_descriptor = cm->runtime_descriptor();
        execution_config.pool = &pool;
        execution_config.post_prepare = [](std::vector<InferStage>&) {};
        execution_config.runtime_input_tensors = &runtime_input_tensors;
        execution_config.init_output_desc =
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
            };
        execution_config.input_lookup = input_lookup;
        execution_config.submission = &submission;
        execution_config.submission_caps = submission_caps;
        execution_config.on_stage =
            [&](InferStage& stage, const std::vector<GpuTensor*>& resolved, GpuCommandBufferHandle command_buffer) {
                execute_infer_stage_with_stateful_contract(
                    state.variable_states,
                    stage,
                    resolved,
                    pool,
                    command_buffer,
                    profiler,
                    [&](InferStage& backend_stage,
                        const std::vector<GpuTensor*>& backend_resolved,
                        GpuCommandBufferHandle backend_command_buffer) {
                        if (gfx_log_debug_enabled() || metal_safe_debug_enabled()) {
                            const auto* stage_descriptor =
                                runtime_stage_descriptor_or_null(backend_stage);
                            const std::string node_name =
                                stage_descriptor && !stage_descriptor->stage_name.empty()
                                    ? stage_descriptor->stage_name
                                    : backend_stage.stage->name();
                            const std::string node_type =
                                stage_descriptor && !stage_descriptor->op_family.empty()
                                    ? stage_descriptor->op_family
                                    : backend_stage.stage->type();
                            auto describe_source_ref =
                                [](const PipelineStageInputLink& input) {
                                    std::ostringstream ref;
                                    switch (input.source_ref.kind) {
                                    case PipelineStageTensorRefKind::Parameter:
                                        ref << "param[" << input.source_ref.index << ":"
                                            << input.source_ref.port << "]";
                                        break;
                                    case PipelineStageTensorRefKind::StageOutput:
                                        ref << "stage[" << input.source_ref.index << "].out"
                                            << input.source_ref.port;
                                        break;
                                    case PipelineStageTensorRefKind::None:
                                    default:
                                        ref << "unbound";
                                        break;
                                    }
                                    return ref.str();
                                };
                            auto static_input_shape_from_descriptor =
                                [&](size_t input_idx, ov::Shape& shape) {
                                    return stage_descriptor &&
                                           input_idx < stage_descriptor->input_bindings.size() &&
                                           parse_static_shape_contract(
                                               stage_descriptor->input_bindings[input_idx].partial_shape,
                                               shape);
                                };
                            auto static_output_shape_from_descriptor =
                                [&](size_t output_idx, ov::Shape& shape) {
                                    return stage_descriptor &&
                                           output_idx < stage_descriptor->output_bindings.size() &&
                                           parse_static_shape_contract(
                                               stage_descriptor->output_bindings[output_idx].partial_shape,
                                               shape);
                                };
                            if (gfx_log_debug_enabled()) {
                                std::ostringstream oss;
                                oss << "Op=" << node_type << " name=" << node_name;
                                for (size_t i = 0; i < backend_resolved.size(); ++i) {
                                    if (!backend_resolved[i])
                                        continue;
                                    const auto& t = *backend_resolved[i];
                                    oss << " in" << i << "_shape=" << shape_to_string(t.shape)
                                        << " in" << i << "_bytes=" << t.buf.size;
                                    if (backend_stage.stage && backend_stage.stage->has_internal_input_bindings() &&
                                        i < backend_stage.inputs.size()) {
                                        oss << " in" << i << "_src="
                                            << describe_source_ref(backend_stage.inputs[i]);
                                    }
                                }
                                for (size_t i = 0; i < backend_stage.outputs.size(); ++i) {
                                    const auto& t = backend_stage.outputs[i];
                                    if (!t)
                                        continue;
                                    oss << " out" << i << "_shape=" << shape_to_string(t->shape)
                                        << " out" << i << "_bytes=" << t->buf.size;
                                }
                                gfx_log_debug("PIPELINE") << oss.str();
                            }
                            const bool internal_input_bindings =
                                backend_stage.stage && backend_stage.stage->has_internal_input_bindings();
                            for (size_t i = 0; i < backend_resolved.size(); ++i) {
                                if (!backend_resolved[i])
                                    continue;
                                ov::Shape in_shape = backend_resolved[i]->shape;
                                if (!internal_input_bindings &&
                                    in_shape.empty()) {
                                    static_input_shape_from_descriptor(i, in_shape);
                                }
                                safe_check(("input" + std::to_string(i)).c_str(),
                                           backend_resolved[i],
                                           in_shape,
                                           !internal_input_bindings);
                            }
                            for (size_t i = 0; i < backend_stage.outputs.size(); ++i) {
                                if (!backend_stage.outputs[i])
                                    continue;
                                ov::Shape out_shape = backend_stage.outputs[i]->shape;
                                if (out_shape.empty()) {
                                    static_output_shape_from_descriptor(i, out_shape);
                                }
                                safe_check(("output" + std::to_string(i)).c_str(),
                                           backend_stage.outputs[i].get(),
                                           out_shape,
                                           false);
                            }
                        }
                        backend_stage.stage->execute(backend_command_buffer);
                    });
            };
        execution_config.profiler = profiler;
        execution_config.error_prefix = "GFX";
        const auto execution_result =
            prepare_and_execute_infer_runtime(std::move(execution_config));
        if (profiler) {
            profiler->set_counter("stage_output_workspace_outputs",
                                  metal_state->stage_output_workspace.last_workspace_outputs);
            profiler->set_counter("stage_output_direct_outputs",
                                  metal_state->stage_output_workspace.last_direct_outputs);
            profiler->set_counter("stage_output_workspace_slots",
                                  metal_state->stage_output_workspace.last_slots_used);
            profiler->set_counter("stage_output_workspace_peak_live",
                                  metal_state->stage_output_workspace.last_peak_live_slots);
        }
        if (gfx_log_debug_enabled()) {
            const auto& submission_tuning = execution_result.submission_tuning;
            gfx_log_debug("InferSubmit") << "Metal submission tuning: slots=" << submission_tuning.slot_count
                                         << " max_stages=" << submission_tuning.config.max_stages_per_submit
                                         << " max_output_bytes="
                                         << submission_tuning.config.max_output_bytes_per_submit
                                         << " max_macs=" << submission_tuning.config.max_macs_per_submit
                                         << " simd=" << submission_caps.preferred_simd_width
                                         << " max_threads=" << submission_caps.max_total_threads_per_group
                                         << " incremental="
                                         << (submission_tuning.config.allow_incremental_submit ? "yes" : "no")
                                         << " pipeline_stages=" << execution_result.pipeline->size();
        }
        return execution_result.completed_command_buffer;
    };

    auto completed_command_buffer = run_pipeline();
    auto& pipeline = metal_state->reusable_pipeline;

    GpuBufferPool output_pool(gpu_alloc);

    auto output_input_lookup = [&](size_t input_idx) -> GpuTensor* {
        return lookup_runtime_input_tensor(runtime_input_tensors, input_idx);
    };
    const auto resources = cm->backend_state()->resources();
    OPENVINO_ASSERT(resources.queue, "GFX: command queue is null");
    InferRequestBackendAccess::bind_outputs_after_infer(
        request,
        cm,
        pipeline,
        output_input_lookup,
        [&](size_t idx,
            GpuTensor& dev,
            const OutputViewInfo& info,
            const ov::Tensor* host_override,
            ov::Tensor* reusable_host,
            BufferHandle* staging_handle) {
            if (gfx_log_debug_enabled()) {
                gfx_log_debug("InferIO") << "output[" << idx << "] dev_buf=" << dev.buf.buffer
                                        << " et=" << (dev.expected_type == ov::element::dynamic
                                                          ? dev.buf.type.get_type_name()
                                                          : dev.expected_type.get_type_name());
            }
            return bind_host_output_metal(dev,
                                          info,
                                          host_override,
                                          reusable_host,
                                          &gpu_alloc,
                                          &output_pool,
                                          staging_handle,
                                          resources.queue,
                                          profiler,
                                          "GFX");
        },
        {},
        profiler,
        profiling,
        "GFX");

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
