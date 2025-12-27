// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/op.hpp"
#include "backends/metal/runtime/backend.hpp"
#include "backends/metal/profiling/profiler.hpp"

#include <cstring>

namespace ov {
namespace gfx_plugin {

MetalOp::MetalOp(std::string name,
                 std::string type,
                 const ov::Shape& output_shape,
                 void* device,
                 void* command_queue)
    : m_name(std::move(name)),
      m_type(std::move(type)),
      m_output_shape(output_shape),
      m_device(static_cast<MetalDeviceHandle>(device)),
      m_command_queue(static_cast<MetalCommandQueueHandle>(command_queue)) {}

MetalOp::~MetalOp() {
    release_inflight_const_buffers();
}

void MetalOp::init(MetalBufferManager* buffer_manager) {
    m_buffer_manager = buffer_manager;
}

void MetalOp::compile(MetalBufferManager* buffer_manager) {
    if (!m_buffer_manager) {
        m_buffer_manager = buffer_manager;
    }
    m_compiled = true;
}

void MetalOp::set_inputs(const std::vector<MetalTensor*>& inputs) {
    m_inputs = inputs;
}

void MetalOp::set_output(MetalTensor* output) {
    m_output = output;
}

void MetalOp::set_outputs(const std::vector<std::unique_ptr<MetalTensor>>& outputs) {
    if (!outputs.empty()) {
        m_output = outputs.front().get();
    }
}

void MetalOp::execute_kernel(ICompiledKernel& kernel,
                             MetalCommandBufferHandle command_buffer,
                             const KernelDispatch& dispatch,
                             const std::vector<KernelArg>& args) {
    KernelExecutionHooks hooks;
    KernelExecutionHooks* hooks_ptr = nullptr;
    if (m_profiling_enabled) {
        hooks.on_begin = [this](GpuCommandEncoderHandle encoder) { start_profiling(encoder); };
        hooks.on_end = [this](GpuCommandEncoderHandle encoder) { stop_profiling_ms(encoder); };
        hooks_ptr = &hooks;
    }
    const auto bound_args = materialize_kernel_args(args);
    kernel.execute(command_buffer, dispatch, bound_args, hooks_ptr);
    flush_inflight_const_buffers(command_buffer);
}

std::shared_ptr<ICompiledKernel> MetalOp::compile_msl_kernel(MetalCodegenBackend& backend,
                                                             mlir::ModuleOp module,
                                                             const std::string& entry_point,
                                                             std::string msl_source,
                                                             std::string* log,
                                                             uint32_t arg_count) {
    KernelPlan plan(module, entry_point, arg_count);
    return backend.compile(plan.to_source_with_msl(std::move(msl_source)), log);
}

std::shared_ptr<ICompiledKernel> MetalOp::compile_msl_kernel(MetalCodegenBackend& backend,
                                                             const KernelSpec& spec,
                                                             mlir::ModuleOp module,
                                                             const std::string& entry_point,
                                                             std::string msl_source,
                                                             std::string* log) {
    KernelPlan plan(module, entry_point, spec.arg_count());
    return backend.compile(plan.to_source_with_msl(std::move(msl_source)), log);
}

MetalTensor& MetalOp::require_output() const {
    OPENVINO_ASSERT(m_output, "Output tensor is not bound for op ", m_name);
    return *m_output;
}

MetalBuffer MetalOp::allocate_temp_buffer(size_t bytes,
                                          ov::element::Type type,
                                          bool persistent,
                                          bool storageModePrivate) {
    OPENVINO_ASSERT(m_buffer_manager, "Buffer manager is not set for op ", m_name);
    return m_buffer_manager->allocate(bytes, type, persistent, storageModePrivate);
}

std::vector<KernelArg> MetalOp::materialize_kernel_args(const std::vector<KernelArg>& args) {
    if (args.empty()) {
        return args;
    }
    std::vector<KernelArg> out;
    out.reserve(args.size());
    for (const auto& arg : args) {
        if (arg.kind != KernelArg::Kind::Bytes) {
            out.push_back(arg);
            continue;
        }
        OPENVINO_ASSERT(m_buffer_manager, "Buffer manager is not set for op ", m_name);
        OPENVINO_ASSERT(arg.bytes, "MetalOp: bytes arg pointer is null for op ", m_name);
        OPENVINO_ASSERT(arg.byte_size > 0, "MetalOp: bytes arg size is zero for op ", m_name);
        auto payload = std::make_shared<std::vector<uint8_t>>(arg.byte_size);
        std::memcpy(payload->data(), arg.bytes, arg.byte_size);
        MetalBuffer buf = m_buffer_manager->wrap_shared(payload->data(), payload->size(), ov::element::u8);
        buf.external = true;
        buf.from_handle = true;
        buf.host_visible = true;
        m_inflight_const_buffers.push_back(buf);
        m_inflight_const_payloads.push_back(std::move(payload));
        out.push_back(make_buffer_arg(arg.index, buf, 0));
    }
    return out;
}

void MetalOp::flush_inflight_const_buffers(MetalCommandBufferHandle command_buffer) {
    if (m_inflight_const_buffers.empty()) {
        return;
    }
#ifdef __OBJC__
    if (!command_buffer) {
        release_inflight_const_buffers();
        return;
    }
    __block auto buffers = std::move(m_inflight_const_buffers);
    __block auto payloads = std::move(m_inflight_const_payloads);
    id<MTLCommandBuffer> cb = static_cast<id<MTLCommandBuffer>>(command_buffer);
    [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
        for (auto& buf : buffers) {
            metal_release_external_buffer(buf);
        }
        (void)payloads;
    }];
#else
    (void)command_buffer;
    release_inflight_const_buffers();
#endif
}

void MetalOp::release_inflight_const_buffers() {
    for (auto& buf : m_inflight_const_buffers) {
        if (buf.valid()) {
            metal_release_external_buffer(buf);
        }
    }
    m_inflight_const_buffers.clear();
    m_inflight_const_payloads.clear();
}

void MetalOp::set_profiler(MetalProfiler* profiler,
                           uint32_t node_id,
                           const std::string& node_name,
                           const std::string& node_type) {
    m_profiler = profiler;
    m_profile_node_id = node_id;
    m_profile_node_name = node_name;
    m_profile_node_type = node_type;
}

void MetalOp::start_profiling(MetalCommandEncoderHandle encoder) {
    if (!m_profiling_enabled)
        return;
    if (m_profiler) {
        const char* name = m_profile_node_name.empty() ? m_name.c_str() : m_profile_node_name.c_str();
        const char* type = m_profile_node_type.empty() ? m_type.c_str() : m_profile_node_type.c_str();
        m_profiler->begin_node(m_profile_node_id, name, type, "GFX");
        m_gpu_sample_begin = m_profiler->gpu_sample_begin(encoder);
    }
}

double MetalOp::stop_profiling_ms(MetalCommandEncoderHandle encoder) {
    if (!m_profiling_enabled)
        return m_last_duration_ms;
    if (m_profiler) {
        const auto sample_end = m_profiler->gpu_sample_end(encoder);
        m_profiler->end_node(m_profile_node_id, std::chrono::microseconds{0}, m_gpu_sample_begin, sample_end);
        m_gpu_sample_begin = -1;
    }
    m_last_duration_ms = 0.0;
    return m_last_duration_ms;
}

}  // namespace gfx_plugin
}  // namespace ov
