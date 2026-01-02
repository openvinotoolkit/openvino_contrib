// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/op.hpp"
#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "backends/metal/runtime/profiling/profiler.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "kernel_ir/gfx_kernel_cache.hpp"
#include "kernel_ir/gfx_kernel_inputs.hpp"
#include "kernel_ir/gfx_kernel_plan.hpp"
#include "openvino/op/constant.hpp"

#include <string>

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
    validate_kernel_args(kernel, args, m_name.c_str());
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

std::shared_ptr<ICompiledKernel> MetalOp::compile_msl_kernel(
    MetalCodegenBackend& backend,
    mlir::ModuleOp module,
    const std::string& entry_point,
    std::function<std::string(mlir::ModuleOp)> msl_generator,
    std::string* log,
    uint32_t arg_count) {
    KernelPlan plan(module, entry_point, arg_count);
    return backend.compile(plan.to_source_with_msl_generator(std::move(msl_generator)), log);
}

std::shared_ptr<ICompiledKernel> MetalOp::compile_msl_kernel(MetalCodegenBackend& backend,
                                                             const KernelSpec& spec,
                                                             mlir::ModuleOp module,
                                                             const std::string& entry_point,
                                                             std::string msl_source,
                                                             std::string* log) {
    KernelFunctionSignature signature{};
    if (spec.node() && module) {
        signature = infer_kernel_signature(module, entry_point);
        const size_t func_inputs = signature.inputs;
        const size_t node_inputs = spec.node()->get_input_size();
        if (func_inputs > 0 && func_inputs <= node_inputs) {
            size_t nonconst_count = 0;
            for (size_t i = 0; i < node_inputs; ++i) {
                auto src = spec.node()->get_input_node_shared_ptr(i);
                if (!ov::as_type_ptr<const ov::op::v0::Constant>(src)) {
                    ++nonconst_count;
                }
            }
            if (func_inputs >= nonconst_count) {
                auto mapping = build_kernel_inputs(spec.node(), func_inputs, spec.name().c_str());
                if (!mapping.kernel_inputs.empty() &&
                    (m_kernel_inputs.empty() || mapping.kernel_inputs.size() > m_kernel_inputs.size())) {
                    m_kernel_inputs = std::move(mapping.kernel_inputs);
                }
            }
        }
    } else if (module) {
        signature = infer_kernel_signature(module, entry_point);
    }
    const uint32_t inferred_total = signature.total();
    uint32_t arg_count = spec.arg_count();
    if (arg_count == 0 && inferred_total) {
        arg_count = inferred_total;
    }
    KernelPlan plan(module, entry_point, arg_count);
    return backend.compile(plan.to_source_with_msl(std::move(msl_source)), log);
}

std::shared_ptr<ICompiledKernel> MetalOp::compile_msl_kernel(
    MetalCodegenBackend& backend,
    const KernelSpec& spec,
    mlir::ModuleOp module,
    const std::string& entry_point,
    std::function<std::string(mlir::ModuleOp)> msl_generator,
    std::string* log) {
    KernelFunctionSignature signature{};
    if (spec.node() && module) {
        signature = infer_kernel_signature(module, entry_point);
        const size_t func_inputs = signature.inputs;
        const size_t node_inputs = spec.node()->get_input_size();
        if (func_inputs > 0 && func_inputs <= node_inputs) {
            size_t nonconst_count = 0;
            for (size_t i = 0; i < node_inputs; ++i) {
                auto src = spec.node()->get_input_node_shared_ptr(i);
                if (!ov::as_type_ptr<const ov::op::v0::Constant>(src)) {
                    ++nonconst_count;
                }
            }
            if (func_inputs >= nonconst_count) {
                auto mapping = build_kernel_inputs(spec.node(), func_inputs, spec.name().c_str());
                if (!mapping.kernel_inputs.empty() &&
                    (m_kernel_inputs.empty() || mapping.kernel_inputs.size() > m_kernel_inputs.size())) {
                    m_kernel_inputs = std::move(mapping.kernel_inputs);
                }
            }
        }
    } else if (module) {
        signature = infer_kernel_signature(module, entry_point);
    }
    const uint32_t inferred_total = signature.total();
    uint32_t arg_count = spec.arg_count();
    if (arg_count == 0 && inferred_total) {
        arg_count = inferred_total;
    }
    KernelPlan plan(module, entry_point, arg_count);
    return backend.compile(plan.to_source_with_msl_generator(std::move(msl_generator)), log);
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
    GpuBufferDesc desc{};
    desc.bytes = bytes;
    desc.type = type;
    desc.usage = storageModePrivate ? BufferUsage::Intermediate : BufferUsage::IO;
    desc.prefer_device_local = storageModePrivate;
    desc.label = m_name.c_str();
    return m_buffer_manager->allocate(desc, persistent);
}

std::vector<KernelArg> MetalOp::materialize_kernel_args(const std::vector<KernelArg>& args) {
    if (args.empty()) {
        return args;
    }
    OPENVINO_ASSERT(m_buffer_manager, "Buffer manager is not set for op ", m_name);
    return materialize_kernel_bytes_args(args, *m_buffer_manager, m_name.c_str());
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
