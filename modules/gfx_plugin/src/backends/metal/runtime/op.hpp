// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <cstdint>

#include "openvino/core/shape.hpp"
#include "openvino/core/except.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "runtime/gpu_backend_base.hpp"
#include "kernel_ir/gfx_kernel_plan.hpp"
#include "kernel_ir/gfx_kernel_spec.hpp"
#include "backends/metal/runtime/metal_memory.hpp"
#include "runtime/gfx_activation.hpp"
#include "runtime/gfx_batchnorm.hpp"

namespace ov {
namespace gfx_plugin {

class MetalProfiler;
class MetalCodegenBackend;

// Visibility helper: keep MetalOp symbols exported for tests and downstream components.
#if defined(__clang__) || defined(__GNUC__)
#    define GFX_OP_API __attribute__((visibility("default")))
#else
#    define GFX_OP_API
#endif

// Abstract base for all Metal GPU operations.
// Stores shared device context and lightweight metadata that concrete ops reuse.
class GFX_OP_API MetalOp {
public:
    MetalOp(std::string name,
            std::string type,
            const ov::Shape& output_shape,
            void* device = nullptr,
            void* command_queue = nullptr);

    virtual ~MetalOp();

    // Optional initialization hook (e.g., compile kernels, cache layouts).
    virtual void init(MetalBufferManager* buffer_manager);
    // Explicit compilation hook (generates kernels/pipelines). Called once per compiled model.
    virtual void compile(MetalBufferManager* buffer_manager);

    // Execute op on device using the provided command buffer (may be null → op may create its own, but
    // pipeline should pass a shared buffer to avoid extra commits).
    virtual void execute(MetalCommandBufferHandle command_buffer) = 0;
    // Optional fusion hook (e.g., Conv + Relu). Default: not supported.
    virtual bool fuse_activation(ActivationKind /*kind*/, float /*alpha*/) { return false; }
    virtual bool fuse_batchnorm(const BatchNormParams& /*params*/) { return false; }

    void set_inputs(const std::vector<MetalTensor*>& inputs);
    void set_output(MetalTensor* output);
    // Optional multi-output binding (used by ops like Split). Default binds first output.
    virtual void set_outputs(const std::vector<std::unique_ptr<MetalTensor>>& outputs);

    const std::vector<MetalTensor*>& inputs() const { return m_inputs; }
    MetalTensor* output() const { return m_output; }

    MetalDeviceHandle device() const { return m_device; }
    MetalCommandQueueHandle command_queue() const { return m_command_queue; }

    const std::string& name() const { return m_name; }
    const std::string& type() const { return m_type; }
    const ov::Shape& output_shape() const { return m_output_shape; }

    void enable_profiling(bool enable) { m_profiling_enabled = enable; }
    void set_profiler(MetalProfiler* profiler,
                      uint32_t node_id,
                      const std::string& node_name,
                      const std::string& node_type);
    double last_exec_duration_ms() const { return m_last_duration_ms; }

protected:
    std::shared_ptr<ICompiledKernel> compile_msl_kernel(MetalCodegenBackend& backend,
                                                        mlir::ModuleOp module,
                                                        const std::string& entry_point,
                                                        std::string msl_source,
                                                        std::string* log = nullptr,
                                                        uint32_t arg_count = 0);
    std::shared_ptr<ICompiledKernel> compile_msl_kernel(
        MetalCodegenBackend& backend,
        mlir::ModuleOp module,
        const std::string& entry_point,
        std::function<std::string(mlir::ModuleOp)> msl_generator,
        std::string* log = nullptr,
        uint32_t arg_count = 0);
    std::shared_ptr<ICompiledKernel> compile_msl_kernel(MetalCodegenBackend& backend,
                                                        const KernelSpec& spec,
                                                        mlir::ModuleOp module,
                                                        const std::string& entry_point,
                                                        std::string msl_source,
                                                        std::string* log = nullptr);
    std::shared_ptr<ICompiledKernel> compile_msl_kernel(
        MetalCodegenBackend& backend,
        const KernelSpec& spec,
        mlir::ModuleOp module,
        const std::string& entry_point,
        std::function<std::string(mlir::ModuleOp)> msl_generator,
        std::string* log = nullptr);

    void execute_kernel(ICompiledKernel& kernel,
                        MetalCommandBufferHandle command_buffer,
                        const KernelDispatch& dispatch,
                        const std::vector<KernelArg>& args);

    template <typename ResolveInputFn>
    void append_kernel_input_args(std::vector<KernelArg>& args,
                                  const std::vector<size_t>& kernel_inputs,
                                  ResolveInputFn&& resolve_input,
                                  const char* stage_name) const {
        ::ov::gfx_plugin::append_kernel_input_args(args,
                                                   kernel_inputs,
                                                   std::forward<ResolveInputFn>(resolve_input),
                                                   stage_name);
    }

    template <typename ResolveInputFn>
    void append_kernel_input_args(std::vector<KernelArg>& args,
                                  size_t input_count,
                                  ResolveInputFn&& resolve_input,
                                  const char* stage_name) const {
        if (!m_kernel_inputs.empty()) {
            if (m_kernel_inputs.size() <= input_count) {
                ::ov::gfx_plugin::append_kernel_input_args(args,
                                                           m_kernel_inputs,
                                                           std::forward<ResolveInputFn>(resolve_input),
                                                           stage_name);
                return;
            }
        }
        ::ov::gfx_plugin::append_kernel_input_args(args,
                                                   input_count,
                                                   std::forward<ResolveInputFn>(resolve_input),
                                                   stage_name);
    }

    template <typename ResolveInputFn>
    void append_kernel_input_args(KernelArgsBuilder& builder,
                                  const std::vector<size_t>& kernel_inputs,
                                  ResolveInputFn&& resolve_input,
                                  const char* /*stage_name*/) const {
        builder.add_inputs(kernel_inputs, std::forward<ResolveInputFn>(resolve_input));
    }

    template <typename ResolveInputFn>
    void append_kernel_input_args(KernelArgsBuilder& builder,
                                  size_t input_count,
                                  ResolveInputFn&& resolve_input,
                                  const char* /*stage_name*/) const {
        if (!m_kernel_inputs.empty()) {
            if (m_kernel_inputs.size() <= input_count) {
                builder.add_inputs(m_kernel_inputs, std::forward<ResolveInputFn>(resolve_input));
                return;
            }
        }
        builder.add_inputs(input_count, std::forward<ResolveInputFn>(resolve_input));
    }

    // Allocate temporary device memory via the shared buffer manager.
    MetalBuffer allocate_temp_buffer(size_t bytes,
                                     ov::element::Type type,
                                     bool persistent = false,
                                     bool storageModePrivate = true);

    MetalBufferManager* buffer_manager() const { return m_buffer_manager; }
    bool is_compiled() const { return m_compiled; }
    void mark_compiled() { m_compiled = true; }

    // Profiling helpers used by derived ops.
    void start_profiling(MetalCommandEncoderHandle encoder = nullptr);
    double stop_profiling_ms(MetalCommandEncoderHandle encoder = nullptr);

    MetalTensor& require_output() const;

private:
    std::vector<KernelArg> materialize_kernel_args(const std::vector<KernelArg>& args);
    void flush_inflight_const_buffers(MetalCommandBufferHandle command_buffer);
    void release_inflight_const_buffers();

    std::string m_name;
    std::string m_type;
    ov::Shape m_output_shape;

    MetalDeviceHandle m_device;
    MetalCommandQueueHandle m_command_queue;
    MetalBufferManager* m_buffer_manager = nullptr;  // non-owning
    bool m_compiled = false;

    std::vector<MetalTensor*> m_inputs;  // non-owning
    MetalTensor* m_output = nullptr;     // non-owning

    bool m_profiling_enabled = false;
    double m_last_duration_ms = 0.0;
    MetalProfiler* m_profiler = nullptr;  // non-owning
    uint32_t m_profile_node_id = 0;
    int32_t m_gpu_sample_begin = -1;
    std::string m_profile_node_name;
    std::string m_profile_node_type;

    std::vector<size_t> m_kernel_inputs;
    std::vector<MetalBuffer> m_inflight_const_buffers;
    std::vector<std::shared_ptr<std::vector<uint8_t>>> m_inflight_const_payloads;
};

}  // namespace gfx_plugin
}  // namespace ov
