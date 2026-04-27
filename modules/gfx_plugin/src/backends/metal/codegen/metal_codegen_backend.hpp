// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>

#include "kernel_ir/gfx_codegen_backend.hpp"
#include "backends/metal/runtime/memory/buffer.hpp"

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

namespace ov {
namespace gfx_plugin {

class MetalBindingSchema;
class MetalDeviceReuseContext;

class MetalCodegenBackend final : public ICodegenBackend {
public:
    explicit MetalCodegenBackend(MetalDeviceHandle device);

    std::shared_ptr<ICompiledKernel> compile(const KernelSource& source,
                                             std::string* log = nullptr) override;

private:
    MetalDeviceHandle m_device = nullptr;
    std::shared_ptr<MetalDeviceReuseContext> m_reuse_context;
};

class MetalCompiledKernel final : public CompiledKernelBase,
                                  public std::enable_shared_from_this<MetalCompiledKernel> {
public:
    explicit MetalCompiledKernel(MetalDeviceHandle device, void* pipeline, uint32_t arg_count = 0);
    MetalCompiledKernel(MetalDeviceHandle device,
                        void* pipeline,
                        std::shared_ptr<const KernelBindingPlan> binding_plan);
    MetalCompiledKernel(MetalDeviceHandle device,
                        void* pipeline,
                        std::shared_ptr<const KernelBindingPlan> binding_plan,
                        std::shared_ptr<void> prepared_binding_cache,
                        std::shared_ptr<MetalBindingSchema> binding_schema);

    size_t clamp_threadgroup_size(size_t desired) const override;
    std::shared_ptr<ICompiledKernel> fork() const override;
    void prewarm_bindings(const std::vector<KernelArg>& args) override;
    void execute(GpuCommandBufferHandle command_buffer,
                 const KernelDispatch& dispatch,
                 const std::vector<KernelArg>& args,
                 const KernelExecutionHooks* hooks = nullptr) override;
    const void* shared_binding_schema_identity() const;

private:
    MetalDeviceHandle m_device = nullptr;
    void* m_pipeline = nullptr;
    std::shared_ptr<MetalBindingSchema> m_binding_schema;
};

}  // namespace gfx_plugin
}  // namespace ov
