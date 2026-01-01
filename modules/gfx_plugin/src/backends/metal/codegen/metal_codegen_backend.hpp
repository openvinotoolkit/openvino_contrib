// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>

#include "mlir_codegen/gfx_codegen_backend.hpp"
#include "backends/metal/runtime/memory/buffer.hpp"

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

namespace ov {
namespace gfx_plugin {

class MetalCodegenBackend final : public ICodegenBackend {
public:
    explicit MetalCodegenBackend(MetalDeviceHandle device);

    std::shared_ptr<ICompiledKernel> compile(const KernelSource& source,
                                             std::string* log = nullptr) override;

private:
    MetalDeviceHandle m_device = nullptr;
};

class MetalCompiledKernel final : public ICompiledKernel {
public:
    explicit MetalCompiledKernel(MetalDeviceHandle device, void* pipeline, uint32_t arg_count = 0);

    uint32_t args_count() const override { return m_args_count; }
    void set_args_count(uint32_t count) override;
    size_t clamp_threadgroup_size(size_t desired) const override;
    void execute(GpuCommandBufferHandle command_buffer,
                 const KernelDispatch& dispatch,
                 const std::vector<KernelArg>& args,
                 const KernelExecutionHooks* hooks = nullptr) override;

private:
    MetalDeviceHandle m_device = nullptr;
    void* m_pipeline = nullptr;
    uint32_t m_args_count = 0;
};

}  // namespace gfx_plugin
}  // namespace ov
