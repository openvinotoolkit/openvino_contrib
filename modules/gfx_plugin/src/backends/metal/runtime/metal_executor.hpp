// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mlir/mlir_stage.hpp"
#include "backends/metal/runtime/memory/buffer.hpp"

namespace ov {
namespace gfx_plugin {

class MetalProfiler;

// Metal-backed MlirStage implementation (pure MLIR→MSL path, no legacy ops).
class MetalStage final : public MlirStage {
public:
    MetalStage(const std::shared_ptr<const ov::Node>& node,
               MetalDeviceHandle device,
               MetalCommandQueueHandle queue);

    void init(GpuBufferManager* buffer_manager) override;
    void compile(GpuBufferManager* buffer_manager) override;
    void execute(GpuCommandBufferHandle command_buffer) override;

    void set_inputs(const std::vector<GpuTensor*>& inputs) override;
    void set_output(GpuTensor* output) override;
    void set_outputs(const std::vector<std::unique_ptr<GpuTensor>>& outputs) override;

    bool fuse_activation(ActivationKind kind, float alpha) override;
    bool fuse_batchnorm(const BatchNormParams& params) override;
    bool fuse_bias(const BiasParams& params) override;

    void enable_profiling(bool enable) override;
    void set_profiler(void* profiler,
                      uint32_t node_id,
                      const std::string& node_name,
                      const std::string& node_type) override;

    std::unique_ptr<GpuStage> clone() const override;

private:
    std::shared_ptr<ICompiledKernel> compile_kernel(const KernelSource& source,
                                                    std::string* log) override;
    void configure_runtime_matmul_kernel_source(KernelSource& source,
                                                const MatMulCodegenDesc& desc) const override;
    KernelExecutionHooks* prepare_profiling(ProfileState& state,
                                            KernelExecutionHooks& hooks) override;
    void finalize_profiling(const ProfileState& state) override;

    MetalDeviceHandle m_device = nullptr;
    [[maybe_unused]] MetalCommandQueueHandle m_queue = nullptr;
};

}  // namespace gfx_plugin
}  // namespace ov
