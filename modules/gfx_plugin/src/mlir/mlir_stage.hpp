// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "kernel_ir/gfx_codegen_backend.hpp"
#include "kernel_ir/gfx_kernel_dispatch.hpp"
#include "mlir/gfx_mlir_kernel_metadata.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/float16.hpp"
#include "runtime/gfx_activation.hpp"
#include "runtime/gfx_batchnorm.hpp"
#include "runtime/gfx_backend_utils.hpp"
#include "runtime/gfx_bias.hpp"
#include "runtime/gfx_stage_policy.hpp"
#include "runtime/gpu_stage.hpp"
#include "runtime/gpu_tensor.hpp"

namespace ov {
namespace gfx_plugin {

struct MlirKernelPlanContext;

class MlirStage : public GpuStage {
public:
    explicit MlirStage(const std::shared_ptr<const ov::Node>& node);
    ~MlirStage() override = default;

    void init(GpuBufferManager* buffer_manager) override;
    void compile(GpuBufferManager* buffer_manager) override;
    void execute(GpuCommandBufferHandle command_buffer) override;
    void enable_profiling(bool enable) override;
    void set_profiler(void* profiler,
                      uint32_t node_id,
                      const std::string& node_name,
                      const std::string& node_type) override;
    void on_command_buffer_complete() override;

    void set_inputs(const std::vector<GpuTensor*>& inputs) override;
    void set_output(GpuTensor* output) override;
    void set_outputs(const std::vector<std::unique_ptr<GpuTensor>>& outputs) override;
    void set_input_transform(size_t input_idx, const GfxInputTransform& transform) override;

    const std::string& name() const override { return m_name; }
    const std::string& type() const override { return m_type; }

    bool fuse_activation(ActivationKind kind, float alpha) override;
    bool fuse_batchnorm(const BatchNormParams& params) override;
    bool fuse_bias(const BiasParams& params) override;

protected:
    struct ConstBufferSet {
        std::vector<GpuTensor> buffers;
        std::vector<bool> present;
    };

    struct ProfileState {
        bool enabled = false;
        std::chrono::steady_clock::time_point cpu_start{};
        int32_t sample_begin = -1;
        int32_t sample_end = -1;
    };

    virtual std::shared_ptr<ICompiledKernel> compile_kernel(const KernelSource& source,
                                                            std::string* log) = 0;
    virtual bool is_vulkan_backend() const { return false; }
    GpuBackend backend_kind() const { return is_vulkan_backend() ? GpuBackend::Vulkan : GpuBackend::Metal; }
    virtual KernelExecutionHooks* prepare_profiling(ProfileState& state,
                                                    KernelExecutionHooks& hooks);
    virtual void finalize_profiling(const ProfileState& state);

    void clone_into(MlirStage& dst) const;
    bool has_absorbed_input_transpose() const;

    void* profiler_handle() const { return m_profiler; }
    uint32_t profile_node_id() const { return m_profile_node_id; }
    const std::string& profile_node_name() const { return m_profile_node_name; }
    const std::string& profile_node_type() const { return m_profile_node_type; }

protected:
    bool m_is_view_op = false;
    std::shared_ptr<const ov::Node> m_node;
    std::shared_ptr<ICompiledKernel> m_kernel;
    std::string m_name;
    std::string m_type;
    std::vector<GpuTensor*> m_inputs;
    std::vector<GpuTensor*> m_outputs;
    std::vector<GfxInputTransform> m_input_transforms;
    std::vector<size_t> m_kernel_inputs;
    size_t m_kernel_input_arg_count = 0;
    std::vector<int32_t> m_kernel_operand_kinds;
    std::vector<int32_t> m_kernel_operand_arg_indices;
    std::vector<int32_t> m_kernel_scalar_args;
    std::vector<GpuTensor> m_kernel_extra_inputs;
    GpuTensor* m_output = nullptr;
    std::shared_ptr<ConstBufferSet> m_const_buffers;
    ov::Shape m_output_shape;
    ov::Shape m_last_input_shape;
    GpuBufferManager* m_buffer_manager = nullptr;
    bool m_profiling_enabled = false;
    ParallelDispatchConfig m_parallel_cfg{};
    bool m_force_single_dispatch = false;
    bool m_has_activation = false;
    ActivationKind m_activation = ActivationKind::Relu;
    float m_activation_alpha = 0.0f;
    bool m_has_bn = false;
    BatchNormParams m_bn_params{};
    bool m_has_bias = false;
    BiasParams m_bias_params{};
    std::vector<ov::float16> m_bias_f16;
    void* m_profiler = nullptr;
    uint32_t m_profile_node_id = 0;
    std::string m_profile_node_name;
    std::string m_profile_node_type;

private:
    void apply_kernel_metadata(const KernelRuntimeMetadata& meta,
                               size_t scalar_inputs);
    void compile_from_plan(MlirKernelPlanContext& plan_ctx,
                           mlir::ModuleOp module,
                           const char* stage_kind);
    GfxStageOptimizationPlan stage_optimization_plan() const;
    bool is_conv_like() const;
    bool is_matmul_like() const;
    const GfxInputTransform* input_transform(size_t input_idx) const;
    ov::Shape compile_time_input_shape(size_t input_idx) const;
    std::vector<int32_t> compile_time_broadcast_strides(size_t input_idx, const ov::Shape& out_shape) const;
    void apply_stage_optimization_attrs(mlir::ModuleOp module,
                                        const GfxStageOptimizationPlan& plan);
    void apply_input_transform_attrs(mlir::ModuleOp module) const;
    void set_parallel_preference(mlir::ModuleOp module);
    void apply_fused_operations(mlir::ModuleOp module);
};

}  // namespace gfx_plugin
}  // namespace ov
