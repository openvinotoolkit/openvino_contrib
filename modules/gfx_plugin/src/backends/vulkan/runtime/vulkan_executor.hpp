// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <cstddef>
#include <string>
#include <vector>

#include "mlir/mlir_stage.hpp"
#include "runtime/gfx_stage_policy.hpp"

namespace ov {
class Node;
}  // namespace ov

namespace ov {
namespace gfx_plugin {

struct LaunchOperandABI {
    bool valid = false;
    std::vector<int32_t> kinds;
    std::vector<int32_t> arg_indices;
    std::vector<int32_t> scalar_values;
    std::vector<uint8_t> scalar_known;
};

class VulkanStage final : public MlirStage {
public:
    explicit VulkanStage(const std::shared_ptr<const ov::Node>& node);
    ~VulkanStage() override;

    void init(GpuBufferManager* buffer_manager) override;
    void compile(GpuBufferManager* buffer_manager) override;
    void execute(GpuCommandBufferHandle command_buffer) override;
    void enable_profiling(bool enable) override;
    void set_profiler(void* profiler,
                      uint32_t node_id,
                      const std::string& node_name,
                      const std::string& node_type) override;

    void set_inputs(const std::vector<GpuTensor*>& inputs) override;
    void set_output(GpuTensor* output) override;
    void set_outputs(const std::vector<std::unique_ptr<GpuTensor>>& outputs) override;

    std::unique_ptr<GpuStage> clone() const override;
    bool fuse_activation(ActivationKind kind, float alpha) override;
    bool fuse_batchnorm(const BatchNormParams& params) override;
    bool fuse_bias(const BiasParams& params) override;
    GpuStageSubmitPolicy submit_policy() const override;

private:
    std::shared_ptr<ICompiledKernel> compile_kernel(const KernelSource& source,
                                                    std::string* log) override;
    std::shared_ptr<ICompiledKernel> compile_specialized_kernel_from_mlir(mlir::ModuleOp module,
                                                                          const std::string& entry_name,
                                                                          uint32_t arg_count,
                                                                          const char* error_prefix);
    void prepare_specialized_kernels();
    void prepare_unary_kernel();
    void prepare_binary_kernel();
    void prepare_binary_same_shape_kernel();
    void prepare_binary_bias_add_kernel();
    void prepare_conv2d_1x1_kernel();
    void prepare_conv2d_3x3_direct_kernel();
    void prepare_conv2d_chunk_kernel();
    void prepare_group_conv2d_kernel();
    void prepare_softmax_kernel();
    void prepare_concat_kernel();
    void prepare_split_kernel();
    void prepare_interpolate_kernel();
    void prepare_transpose_kernel();
    void prepare_convert_kernel();
    bool is_vulkan_backend() const override { return true; }
    bool prefer_specialized_concat_execution() const override { return true; }
    KernelExecutionHooks* prepare_profiling(ProfileState& state,
                                            KernelExecutionHooks& hooks) override;
    void finalize_profiling(const ProfileState& state) override;

    // Chunked Split support to stay within mobile descriptor limits.
    std::shared_ptr<ICompiledKernel> m_split_single_kernel;
    std::shared_ptr<ICompiledKernel> m_concat_single_kernel;
    ov::element::Type m_split_elem_type{};
    ov::element::Type m_concat_elem_type{};
    std::shared_ptr<ICompiledKernel> m_softmax_row_kernel;
    ov::element::Type m_softmax_elem_type{};
    bool m_softmax_log_kernel = false;
    std::shared_ptr<ICompiledKernel> m_conv2d_1x1_kernel;
    ov::element::Type m_conv2d_1x1_elem_type{};
    bool m_conv2d_1x1_force_chunked_fallback = false;
    std::shared_ptr<ICompiledKernel> m_conv2d_3x3_direct_kernel;
    ov::element::Type m_conv2d_3x3_direct_elem_type{};
    uint32_t m_conv2d_3x3_direct_oc_block = 1;
    uint32_t m_conv2d_3x3_direct_threads_per_group = 64;
    std::string m_conv2d_3x3_direct_variant;
    bool m_conv2d_3x3_force_safe_variant = false;
    bool m_conv2d_3x3_force_chunked_fallback = false;
    std::shared_ptr<ICompiledKernel> m_conv2d_chunk_kernel;
    ov::element::Type m_conv2d_chunk_elem_type{};
    LaunchOperandABI m_conv2d_chunk_launch_abi;
    std::shared_ptr<ICompiledKernel> m_group_conv2d_kernel;
    ov::element::Type m_group_conv2d_elem_type{};
    std::shared_ptr<ICompiledKernel> m_interpolate_kernel;
    ov::element::Type m_interpolate_elem_type{};
    std::shared_ptr<ICompiledKernel> m_transpose_kernel;
    ov::element::Type m_transpose_elem_type{};
    std::shared_ptr<ICompiledKernel> m_convert_linear_kernel;
    ov::element::Type m_convert_src_elem_type{};
    ov::element::Type m_convert_dst_elem_type{};
    std::shared_ptr<ICompiledKernel> m_linear_unary_kernel;
    std::shared_ptr<ICompiledKernel> m_linear_binary_kernel;
    std::shared_ptr<ICompiledKernel> m_linear_binary_same_shape_kernel;
    std::shared_ptr<ICompiledKernel> m_binary_bias_add_kernel;
    ov::element::Type m_linear_unary_elem_type{};
    ov::element::Type m_linear_binary_elem_type{};
    ov::element::Type m_linear_binary_same_shape_elem_type{};
    ov::element::Type m_binary_bias_add_elem_type{};
    size_t m_linear_binary_rank = 0;
    LaunchOperandABI m_linear_unary_launch_abi;
    LaunchOperandABI m_linear_binary_launch_abi;
    std::vector<int32_t> m_linear_unary_scalar_args;
    std::string m_linear_unary_key;
    std::string m_linear_binary_key;
    std::string m_linear_binary_same_shape_key;

    void execute_split_chunked(GpuCommandBufferHandle command_buffer);
    void execute_concat_chunked(GpuCommandBufferHandle command_buffer);
    void execute_softmax_chunked(GpuCommandBufferHandle command_buffer);
    void execute_conv2d_1x1_chunked(GpuCommandBufferHandle command_buffer);
    void execute_conv2d_3x3_direct(GpuCommandBufferHandle command_buffer);
    void execute_conv2d_chunked(GpuCommandBufferHandle command_buffer);
    void execute_group_conv2d_chunked(GpuCommandBufferHandle command_buffer);
    void execute_interpolate_chunked(GpuCommandBufferHandle command_buffer);
    void execute_transpose_chunked(GpuCommandBufferHandle command_buffer);
    void execute_convert_chunked(GpuCommandBufferHandle command_buffer);
    void execute_unary_chunked(GpuCommandBufferHandle command_buffer);
    void execute_binary_chunked(GpuCommandBufferHandle command_buffer);
    void execute_binary_same_shape(GpuCommandBufferHandle command_buffer);
    void execute_binary_bias_add(GpuCommandBufferHandle command_buffer);
    mlir::ModuleOp build_split_single_module(mlir::MLIRContext& ctx, const ov::element::Type& et);
    mlir::ModuleOp build_concat_single_module(mlir::MLIRContext& ctx, const ov::element::Type& et);
    mlir::ModuleOp build_softmax_row_module(mlir::MLIRContext& ctx,
                                            const ov::element::Type& et,
                                            bool log_softmax);
    mlir::ModuleOp build_interpolate_module(mlir::MLIRContext& ctx,
                                            const ov::element::Type& et);
    mlir::ModuleOp build_conv2d_1x1_module(mlir::MLIRContext& ctx,
                                           const ov::element::Type& et);
    mlir::ModuleOp build_conv2d_3x3_direct_module(mlir::MLIRContext& ctx,
                                                  const ov::element::Type& et,
                                                  uint32_t output_channel_block,
                                                  uint32_t threads_per_group,
                                                  const std::string& variant);
    mlir::ModuleOp build_transpose_module(mlir::MLIRContext& ctx,
                                          const ov::element::Type& et);
    mlir::ModuleOp build_convert_linear_module(mlir::MLIRContext& ctx,
                                               const ov::element::Type& src_et,
                                               const ov::element::Type& dst_et);
    mlir::ModuleOp build_conv2d_chunk_module(mlir::MLIRContext& ctx,
                                             const ov::element::Type& et);
    mlir::ModuleOp build_group_conv2d_chunk_module(mlir::MLIRContext& ctx,
                                                   const ov::element::Type& et);
    mlir::ModuleOp build_linear_unary_module(mlir::MLIRContext& ctx,
                                             const ov::element::Type& et,
                                             const std::string& op_key);
    mlir::ModuleOp build_linear_binary_module(mlir::MLIRContext& ctx,
                                              const ov::element::Type& et,
                                              const std::string& op_key,
                                              size_t meta_rank);
    mlir::ModuleOp build_linear_binary_same_shape_module(mlir::MLIRContext& ctx,
                                                         const ov::element::Type& et,
                                                         const std::string& op_key);
    mlir::ModuleOp build_binary_bias_add_module(mlir::MLIRContext& ctx,
                                                const ov::element::Type& et);
    bool should_use_unary_chunked() const;
    bool should_use_concat_chunked() const;
    bool should_use_softmax_chunked() const;
    bool should_use_binary_same_shape() const;
    bool should_use_binary_chunked() const;
    bool should_use_binary_bias_add() const;
    bool should_use_conv2d_1x1_chunked() const;
    bool should_use_conv2d_3x3_direct() const;
    bool should_use_conv2d_chunked() const;
    bool should_use_group_conv2d_chunked() const;
    GfxStageOptimizationPlan optimization_plan() const;
    GfxConvRoutePlan conv_route_plan() const;
    bool should_use_interpolate_chunked() const;
    bool should_use_transpose_chunked() const;
    bool should_use_convert_chunked() const;
    bool should_skip_generic_kernel_compile(const GfxStageOptimizationPlan& plan) const override;
};

}  // namespace gfx_plugin
}  // namespace ov
