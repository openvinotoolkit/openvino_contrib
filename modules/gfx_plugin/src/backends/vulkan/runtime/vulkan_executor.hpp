// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <cstddef>
#include <string>
#include <vector>

#include "mlir/mlir_stage.hpp"

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

private:
    std::shared_ptr<ICompiledKernel> compile_kernel(const KernelSource& source,
                                                    std::string* log) override;
    bool is_vulkan_backend() const override { return true; }
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
    std::shared_ptr<ICompiledKernel> m_conv2d_chunk_kernel;
    ov::element::Type m_conv2d_chunk_elem_type{};
    LaunchOperandABI m_conv2d_chunk_launch_abi;
    std::shared_ptr<ICompiledKernel> m_linear_unary_kernel;
    std::shared_ptr<ICompiledKernel> m_linear_binary_kernel;
    ov::element::Type m_linear_unary_elem_type{};
    ov::element::Type m_linear_binary_elem_type{};
    size_t m_linear_binary_rank = 0;
    LaunchOperandABI m_linear_unary_launch_abi;
    std::vector<int32_t> m_linear_unary_scalar_args;
    std::string m_linear_unary_key;
    std::string m_linear_binary_key;

    void execute_split_chunked(GpuCommandBufferHandle command_buffer);
    void execute_concat_chunked(GpuCommandBufferHandle command_buffer);
    void execute_softmax_chunked(GpuCommandBufferHandle command_buffer);
    void execute_conv2d_chunked(GpuCommandBufferHandle command_buffer);
    void execute_unary_chunked(GpuCommandBufferHandle command_buffer);
    void execute_binary_chunked(GpuCommandBufferHandle command_buffer);
    mlir::ModuleOp build_split_single_module(mlir::MLIRContext& ctx, const ov::element::Type& et);
    mlir::ModuleOp build_concat_single_module(mlir::MLIRContext& ctx, const ov::element::Type& et);
    mlir::ModuleOp build_softmax_row_module(mlir::MLIRContext& ctx,
                                            const ov::element::Type& et,
                                            bool log_softmax);
    mlir::ModuleOp build_conv2d_chunk_module(mlir::MLIRContext& ctx,
                                             const ov::element::Type& et);
    mlir::ModuleOp build_linear_unary_module(mlir::MLIRContext& ctx,
                                             const ov::element::Type& et,
                                             const std::string& op_key);
    mlir::ModuleOp build_linear_binary_module(mlir::MLIRContext& ctx,
                                              const ov::element::Type& et,
                                              const std::string& op_key,
                                              size_t meta_rank);
    bool should_use_unary_chunked() const;
    bool should_use_softmax_chunked() const;
    bool should_use_binary_chunked() const;
    bool should_use_conv2d_chunked() const;
};

}  // namespace gfx_plugin
}  // namespace ov
