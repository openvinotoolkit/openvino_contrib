// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "runtime/gfx_activation.hpp"
#include "runtime/gfx_batchnorm.hpp"
#include "runtime/gfx_bias.hpp"
#include "runtime/gpu_backend_base.hpp"
#include "runtime/gpu_stage.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov {
class Node;
}  // namespace ov

namespace ov {
namespace gfx_plugin {

class VulkanStage final : public GpuStage {
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

    const std::string& name() const override { return m_name; }
    const std::string& type() const override { return m_type; }

    std::unique_ptr<GpuStage> clone() const override;
    bool fuse_activation(ActivationKind kind, float alpha) override;
    bool fuse_batchnorm(const BatchNormParams& params) override;
    bool fuse_bias(const BiasParams& params) override;

private:
    struct ConstBufferSet {
        std::vector<GpuTensor> buffers;
        std::vector<bool> present;
        ~ConstBufferSet();
    };

    bool m_is_view_op = false;
    std::shared_ptr<const ov::Node> m_node;
    std::shared_ptr<ICompiledKernel> m_kernel;
    std::string m_name;
    std::string m_type;
    std::vector<GpuTensor*> m_inputs;
    std::vector<GpuTensor*> m_outputs;
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
    bool m_parallel_dispatch = false;
    size_t m_parallel_loop_dims = 0;
    uint32_t m_dispatch_tile_h = 1;
    uint32_t m_dispatch_tile_w = 1;
    uint32_t m_dispatch_threads_h = 1;
    uint32_t m_dispatch_threads_w = 1;
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
};

}  // namespace gfx_plugin
}  // namespace ov
