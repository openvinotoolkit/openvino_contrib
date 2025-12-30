// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "runtime/gpu_backend_base.hpp"
#include "runtime/gpu_stage.hpp"

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
    GpuTensor* m_output = nullptr;
    std::shared_ptr<ConstBufferSet> m_const_buffers;
    ov::Shape m_output_shape;
    ov::Shape m_last_input_shape;
    bool m_softmax_tiled = false;
    GpuBuffer m_softmax_params;
    bool m_profiling_enabled = false;
    void* m_profiler = nullptr;
    uint32_t m_profile_node_id = 0;
    std::string m_profile_node_name;
    std::string m_profile_node_type;
};

}  // namespace gfx_plugin
}  // namespace ov
