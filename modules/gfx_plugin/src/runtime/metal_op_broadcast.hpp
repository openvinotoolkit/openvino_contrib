// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "runtime/metal_op.hpp"
#include "kernel_codegen/metal_kernel_compiler.hpp"

namespace ov {
namespace gfx_plugin {

class GFX_OP_API MetalBroadcastOp : public MetalOp {
public:
    MetalBroadcastOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);
    ~MetalBroadcastOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;

private:
    std::shared_ptr<const ov::Node> m_node;
    ov::element::Type m_element_type{ov::element::f32};
    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    id<MTLComputePipelineState> m_pipeline = nil;

    std::vector<int> m_in_dims;
    std::vector<int> m_out_dims;
    std::vector<int> m_in_strides;
    std::vector<int> m_axes;
    uint32_t m_num_elems = 0;
};

}  // namespace gfx_plugin
}  // namespace ov

