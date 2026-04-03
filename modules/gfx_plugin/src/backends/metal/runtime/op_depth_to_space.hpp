// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "kernel_ir/gfx_codegen_desc.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "backends/metal/runtime/op.hpp"

namespace ov {
namespace gfx_plugin {

class GFX_OP_API MetalDepthToSpaceOp : public MetalOp {
public:
    MetalDepthToSpaceOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);
    ~MetalDepthToSpaceOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;

private:
    void parse_depth_to_space(const std::shared_ptr<const ov::Node>& node);

    std::shared_ptr<const ov::Node> m_node;
    ov::element::Type m_element_type{ov::element::f32};
    uint32_t m_block = 1;
    uint32_t m_mode = 0; // 0 blocks_first, 1 depth_first
    DepthToSpaceCodegenDesc m_desc{};

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    std::shared_ptr<ICompiledKernel> m_kernel;
};

}  // namespace gfx_plugin
}  // namespace ov
