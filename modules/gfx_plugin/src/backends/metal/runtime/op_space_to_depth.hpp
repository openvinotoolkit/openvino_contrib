// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "mlir_codegen/codegen_desc.hpp"
#include "openvino/op/space_to_depth.hpp"
#include "backends/metal/runtime/op.hpp"

namespace ov {
namespace gfx_plugin {

class GFX_OP_API MetalSpaceToDepthOp : public MetalOp {
public:
    MetalSpaceToDepthOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);
    ~MetalSpaceToDepthOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;

private:
    void parse_space_to_depth(const std::shared_ptr<const ov::Node>& node);

    std::shared_ptr<const ov::Node> m_node;
    ov::element::Type m_element_type{ov::element::f32};
    uint32_t m_block = 1;
    uint32_t m_mode = 0; // 0 blocks_first, 1 depth_first
    SpaceToDepthCodegenDesc m_desc{};

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    std::shared_ptr<ICompiledKernel> m_kernel;
};

}  // namespace gfx_plugin
}  // namespace ov
