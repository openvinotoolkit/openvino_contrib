// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "mlir/codegen/codegen_desc.hpp"
#include "backends/metal/runtime/op.hpp"

namespace ov {
namespace gfx_plugin {

class GFX_OP_API MetalSelectOp : public MetalOp {
public:
    MetalSelectOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);
    ~MetalSelectOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;

private:
    std::shared_ptr<const ov::Node> m_node;
    ov::element::Type m_element_type{ov::element::f32};
    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    std::shared_ptr<ICompiledKernel> m_kernel;
    uint32_t m_num_elems = 0;
};

}  // namespace gfx_plugin
}  // namespace ov
