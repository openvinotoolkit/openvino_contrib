// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "backends/metal/runtime/op.hpp"

namespace ov {
namespace gfx_plugin {

class GFX_OP_API MetalRangeOp : public MetalOp {
public:
    MetalRangeOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);
    ~MetalRangeOp() override = default;

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
    double m_start = 0.0;
    double m_step = 1.0;
};

}  // namespace gfx_plugin
}  // namespace ov

