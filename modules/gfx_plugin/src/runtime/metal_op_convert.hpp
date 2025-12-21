// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/op/convert.hpp"
#include "runtime/metal_op.hpp"

namespace ov {
namespace gfx_plugin {

class GFX_OP_API MetalConvertOp : public MetalOp {
public:
    MetalConvertOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);
    ~MetalConvertOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;

private:
    std::shared_ptr<const ov::Node> m_node;
    ov::element::Type m_src_type{ov::element::dynamic};
    ov::element::Type m_dst_type{ov::element::dynamic};
    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    id<MTLComputePipelineState> m_pipeline = nil;
};

}  // namespace gfx_plugin
}  // namespace ov
