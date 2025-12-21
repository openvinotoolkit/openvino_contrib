// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "openvino/op/slice.hpp"
#include "runtime/metal_op.hpp"

namespace ov {
namespace gfx_plugin {

class GFX_OP_API MetalSliceOp : public MetalOp {
public:
    MetalSliceOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);
    ~MetalSliceOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;

private:
    void parse_slice(const std::shared_ptr<const ov::Node>& node);

    std::shared_ptr<const ov::Node> m_node;
    ov::element::Type m_element_type{ov::element::f32};
    std::vector<uint32_t> m_in_stride;
    std::vector<uint32_t> m_out_shape;
    std::vector<int32_t> m_starts;
    std::vector<uint32_t> m_steps;

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    id<MTLComputePipelineState> m_pipeline = nil;
};

}  // namespace gfx_plugin
}  // namespace ov
