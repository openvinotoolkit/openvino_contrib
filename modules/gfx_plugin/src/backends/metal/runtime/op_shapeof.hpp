// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "openvino/op/shape_of.hpp"
#include "kernel_ir/gfx_codegen_desc.hpp"
#include "backends/metal/runtime/op.hpp"

namespace ov {
namespace gfx_plugin {

class GFX_OP_API MetalShapeOfOp : public MetalOp {
public:
    MetalShapeOfOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);
    ~MetalShapeOfOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;

private:
    std::shared_ptr<const ov::Node> m_node;
    ov::element::Type m_element_type{ov::element::i64};
    std::vector<int64_t> m_shape_vals;

    ShapeOfCodegenDesc m_desc{};

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    std::shared_ptr<ICompiledKernel> m_kernel;
};

}  // namespace gfx_plugin
}  // namespace ov
