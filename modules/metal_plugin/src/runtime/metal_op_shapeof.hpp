// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "openvino/op/shape_of.hpp"
#include "mlir_codegen/codegen_common.hpp"
#include "runtime/metal_op.hpp"

namespace ov {
namespace metal_plugin {

class METAL_OP_API MetalShapeOfOp : public MetalOp {
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
    id<MTLComputePipelineState> m_pipeline = nil;
};

}  // namespace metal_plugin
}  // namespace ov
