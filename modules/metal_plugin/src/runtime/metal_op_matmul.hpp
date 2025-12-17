// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "kernel_ir/kernel_ir_common.hpp"
#include "openvino/op/matmul.hpp"
#include "runtime/metal_op.hpp"

namespace ov {
namespace metal_plugin {

class METAL_OP_API MetalMatMulOp : public MetalOp {
public:
    MetalMatMulOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);
    ~MetalMatMulOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void execute() override;

private:
    void fill_desc_from_node(const std::shared_ptr<const ov::Node>& node);

    std::shared_ptr<const ov::Node> m_node;
    KernelOp m_desc{};
    ov::Shape m_shape_a;
    ov::Shape m_shape_b;
    ov::element::Type m_element_type{ov::element::f32};
    MetalTensor m_constA;
    MetalTensor m_constB;

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    id<MTLComputePipelineState> m_pipeline = nil;
};

}  // namespace metal_plugin
}  // namespace ov
