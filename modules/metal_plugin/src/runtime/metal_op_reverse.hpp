// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "openvino/op/reverse.hpp"
#include "runtime/metal_op.hpp"
#include "mlir_codegen/codegen_common.hpp"

namespace ov {
namespace metal_plugin {

class METAL_OP_API MetalReverseOp final : public MetalOp {
public:
    MetalReverseOp(const std::shared_ptr<const ov::op::v1::Reverse>& node,
                   MetalDeviceHandle device,
                   MetalCommandQueueHandle queue);

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle cmd_buf_handle) override;

private:
    void parse_axes();

    std::shared_ptr<const ov::op::v1::Reverse> m_node;
    ReverseCodegenDesc m_desc{};
    ov::element::Type m_element_type{ov::element::f32};

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    id<MTLComputePipelineState> m_pipeline = nil;
};

}  // namespace metal_plugin
}  // namespace ov
