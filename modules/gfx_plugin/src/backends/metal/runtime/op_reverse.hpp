// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "openvino/op/reverse.hpp"
#include "backends/metal/runtime/op.hpp"
#include "mlir_codegen/codegen_desc.hpp"

namespace ov {
namespace gfx_plugin {

class GFX_OP_API MetalReverseOp final : public MetalOp {
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
    std::shared_ptr<ICompiledKernel> m_kernel;
};

}  // namespace gfx_plugin
}  // namespace ov
