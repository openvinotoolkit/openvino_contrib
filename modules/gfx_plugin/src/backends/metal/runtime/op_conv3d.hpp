// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "backends/metal/runtime/op.hpp"
#include "mlir/codegen/codegen_desc.hpp"
#include "openvino/op/convolution.hpp"

namespace ov {
namespace gfx_plugin {

class GFX_OP_API MetalConv3DOp : public MetalOp {
public:
    MetalConv3DOp(const std::shared_ptr<const ov::op::v1::Convolution>& node,
                  MetalDeviceHandle device,
                  MetalCommandQueueHandle queue);
    ~MetalConv3DOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;

private:
    void prepare_weights();
    size_t element_size() const { return m_element_type.size(); }

    std::shared_ptr<const ov::op::v1::Convolution> m_node;
    ov::element::Type m_element_type{ov::element::f32};
    Conv3DCodegenDesc m_desc{};
    MetalBuffer m_weights;

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    std::shared_ptr<ICompiledKernel> m_kernel;
};

}  // namespace gfx_plugin
}  // namespace ov
