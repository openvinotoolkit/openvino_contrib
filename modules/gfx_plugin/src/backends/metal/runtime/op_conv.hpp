// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "openvino/op/convolution.hpp"
#include "backends/metal/runtime/op.hpp"
#include "kernel_ir/gfx_codegen_desc.hpp"

namespace ov {
namespace gfx_plugin {

// Metal compute implementation of 2D convolution using generated MSL kernel (no MPSGraph).
class MetalConvOp : public MetalOp {
public:
    MetalConvOp(const std::shared_ptr<const ov::op::v1::Convolution>& node,
                MetalDeviceHandle device,
                MetalCommandQueueHandle queue);
    ~MetalConvOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;
    bool fuse_activation(ActivationKind kind, float alpha = 0.0f) override;

private:
    void prepare_weights();

    std::shared_ptr<const ov::op::v1::Convolution> m_node;
    Conv2DCodegenDesc m_desc{};
    ov::element::Type m_element_type{ov::element::f32};
    bool m_has_activation = false;
    ActivationKind m_activation = ActivationKind::Relu;
    float m_activation_alpha = 0.0f;

    MetalBuffer m_weights{};

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    std::shared_ptr<ICompiledKernel> m_kernel;
};

}  // namespace gfx_plugin
}  // namespace ov
