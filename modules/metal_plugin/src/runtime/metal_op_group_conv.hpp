// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/op/group_conv.hpp"
#include "runtime/metal_op.hpp"
#include "mlir_codegen/codegen_common.hpp"
#include "kernel_codegen/metal_kernel_compiler.hpp"

namespace ov {
namespace metal_plugin {

// Metal compute implementation of 2D GroupConvolution using generated MSL kernel.
class MetalGroupConvOp : public MetalOp {
public:
    MetalGroupConvOp(const std::shared_ptr<const ov::op::v1::GroupConvolution>& node,
                     MetalDeviceHandle device,
                     MetalCommandQueueHandle queue);
    ~MetalGroupConvOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;
    bool fuse_activation(ActivationKind kind, float alpha = 0.0f) override;

private:
    void prepare_weights();

    std::shared_ptr<const ov::op::v1::GroupConvolution> m_node;
    Conv2DCodegenDesc m_desc{};
    ov::element::Type m_element_type{ov::element::f32};
    bool m_has_activation = false;
    ActivationKind m_activation = ActivationKind::Relu;
    float m_activation_alpha = 0.0f;

    MetalBuffer m_weights{};

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    id<MTLComputePipelineState> m_pipeline = nil;
};

}  // namespace metal_plugin
}  // namespace ov
