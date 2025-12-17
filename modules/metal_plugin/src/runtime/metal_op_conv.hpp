// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "openvino/op/convolution.hpp"
#include "runtime/metal_op.hpp"
#include "kernel_ir/kernel_ir_common.hpp"
#include "kernel_codegen/metal_kernel_compiler.hpp"

namespace ov {
namespace metal_plugin {

// Metal compute implementation of 2D convolution using generated MSL kernel (no MPSGraph).
class MetalConvOp : public MetalOp {
public:
    MetalConvOp(const std::shared_ptr<const ov::op::v1::Convolution>& node,
                MetalDeviceHandle device,
                MetalCommandQueueHandle queue);
    ~MetalConvOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void execute() override;

private:
    void prepare_weights();
    void compile_pipeline();

    std::shared_ptr<const ov::op::v1::Convolution> m_node;
    KernelOp m_desc{};

    MetalBuffer m_weights{};

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    id<MTLComputePipelineState> m_pipeline = nil;
};

}  // namespace metal_plugin
}  // namespace ov
