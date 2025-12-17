// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/op/batch_norm.hpp"
#include "runtime/metal_op.hpp"
#include "kernel_ir/kernel_ir_common.hpp"

namespace ov {
namespace metal_plugin {

class METAL_OP_API MetalBatchNormOp : public MetalOp {
public:
    MetalBatchNormOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);
    ~MetalBatchNormOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void execute() override;

private:
    void parse_bn(const std::shared_ptr<const ov::Node>& node);

    KernelOp m_desc{};
    ov::element::Type m_element_type{ov::element::f32};
    std::vector<float> m_params;  // gamma | beta | mean | var | eps

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    id<MTLComputePipelineState> m_pipeline = nil;
    MetalBuffer m_params_buf;
};

}  // namespace metal_plugin
}  // namespace ov
