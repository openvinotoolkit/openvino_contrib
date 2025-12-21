// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/op/batch_norm.hpp"
#include "runtime/metal_op.hpp"
#include "mlir_codegen/codegen_common.hpp"

namespace ov {
namespace gfx_plugin {

class GFX_OP_API MetalBatchNormOp : public MetalOp {
public:
    MetalBatchNormOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);
    ~MetalBatchNormOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;

private:
    void parse_bn(const std::shared_ptr<const ov::Node>& node);

    std::shared_ptr<const ov::Node> m_node;
    BatchNorm2DCodegenDesc m_desc{};
    ov::element::Type m_element_type{ov::element::f32};
    std::vector<float> m_params;  // gamma | beta | mean | var | eps

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    id<MTLComputePipelineState> m_pipeline = nil;
    MetalBuffer m_params_buf;
};

}  // namespace gfx_plugin
}  // namespace ov
