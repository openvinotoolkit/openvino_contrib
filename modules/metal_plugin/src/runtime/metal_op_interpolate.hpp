// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/op/interpolate.hpp"
#include "mlir_codegen/codegen_common.hpp"
#include "runtime/metal_op.hpp"

namespace ov {
namespace metal_plugin {

class METAL_OP_API MetalInterpolateOp : public MetalOp {
public:
    MetalInterpolateOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);
    ~MetalInterpolateOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;

private:
    void parse_interpolate(const std::shared_ptr<const ov::Node>& node);

    std::shared_ptr<const ov::Node> m_node;
    ov::element::Type m_element_type{ov::element::f32};
    InterpolateCodegenDesc m_desc{};

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    id<MTLComputePipelineState> m_pipeline = nil;
};

}  // namespace metal_plugin
}  // namespace ov
