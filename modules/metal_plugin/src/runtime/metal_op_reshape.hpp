// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "runtime/metal_op.hpp"
#include "mlir_codegen/codegen_common.hpp"

namespace ov {
namespace metal_plugin {

class METAL_OP_API MetalReshapeOp : public MetalOp {
public:
    MetalReshapeOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);
    ~MetalReshapeOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;

private:
    std::shared_ptr<const ov::Node> m_node;
    ov::Shape m_target_shape;
    ov::element::Type m_element_type{ov::element::dynamic};
};

class METAL_OP_API MetalTransposeOp : public MetalOp {
public:
    MetalTransposeOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);
    ~MetalTransposeOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;

private:
    void build_desc(const std::shared_ptr<const ov::Node>& node);

    std::shared_ptr<const ov::Node> m_node;
    struct TransposeRuntimeDesc {
        std::vector<int64_t> in_shape;
        std::vector<int64_t> out_shape;
        std::vector<int64_t> perm;
    } m_desc;
    ov::element::Type m_element_type{ov::element::f32};

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    id<MTLComputePipelineState> m_pipeline = nil;
};

}  // namespace metal_plugin
}  // namespace ov
