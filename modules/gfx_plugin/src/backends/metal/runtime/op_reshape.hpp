// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "backends/metal/runtime/op.hpp"
#include "mlir_codegen/codegen_desc.hpp"

namespace ov {
namespace gfx_plugin {

class GFX_OP_API MetalReshapeOp : public MetalOp {
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

class GFX_OP_API MetalTransposeOp : public MetalOp {
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
    std::shared_ptr<ICompiledKernel> m_kernel;
};

}  // namespace gfx_plugin
}  // namespace ov
