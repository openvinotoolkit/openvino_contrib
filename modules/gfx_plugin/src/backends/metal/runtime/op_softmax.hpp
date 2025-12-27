// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "mlir_codegen/codegen_desc.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/log_softmax.hpp"
#include "backends/metal/runtime/op.hpp"

namespace ov {
namespace gfx_plugin {

class GFX_OP_API MetalSoftmaxOp : public MetalOp {
public:
    MetalSoftmaxOp(const std::shared_ptr<const ov::Node>& node,
                   void* device,
                   void* queue,
                   bool log_softmax = false);
    ~MetalSoftmaxOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;

private:
    int64_t m_axis = -1;
    std::shared_ptr<const ov::Node> m_node;
    ov::element::Type m_element_type{ov::element::f32};
    SoftmaxCodegenDesc m_desc{};
    bool m_log_softmax = false;

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    std::shared_ptr<ICompiledKernel> m_kernel;
};

}  // namespace gfx_plugin
}  // namespace ov
