// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "mlir_codegen/codegen_desc.hpp"
#include "backends/metal/runtime/op.hpp"

namespace ov {
namespace gfx_plugin {

class GFX_OP_API MetalReduceOp : public MetalOp {
public:
    MetalReduceOp(const std::shared_ptr<const ov::Node>& node,
                  ReduceKind kind,
                  void* device,
                  void* queue);
    ~MetalReduceOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;

private:
    ReduceKind m_kind;
    std::shared_ptr<const ov::Node> m_node;
    ov::element::Type m_element_type{ov::element::f32};
    std::vector<int> m_axes_mask;
    std::vector<int> m_out_dims;
    std::vector<int> m_in_dims;
    std::vector<int> m_in_strides;
    std::vector<int> m_reduce_dims;
    uint32_t m_num_elems = 0;
    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    std::shared_ptr<ICompiledKernel> m_kernel;
};

}  // namespace gfx_plugin
}  // namespace ov
