// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "mlir_codegen/codegen_common.hpp"
#include "runtime/metal_op.hpp"
#include "kernel_codegen/metal_kernel_compiler.hpp"

namespace ov {
namespace metal_plugin {

class METAL_OP_API MetalReduceOp : public MetalOp {
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
    id<MTLComputePipelineState> m_pipeline = nil;
};

}  // namespace metal_plugin
}  // namespace ov
