// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <vector>

#include "kernel_ir/kernel_ir_common.hpp"
#include "runtime/metal_op.hpp"
#include "kernel_codegen/metal_kernel_compiler.hpp"

namespace ov {
namespace metal_plugin {

class METAL_OP_API MetalElementwiseOp : public MetalOp {
public:
    MetalElementwiseOp(const std::shared_ptr<const ov::Node>& node,
                       KernelOpKind kind,
                       void* device,
                       void* queue);
    ~MetalElementwiseOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void execute() override;

protected:
    KernelOpKind m_kind;
    std::shared_ptr<const ov::Node> m_node;
    ov::element::Type m_element_type{ov::element::f32};
    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    id<MTLComputePipelineState> m_pipeline = nil;

    // Broadcasting metadata (ints for Metal constant buffers).
    std::vector<int> m_out_dims;
    std::vector<int> m_stride0;
    std::vector<int> m_stride1;
    size_t m_num_elems = 0;

    // Optional constant inputs cached on device.
    MetalTensor m_const0;
    MetalTensor m_const1;

    // Recompute shapes/strides from runtime tensors if needed.
    void refresh_shapes_from_inputs();
};

class METAL_OP_API MetalAddOp final : public MetalElementwiseOp {
public:
    MetalAddOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, KernelOpKind::ElementwiseAdd, device, queue) {}
};

class METAL_OP_API MetalSubOp final : public MetalElementwiseOp {
public:
    MetalSubOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, KernelOpKind::ElementwiseSub, device, queue) {}
};

class METAL_OP_API MetalMulOp final : public MetalElementwiseOp {
public:
    MetalMulOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, KernelOpKind::ElementwiseMul, device, queue) {}
};

class METAL_OP_API MetalDivOp final : public MetalElementwiseOp {
public:
    MetalDivOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, KernelOpKind::ElementwiseDiv, device, queue) {}
};

}  // namespace metal_plugin
}  // namespace ov
