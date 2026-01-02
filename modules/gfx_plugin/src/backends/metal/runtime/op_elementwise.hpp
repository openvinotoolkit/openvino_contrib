// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "kernel_ir/gfx_codegen_desc.hpp"
#include "backends/metal/runtime/op.hpp"

namespace ov {
namespace gfx_plugin {

class GFX_OP_API MetalElementwiseOp : public MetalOp {
public:
    MetalElementwiseOp(const std::shared_ptr<const ov::Node>& node,
                       EltwiseKind kind,
                       void* device,
                       void* queue);
    ~MetalElementwiseOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;

protected:
    EltwiseKind m_kind;
    std::shared_ptr<const ov::Node> m_node;
    ov::element::Type m_element_type{ov::element::f32};
    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    std::shared_ptr<ICompiledKernel> m_kernel;

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

class GFX_OP_API MetalAddOp final : public MetalElementwiseOp {
public:
    MetalAddOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, EltwiseKind::Add, device, queue) {}
};

class GFX_OP_API MetalSubOp final : public MetalElementwiseOp {
public:
    MetalSubOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, EltwiseKind::Sub, device, queue) {}
};

class GFX_OP_API MetalMulOp final : public MetalElementwiseOp {
public:
    MetalMulOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, EltwiseKind::Mul, device, queue) {}
};

class GFX_OP_API MetalDivOp final : public MetalElementwiseOp {
public:
    MetalDivOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, EltwiseKind::Div, device, queue) {}
};

class GFX_OP_API MetalPowOp final : public MetalElementwiseOp {
public:
    MetalPowOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, EltwiseKind::Pow, device, queue) {}
};

class GFX_OP_API MetalModOp final : public MetalElementwiseOp {
public:
    MetalModOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, EltwiseKind::Mod, device, queue) {}
};

class GFX_OP_API MetalFloorModOp final : public MetalElementwiseOp {
public:
    MetalFloorModOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, EltwiseKind::FloorMod, device, queue) {}
};

class GFX_OP_API MetalPreluOp final : public MetalElementwiseOp {
public:
    MetalPreluOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, EltwiseKind::Prelu, device, queue) {}
};

class GFX_OP_API MetalSquaredDiffOp final : public MetalElementwiseOp {
public:
    MetalSquaredDiffOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, EltwiseKind::SquaredDiff, device, queue) {}
};

class GFX_OP_API MetalMinOp final : public MetalElementwiseOp {
public:
    MetalMinOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, EltwiseKind::Min, device, queue) {}
};

class GFX_OP_API MetalMaxOp final : public MetalElementwiseOp {
public:
    MetalMaxOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, EltwiseKind::Max, device, queue) {}
};

class GFX_OP_API MetalLogicalAndOp final : public MetalElementwiseOp {
public:
    MetalLogicalAndOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, EltwiseKind::LogicalAnd, device, queue) {}
};

class GFX_OP_API MetalLogicalOrOp final : public MetalElementwiseOp {
public:
    MetalLogicalOrOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, EltwiseKind::LogicalOr, device, queue) {}
};

class GFX_OP_API MetalLogicalXorOp final : public MetalElementwiseOp {
public:
    MetalLogicalXorOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, EltwiseKind::LogicalXor, device, queue) {}
};

class GFX_OP_API MetalEqualOp final : public MetalElementwiseOp {
public:
    MetalEqualOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, EltwiseKind::Equal, device, queue) {}
};

class GFX_OP_API MetalNotEqualOp final : public MetalElementwiseOp {
public:
    MetalNotEqualOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, EltwiseKind::NotEqual, device, queue) {}
};

class GFX_OP_API MetalLessOp final : public MetalElementwiseOp {
public:
    MetalLessOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, EltwiseKind::Less, device, queue) {}
};

class GFX_OP_API MetalGreaterOp final : public MetalElementwiseOp {
public:
    MetalGreaterOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, EltwiseKind::Greater, device, queue) {}
};

class GFX_OP_API MetalLessEqualOp final : public MetalElementwiseOp {
public:
    MetalLessEqualOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, EltwiseKind::LessEqual, device, queue) {}
};

class GFX_OP_API MetalGreaterEqualOp final : public MetalElementwiseOp {
public:
    MetalGreaterEqualOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalElementwiseOp(node, EltwiseKind::GreaterEqual, device, queue) {}
};

}  // namespace gfx_plugin
}  // namespace ov
