// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "kernel_ir/kernel_ir_common.hpp"
#include "runtime/metal_op.hpp"
#include "kernel_codegen/metal_kernel_compiler.hpp"

namespace ov {
namespace metal_plugin {

class METAL_OP_API MetalActivationOp : public MetalOp {
public:
    MetalActivationOp(const std::shared_ptr<const ov::Node>& node,
                      ActivationKind kind,
                      float alpha,
                      void* device,
                      void* queue);
    ~MetalActivationOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void execute() override;

protected:
    ActivationKind m_kind;
    float m_alpha;
    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    id<MTLComputePipelineState> m_pipeline = nil;
};

class METAL_OP_API MetalReluOp final : public MetalActivationOp {
public:
    MetalReluOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Relu, /*alpha=*/0.f, device, queue) {}
};

class METAL_OP_API MetalSigmoidOp final : public MetalActivationOp {
public:
    MetalSigmoidOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Sigmoid, /*alpha=*/0.f, device, queue) {}
};

class METAL_OP_API MetalTanhOp final : public MetalActivationOp {
public:
    MetalTanhOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Tanh, /*alpha=*/0.f, device, queue) {}
};

class METAL_OP_API MetalEluOp final : public MetalActivationOp {
public:
    MetalEluOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue, float alpha)
        : MetalActivationOp(node, ActivationKind::Elu, alpha, device, queue) {}
};

}  // namespace metal_plugin
}  // namespace ov
