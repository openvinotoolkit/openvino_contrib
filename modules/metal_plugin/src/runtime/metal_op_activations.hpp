// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "mlir_codegen/codegen_common.hpp"
#include "runtime/metal_op.hpp"
#include "kernel_codegen/metal_kernel_compiler.hpp"

namespace ov {
namespace metal_plugin {

class METAL_OP_API MetalActivationOp : public MetalOp {
public:
    MetalActivationOp(const std::shared_ptr<const ov::Node>& node,
                      ActivationKind kind,
                      float alpha,
                      double clamp_min,
                      double clamp_max,
                      void* device,
                      void* queue);
    ~MetalActivationOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;

protected:
    ActivationKind m_kind;
    float m_alpha;
    double m_clamp_min;
    double m_clamp_max;
    std::shared_ptr<const ov::Node> m_node;
    ov::element::Type m_element_type{ov::element::f32};
    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    id<MTLComputePipelineState> m_pipeline = nil;
};

class METAL_OP_API MetalReluOp final : public MetalActivationOp {
public:
    MetalReluOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Relu, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalSigmoidOp final : public MetalActivationOp {
public:
    MetalSigmoidOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Sigmoid, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalTanhOp final : public MetalActivationOp {
public:
    MetalTanhOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Tanh, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalEluOp final : public MetalActivationOp {
public:
    MetalEluOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue, float alpha)
        : MetalActivationOp(node, ActivationKind::Elu, alpha, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalAbsOp final : public MetalActivationOp {
public:
    MetalAbsOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Abs, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalSignOp final : public MetalActivationOp {
public:
    MetalSignOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Sign, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalGeluOp final : public MetalActivationOp {
public:
    MetalGeluOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Gelu, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalSwishOp final : public MetalActivationOp {
public:
    MetalSwishOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Swish, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalHSwishOp final : public MetalActivationOp {
public:
    MetalHSwishOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::HSwish, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalHSigmoidOp final : public MetalActivationOp {
public:
    MetalHSigmoidOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::HSigmoid, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalSoftPlusOp final : public MetalActivationOp {
public:
    MetalSoftPlusOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::SoftPlus, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalMishOp final : public MetalActivationOp {
public:
    MetalMishOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Mish, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalSoftSignOp final : public MetalActivationOp {
public:
    MetalSoftSignOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::SoftSign, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalLogicalNotOp final : public MetalActivationOp {
public:
    MetalLogicalNotOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::LogicalNot, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalClampOp final : public MetalActivationOp {
public:
    MetalClampOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue, double min_v, double max_v)
        : MetalActivationOp(node, ActivationKind::Clamp, /*alpha=*/0.f, min_v, max_v, device, queue) {}
};

class METAL_OP_API MetalExpOp final : public MetalActivationOp {
public:
    MetalExpOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Exp, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalLogOp final : public MetalActivationOp {
public:
    MetalLogOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Log, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalSqrtOp final : public MetalActivationOp {
public:
    MetalSqrtOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Sqrt, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalFloorOp final : public MetalActivationOp {
public:
    MetalFloorOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Floor, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalCeilOp final : public MetalActivationOp {
public:
    MetalCeilOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Ceil, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalNegativeOp final : public MetalActivationOp {
public:
    MetalNegativeOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Negative, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalSinOp final : public MetalActivationOp {
public:
    MetalSinOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Sin, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalCosOp final : public MetalActivationOp {
public:
    MetalCosOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Cos, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalTanOp final : public MetalActivationOp {
public:
    MetalTanOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Tan, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalErfOp final : public MetalActivationOp {
public:
    MetalErfOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Erf, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalAsinOp final : public MetalActivationOp {
public:
    MetalAsinOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Asin, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalAcosOp final : public MetalActivationOp {
public:
    MetalAcosOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Acos, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalAtanOp final : public MetalActivationOp {
public:
    MetalAtanOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Atan, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalAsinhOp final : public MetalActivationOp {
public:
    MetalAsinhOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Asinh, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalAcoshOp final : public MetalActivationOp {
public:
    MetalAcoshOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Acosh, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalAtanhOp final : public MetalActivationOp {
public:
    MetalAtanhOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Atanh, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalSinhOp final : public MetalActivationOp {
public:
    MetalSinhOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Sinh, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalCoshOp final : public MetalActivationOp {
public:
    MetalCoshOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::Cosh, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalRoundEvenOp final : public MetalActivationOp {
public:
    MetalRoundEvenOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::RoundEven, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

class METAL_OP_API MetalRoundAwayOp final : public MetalActivationOp {
public:
    MetalRoundAwayOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
        : MetalActivationOp(node, ActivationKind::RoundAway, /*alpha=*/0.f, /*clamp_min=*/0.0, /*clamp_max=*/0.0, device, queue) {}
};

}  // namespace metal_plugin
}  // namespace ov
