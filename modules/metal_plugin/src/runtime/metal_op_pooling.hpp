// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "kernel_ir/kernel_ir_common.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/max_pool.hpp"
#include "runtime/metal_op.hpp"

namespace ov {
namespace metal_plugin {

// Shared implementation for 2D pooling (Max/Avg) using generated Metal kernels.
class METAL_OP_API MetalPoolOp : public MetalOp {
public:
    MetalPoolOp(const std::shared_ptr<const ov::Node>& node,
                KernelOpKind kind,
                bool exclude_pad,
                void* device,
                void* queue);
    ~MetalPoolOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void execute() override;

protected:
    KernelOpKind m_kind;
    ov::element::Type m_element_type{ov::element::f32};
    KernelOp m_desc{};
    ov::op::RoundingType m_rounding{ov::op::RoundingType::FLOOR};

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    id<MTLComputePipelineState> m_pipeline = nil;
    bool m_compiled = false;
};

class METAL_OP_API MetalMaxPoolOp final : public MetalPoolOp {
public:
    MetalMaxPoolOp(const std::shared_ptr<const ov::op::v1::MaxPool>& node,
                   void* device,
                   void* queue);
};

class METAL_OP_API MetalAvgPoolOp final : public MetalPoolOp {
public:
    MetalAvgPoolOp(const std::shared_ptr<const ov::op::v1::AvgPool>& node,
                   void* device,
                   void* queue);
};

}  // namespace metal_plugin
}  // namespace ov
