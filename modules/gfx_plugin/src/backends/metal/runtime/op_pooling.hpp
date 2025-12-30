// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "mlir/codegen/codegen_desc.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/max_pool.hpp"
#include "backends/metal/runtime/op.hpp"

namespace ov {
namespace gfx_plugin {

// Shared implementation for 2D pooling (Max/Avg) using generated Metal kernels.
class GFX_OP_API MetalPoolOp : public MetalOp {
public:
    MetalPoolOp(const std::shared_ptr<const ov::Node>& node,
                bool is_avg,
                bool exclude_pad,
                void* device,
                void* queue);
    ~MetalPoolOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;

protected:
    bool m_is_avg = false;
    std::shared_ptr<const ov::Node> m_node;
    ov::element::Type m_element_type{ov::element::f32};
    Pool2DCodegenDesc m_desc{};
    ov::op::RoundingType m_rounding{ov::op::RoundingType::FLOOR};

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    std::shared_ptr<ICompiledKernel> m_kernel;
};

class GFX_OP_API MetalMaxPoolOp final : public MetalPoolOp {
public:
    MetalMaxPoolOp(const std::shared_ptr<const ov::op::v1::MaxPool>& node,
                   void* device,
                   void* queue);
};

class GFX_OP_API MetalAvgPoolOp final : public MetalPoolOp {
public:
    MetalAvgPoolOp(const std::shared_ptr<const ov::op::v1::AvgPool>& node,
                   void* device,
                   void* queue);
};

}  // namespace gfx_plugin
}  // namespace ov
