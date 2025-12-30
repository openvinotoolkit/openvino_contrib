// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "openvino/op/gather.hpp"
#include "mlir/codegen/codegen_desc.hpp"
#include "backends/metal/runtime/op.hpp"

namespace ov {
namespace gfx_plugin {

class GFX_OP_API MetalGatherOp : public MetalOp {
public:
    MetalGatherOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);
    ~MetalGatherOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;

private:
    void parse_gather(const std::shared_ptr<const ov::Node>& node);

    std::shared_ptr<const ov::Node> m_node;
    ov::element::Type m_element_type{ov::element::f32};
    ov::element::Type m_index_type{ov::element::i64};
    int64_t m_axis = 0;
    uint64_t m_outer = 0;
    uint64_t m_inner = 0;
    uint64_t m_axis_dim = 0;
    uint64_t m_indices_count = 0;

    MetalTensor m_const_indices{};

    GatherCodegenDesc m_desc{};

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    std::shared_ptr<ICompiledKernel> m_kernel;
};

}  // namespace gfx_plugin
}  // namespace ov
