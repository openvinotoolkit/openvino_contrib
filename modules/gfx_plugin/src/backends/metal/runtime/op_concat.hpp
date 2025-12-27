// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "openvino/op/concat.hpp"
#include "backends/metal/runtime/op.hpp"

namespace ov {
namespace gfx_plugin {

class GFX_OP_API MetalConcatOp : public MetalOp {
public:
    MetalConcatOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);
    ~MetalConcatOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;

private:
    void compute_layout(const std::shared_ptr<const ov::Node>& node);

    std::shared_ptr<const ov::Node> m_node;
    int64_t m_axis = 0;
    uint64_t m_outer = 0;
    uint64_t m_inner = 0;
    std::vector<uint64_t> m_axis_sizes;
    std::vector<uint64_t> m_axis_offsets;
    ov::element::Type m_element_type{ov::element::f32};
    std::vector<MetalTensor> m_const_inputs;

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    std::shared_ptr<ICompiledKernel> m_kernel;
};

}  // namespace gfx_plugin
}  // namespace ov
