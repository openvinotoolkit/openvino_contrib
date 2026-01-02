// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <memory>

#include "openvino/op/gather_nd.hpp"
#include "kernel_ir/gfx_codegen_desc.hpp"
#include "backends/metal/runtime/op.hpp"

namespace ov {
namespace gfx_plugin {

class MetalGatherNDOp : public MetalOp {
public:
    MetalGatherNDOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;

private:
    void parse_gathernd(const std::shared_ptr<const ov::Node>& node);

    std::shared_ptr<const ov::Node> m_node;
    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;

    ov::element::Type m_element_type{ov::element::dynamic};
    ov::element::Type m_index_type{ov::element::i64};
    uint64_t m_inner = 1;
    uint64_t m_num_indices = 1;
    uint64_t m_k = 1;
    uint64_t m_total = 1;
    std::array<uint32_t, 8> m_strides{};
    std::array<uint32_t, 8> m_dims{};

    MetalTensor m_const_indices;
    std::shared_ptr<ICompiledKernel> m_kernel;
    GatherNDCodegenDesc m_desc{};
};

}  // namespace gfx_plugin
}  // namespace ov
