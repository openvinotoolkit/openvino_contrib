// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "kernel_ir/gfx_codegen_desc.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "backends/metal/runtime/op.hpp"

namespace ov {
namespace gfx_plugin {

class GFX_OP_API MetalScatterNDUpdateOp : public MetalOp {
public:
    MetalScatterNDUpdateOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);
    ~MetalScatterNDUpdateOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;

private:
    void parse_scatter_nd_update(const std::shared_ptr<const ov::Node>& node);

    std::shared_ptr<const ov::Node> m_node;
    ov::element::Type m_element_type{ov::element::f32};
    ov::element::Type m_index_type{ov::element::i64};
    uint64_t m_inner = 0;
    uint64_t m_num_indices = 0;
    uint64_t m_k = 0;
    uint64_t m_total_updates = 0;
    uint64_t m_total_data = 0;

    std::array<uint32_t, ScatterNDUpdateCodegenDesc::kMaxDims> m_dims{};
    std::array<uint32_t, ScatterNDUpdateCodegenDesc::kMaxDims> m_strides{};

    MetalTensor m_const_indices{};
    MetalTensor m_const_updates{};

    ScatterNDUpdateCodegenDesc m_desc{};

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    std::shared_ptr<ICompiledKernel> m_kernel_init;
    std::shared_ptr<ICompiledKernel> m_kernel_update;
};

}  // namespace gfx_plugin
}  // namespace ov
