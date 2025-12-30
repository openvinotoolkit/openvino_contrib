// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "backends/metal/runtime/op.hpp"
#include "mlir/codegen/codegen_desc.hpp"

namespace ov {
namespace gfx_plugin {

class GFX_OP_API MetalTopKOp final : public MetalOp {
public:
    MetalTopKOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);
    ~MetalTopKOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;
    void set_outputs(const std::vector<std::unique_ptr<MetalTensor>>& outputs) override;

private:
    void parse_topk(const std::shared_ptr<const ov::Node>& node);

    std::shared_ptr<const ov::Node> m_node;
    std::vector<MetalTensor*> m_outputs;
    ov::Shape m_input_shape;
    ov::Shape m_output_shape;
    ov::element::Type m_element_type{ov::element::f32};
    ov::element::Type m_index_type{ov::element::i32};
    uint32_t m_axis = 0;
    uint32_t m_k = 0;
    uint32_t m_axis_len = 0;
    uint32_t m_outer = 1;
    uint32_t m_inner = 1;
    bool m_mode_max = true;
    TopKSortType m_sort_type = TopKSortType::SortValues;

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    std::shared_ptr<ICompiledKernel> m_kernel;
};

}  // namespace gfx_plugin
}  // namespace ov
