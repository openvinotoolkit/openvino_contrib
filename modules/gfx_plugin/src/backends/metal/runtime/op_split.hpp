// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "backends/metal/runtime/op.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"

namespace ov {
namespace gfx_plugin {

// Metal implementation of Split / VariadicSplit using MLIR-generated MSL kernels.
class GFX_OP_API MetalSplitOp : public MetalOp {
public:
    MetalSplitOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);
    ~MetalSplitOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void set_outputs(const std::vector<std::unique_ptr<MetalTensor>>& outputs) override;
    void execute(MetalCommandBufferHandle command_buffer) override;

private:
    void parse_split(const std::shared_ptr<const ov::Node>& node);
    size_t element_size() const { return m_element_type.size(); }

    std::shared_ptr<const ov::Node> m_node;
    int64_t m_axis = 0;
    std::vector<size_t> m_split_sizes;
    bool m_is_variadic = false;
    size_t m_num_splits = 0;
    ov::Shape m_input_shape;
    ov::element::Type m_element_type{ov::element::f32};

    std::vector<MetalTensor*> m_outputs;

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    std::shared_ptr<ICompiledKernel> m_kernel;
};

}  // namespace gfx_plugin
}  // namespace ov
