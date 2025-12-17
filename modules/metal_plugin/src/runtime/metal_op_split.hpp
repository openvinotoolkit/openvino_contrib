// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "runtime/metal_op.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"

namespace ov {
namespace metal_plugin {

// Metal implementation of Split / VariadicSplit using blit copies on GPU.
class METAL_OP_API MetalSplitOp : public MetalOp {
public:
    MetalSplitOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);
    ~MetalSplitOp() override = default;

    void set_outputs(const std::vector<std::unique_ptr<MetalTensor>>& outputs) override;
    void execute() override;

private:
    void parse_split(const std::shared_ptr<const ov::Node>& node);
    size_t element_size() const { return m_element_type.size(); }

    int64_t m_axis = 0;
    std::vector<size_t> m_split_sizes;
    ov::Shape m_input_shape;
    ov::element::Type m_element_type{ov::element::f32};

    std::vector<MetalTensor*> m_outputs;

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
};

}  // namespace metal_plugin
}  // namespace ov
