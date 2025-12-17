// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "openvino/op/concat.hpp"
#include "runtime/metal_op.hpp"

namespace ov {
namespace metal_plugin {

class METAL_OP_API MetalConcatOp : public MetalOp {
public:
    MetalConcatOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);
    ~MetalConcatOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void execute() override;

private:
    void compute_layout(const std::shared_ptr<const ov::Node>& node);

    int64_t m_axis = 0;
    uint64_t m_outer = 0;
    uint64_t m_inner = 0;
    std::vector<uint64_t> m_axis_sizes;
    std::vector<uint64_t> m_axis_offsets;
    ov::element::Type m_element_type{ov::element::f32};

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
};

}  // namespace metal_plugin
}  // namespace ov
