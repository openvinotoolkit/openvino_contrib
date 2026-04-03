// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "kernel_ir/gfx_codegen_desc.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/core/type/float16.hpp"
#include "backends/metal/runtime/op.hpp"

namespace ov {
namespace gfx_plugin {

class GFX_OP_API MetalMatMulOp : public MetalOp {
public:
    MetalMatMulOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue);
    ~MetalMatMulOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;
    bool fuse_activation(ActivationKind kind, float alpha = 0.0f) override;
    bool fuse_bias(const BiasParams& params) override;

private:
    void fill_desc_from_node(const std::shared_ptr<const ov::Node>& node);

    std::shared_ptr<const ov::Node> m_node;
    MatMulCodegenDesc m_desc{};
    ov::Shape m_shape_a;
    ov::Shape m_shape_b;
    ov::element::Type m_element_type{ov::element::f32};
    bool m_has_activation = false;
    ActivationKind m_activation = ActivationKind::Relu;
    float m_activation_alpha = 0.0f;
    bool m_has_bias = false;
    BiasParams m_bias_params{};
    MetalTensor m_constA;
    MetalTensor m_constB;
    MetalBuffer m_bias{};
    std::vector<ov::float16> m_bias_f16;

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    std::shared_ptr<ICompiledKernel> m_kernel;
};

}  // namespace gfx_plugin
}  // namespace ov
