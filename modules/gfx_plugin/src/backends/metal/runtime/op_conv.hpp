// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "openvino/op/convolution.hpp"
#include "openvino/core/type/float16.hpp"
#include "backends/metal/runtime/op.hpp"
#include "kernel_ir/gfx_codegen_desc.hpp"
#include "runtime/gfx_batchnorm.hpp"

namespace ov {
namespace gfx_plugin {

// Metal compute implementation of 2D convolution using generated MSL kernel (no MPSGraph).
class MetalConvOp : public MetalOp {
public:
    MetalConvOp(const std::shared_ptr<const ov::op::v1::Convolution>& node,
                MetalDeviceHandle device,
                MetalCommandQueueHandle queue);
    ~MetalConvOp() override = default;

    void init(MetalBufferManager* buffer_manager) override;
    void compile(MetalBufferManager* buffer_manager) override;
    void execute(MetalCommandBufferHandle command_buffer) override;
    bool fuse_activation(ActivationKind kind, float alpha = 0.0f) override;
    bool fuse_batchnorm(const BatchNormParams& params) override;
    bool fuse_bias(const BiasParams& params) override;

private:
    void prepare_weights();
    void prepare_batchnorm();
    void prepare_bias();

    std::shared_ptr<const ov::op::v1::Convolution> m_node;
    Conv2DCodegenDesc m_desc{};
    ov::element::Type m_element_type{ov::element::f32};
    bool m_has_activation = false;
    ActivationKind m_activation = ActivationKind::Relu;
    float m_activation_alpha = 0.0f;
    bool m_has_bn = false;
    BatchNormParams m_bn_params{};
    bool m_has_bias = false;
    BiasParams m_bias_params{};
    std::vector<ov::float16> m_bn_gamma_f16;
    std::vector<ov::float16> m_bn_beta_f16;
    std::vector<ov::float16> m_bn_mean_f16;
    std::vector<ov::float16> m_bn_var_f16;
    std::vector<ov::float16> m_bias_f16;

    MetalBuffer m_weights{};
    MetalBuffer m_bias{};
    MetalBuffer m_bn_gamma{};
    MetalBuffer m_bn_beta{};
    MetalBuffer m_bn_mean{};
    MetalBuffer m_bn_var{};

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    std::shared_ptr<ICompiledKernel> m_kernel;
};

}  // namespace gfx_plugin
}  // namespace ov
