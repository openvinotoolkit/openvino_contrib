// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/metal_op_factory.hpp"

#include <string>
#include <utility>

#include "openvino/core/validation_util.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "runtime/metal_logger.hpp"
#include "runtime/metal_op_conv.hpp"
#include "runtime/metal_op_matmul.hpp"
#include "runtime/metal_op_activations.hpp"
#include "runtime/metal_op_elementwise.hpp"
#include "runtime/metal_op_pooling.hpp"
#include "runtime/metal_op_softmax.hpp"
#include "runtime/metal_op_split.hpp"
#include "runtime/metal_op_concat.hpp"
#include "runtime/metal_op_reshape.hpp"
#include "runtime/metal_op_batchnorm.hpp"
#include "runtime/metal_op_convert.hpp"

namespace ov {
namespace metal_plugin {

namespace {

// Temporary stub that allows wiring of the pipeline before real kernels land.
class StubMetalOp final : public MetalOp {
public:
    StubMetalOp(std::string name,
                std::string type,
                const ov::Shape& output_shape,
                MetalDeviceHandle device,
                MetalCommandQueueHandle queue)
        : MetalOp(std::move(name), std::move(type), output_shape, device, queue) {}

    void execute() override {
        // Profiling hooks remain usable even for stubs.
        start_profiling();
        stop_profiling_ms();
    }
};

inline ov::Shape output_shape_or_empty(const std::shared_ptr<const ov::Node>& node) {
    if (!node || node->get_output_size() == 0)
        return {};
    const auto& pshape = node->get_output_partial_shape(0);
    return pshape.is_static() ? node->get_output_shape(0) : ov::Shape{};
}

bool is_input_or_output(const std::shared_ptr<const ov::Node>& node) {
    return ov::as_type_ptr<const ov::op::v0::Parameter>(node) ||
           ov::as_type_ptr<const ov::op::v0::Result>(node) ||
           ov::as_type_ptr<const ov::op::v0::Constant>(node);
}

}  // namespace

std::unique_ptr<MetalOp> MetalOpFactory::create(const std::shared_ptr<const ov::Node>& node,
                                                void* device,
                                                void* queue) {
    MetalDeviceHandle dev = static_cast<MetalDeviceHandle>(device);
    MetalCommandQueueHandle q = static_cast<MetalCommandQueueHandle>(queue);
    if (!node || is_input_or_output(node))
        return nullptr;

    const auto shape = output_shape_or_empty(node);
    const auto name = node->get_friendly_name();
    const auto type_name = std::string{node->get_type_name()};

    auto make_stub = [&](const std::string& mapped_type) {
        METAL_LOG_DEBUG("OpFactory", "Mapped " << name << " (" << type_name << ") -> " << mapped_type);
        return std::make_unique<StubMetalOp>(name, mapped_type, shape, dev, q);
    };

    if (ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
        auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(node);
        // Only handle 2D NCHW here; let the MLIR path take 3D/other ranks.
        const auto in_pshape = conv->get_input_partial_shape(0);
        if (in_pshape.rank().is_static() && in_pshape.size() == 4) {
            return std::make_unique<MetalConvOp>(conv, dev, q);
        }
        return nullptr;
    }
    if (ov::as_type_ptr<const ov::op::v0::MatMul>(node)) {
        // Support static ranks 2–4; otherwise defer to MLIR.
        const auto& ps0 = node->get_input_partial_shape(0);
        const auto& ps1 = node->get_input_partial_shape(1);
        const auto et = node->get_output_element_type(0);
        const bool et_ok = (et == ov::element::f32 || et == ov::element::f16);
        if (ps0.rank().is_static() && ps1.rank().is_static() &&
            ps0.size() >= 2 && ps0.size() <= 4 && ps1.size() >= 2 && ps1.size() <= 4 &&
            et_ok) {
            return std::make_unique<MetalMatMulOp>(node, dev, q);
        }
        return nullptr;
    }
    if (ov::as_type_ptr<const ov::op::v0::Relu>(node)) {
        return std::make_unique<MetalReluOp>(node, dev, q);
    }
    if (ov::as_type_ptr<const ov::op::v0::Sigmoid>(node)) {
        return std::make_unique<MetalSigmoidOp>(node, dev, q);
    }
    if (ov::as_type_ptr<const ov::op::v0::Tanh>(node)) {
        return std::make_unique<MetalTanhOp>(node, dev, q);
    }
    if (auto elu = ov::as_type_ptr<const ov::op::v0::Elu>(node)) {
        return std::make_unique<MetalEluOp>(node, dev, q, static_cast<float>(elu->get_alpha()));
    }
    if (ov::as_type_ptr<const ov::op::v1::Add>(node)) {
        return std::make_unique<MetalAddOp>(node, dev, q);
    }
    if (ov::as_type_ptr<const ov::op::v1::Subtract>(node)) {
        return std::make_unique<MetalSubOp>(node, dev, q);
    }
    if (ov::as_type_ptr<const ov::op::v1::Multiply>(node)) {
        return std::make_unique<MetalMulOp>(node, dev, q);
    }
    if (ov::as_type_ptr<const ov::op::v1::Divide>(node)) {
        return std::make_unique<MetalDivOp>(node, dev, q);
    }
    if (auto mp = ov::as_type_ptr<const ov::op::v1::MaxPool>(node)) {
        return std::make_unique<MetalMaxPoolOp>(mp, dev, q);
    }
    if (auto ap = ov::as_type_ptr<const ov::op::v1::AvgPool>(node)) {
        return std::make_unique<MetalAvgPoolOp>(ap, dev, q);
    }
    if (ov::as_type_ptr<const ov::op::v1::Softmax>(node) ||
        ov::as_type_ptr<const ov::op::v8::Softmax>(node)) {
        // MetalSoftmaxOp handles dynamic shapes at runtime using input tensor shape.
        return std::make_unique<MetalSoftmaxOp>(node, dev, q);
    }
    if (ov::as_type_ptr<const ov::op::v1::Split>(node) ||
        ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node)) {
        return std::make_unique<MetalSplitOp>(node, dev, q);
    }
    if (ov::as_type_ptr<const ov::op::v0::Concat>(node)) {
        const auto& pshape = node->get_output_partial_shape(0);
        if (pshape.rank().is_static()) {
            return std::make_unique<MetalConcatOp>(node, dev, q);
        }
        return nullptr;
    }
    if (ov::as_type_ptr<const ov::op::v1::Reshape>(node)) {
        return std::make_unique<MetalReshapeOp>(node, dev, q);
    }
    if (ov::as_type_ptr<const ov::op::v1::Transpose>(node)) {
        auto perm_const = ov::as_type_ptr<const ov::op::v0::Constant>(node->input_value(1).get_node_shared_ptr());
        if (perm_const) {
            return std::make_unique<MetalTransposeOp>(node, dev, q);
        }
        return nullptr;
    }
    if (ov::as_type_ptr<const ov::op::v0::BatchNormInference>(node) ||
        ov::as_type_ptr<const ov::op::v5::BatchNormInference>(node)) {
        // Require rank-4 static shape and constant params; otherwise delegate to MLIR.
        const auto& pshape = node->get_input_partial_shape(0);
        bool params_const = ov::as_type_ptr<const ov::op::v0::Constant>(node->input_value(1).get_node_shared_ptr()) &&
                            ov::as_type_ptr<const ov::op::v0::Constant>(node->input_value(2).get_node_shared_ptr()) &&
                            ov::as_type_ptr<const ov::op::v0::Constant>(node->input_value(3).get_node_shared_ptr()) &&
                            ov::as_type_ptr<const ov::op::v0::Constant>(node->input_value(4).get_node_shared_ptr());
        if (pshape.rank().is_static() && pshape.size() == 4 && params_const &&
            node->get_output_element_type(0) == ov::element::f32) {
            return std::make_unique<MetalBatchNormOp>(node, dev, q);
        }
        return nullptr;
    }
    if (ov::as_type_ptr<const ov::op::v0::Convert>(node)) {
        auto src = node->get_input_element_type(0);
        auto dst = node->get_output_element_type(0);
        auto supported = [&](const ov::element::Type& t) {
            return t == ov::element::f32 || t == ov::element::f16 ||
                   t == ov::element::i32 || t == ov::element::i64;
        };
        if (supported(src) && supported(dst)) {
            return std::make_unique<MetalConvertOp>(node, dev, q);
        }
        return nullptr;
    }
    // Unsupported op for MetalOp pipeline.
    METAL_LOG_DEBUG("OpFactory", "No MetalOp mapping for " << name << " (" << type_name << ")");
    return nullptr;
}

}  // namespace metal_plugin
}  // namespace ov
