// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/metal_op_factory.hpp"

#include <string>
#include <utility>
#include <vector>
#include <functional>

#include "openvino/core/validation_util.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/asin.hpp"
#include "openvino/op/acos.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/asinh.hpp"
#include "openvino/op/acosh.hpp"
#include "openvino/op/atanh.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/tan.hpp"
#include "openvino/op/sinh.hpp"
#include "openvino/op/cosh.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/hswish.hpp"
#include "openvino/op/hsigmoid.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/mish.hpp"
#include "openvino/op/softsign.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/space_to_depth.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reverse.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/topk.hpp"
#include "runtime/metal_logger.hpp"
#include "runtime/metal_op_conv.hpp"
#include "runtime/metal_op_conv3d.hpp"
#include "runtime/metal_op_group_conv.hpp"
#include "runtime/metal_op_matmul.hpp"
#include "runtime/metal_op_activations.hpp"
#include "runtime/metal_op_elementwise.hpp"
#include "runtime/metal_op_pooling.hpp"
#include "runtime/metal_op_softmax.hpp"
#include "runtime/metal_op_split.hpp"
#include "runtime/metal_op_concat.hpp"
#include "runtime/metal_op_depth_to_space.hpp"
#include "runtime/metal_op_reshape.hpp"
#include "runtime/metal_op_batchnorm.hpp"
#include "runtime/metal_op_convert.hpp"
#include "runtime/metal_op_interpolate.hpp"
#include "runtime/metal_op_gather.hpp"
#include "runtime/metal_op_gather_elements.hpp"
#include "runtime/metal_op_gathernd.hpp"
#include "runtime/metal_op_scatter_elements_update.hpp"
#include "runtime/metal_op_scatter_nd_update.hpp"
#include "runtime/metal_op_space_to_depth.hpp"
#include "runtime/metal_op_slice.hpp"
#include "runtime/metal_op_shapeof.hpp"
#include "runtime/metal_op_select.hpp"
#include "runtime/metal_op_reduce.hpp"
#include "runtime/metal_op_pad.hpp"
#include "runtime/metal_op_tile.hpp"
#include "runtime/metal_op_broadcast.hpp"
#include "runtime/metal_op_range.hpp"
#include "runtime/metal_op_reverse.hpp"
#include "runtime/metal_op_topk.hpp"

namespace ov {
namespace metal_plugin {

namespace {

bool is_static_shape(const ov::PartialShape& ps) {
    if (!ps.rank().is_static())
        return false;
    for (const auto& d : ps) {
        if (!d.is_static())
            return false;
    }
    return true;
}

bool is_input_or_output(const std::shared_ptr<const ov::Node>& node) {
    return ov::as_type_ptr<const ov::op::v0::Parameter>(node) ||
           ov::as_type_ptr<const ov::op::v0::Result>(node) ||
           ov::as_type_ptr<const ov::op::v0::Constant>(node);
}

bool is_broadcastable_to(const ov::Shape& target, const ov::Shape& src) {
    if (src.size() > target.size())
        return false;
    const size_t rank = target.size();
    const size_t src_rank = src.size();
    for (size_t i = 0; i < rank; ++i) {
        const size_t t = target[rank - 1 - i];
        const size_t s = (i < src_rank) ? src[src_rank - 1 - i] : 1;
        if (s != 1 && s != t) {
            return false;
        }
    }
    return true;
}

}  // namespace

using CreateFn = std::function<std::unique_ptr<MetalOp>(const std::shared_ptr<const ov::Node>&, void*, void*)>;

std::vector<CreateFn>& registry() {
    static std::vector<CreateFn> entries;
    return entries;
}

void register_ops_once() {
    static bool registered = false;
    if (registered)
        return;
    registered = true;

    auto& r = registry();

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(node);
        if (!conv)
            return nullptr;
        const auto in_pshape = conv->get_input_partial_shape(0);
        if (in_pshape.rank().is_static() && in_pshape.size() == 5 &&
            is_static_shape(conv->get_input_partial_shape(0)) &&
            is_static_shape(conv->get_input_partial_shape(1))) {
            return std::make_unique<MetalConv3DOp>(conv, device, queue);
        }
        return nullptr;
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        auto gconv = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node);
        if (!gconv)
            return nullptr;
        const auto in_pshape = gconv->get_input_partial_shape(0);
        const auto w_pshape = gconv->get_input_partial_shape(1);
        if (in_pshape.rank().is_static() && in_pshape.size() == 4 &&
            w_pshape.rank().is_static() && w_pshape.size() == 5 &&
            is_static_shape(in_pshape) && is_static_shape(w_pshape)) {
            return std::make_unique<MetalGroupConvOp>(gconv, device, queue);
        }
        return nullptr;
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(node);
        if (!conv)
            return nullptr;
        const auto in_pshape = conv->get_input_partial_shape(0);
        if (in_pshape.rank().is_static() && in_pshape.size() == 4 &&
            is_static_shape(conv->get_input_partial_shape(0)) &&
            is_static_shape(conv->get_input_partial_shape(1))) {
            return std::make_unique<MetalConvOp>(conv, device, queue);
        }
        return nullptr;
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::MatMul>(node))
            return nullptr;
        const auto& ps0 = node->get_input_partial_shape(0);
        const auto& ps1 = node->get_input_partial_shape(1);
        const auto et = node->get_output_element_type(0);
        const bool et_ok = (et == ov::element::f32 || et == ov::element::f16);
        if (ps0.rank().is_static() && ps1.rank().is_static() &&
            ps0.size() >= 2 && ps0.size() <= 4 && ps1.size() >= 2 && ps1.size() <= 4 &&
            et_ok && is_static_shape(ps0) && is_static_shape(ps1)) {
            return std::make_unique<MetalMatMulOp>(node, device, queue);
        }
        return nullptr;
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Relu>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        return std::make_unique<MetalReluOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Sigmoid>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        return std::make_unique<MetalSigmoidOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Tanh>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        return std::make_unique<MetalTanhOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        auto elu = ov::as_type_ptr<const ov::op::v0::Elu>(node);
        if (!elu)
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        return std::make_unique<MetalEluOp>(node, device, queue, static_cast<float>(elu->get_alpha()));
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Abs>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        return std::make_unique<MetalAbsOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Sign>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        return std::make_unique<MetalSignOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        auto clamp = ov::as_type_ptr<const ov::op::v0::Clamp>(node);
        if (!clamp)
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        return std::make_unique<MetalClampOp>(node, device, queue, clamp->get_min(), clamp->get_max());
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::LogicalNot>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        return std::make_unique<MetalLogicalNotOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Gelu>(node) &&
            !ov::as_type_ptr<const ov::op::v7::Gelu>(node)) {
            return nullptr;
        }
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        return std::make_unique<MetalGeluOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v4::Swish>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        return std::make_unique<MetalSwishOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v4::HSwish>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalHSwishOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v5::HSigmoid>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalHSigmoidOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v4::SoftPlus>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalSoftPlusOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v4::Mish>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalMishOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v9::SoftSign>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalSoftSignOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Exp>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalExpOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Log>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalLogOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Sqrt>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalSqrtOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Floor>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalFloorOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Ceiling>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalCeilOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Negative>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalNegativeOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Sin>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalSinOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Cos>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalCosOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Tan>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalTanOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Erf>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalErfOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Asin>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalAsinOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Acos>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalAcosOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Atan>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalAtanOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v3::Asinh>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalAsinhOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v3::Acosh>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalAcoshOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v3::Atanh>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalAtanhOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Sinh>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalSinhOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Cosh>(node))
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalCoshOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        auto round = ov::as_type_ptr<const ov::op::v5::Round>(node);
        if (!round)
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        if (round->get_mode() == ov::op::v5::Round::RoundMode::HALF_TO_EVEN) {
            return std::make_unique<MetalRoundEvenOp>(node, device, queue);
        }
        return std::make_unique<MetalRoundAwayOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::Add>(node))
            return nullptr;
        return std::make_unique<MetalAddOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::Subtract>(node))
            return nullptr;
        return std::make_unique<MetalSubOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::Multiply>(node))
            return nullptr;
        return std::make_unique<MetalMulOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::Divide>(node))
            return nullptr;
        return std::make_unique<MetalDivOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::Power>(node))
            return nullptr;
        return std::make_unique<MetalPowOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::Mod>(node))
            return nullptr;
        return std::make_unique<MetalModOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::FloorMod>(node))
            return nullptr;
        return std::make_unique<MetalFloorModOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::PRelu>(node))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        if (is_static_shape(node->get_input_partial_shape(0)) &&
            is_static_shape(node->get_input_partial_shape(1)) &&
            is_static_shape(node->get_output_partial_shape(0))) {
            const auto out_shape = node->get_output_shape(0);
            if (!is_broadcastable_to(out_shape, node->get_input_shape(0)) ||
                !is_broadcastable_to(out_shape, node->get_input_shape(1))) {
                return nullptr;
            }
        }
        return std::make_unique<MetalPreluOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::SquaredDifference>(node))
            return nullptr;
        return std::make_unique<MetalSquaredDiffOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::Minimum>(node))
            return nullptr;
        return std::make_unique<MetalMinOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::Maximum>(node))
            return nullptr;
        return std::make_unique<MetalMaxOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::LogicalAnd>(node))
            return nullptr;
        return std::make_unique<MetalLogicalAndOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::LogicalOr>(node))
            return nullptr;
        return std::make_unique<MetalLogicalOrOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::LogicalXor>(node))
            return nullptr;
        return std::make_unique<MetalLogicalXorOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::Equal>(node))
            return nullptr;
        return std::make_unique<MetalEqualOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::NotEqual>(node))
            return nullptr;
        return std::make_unique<MetalNotEqualOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::Less>(node))
            return nullptr;
        return std::make_unique<MetalLessOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::Greater>(node))
            return nullptr;
        return std::make_unique<MetalGreaterOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::LessEqual>(node))
            return nullptr;
        return std::make_unique<MetalLessEqualOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::GreaterEqual>(node))
            return nullptr;
        return std::make_unique<MetalGreaterEqualOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::ReduceSum>(node))
            return nullptr;
        auto axes_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(node->input_value(1).get_node_shared_ptr());
        if (!axes_const)
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)) || !is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        return std::make_unique<MetalReduceOp>(node, ReduceKind::Sum, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::ReduceMean>(node))
            return nullptr;
        auto axes_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(node->input_value(1).get_node_shared_ptr());
        if (!axes_const)
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)) || !is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        return std::make_unique<MetalReduceOp>(node, ReduceKind::Mean, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::ReduceMax>(node))
            return nullptr;
        auto axes_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(node->input_value(1).get_node_shared_ptr());
        if (!axes_const)
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)) || !is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        return std::make_unique<MetalReduceOp>(node, ReduceKind::Max, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::ReduceMin>(node))
            return nullptr;
        auto axes_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(node->input_value(1).get_node_shared_ptr());
        if (!axes_const)
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)) || !is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        return std::make_unique<MetalReduceOp>(node, ReduceKind::Min, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::ReduceProd>(node))
            return nullptr;
        auto axes_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(node->input_value(1).get_node_shared_ptr());
        if (!axes_const)
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)) || !is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        return std::make_unique<MetalReduceOp>(node, ReduceKind::Prod, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v4::ReduceL1>(node))
            return nullptr;
        auto axes_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(node->input_value(1).get_node_shared_ptr());
        if (!axes_const)
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)) || !is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalReduceOp>(node, ReduceKind::L1, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v4::ReduceL2>(node))
            return nullptr;
        auto axes_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(node->input_value(1).get_node_shared_ptr());
        if (!axes_const)
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)) || !is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalReduceOp>(node, ReduceKind::L2, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        auto pad_base = std::dynamic_pointer_cast<const ov::op::util::PadBase>(node);
        if (!pad_base)
            return nullptr;
        if (pad_base->get_pad_mode() != ov::op::PadMode::CONSTANT)
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)) ||
            !is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        auto pads_begin_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(
            node->input_value(1).get_node_shared_ptr());
        auto pads_end_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(
            node->input_value(2).get_node_shared_ptr());
        if (!pads_begin_const || !pads_end_const)
            return nullptr;
        auto pads_begin = pads_begin_const->cast_vector<int64_t>();
        auto pads_end = pads_end_const->cast_vector<int64_t>();
        if (pads_begin.size() != pads_end.size())
            return nullptr;
        for (size_t i = 0; i < pads_begin.size(); ++i) {
            if (pads_begin[i] < 0 || pads_end[i] < 0)
                return nullptr;
        }
        if (node->get_input_size() >= 4) {
            auto pad_val = std::dynamic_pointer_cast<const ov::op::v0::Constant>(
                node->input_value(3).get_node_shared_ptr());
            if (!pad_val)
                return nullptr;
        }
        return std::make_unique<MetalPadOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Tile>(node))
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)) ||
            !is_static_shape(node->get_output_partial_shape(0))) {
            return nullptr;
        }
        auto reps_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(
            node->input_value(1).get_node_shared_ptr());
        if (!reps_const)
            return nullptr;
        auto reps = reps_const->cast_vector<int64_t>();
        if (reps.size() != node->get_input_shape(0).size())
            return nullptr;
        for (auto r : reps) {
            if (r <= 0)
                return nullptr;
        }
        return std::make_unique<MetalTileOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        auto b3 = ov::as_type_ptr<const ov::op::v3::Broadcast>(node);
        auto b1 = ov::as_type_ptr<const ov::op::v1::Broadcast>(node);
        if (!b3 && !b1)
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)) ||
            !is_static_shape(node->get_output_partial_shape(0))) {
            return nullptr;
        }
        const auto out_shape = node->get_output_shape(0);
        const auto in_shape = node->get_input_shape(0);
        if (out_shape.size() > 8 || in_shape.size() > 8)
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32 &&
            et != ov::element::i32 && et != ov::element::i64) {
            return nullptr;
        }
        auto target_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(
            node->input_value(1).get_node_shared_ptr());
        if (!target_const)
            return nullptr;
        auto target = target_const->cast_vector<int64_t>();
        if (target.size() != out_shape.size())
            return nullptr;
        for (size_t i = 0; i < target.size(); ++i) {
            if (target[i] != static_cast<int64_t>(out_shape[i]))
                return nullptr;
        }
        if (b3) {
            auto spec = b3->get_broadcast_spec();
            if (spec.m_type == ov::op::BroadcastType::NONE || spec.m_type == ov::op::BroadcastType::EXPLICIT) {
                auto axes_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(
                    node->input_value(2).get_node_shared_ptr());
                if (!axes_const)
                    return nullptr;
            }
        }
        if (b1) {
            auto spec = b1->get_broadcast_spec();
            if (spec.m_type == ov::op::AutoBroadcastType::NONE) {
                auto axes_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(
                    node->input_value(2).get_node_shared_ptr());
                if (!axes_const)
                    return nullptr;
            }
        }
        return std::make_unique<MetalBroadcastOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        auto r4 = ov::as_type_ptr<const ov::op::v4::Range>(node);
        auto r0 = ov::as_type_ptr<const ov::op::v0::Range>(node);
        if (!r4 && !r0)
            return nullptr;
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto out_shape = node->get_output_shape(0);
        if (out_shape.size() != 1)
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32 &&
            et != ov::element::i32 && et != ov::element::i64) {
            return nullptr;
        }
        auto c0 = std::dynamic_pointer_cast<const ov::op::v0::Constant>(
            node->input_value(0).get_node_shared_ptr());
        auto c1 = std::dynamic_pointer_cast<const ov::op::v0::Constant>(
            node->input_value(1).get_node_shared_ptr());
        auto c2 = std::dynamic_pointer_cast<const ov::op::v0::Constant>(
            node->input_value(2).get_node_shared_ptr());
        if (!c0 || !c1 || !c2)
            return nullptr;
        return std::make_unique<MetalRangeOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        auto rev = ov::as_type_ptr<const ov::op::v1::Reverse>(node);
        if (!rev)
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)) ||
            !is_static_shape(node->get_output_partial_shape(0))) {
            return nullptr;
        }
        const auto in_shape = node->get_input_shape(0);
        if (in_shape.size() > ReverseCodegenDesc::kMaxDims)
            return nullptr;
        auto axes_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(
            node->input_value(1).get_node_shared_ptr());
        if (!axes_const)
            return nullptr;
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32 &&
            et != ov::element::i32 && et != ov::element::i64) {
            return nullptr;
        }
        return std::make_unique<MetalReverseOp>(rev, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::Select>(node))
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)) ||
            !is_static_shape(node->get_input_partial_shape(1)) ||
            !is_static_shape(node->get_input_partial_shape(2))) {
            return nullptr;
        }
        if (!is_static_shape(node->get_output_partial_shape(0)))
            return nullptr;
        const auto out_shape = node->get_output_shape(0);
        if (!is_broadcastable_to(out_shape, node->get_input_shape(0)) ||
            !is_broadcastable_to(out_shape, node->get_input_shape(1)) ||
            !is_broadcastable_to(out_shape, node->get_input_shape(2))) {
            return nullptr;
        }
        return std::make_unique<MetalSelectOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        auto mp = ov::as_type_ptr<const ov::op::v1::MaxPool>(node);
        if (!mp)
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)))
            return nullptr;
        return std::make_unique<MetalMaxPoolOp>(mp, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        auto ap = ov::as_type_ptr<const ov::op::v1::AvgPool>(node);
        if (!ap)
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)))
            return nullptr;
        return std::make_unique<MetalAvgPoolOp>(ap, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::Softmax>(node) &&
            !ov::as_type_ptr<const ov::op::v8::Softmax>(node)) {
            return nullptr;
        }
        return std::make_unique<MetalSoftmaxOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v5::LogSoftmax>(node))
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)) ||
            !is_static_shape(node->get_output_partial_shape(0))) {
            return nullptr;
        }
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        return std::make_unique<MetalSoftmaxOp>(node, device, queue, /*log_softmax=*/true);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        auto t1 = std::dynamic_pointer_cast<const ov::op::v1::TopK>(node);
        auto t3 = std::dynamic_pointer_cast<const ov::op::v3::TopK>(node);
        auto t11 = std::dynamic_pointer_cast<const ov::op::v11::TopK>(node);
        auto base = std::dynamic_pointer_cast<const ov::op::util::TopKBase>(node);
        if (!t1 && !t3 && !t11)
            return nullptr;
        if (!base)
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)) ||
            !is_static_shape(node->get_output_partial_shape(0)) ||
            !is_static_shape(node->get_output_partial_shape(1))) {
            return nullptr;
        }
        const auto et = node->get_output_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return nullptr;
        const auto idx_et = node->get_output_element_type(1);
        if (idx_et != ov::element::i32 && idx_et != ov::element::i64)
            return nullptr;
        const size_t k = base->get_k();
        if (k == 0)
            return nullptr;
        const auto in_shape = node->get_input_shape(0);
        if (in_shape.empty())
            return nullptr;
        const uint64_t axis = base->get_axis();
        if (axis >= in_shape.size())
            return nullptr;
        const uint64_t axis_len = in_shape[static_cast<size_t>(axis)];
        if (axis_len == 0 || k > axis_len)
            return nullptr;
        constexpr size_t kMaxK = 256;
        if (k > kMaxK)
            return nullptr;
        return std::make_unique<MetalTopKOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::Split>(node) &&
            !ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node)) {
            return nullptr;
        }
        if (!is_static_shape(node->get_input_partial_shape(0)))
            return nullptr;
        return std::make_unique<MetalSplitOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Concat>(node))
            return nullptr;
        const auto& pshape = node->get_output_partial_shape(0);
        if (pshape.rank().is_static() && is_static_shape(pshape)) {
            return std::make_unique<MetalConcatOp>(node, device, queue);
        }
        return nullptr;
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::Reshape>(node))
            return nullptr;
        return std::make_unique<MetalReshapeOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Squeeze>(node) &&
            !ov::as_type_ptr<const ov::op::v0::Unsqueeze>(node)) {
            return nullptr;
        }
        return std::make_unique<MetalReshapeOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        auto d2s = ov::as_type_ptr<const ov::op::v0::DepthToSpace>(node);
        if (!d2s)
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)) ||
            !is_static_shape(node->get_output_partial_shape(0))) {
            return nullptr;
        }
        const auto in_shape = node->get_input_shape(0);
        const auto out_shape = node->get_output_shape(0);
        if (in_shape.size() != 4 || out_shape.size() != 4)
            return nullptr;
        const size_t block = d2s->get_block_size();
        if (block == 0)
            return nullptr;
        if (in_shape[1] != out_shape[1] * block * block)
            return nullptr;
        if (out_shape[2] != in_shape[2] * block || out_shape[3] != in_shape[3] * block)
            return nullptr;
        return std::make_unique<MetalDepthToSpaceOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        auto s2d = ov::as_type_ptr<const ov::op::v0::SpaceToDepth>(node);
        if (!s2d)
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)) ||
            !is_static_shape(node->get_output_partial_shape(0))) {
            return nullptr;
        }
        const auto in_shape = node->get_input_shape(0);
        const auto out_shape = node->get_output_shape(0);
        if (in_shape.size() != 4 || out_shape.size() != 4)
            return nullptr;
        const size_t block = s2d->get_block_size();
        if (block == 0)
            return nullptr;
        if (in_shape[2] % block != 0 || in_shape[3] % block != 0)
            return nullptr;
        if (out_shape[1] != in_shape[1] * block * block)
            return nullptr;
        if (out_shape[2] != in_shape[2] / block || out_shape[3] != in_shape[3] / block)
            return nullptr;
        return std::make_unique<MetalSpaceToDepthOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        auto s3 = ov::as_type_ptr<const ov::op::v3::ScatterElementsUpdate>(node);
        auto s12 = ov::as_type_ptr<const ov::op::v12::ScatterElementsUpdate>(node);
        if (!s3 && !s12)
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)) ||
            !is_static_shape(node->get_input_partial_shape(1)) ||
            !is_static_shape(node->get_input_partial_shape(2)) ||
            !is_static_shape(node->get_output_partial_shape(0))) {
            return nullptr;
        }
        const auto data_shape = node->get_input_shape(0);
        const auto idx_shape = node->get_input_shape(1);
        const auto upd_shape = node->get_input_shape(2);
        if (data_shape.empty() || idx_shape.empty() || upd_shape.empty())
            return nullptr;
        if (data_shape.size() != idx_shape.size())
            return nullptr;
        if (idx_shape != upd_shape)
            return nullptr;
        if (data_shape.size() > ScatterElementsUpdateCodegenDesc::kMaxDims)
            return nullptr;
        auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(node->get_input_node_shared_ptr(3));
        if (!axis_const)
            return nullptr;
        if (s12) {
            if (s12->get_reduction() != ov::op::v12::ScatterElementsUpdate::Reduction::NONE)
                return nullptr;
            if (!s12->get_use_init_val())
                return nullptr;
        }
        return std::make_unique<MetalScatterElementsUpdateOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        auto s3 = ov::as_type_ptr<const ov::op::v3::ScatterNDUpdate>(node);
        auto s15 = ov::as_type_ptr<const ov::op::v15::ScatterNDUpdate>(node);
        if (!s3 && !s15)
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)) ||
            !is_static_shape(node->get_input_partial_shape(1)) ||
            !is_static_shape(node->get_input_partial_shape(2)) ||
            !is_static_shape(node->get_output_partial_shape(0))) {
            return nullptr;
        }
        const auto data_shape = node->get_input_shape(0);
        const auto idx_shape = node->get_input_shape(1);
        const auto upd_shape = node->get_input_shape(2);
        if (data_shape.empty() || idx_shape.empty() || upd_shape.empty())
            return nullptr;
        if (idx_shape.back() == 0 || idx_shape.back() > data_shape.size() ||
            idx_shape.back() > ScatterNDUpdateCodegenDesc::kMaxDims) {
            return nullptr;
        }
        if (s15) {
            if (s15->get_reduction() != ov::op::v15::ScatterNDUpdate::Reduction::NONE)
                return nullptr;
        }
        return std::make_unique<MetalScatterNDUpdateOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::Transpose>(node))
            return nullptr;
        auto perm_const = ov::as_type_ptr<const ov::op::v0::Constant>(node->input_value(1).get_node_shared_ptr());
        if (perm_const && is_static_shape(node->get_input_partial_shape(0)) &&
            is_static_shape(node->get_output_partial_shape(0))) {
            return std::make_unique<MetalTransposeOp>(node, device, queue);
        }
        return nullptr;
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Interpolate>(node) &&
            !ov::as_type_ptr<const ov::op::v4::Interpolate>(node) &&
            !ov::as_type_ptr<const ov::op::v11::Interpolate>(node)) {
            return nullptr;
        }
        if (!is_static_shape(node->get_input_partial_shape(0)) ||
            !is_static_shape(node->get_output_partial_shape(0))) {
            return nullptr;
        }
        if (node->get_input_partial_shape(0).rank().is_static() &&
            node->get_input_partial_shape(0).rank().get_length() != 4) {
            return nullptr;
        }
        return std::make_unique<MetalInterpolateOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v1::Gather>(node) &&
            !ov::as_type_ptr<const ov::op::v7::Gather>(node) &&
            !ov::as_type_ptr<const ov::op::v8::Gather>(node)) {
            return nullptr;
        }
        if (!is_static_shape(node->get_input_partial_shape(0)) ||
            !is_static_shape(node->get_input_partial_shape(1)) ||
            !is_static_shape(node->get_output_partial_shape(0))) {
            return nullptr;
        }
        auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(node->get_input_node_shared_ptr(2));
        if (!axis_const)
            return nullptr;
        return std::make_unique<MetalGatherOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v6::GatherElements>(node)) {
            return nullptr;
        }
        if (!is_static_shape(node->get_input_partial_shape(0)) ||
            !is_static_shape(node->get_input_partial_shape(1)) ||
            !is_static_shape(node->get_output_partial_shape(0))) {
            return nullptr;
        }
        const auto data_shape = node->get_input_shape(0);
        const auto idx_shape = node->get_input_shape(1);
        if (data_shape.empty() || idx_shape.empty())
            return nullptr;
        if (data_shape.size() != idx_shape.size())
            return nullptr;
        if (data_shape.size() > GatherElementsCodegenDesc::kMaxDims)
            return nullptr;
        return std::make_unique<MetalGatherElementsOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        auto g5 = ov::as_type_ptr<const ov::op::v5::GatherND>(node);
        auto g8 = ov::as_type_ptr<const ov::op::v8::GatherND>(node);
        if (!g5 && !g8)
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)) ||
            !is_static_shape(node->get_input_partial_shape(1)) ||
            !is_static_shape(node->get_output_partial_shape(0))) {
            return nullptr;
        }
        const size_t batch_dims = g5 ? g5->get_batch_dims() : g8->get_batch_dims();
        if (batch_dims != 0)
            return nullptr;
        const auto data_shape = node->get_input_shape(0);
        const auto idx_shape = node->get_input_shape(1);
        if (data_shape.empty() || idx_shape.empty())
            return nullptr;
        const size_t k = idx_shape.back();
        if (k == 0 || k > data_shape.size() || k > 8)
            return nullptr;
        return std::make_unique<MetalGatherNDOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v8::Slice>(node))
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)) ||
            !is_static_shape(node->get_output_partial_shape(0))) {
            return nullptr;
        }
        return std::make_unique<MetalSliceOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::ShapeOf>(node) &&
            !ov::as_type_ptr<const ov::op::v3::ShapeOf>(node))
            return nullptr;
        if (!node->get_input_partial_shape(0).rank().is_static())
            return nullptr;
        return std::make_unique<MetalShapeOfOp>(node, device, queue);
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::BatchNormInference>(node) &&
            !ov::as_type_ptr<const ov::op::v5::BatchNormInference>(node)) {
            return nullptr;
        }
        const auto& pshape = node->get_input_partial_shape(0);
        bool params_const = ov::as_type_ptr<const ov::op::v0::Constant>(node->input_value(1).get_node_shared_ptr()) &&
                            ov::as_type_ptr<const ov::op::v0::Constant>(node->input_value(2).get_node_shared_ptr()) &&
                            ov::as_type_ptr<const ov::op::v0::Constant>(node->input_value(3).get_node_shared_ptr()) &&
                            ov::as_type_ptr<const ov::op::v0::Constant>(node->input_value(4).get_node_shared_ptr());
        if (pshape.rank().is_static() && pshape.size() == 4 && params_const &&
            is_static_shape(pshape) &&
            node->get_output_element_type(0) == ov::element::f32) {
            return std::make_unique<MetalBatchNormOp>(node, device, queue);
        }
        return nullptr;
    });

    r.push_back([](const std::shared_ptr<const ov::Node>& node, void* device, void* queue) -> std::unique_ptr<MetalOp> {
        if (!ov::as_type_ptr<const ov::op::v0::Convert>(node))
            return nullptr;
        if (!is_static_shape(node->get_input_partial_shape(0)))
            return nullptr;
        auto src = node->get_input_element_type(0);
        auto dst = node->get_output_element_type(0);
        auto supported = [&](const ov::element::Type& t) {
            return t == ov::element::f32 || t == ov::element::f16 ||
                   t == ov::element::i32 || t == ov::element::i64 ||
                   t == ov::element::u8 || t == ov::element::i8;
        };
        if (supported(src) && supported(dst)) {
            return std::make_unique<MetalConvertOp>(node, device, queue);
        }
        return nullptr;
    });
}

std::unique_ptr<MetalOp> MetalOpFactory::create(const std::shared_ptr<const ov::Node>& node,
                                                void* device,
                                                void* queue) {
    MetalDeviceHandle dev = static_cast<MetalDeviceHandle>(device);
    MetalCommandQueueHandle q = static_cast<MetalCommandQueueHandle>(queue);
    if (!node || is_input_or_output(node))
        return nullptr;

    register_ops_once();
    for (const auto& entry : registry()) {
        if (auto op = entry(node, dev, q)) {
            return op;
        }
    }

    METAL_LOG_DEBUG("OpFactory",
                    "No MetalOp mapping for " << node->get_friendly_name()
                                              << " (" << node->get_type_name() << ")");
    return nullptr;
}

std::unique_ptr<MetalOp> MetalOpFactory::clone(const MetalOp& op) {
    if (auto p = dynamic_cast<const MetalConvOp*>(&op))
        return std::make_unique<MetalConvOp>(*p);
    if (auto p = dynamic_cast<const MetalConv3DOp*>(&op))
        return std::make_unique<MetalConv3DOp>(*p);
    if (auto p = dynamic_cast<const MetalGroupConvOp*>(&op))
        return std::make_unique<MetalGroupConvOp>(*p);
    if (auto p = dynamic_cast<const MetalMatMulOp*>(&op))
        return std::make_unique<MetalMatMulOp>(*p);
    if (auto p = dynamic_cast<const MetalActivationOp*>(&op))
        return std::make_unique<MetalActivationOp>(*p);
    if (auto p = dynamic_cast<const MetalElementwiseOp*>(&op))
        return std::make_unique<MetalElementwiseOp>(*p);
    if (auto p = dynamic_cast<const MetalPoolOp*>(&op))
        return std::make_unique<MetalPoolOp>(*p);
    if (auto p = dynamic_cast<const MetalSoftmaxOp*>(&op))
        return std::make_unique<MetalSoftmaxOp>(*p);
    if (auto p = dynamic_cast<const MetalSplitOp*>(&op))
        return std::make_unique<MetalSplitOp>(*p);
    if (auto p = dynamic_cast<const MetalConcatOp*>(&op))
        return std::make_unique<MetalConcatOp>(*p);
    if (auto p = dynamic_cast<const MetalReshapeOp*>(&op))
        return std::make_unique<MetalReshapeOp>(*p);
    if (auto p = dynamic_cast<const MetalTransposeOp*>(&op))
        return std::make_unique<MetalTransposeOp>(*p);
    if (auto p = dynamic_cast<const MetalBatchNormOp*>(&op))
        return std::make_unique<MetalBatchNormOp>(*p);
    if (auto p = dynamic_cast<const MetalConvertOp*>(&op))
        return std::make_unique<MetalConvertOp>(*p);
    if (auto p = dynamic_cast<const MetalInterpolateOp*>(&op))
        return std::make_unique<MetalInterpolateOp>(*p);
    if (auto p = dynamic_cast<const MetalDepthToSpaceOp*>(&op))
        return std::make_unique<MetalDepthToSpaceOp>(*p);
    if (auto p = dynamic_cast<const MetalSliceOp*>(&op))
        return std::make_unique<MetalSliceOp>(*p);
    if (auto p = dynamic_cast<const MetalGatherOp*>(&op))
        return std::make_unique<MetalGatherOp>(*p);
    if (auto p = dynamic_cast<const MetalGatherElementsOp*>(&op))
        return std::make_unique<MetalGatherElementsOp>(*p);
    if (auto p = dynamic_cast<const MetalGatherNDOp*>(&op))
        return std::make_unique<MetalGatherNDOp>(*p);
    if (auto p = dynamic_cast<const MetalScatterElementsUpdateOp*>(&op))
        return std::make_unique<MetalScatterElementsUpdateOp>(*p);
    if (auto p = dynamic_cast<const MetalScatterNDUpdateOp*>(&op))
        return std::make_unique<MetalScatterNDUpdateOp>(*p);
    if (auto p = dynamic_cast<const MetalSpaceToDepthOp*>(&op))
        return std::make_unique<MetalSpaceToDepthOp>(*p);
    if (auto p = dynamic_cast<const MetalShapeOfOp*>(&op))
        return std::make_unique<MetalShapeOfOp>(*p);
    if (auto p = dynamic_cast<const MetalSelectOp*>(&op))
        return std::make_unique<MetalSelectOp>(*p);
    if (auto p = dynamic_cast<const MetalReduceOp*>(&op))
        return std::make_unique<MetalReduceOp>(*p);
    if (auto p = dynamic_cast<const MetalPadOp*>(&op))
        return std::make_unique<MetalPadOp>(*p);
    if (auto p = dynamic_cast<const MetalTileOp*>(&op))
        return std::make_unique<MetalTileOp>(*p);
    if (auto p = dynamic_cast<const MetalBroadcastOp*>(&op))
        return std::make_unique<MetalBroadcastOp>(*p);
    if (auto p = dynamic_cast<const MetalRangeOp*>(&op))
        return std::make_unique<MetalRangeOp>(*p);
    if (auto p = dynamic_cast<const MetalReverseOp*>(&op))
        return std::make_unique<MetalReverseOp>(*p);
    if (auto p = dynamic_cast<const MetalTopKOp*>(&op))
        return std::make_unique<MetalTopKOp>(*p);
    METAL_LOG_WARN("OpFactory", "Clone requested for unsupported MetalOp type: " << op.type());
    return nullptr;
}

}  // namespace metal_plugin
}  // namespace ov
