// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <openvino/core/except.hpp>
#include <openvino/op/constant.hpp>
#include <sstream>

#include "cuda_operation_registry.hpp"
#include "interpolate_cubic.hpp"
#include "interpolate_linear.hpp"
#include "interpolate_nearest.hpp"

namespace ov {
namespace nvidia_gpu {

static OperationBase::Ptr interpolateFactory(const CreationContext& context,
                                             const std::shared_ptr<ov::Node>& in_node,
                                             OperationBase::IndexCollection&& inputIds,
                                             OperationBase::IndexCollection&& outputIds) {
    auto node = std::dynamic_pointer_cast<ov::op::v4::Interpolate>(in_node);
    OPENVINO_ASSERT(node);

    using InterpolateMode = ov::op::v4::Interpolate::InterpolateMode;
    using IndexCollection = OperationBase::IndexCollection;
    std::stringstream exception_msg;

    switch (node->get_attrs().mode) {
        case InterpolateMode::NEAREST:
            try {
                return std::make_shared<InterpolateNearestOp>(
                    context, *node, IndexCollection{inputIds}, IndexCollection{outputIds});
            } catch (const std::exception& e) {
                exception_msg << "failed to create InterpolateNearestOp: " << e.what() << "\n";
            }
            break;
        case InterpolateMode::LINEAR:
            try {
                return std::make_shared<InterpolateLinearOp>(
                    context, *node, IndexCollection{inputIds}, IndexCollection{outputIds});
            } catch (const std::exception& e) {
                exception_msg << "failed to create InterpolateLinearOp: " << e.what() << "\n";
            }
            break;
        case InterpolateMode::CUBIC:
            try {
                return std::make_shared<InterpolateCubicOp>(
                    context, *node, IndexCollection{inputIds}, IndexCollection{outputIds});
            } catch (const std::exception& e) {
                exception_msg << "failed to create InterpolateCubicOp: " << e.what() << "\n";
            }
            break;
        default:
            exception_msg << "not implemented.\n";
            break;
    };

    throw_ov_exception(fmt::format("Interpolate node is not supported: {}", exception_msg.str()));
}

OPERATION_REGISTER_FACTORY(interpolateFactory, Interpolate);

}  // namespace nvidia_gpu
}  // namespace ov
