// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <cuda_operation_base.hpp>
#include <cuda_operation_registry.hpp>
#include <openvino/op/clamp.hpp>
#include <sstream>

#include "clamp_cuda.hpp"
#include "clamp_cudnn.hpp"
#include "clipped_relu_cudnn.hpp"

namespace ov {
namespace nvidia_gpu {

using IndexCollection = OperationBase::IndexCollection;

static OperationBase::Ptr clampFactory(const CreationContext& context,
                                       const std::shared_ptr<ov::Node>& node,
                                       IndexCollection&& inputIds,
                                       IndexCollection&& outputIds) {
    const ov::op::v0::Clamp& node_op{downcast<const ov::op::v0::Clamp>(node)};

    const IndexCollection inputs{inputIds};
    const IndexCollection outputs{outputIds};

    std::stringstream exception_msg;
    try {
        return std::make_shared<ClippedReluCuDnnOp>(
            context, node_op, IndexCollection{inputIds}, IndexCollection{outputIds});
    } catch (const std::exception& e) {
        exception_msg << "Failed to create ClippedReluCuDnnOp implementation: " << e.what();
    }
    // TODO: ClampCuDnnOp is disabled now due to performance lower then both, ClippedReluCuDnnOp and ClampCudaOp
    // versions (CUDA 11.2 + cuDNN 8.1.0).
    // It may be enabled in the future if becomes faster in a newer cuDNN version
    //
    // try {
    //     return std::make_shared<ClampCuDnnOp>(context, node_op, IndexCollection{inputIds},
    //     IndexCollection{outputIds});
    // } catch (const std::exception& e) {
    //     exception_msg << "\nFailed to create ClampCuDnnOp implementation: " << e.what();
    // }
    try {
        return std::make_shared<ClampCudaOp>(context, node_op, IndexCollection{inputIds}, IndexCollection{outputIds});
    } catch (const std::exception& e) {
        exception_msg << "\nFailed to create ClampCudaOp implementation: " << e.what();
    }
    throw_ov_exception(fmt::format("Clamp node is not supported:\n{}", exception_msg.str()));
}

OPERATION_REGISTER_FACTORY(clampFactory, Clamp)

}  // namespace nvidia_gpu
}  // namespace ov
