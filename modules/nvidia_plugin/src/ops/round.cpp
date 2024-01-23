// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernels/round.hpp"

#include <cuda_operation_registry.hpp>

#include "converters.hpp"
#include "openvino/op/round.hpp"
#include "round.hpp"

namespace ov {
namespace nvidia_gpu {

RoundOp::RoundOp(const CreationContext& context,
                 const NodeOp& node,
                 IndexCollection&& inputIds,
                 IndexCollection&& outputIds)
    : OperationBase{context, node, move(inputIds), move(outputIds)} {
    OPENVINO_ASSERT(node.get_input_size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());

    const auto& element_type = node.get_input_element_type(0);
    const auto& out_element_type = node.get_output_element_type(0);
    if (out_element_type != element_type) {
        throw_ov_exception(
            fmt::format("RoundOp: output type should be the same as input type, input type: {}, output type: {}",
                        element_type.get_type_name(),
                        out_element_type.get_type_name()));
    }
    const size_t num_elements = ov::shape_size(node.get_input_shape(0));
    OPENVINO_ASSERT(ov::shape_size(node.get_output_shape(0)) == num_elements, "Node name: ", GetName());

    const size_t max_threads_per_block = context.device().props().maxThreadsPerBlock;
    kernel_ =
        kernel::Round{convertDataType<ov::nvidia_gpu::kernel::Type_t>(element_type), max_threads_per_block, num_elements};
}

void RoundOp::Execute(const InferenceRequestContext& context,
                      Inputs inputTensors,
                      Outputs outputTensors,
                      const Workbuffers& workbuffers) const {
    OPENVINO_ASSERT(kernel_, "Node name: ", GetName());
    OPENVINO_ASSERT(inputTensors.size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(outputTensors.size() == 1, "Node name: ", GetName());

    (*kernel_)(context.getThreadContext().stream().get(), inputTensors[0].get(), outputTensors[0].get());
}

CudaGraphCompatibility RoundOp::GetCudaGraphCompatibility() const { return CudaGraphCompatibility::FULL; }

OPERATION_REGISTER(RoundOp, Round);

}  // namespace nvidia_gpu
}  // namespace ov
