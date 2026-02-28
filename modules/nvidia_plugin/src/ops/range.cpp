// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "range.hpp"

#include <fmt/format.h>

#include <cuda_operation_registry.hpp>

#include "openvino/op/constant.hpp"
#include "openvino/op/range.hpp"

#include "converters.hpp"
#include "kernels/details/cuda_type_traits.hpp"
#include "kernels/range.hpp"

namespace ov {
namespace nvidia_gpu {

static constexpr auto OUTPUT_INDX = 0u;

RangeOp::RangeOp(const CreationContext& context,
                 const ov::Node& node,
                 IndexCollection&& inputIds,
                 IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)),
      output_size_(shape_size(node.get_output_shape(OUTPUT_INDX))) {
    if (ov::as_type_ptr<op::v0::Constant>(node.get_input_node_shared_ptr(START_INDX)) == nullptr ||
        ov::as_type_ptr<op::v0::Constant>(node.get_input_node_shared_ptr(STOP_INDX)) == nullptr ||
        ov::as_type_ptr<op::v0::Constant>(node.get_input_node_shared_ptr(STEP_INDX)) == nullptr) {
        // TODO: Implement the dynamic shapes support for the Range operation
        throw_ov_exception("The dynamic shape is not supported for Range operation. All Range inputs must be constants.");
    }
    auto inputStart_type = node.get_input_element_type(START_INDX);
    auto inputStop_type = node.get_input_element_type(STOP_INDX);
    auto inputStep_type = node.get_input_element_type(STEP_INDX);
    auto output_type = node.get_output_element_type(OUTPUT_INDX);
    size_t max_size = shape_size(node.get_output_shape(OUTPUT_INDX));
    const auto& prop = context.device().props();
    unsigned max_threads_per_block = prop.maxThreadsPerBlock;
    unsigned blocks_number = 1 + max_size / max_threads_per_block;
    unsigned threads_per_block = (blocks_number == 1) ? max_size : max_threads_per_block;
    kernel_op_ = kernel::RangeKernelOp(max_size,
                                       blocks_number,
                                       threads_per_block,
                                       convertDataType<ov::nvidia_gpu::kernel::Type_t>(inputStart_type),
                                       convertDataType<ov::nvidia_gpu::kernel::Type_t>(inputStop_type),
                                       convertDataType<ov::nvidia_gpu::kernel::Type_t>(inputStep_type),
                                       convertDataType<ov::nvidia_gpu::kernel::Type_t>(output_type));
}

void RangeOp::Execute(const InferenceRequestContext& context,
                      Inputs inputs,
                      Outputs outputs,
                      const Workbuffers& workbuffers) const {
    OPENVINO_ASSERT(inputs.size() == 3, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(kernel_op_, "Node name: ", GetName());
    (*kernel_op_)(context.getThreadContext().stream().get(),
                  inputs[START_INDX].get(),
                  inputs[STEP_INDX].get(),
                  output_size_,
                  outputs[OUTPUT_INDX].get());
}

CudaGraphCompatibility RangeOp::GetCudaGraphCompatibilityImpl() const { return CudaGraphCompatibility::FULL; }

OPERATION_REGISTER(RangeOp, Range);
}  // namespace nvidia_gpu
}  // namespace ov
