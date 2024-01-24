// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "components/numpy_broadcast_params.h"
#include "converters.hpp"

namespace ov {
namespace nvidia_gpu {

template <typename nGraphNode, typename Kernel>
class ElementwiseUnaryOp : public OperationBase {
public:
    using NodeOp = nGraphNode;
    ElementwiseUnaryOp(const CreationContext& context,
                       const NodeOp& node,
                       IndexCollection&& inputIds,
                       IndexCollection&& outputIds)
        : OperationBase{context, node, move(inputIds), move(outputIds)} {
        OPENVINO_ASSERT(node.get_input_size() == 1, "Node name: ", GetName());
        OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());
        const auto input_element_type = node.get_input_element_type(0);
        const auto output_element_type = node.get_output_element_type(0);
        OPENVINO_ASSERT(input_element_type == output_element_type, "Node name: ", GetName());
        const auto input_shape = node.get_input_shape(0);
        const auto output_shape = node.get_output_shape(0);
        OPENVINO_ASSERT(input_shape == output_shape, "Node name: ", GetName());
        size_t num_elements = ov::shape_size(input_shape);
        const size_t max_threads_per_block = context.device().props().maxThreadsPerBlock;
        kernel_ = Kernel{
            convertDataType<ov::nvidia_gpu::kernel::Type_t>(input_element_type), max_threads_per_block, num_elements};
    }

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override {
        OPENVINO_ASSERT(kernel_, "Node name: ", GetName());
        OPENVINO_ASSERT(inputTensors.size() == 1, "Node name: ", GetName());
        OPENVINO_ASSERT(outputTensors.size() == 1, "Node name: ", GetName());
        const auto& stream = context.getThreadContext().stream();
        (*kernel_)(stream.get(), inputTensors[0].get(), outputTensors[0].get());
    }

    CudaGraphCompatibility GetCudaGraphCompatibility() const override { return CudaGraphCompatibility::FULL; }

private:
    std::optional<Kernel> kernel_;
};

}  // namespace nvidia_gpu
}  // namespace ov
