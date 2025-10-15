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
class ElementwiseBinaryOp : public OperationBase {
public:
    using NodeOp = nGraphNode;
    ElementwiseBinaryOp(const CreationContext& context,
                        const NodeOp& node,
                        IndexCollection&& inputIds,
                        IndexCollection&& outputIds)
        : OperationBase{context, node, move(inputIds), move(outputIds)},
          in0_broadcast_params_{NumpyBroadcastParams::create(node.get_input_shape(0), node.get_output_shape(0))},
          in1_broadcast_params_{NumpyBroadcastParams::create(node.get_input_shape(1), node.get_output_shape(0))} {
        OPENVINO_ASSERT(node.get_input_size() == 2, "Node name: ", GetName());
        OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());

        const auto element_type = node.get_output_element_type(0);
        const bool types_are_expected =
            (element_type == node.get_input_element_type(0)) && (element_type == node.get_input_element_type(1));
        if (!types_are_expected) {
            throw_ov_exception("Element types combination is not supported");
        }

        in0_broadcast_params_->addWorkbufferRequests(immutable_buffer_sizes_);
        in1_broadcast_params_->addWorkbufferRequests(immutable_buffer_sizes_);

        const size_t max_threads_per_block = context.device().props().maxThreadsPerBlock;
        const size_t out_num_elements = ov::shape_size(node.get_output_shape(0));
        kernel_ = Kernel{
            convertDataType<ov::nvidia_gpu::kernel::Type_t>(element_type), out_num_elements, max_threads_per_block};
    }

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override {
        OPENVINO_ASSERT(kernel_, "Node name: ", GetName());
        OPENVINO_ASSERT(inputTensors.size() == 2, "Node name: ", GetName());
        OPENVINO_ASSERT(outputTensors.size() == 1, "Node name: ", GetName());
        auto& stream = context.getThreadContext().stream();

        (*kernel_)(stream.get(),
                   static_cast<const void*>(inputTensors[0].get()),
                   in0_broadcast_params_->mapper(workbuffers.immutable_buffers),
                   static_cast<const void*>(inputTensors[1].get()),
                   in1_broadcast_params_->mapper(workbuffers.immutable_buffers),
                   static_cast<void*>(outputTensors[0].get()));
    }

    CudaGraphCompatibility GetCudaGraphCompatibility() const override { return CudaGraphCompatibility::FULL; }

    void InitSharedImmutableWorkbuffers(const IOperationExec::Buffers& buffers) override {
        in0_broadcast_params_->initWorkbuffers(buffers);
        in1_broadcast_params_->initWorkbuffers(buffers);
    }

    WorkbufferRequest GetWorkBufferRequest() const override { return {immutable_buffer_sizes_, {}}; }

private:
    std::vector<WorkbufferRequest::size_in_bytes_t> immutable_buffer_sizes_;
    std::unique_ptr<NumpyBroadcastParams> in0_broadcast_params_;
    std::unique_ptr<NumpyBroadcastParams> in1_broadcast_params_;

    std::optional<Kernel> kernel_;
};

}  // namespace nvidia_gpu
}  // namespace ov
