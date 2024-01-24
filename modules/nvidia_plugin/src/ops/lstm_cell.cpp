// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_cell.hpp"

#include <cuda_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <utility>
#include <vector>

namespace ov {
namespace nvidia_gpu {

LSTMCellOp::LSTMCellOp(const CreationContext& context,
                       const ov::Node& node,
                       IndexCollection&& inputIds,
                       IndexCollection&& outputIds)
    : OperationCuDnn(context, node, std::move(inputIds), std::move(outputIds)),
      params_{node},
      descs_{{context, params_}} {}

void LSTMCellOp::Execute(const InferenceRequestContext& context,
                         Inputs inputs,
                         Outputs outputs,
                         const Workbuffers& workbuffers) const {
    OPENVINO_ASSERT(inputs.size() == 6, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 2, "Node name: ", GetName());

    const auto& ib = workbuffers.immutable_buffers;
    const auto& mb = workbuffers.mutable_buffers;
    OPENVINO_ASSERT(ib.size() == 1 || ib.size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(mb.size() == 1 || mb.size() == 2, "Node name: ", GetName());

    const auto weight_space = ib.size() > 1 ? ib[1].get() : nullptr;
    auto y_output = mb[0].get();
    auto work_space = mb.size() > 1 ? mb[1].get() : nullptr;

    context.getThreadContext().dnnHandle().rnnForward(descs_.rnnDesc(),
                                                      descs_.dnnForwardMode(),
                                                      static_cast<const int32_t*>(ib[0].get()),
                                                      descs_.xDesc(),
                                                      inputs[0].get(),
                                                      descs_.yDesc(),
                                                      y_output,
                                                      descs_.hDesc(),
                                                      inputs[1].get(),
                                                      outputs[0].get(),
                                                      descs_.cDesc(),
                                                      inputs[2].get(),
                                                      outputs[1].get(),
                                                      descs_.weightSpaceSize(),
                                                      weight_space,
                                                      descs_.workSpaceSize(),
                                                      work_space,
                                                      0,
                                                      nullptr);
}

CudaGraphCompatibility LSTMCellOp::GetCudaGraphCompatibility() const { return CudaGraphCompatibility::FULL; }

void LSTMCellOp::InitSharedImmutableWorkbuffers(const IOperationExec::Buffers& buffers) {
    OPENVINO_ASSERT(buffers.size() == 1 || buffers.size() == 2, "Node name: ", GetName());

    descs_.initDevSeqLengthArray(buffers[0]);

    if (buffers.size() == 1) {
        return;
    }
    descs_.initWeightSpace(buffers[1]);
}

WorkbufferRequest LSTMCellOp::GetWorkBufferRequest() const {
    std::vector<WorkbufferRequest::size_in_bytes_t> immut_sizes;
    immut_sizes.push_back(descs_.seqLengthArraySizeBytes());
    const auto weight_space_size = descs_.weightSpaceSize();
    if (weight_space_size != 0) {
        immut_sizes.push_back(weight_space_size);
    }

    std::vector<WorkbufferRequest::size_in_bytes_t> mut_sizes;
    mut_sizes.push_back(descs_.ySizeBytes());
    const auto work_space_size = descs_.workSpaceSize();
    if (work_space_size != 0) {
        mut_sizes.push_back(work_space_size);
    }

    return {std::move(immut_sizes), std::move(mut_sizes)};
}

OPERATION_REGISTER(LSTMCellOp, LSTMCell);

}  // namespace nvidia_gpu
}  // namespace ov
