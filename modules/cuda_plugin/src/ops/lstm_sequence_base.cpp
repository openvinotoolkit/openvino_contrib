// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_sequence_base.hpp"

#include <gsl/gsl_assert>
#include <utility>
#include <vector>

namespace CUDAPlugin {

LSTMSequenceOpBase::LSTMSequenceOpBase(const CreationContext& context,
                                       const LSTMSequenceParams& params,
                                       const Config& config,
                                       const NodeOp& node,
                                       IndexCollection&& inputIds,
                                       IndexCollection&& outputIds)
    : OperationCuDnn(context, node, std::move(inputIds), std::move(outputIds)),
      params_{params},
      descs_{context, params_, config} {}

void LSTMSequenceOpBase::Execute(const InferenceRequestContext& context,
                                 Inputs inputs,
                                 Outputs outputs,
                                 const Workbuffers& workbuffers) const {
    using ArgIndices = CUDAPlugin::RNN::Details::LSTMSequenceArgIndices;
    Expects(inputs.size() == 7);
    Expects(outputs.size() == 3);

    const auto& ib = workbuffers.immutable_buffers;
    const auto& mb = workbuffers.mutable_buffers;

    if (x_adapter) x_adapter->execute(context, inputs[ArgIndices::x], mb);
    if (hx_adapter) hx_adapter->execute(context, inputs[ArgIndices::hidden_input], mb);
    if (cx_adapter) cx_adapter->execute(context, inputs[ArgIndices::cell_input], mb);

    const void* const api_x = x_adapter ? x_adapter->dnnApiPtr(mb) : inputs[ArgIndices::x].get();
    const void* const api_hx = hx_adapter ? hx_adapter->dnnApiPtr(mb) : inputs[ArgIndices::hidden_input].get();
    const void* const api_cx = cx_adapter ? cx_adapter->dnnApiPtr(mb) : inputs[ArgIndices::cell_input].get();

    void* const api_y = y_adapter ? y_adapter->dnnApiPtr(mb) : outputs[ArgIndices::y].get();
    void* const api_hy = hy_adapter ? hy_adapter->dnnApiPtr(mb) : outputs[ArgIndices::hidden_output].get();
    void* const api_cy = cy_adapter ? cy_adapter->dnnApiPtr(mb) : outputs[ArgIndices::cell_output].get();

    const auto& dnnHandle = context.getThreadContext().dnnHandle();
    dnnHandle.rnnForward(descs_.rnnDesc(),
                         descs_.dnnForwardMode(),
                         static_cast<const int32_t*>(ib_seq_lengths_.requiredPtr(ib)),
                         descs_.xDesc(),
                         api_x,
                         descs_.yDesc(),
                         api_y,
                         descs_.hDesc(),
                         api_hx,
                         api_hy,
                         descs_.cDesc(),
                         api_cx,
                         api_cy,
                         ib_weight_space_.size(),
                         ib_weight_space_.requiredPtr(ib),
                         mb_work_space_.size(),
                         mb_work_space_.optionalPtr(mb),
                         0,
                         nullptr);

    if (y_adapter) y_adapter->execute(context, mb, outputs[ArgIndices::y]);
    if (hy_adapter) hy_adapter->execute(context, mb, outputs[ArgIndices::hidden_output]);
    if (cy_adapter) cy_adapter->execute(context, mb, outputs[ArgIndices::cell_output]);
}

void LSTMSequenceOpBase::InitSharedImmutableWorkbuffers(const IOperationExec::Buffers& buffers) {
    descs_.initDevSeqLengthArray(CUDA::DevicePointer<void*>{ib_seq_lengths_.requiredPtr(buffers)});
    descs_.initWeightSpace(CUDA::DevicePointer<void*>{ib_weight_space_.requiredPtr(buffers)});
}

WorkbufferRequest LSTMSequenceOpBase::GetWorkBufferRequest() const {
    std::vector<WorkbufferRequest::size_in_bytes_t> immut_sizes;
    ib_seq_lengths_.addRequest(immut_sizes, descs_.seqLengthArraySizeBytes());
    ib_weight_space_.addRequest(immut_sizes, descs_.weightSpaceSize());

    std::vector<WorkbufferRequest::size_in_bytes_t> mut_sizes;
    mb_work_space_.addRequest(mut_sizes, descs_.workSpaceSize());

    if (x_adapter) x_adapter->requestWorkbuffer(mut_sizes);
    if (hx_adapter) hx_adapter->requestWorkbuffer(mut_sizes);
    if (cx_adapter) cx_adapter->requestWorkbuffer(mut_sizes);
    if (y_adapter) y_adapter->requestWorkbuffer(mut_sizes);
    if (hy_adapter) hy_adapter->requestWorkbuffer(mut_sizes);
    if (cy_adapter) cy_adapter->requestWorkbuffer(mut_sizes);

    return {std::move(immut_sizes), std::move(mut_sizes)};
}

}  // namespace CUDAPlugin
