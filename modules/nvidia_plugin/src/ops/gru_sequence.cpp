// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gru_sequence.hpp"

#include <cuda_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <utility>
#include <vector>

namespace ov {
namespace nvidia_gpu {

GRUSequenceOp::GRUSequenceOp(const CreationContext& context,
                             const NodeOp& node,
                             IndexCollection&& inputIds,
                             IndexCollection&& outputIds)
    : OperationCuDnn(context, node, std::move(inputIds), std::move(outputIds)),
      params_{node},
      descs_{context, params_, config()},
      graph_compatibility_{RNN::Details::isRNNSequenceCudaGraphCompatible(context.device())
                               ? CudaGraphCompatibility::FULL
                               : CudaGraphCompatibility::NONE} {
    ib_seq_lengths_.addRequest(immut_sizes_, descs_.seqLengthArraySizeBytes());
    ib_weight_space_.addRequest(immut_sizes_, descs_.weightSpaceSize());

    mb_work_space_.addRequest(mut_sizes_, descs_.workSpaceSize());
}

GRUSequenceOp::Config GRUSequenceOp::config() {
    GRUSequenceOp::Config config{};
    config.rnn_data_layout = CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
    return config;
}

void GRUSequenceOp::Execute(const InferenceRequestContext& context,
                            Inputs inputs,
                            Outputs outputs,
                            const Workbuffers& workbuffers) const {
    using ArgIndices = ov::nvidia_gpu::RNN::Details::GRUSequenceArgIndices;
    OPENVINO_ASSERT(inputs.size() == 6, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 2, "Node name: ", GetName());

    const auto& ib = workbuffers.immutable_buffers;
    const auto& mb = workbuffers.mutable_buffers;

    const void* const api_x = inputs[ArgIndices::x].get();
    const void* const api_hx = inputs[ArgIndices::hidden_input].get();

    void* const api_y = outputs[ArgIndices::y].get();
    void* const api_hy = outputs[ArgIndices::hidden_output].get();

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
                         std::nullopt,
                         nullptr,
                         nullptr,
                         ib_weight_space_.size(),
                         ib_weight_space_.requiredPtr(ib),
                         mb_work_space_.size(),
                         mb_work_space_.optionalPtr(mb),
                         0,
                         nullptr);
}

CudaGraphCompatibility GRUSequenceOp::GetCudaGraphCompatibility() const { return graph_compatibility_; }

void GRUSequenceOp::InitSharedImmutableWorkbuffers(const IOperationExec::Buffers& buffers) {
    descs_.initDevSeqLengthArray(CUDA::DevicePointer<void*>{ib_seq_lengths_.requiredPtr(buffers)});
    descs_.initWeightSpace(CUDA::DevicePointer<void*>{ib_weight_space_.requiredPtr(buffers)});
}

WorkbufferRequest GRUSequenceOp::GetWorkBufferRequest() const { return {immut_sizes_, mut_sizes_}; }

OPERATION_REGISTER(GRUSequenceOp, GRUSequence);

}  // namespace nvidia_gpu
}  // namespace ov
