// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gru_cell.hpp"

#include <cuda_operation_registry.hpp>
#include <gsl/gsl_assert>
#include <optional>
#include <utility>
#include <vector>

namespace CUDAPlugin {

GRUCellOp::GRUCellOp(const CreationContext& context,
                     const ov::Node& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds)
    : OperationCuDnn(context, node, std::move(inputIds), std::move(outputIds)),
      params_{dynamic_cast<const ov::op::v3::GRUCell&>(node)},
      descs_{{context, params_}} {}

void GRUCellOp::Execute(const InferenceRequestContext& context,
                        Inputs inputs,
                        Outputs outputs,
                        const Workbuffers& workbuffers) const {
    using CUDAPlugin::RNN::Details::GRUCellArgIndices;

    Expects(inputs.size() == 5);
    Expects(outputs.size() == 1);

    const auto& ib = workbuffers.immutable_buffers;
    const auto& mb = workbuffers.mutable_buffers;
    Expects(ib.size() == 1 || ib.size() == 2);
    Expects(mb.size() == 1 || mb.size() == 2);

    const auto dev_seq_lenghts = static_cast<const int32_t*>(ib[0].get());
    const auto weight_space = ib.size() > 1 ? ib[1].get() : nullptr;
    auto y_output = mb[0].get();
    auto work_space = mb.size() > 1 ? mb[1].get() : nullptr;

    context.getThreadContext().dnnHandle().rnnForward(descs_.rnnDesc(),
                                                      descs_.dnnForwardMode(),
                                                      dev_seq_lenghts,
                                                      descs_.xDesc(),
                                                      inputs[GRUCellArgIndices::x].get(),
                                                      descs_.yDesc(),
                                                      y_output,
                                                      descs_.hDesc(),
                                                      inputs[GRUCellArgIndices::hidden_input].get(),
                                                      outputs[GRUCellArgIndices::hidden_output].get(),
                                                      std::nullopt,
                                                      nullptr,
                                                      nullptr,
                                                      descs_.weightSpaceSize(),
                                                      weight_space,
                                                      descs_.workSpaceSize(),
                                                      work_space,
                                                      0,
                                                      nullptr);
}

void GRUCellOp::InitSharedImmutableWorkbuffers(const IOperationExec::Buffers& buffers) {
    Expects(buffers.size() == 1 || buffers.size() == 2);

    descs_.initDevSeqLengthArray(buffers[0]);

    if (buffers.size() == 1) {
        return;
    }
    descs_.initWeightSpace(buffers[1]);
}

WorkbufferRequest GRUCellOp::GetWorkBufferRequest() const {
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

OPERATION_REGISTER(GRUCellOp, GRUCell);

}  // namespace CUDAPlugin
