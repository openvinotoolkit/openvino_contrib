// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "pad.hpp"

#include <ngraph/validation_util.hpp>
#include <cuda/runtime.hpp>
#include <cuda_operation_registry.hpp>
#include <gsl/gsl_assert>
#include <memory>
#include <cstddef>

#include "converters.hpp"

namespace CUDAPlugin {

static bool isNCHWConvolutionPadding(const PadOp::NodeOp& node) {
    auto padsBegin = ov::get_constant_from_source(node.input_value(1));
    auto padsEnd = ov::get_constant_from_source(node.input_value(2));
    const auto padsBeginCoord = padsBegin->cast_vector<size_t>();
    const auto padsEndCoord = padsEnd->cast_vector<size_t>();
    return node.get_input_shape(0).size() == 4 && padsBeginCoord[0] == 0 && padsBeginCoord[1] == 0 &&
           padsEndCoord[0] == 0 && padsEndCoord[1] == 0;
}

PadOp::PadOp(const CreationContext& context,
             const NodeOp& node,
             IndexCollection&& inputIds,
             IndexCollection&& outputIds)
    : OperationBase{context, node, move(inputIds), move(outputIds)},
      kernel_{eltwise::KernelExecAttrs{
                  node.get_output_shape(0),
                  kernel::ConstModePad::kWarpsPerBlock * static_cast<unsigned>(context.device().props().warpSize),
                  kernel::ConstModePad::kElementsPerThread},
              node.get_output_element_type(0),
              node.get_output_shape(0).size(),
              context.device().props().maxThreadsPerBlock,
              ov::shape_size(node.get_output_shape(0)),
              isNCHWConvolutionPadding(node)},
      src_shape_{node.get_input_shape(0)},
      dst_shape_{node.get_output_shape(0)} {
    Expects(ov::op::PadMode::CONSTANT == node.get_pad_mode());
}

void PadOp::Execute(const InferenceRequestContext& context,
                    Inputs inputTensors,
                    Outputs outputTensors,
                    const Workbuffers& workbuffers) const {
    kernel_(context.getThreadContext().stream().get(),
            inputTensors[InputIndex::kSrc].get(),
            outputTensors[OutputIndex::kDst].get(),
            inputTensors[InputIndex::kPadsBegin].get(),
            workbuffers.immutable_buffers[WorkbufferIndex::kSrcShape].cast<const std::size_t*>().get(),
            workbuffers.immutable_buffers[WorkbufferIndex::kDstShape].cast<const std::size_t*>().get(),
            inputTensors[InputIndex::kPadValue].get());
}

WorkbufferRequest PadOp::GetWorkBufferRequest() const {
    auto rank = src_shape_.size();
    return {{rank * sizeof(std::size_t), rank * sizeof(std::size_t)}, {}};
}

void PadOp::InitSharedImmutableWorkbuffers(const Buffers& devicePointers) {
    CUDA::DefaultStream::stream().upload(
        devicePointers[WorkbufferIndex::kSrcShape], src_shape_.data(), src_shape_.size() * sizeof(std::size_t));
    CUDA::DefaultStream::stream().upload(
        devicePointers[WorkbufferIndex::kDstShape], dst_shape_.data(), src_shape_.size() * sizeof(std::size_t));
}

OPERATION_REGISTER(PadOp, Pad);

}  // namespace CUDAPlugin
