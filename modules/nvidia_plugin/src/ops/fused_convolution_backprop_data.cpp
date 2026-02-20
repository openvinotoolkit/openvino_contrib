// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_convolution_backprop_data.hpp"

#include <cudnn.h>

#include <cuda_operation_registry.hpp>
#include <gsl/span_ext>
#include <openvino/core/except.hpp>
#include <openvino/op/constant.hpp>

#include "cuda/constant_factory.hpp"

namespace ov {
namespace nvidia_gpu {

FusedConvolutionBackpropDataOp::FusedConvolutionBackpropDataOp(const CreationContext& context,
                                                               const NodeOp& node,
                                                               IndexCollection&& inputIds,
                                                               IndexCollection&& outputIds)
    : OperationCuDnn(context, node, std::move(inputIds), std::move(outputIds)),
      params_{node},
      conv_descs_{context, params_.conv_},
      add_in_bytes_{ov::element::Type(params_.conv_.element_type_).size() * ov::shape_size(params_.add_shape_)} {
    const auto size = ov::element::Type(params_.conv_.element_type_).size();
    conv_in_bytes_ = size * ov::shape_size(params_.conv_.dinput_shape_);
    add_in_bytes_ = size * ov::shape_size(params_.add_shape_);
    OPENVINO_ASSERT(conv_in_bytes_ >= add_in_bytes_, "Node name: ", GetName());

    ov::Output<ov::Node> addNode;
    if (node.get_input_size() == 4) {
        addNode = node.input(3).get_source_output();
    } else {
        addNode = node.input(2).get_source_output();
    }

    auto addConstant = dynamic_cast<ov::op::v0::Constant*>(addNode.get_node());
    add_node_ = addConstant->shared_from_this();
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(addConstant->get_data_ptr());
    add_constant_ = gsl::make_span(ptr, GetBufferSize(addNode.get_node()->output(0)));
    OPENVINO_ASSERT(add_constant_.size_bytes() == add_in_bytes_, "Node name: ", GetName());
}

void FusedConvolutionBackpropDataOp::Execute(const InferenceRequestContext& context,
                                             Inputs inputs,
                                             Outputs outputs,
                                             const Workbuffers& workbuffers) const {
    using ArgIndices3Ins = Convolution::Details::FusedConvolutionBackwardDataIndices<3>;
    using ArgIndices4Ins = Convolution::Details::FusedConvolutionBackwardDataIndices<4>;

    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());

    void* workbuffer = workbuffers.mutable_buffers.empty() ? nullptr : workbuffers.mutable_buffers[0].get();

    const auto& threadContext = context.getThreadContext();
    const auto& dnnHandle = threadContext.dnnHandle();
    const auto& stream = threadContext.stream();
    if (inputs.size() == 4 && conv_in_bytes_) {
        stream.transfer(outputs[ArgIndices4Ins::dinput], workbuffers.immutable_buffers.at(0), conv_in_bytes_);
    } else {
        stream.transfer(outputs[ArgIndices3Ins::dinput], workbuffers.immutable_buffers.at(0), conv_in_bytes_);
    }
    throwIfError(::cudnnConvolutionBackwardData(dnnHandle.get(),
                                                &CUDA::NumericConst<CUDA::constants::one>(conv_descs_.ElementType()),
                                                conv_descs_.Filter().get(),
                                                inputs[ArgIndices3Ins::filter].get(),
                                                conv_descs_.dOutput().get(),
                                                inputs[ArgIndices3Ins::doutput].get(),
                                                conv_descs_.Conv().get(),
                                                conv_descs_.Algo().algo,
                                                workbuffer,
                                                conv_descs_.Algo().memory,
                                                &CUDA::NumericConst<CUDA::constants::one>(conv_descs_.ElementType()),
                                                conv_descs_.dInput().get(),
                                                outputs[ArgIndices3Ins::dinput].get()));
}

CudaGraphCompatibility FusedConvolutionBackpropDataOp::GetCudaGraphCompatibility() const {
    return CudaGraphCompatibility::FULL;
}

void FusedConvolutionBackpropDataOp::InitSharedImmutableWorkbuffers(const IOperationExec::Buffers& buffers) {
    OPENVINO_ASSERT(buffers.size() == 1, "Node name: ", GetName());
    const size_t repeat = conv_in_bytes_ / add_in_bytes_;
    for (size_t i = 0; i < repeat; ++i) {
        CUDA::DefaultStream::stream().upload(buffers[0] + i * add_in_bytes_, add_constant_.data(), add_in_bytes_);
    }
    add_node_.reset();
}

WorkbufferRequest FusedConvolutionBackpropDataOp::GetWorkBufferRequest() const {
    if (conv_descs_.Algo().memory != 0)
        return {{conv_in_bytes_}, {conv_descs_.Algo().memory}};
    else
        return {{conv_in_bytes_}, {}};
}

std::size_t FusedConvolutionBackpropDataOp::GetBufferSize(const ov::Output<ov::Node>& output) {
    return output.get_element_type().size() * shape_size(output.get_shape());
}

OPERATION_REGISTER(FusedConvolutionBackpropDataOp, FusedConvBackpropData);

}  // namespace nvidia_gpu
}  // namespace ov
