// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_convolution_backprop_data.hpp"

#include <cudnn.h>

#include <cuda_operation_registry.hpp>
#include <details/ie_exception.hpp>
#include <gsl/gsl_assert>
#include <gsl/span_ext>
#include <ngraph/op/constant.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "constant_factory.hpp"
#include "converters.hpp"
#include "fused_convolution.hpp"

namespace CUDAPlugin {

FusedConvolutionBackpropDataOp::FusedConvolutionBackpropDataOp(const CUDA::CreationContext& context,
                                                               const NodeOp& node,
                                                               IndexCollection&& inputIds,
                                                               IndexCollection&& outputIds)
    : OperationCuDnn(context, node, std::move(inputIds), std::move(outputIds)),
      params_{node},
      conv_descs_{context, params_.conv_},
      add_in_bytes_{ngraph::element::Type(params_.conv_.element_type_).size() *
                    ngraph::shape_size(params_.add_shape_)} {
    const auto size = ngraph::element::Type(params_.conv_.element_type_).size();
    conv_in_bytes_ = size * ngraph::shape_size(params_.conv_.dinput_shape_);
    add_in_bytes_ = size * ngraph::shape_size(params_.add_shape_);
    Expects(conv_in_bytes_ >= add_in_bytes_);

    ngraph::Output<ngraph::Node> addNode;
    if (node.get_input_size() == 4) {
        addNode = node.input(3).get_source_output();
    } else {
        addNode = node.input(2).get_source_output();
    }

    auto addConstant = dynamic_cast<ngraph::op::v0::Constant*>(addNode.get_node());
    add_node_ = addConstant->shared_from_this();
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(addConstant->get_data_ptr());
    add_constant_ = gsl::make_span(ptr, GetBufferSize(addNode.get_node()->output(0)));
    Expects(add_constant_.size_bytes() == add_in_bytes_);
}

void FusedConvolutionBackpropDataOp::Execute(const InferenceRequestContext& context,
                                             Inputs inputs,
                                             Outputs outputs,
                                             const Workbuffers& workbuffers) const {
    using ArgIndices3Ins = Convolution::Details::FusedConvolutionBackwardDataIndices<3>;
    using ArgIndices4Ins = Convolution::Details::FusedConvolutionBackwardDataIndices<4>;

    Expects(outputs.size() == 1);

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
                                                &NumericConst<constants::one>(conv_descs_.ElementType()),
                                                conv_descs_.Filter().get(),
                                                inputs[ArgIndices3Ins::filter].get(),
                                                conv_descs_.dOutput().get(),
                                                inputs[ArgIndices3Ins::doutput].get(),
                                                conv_descs_.Conv().get(),
                                                conv_descs_.Algo().algo,
                                                workbuffer,
                                                conv_descs_.Algo().memory,
                                                &NumericConst<constants::one>(conv_descs_.ElementType()),
                                                conv_descs_.dInput().get(),
                                                outputs[ArgIndices3Ins::dinput].get()));
}

void FusedConvolutionBackpropDataOp::InitSharedImmutableWorkbuffers(const IOperationExec::Buffers& buffers) {
    Expects(buffers.size() == 1);
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

std::size_t FusedConvolutionBackpropDataOp::GetBufferSize(const ngraph::Output<ngraph::Node>& output) {
    return output.get_element_type().size() * shape_size(output.get_shape());
}

OPERATION_REGISTER(FusedConvolutionBackpropDataOp, FusedConvBackpropData);

}  // namespace CUDAPlugin
