// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fake_quantize.hpp"

#include <fenv.h>

#include <cuda_operation_registry.hpp>

#include "converters.hpp"

namespace ov {
namespace nvidia_gpu {

enum InputIdx { ARG, INPUT_LOW, INPUT_HIGH, OUTPUT_LOW, OUTPUT_HIGH };

FakeQuantizeOp::FakeQuantizeOp(const CreationContext &context,
                               const NodeOp &node,
                               IndexCollection &&inputIds,
                               IndexCollection &&outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)),
      in_low_broadcast_params_{NumpyBroadcastParams::create(node.get_input_shape(INPUT_LOW), node.get_output_shape(0))},
      in_high_broadcast_params_{
          NumpyBroadcastParams::create(node.get_input_shape(INPUT_HIGH), node.get_output_shape(0))},
      out_low_broadcast_params_{
          NumpyBroadcastParams::create(node.get_input_shape(OUTPUT_LOW), node.get_output_shape(0))},
      out_high_broadcast_params_{
          NumpyBroadcastParams::create(node.get_input_shape(OUTPUT_HIGH), node.get_output_shape(0))} {
    const ov::element::Type element_type{node.get_input_element_type(0)};
    const std::size_t levels = node.get_levels();

    OPENVINO_ASSERT(levels > 1U, "Node name: ", GetName());
    OPENVINO_ASSERT(node.get_input_shape(0).size() == node.get_output_shape(0).size(), "Node name: ", GetName());

    in_low_broadcast_params_->addWorkbufferRequests(immutable_buffer_sizes_);
    in_high_broadcast_params_->addWorkbufferRequests(immutable_buffer_sizes_);
    out_low_broadcast_params_->addWorkbufferRequests(immutable_buffer_sizes_);
    out_high_broadcast_params_->addWorkbufferRequests(immutable_buffer_sizes_);

    const size_t output_size = ov::shape_size(node.get_output_shape(0));
    const auto max_threads_per_block = static_cast<unsigned>(context.device().props().maxThreadsPerBlock);

    kernel_ = kernel::FakeQuantize{
        convertDataType<ov::nvidia_gpu::kernel::Type_t>(element_type), output_size, max_threads_per_block, levels};
}

CudaGraphCompatibility FakeQuantizeOp::GetCudaGraphCompatibility() const { return CudaGraphCompatibility::FULL; }

void FakeQuantizeOp::Execute(const InferenceRequestContext &context,
                             Inputs inputTensors,
                             Outputs outputTensors,
                             const Workbuffers &workbuffers) const {
    OPENVINO_ASSERT(kernel_, "Node name: ", GetName());
    auto &stream = context.getThreadContext().stream();

    (*kernel_)(stream.get(),
               inputTensors[ARG].get(),
               inputTensors[INPUT_LOW].get(),
               inputTensors[INPUT_HIGH].get(),
               inputTensors[OUTPUT_LOW].get(),
               inputTensors[OUTPUT_HIGH].get(),
               in_low_broadcast_params_->mapper(workbuffers.immutable_buffers),
               in_high_broadcast_params_->mapper(workbuffers.immutable_buffers),
               out_low_broadcast_params_->mapper(workbuffers.immutable_buffers),
               out_high_broadcast_params_->mapper(workbuffers.immutable_buffers),
               outputTensors[0].get());
}

void FakeQuantizeOp::InitSharedImmutableWorkbuffers(const Buffers &buffers) {
    in_low_broadcast_params_->initWorkbuffers(buffers);
    in_high_broadcast_params_->initWorkbuffers(buffers);
    out_low_broadcast_params_->initWorkbuffers(buffers);
    out_high_broadcast_params_->initWorkbuffers(buffers);
}

WorkbufferRequest FakeQuantizeOp::GetWorkBufferRequest() const { return {immutable_buffer_sizes_, {}}; }

OPERATION_REGISTER(FakeQuantizeOp, FakeQuantize);
}  // namespace nvidia_gpu
}  // namespace ov
