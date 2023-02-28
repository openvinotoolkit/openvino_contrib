// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor_iterator.hpp"

#include <cpp/ie_cnn_network.h>

#include <cstdint>
#include <cuda_op_buffers_extractor.hpp>
#include <cuda_profiler.hpp>
#include <kernels/details/cuda_type_traits.hpp>
#include <kernels/details/tensor_helpers.hpp>
#include <kernels/insert.hpp>
#include <kernels/slice.hpp>

#include "converters.hpp"
#include "cuda_operation_registry.hpp"
#include "parameter.hpp"
#include "result.hpp"

namespace ov {
namespace nvidia_gpu {

TensorIteratorOp::TensorIteratorOp(const CreationContext& context,
                                   const NodeOp& op,
                                   IndexCollection&& inputIds,
                                   IndexCollection&& outputIds)
    : SubGraph(context, op, std::move(inputIds), std::move(outputIds)), num_iterations_{op.get_num_iterations()} {
    // Set trip count, initial execution condition, num iteration primitives
    // they should be mutable_data to prevent from being optimized out
    if (num_iterations_ < 0) {
        throw std::runtime_error("tensor iterator's num_iteration cannot be negative");
    }

    inputs_info_.reserve(op.inputs().size());
    for (auto& input : op.inputs()) {
        inputs_info_.emplace_back(getTensorByteSize(input), input.get_element_type(), input.get_shape());
    }

    outputs_info_.reserve(op.outputs().size());
    for (auto& output : op.outputs()) {
        outputs_info_.emplace_back(getTensorByteSize(output), output.get_element_type(), output.get_shape());
    }

    // Get body topology from ngraph func1tion
    InferenceEngine::CNNNetwork body_network(op.get_body());

    // Setup input_primitive_maps/ output_primitive_maps and back_edges
    const auto& loop_input_descs = op.get_input_descriptions();
    const auto& loop_output_descs = op.get_output_descriptions();

    // Set input mapping & back edges
    for (const auto& loop_input_desc : loop_input_descs) {
        inputs_parameters_map_[loop_input_desc->m_input_index] = loop_input_desc->m_body_parameter_index;

        // Set invariant input
        if (const auto& invariantInput =
                std::dynamic_pointer_cast<ov::op::util::SubGraphOp::InvariantInputDescription>(loop_input_desc)) {
            invariant_inputs_.push_back(invariantInput->m_input_index);
        }

        // Set input mapping
        if (const auto& sliceInfo = std::dynamic_pointer_cast<NodeOp::SliceInputDescription>(loop_input_desc)) {
            // sliced input
            portmap_inputs_[loop_input_desc->m_input_index] = PortMap{
                sliceInfo->m_start,
                sliceInfo->m_stride,
                sliceInfo->m_part_size,
                sliceInfo->m_end,
                sliceInfo->m_axis,
            };
        }

        // set back edges
        if (const auto& mergedInput = std::dynamic_pointer_cast<NodeOp::MergedInputDescription>(loop_input_desc)) {
            // backedge
            results_parameters_map_[mergedInput->m_body_value_index] = mergedInput->m_body_parameter_index;
        }
    }

    // Set output mapping
    for (const auto& loop_output_desc : loop_output_descs) {
        results_outputs_map_[loop_output_desc->m_body_value_index] = loop_output_desc->m_output_index;

        if (const auto& concatOutput = std::dynamic_pointer_cast<NodeOp::ConcatOutputDescription>(loop_output_desc)) {
            // concat output
            portmap_outputs_[loop_output_desc->m_output_index] = PortMap{
                concatOutput->m_start,
                concatOutput->m_stride,
                concatOutput->m_part_size,
                concatOutput->m_end,
                concatOutput->m_axis,
            };
        }
        if (const auto& bodyOutput = std::dynamic_pointer_cast<NodeOp::BodyOutputDescription>(loop_output_desc)) {
            size_t iterations;
            if (bodyOutput->m_iteration == -1) {
                iterations = num_iterations_ - 1;
            } else {
                iterations = bodyOutput->m_iteration;
            }
            if (iterations_results_map_.count(iterations) == 0) {
                iterations_results_map_[iterations] = std::vector<uint64_t>{};
            }
            iterations_results_map_[iterations].push_back(bodyOutput->m_body_value_index);
        }
    }
    max_threads_per_block_ = context.device().props().maxThreadsPerBlock;

    for (const auto& [inputIdx, portMap] : portmap_inputs_) {
        const auto inputShape = inputs_info_[inputIdx].shape_;
        const auto inputType = inputs_info_[inputIdx].type_;

        kernel::Type_t element_type = convertDataType<ov::nvidia_gpu::kernel::Type_t>(inputType);
        kernel::Slice::Props props;
        std::copy(inputShape.begin(), inputShape.end(), props.old_shape);
        std::copy(inputShape.begin(), inputShape.end(), props.new_shape);
        props.axe = portMap.axis;
        props.new_shape[props.axe] = portMap.part_size;
        kernelmap_inputs_.emplace(inputIdx, kernel::Slice(element_type, props, max_threads_per_block_));
    }

    for (const auto& [resultIdx, outputIdx] : results_outputs_map_) {
        if (portmap_outputs_.count(outputIdx) > 0) {
            const auto& resultShape = results_info_[resultIdx].shape_;
            const auto outputShape = outputs_info_[outputIdx].shape_;
            const auto outputType = outputs_info_[outputIdx].type_;
            const auto portMap = portmap_outputs_.at(outputIdx);

            kernel::Type_t element_type = convertDataType<kernel::Type_t>(outputType);
            kernel::Insert::Props props;
            std::copy(resultShape.begin(), resultShape.end(), props.old_shape);
            std::copy(outputShape.begin(), outputShape.end(), props.new_shape);
            props.axe = portMap.axis;
            kernelmap_outputs_.emplace(outputIdx, kernel::Insert(element_type, props, max_threads_per_block_));
        }
    }
}

void TensorIteratorOp::Execute(const InferenceRequestContext& context,
                               Inputs inputTensors,
                               Outputs outputTensors,
                               const Workbuffers& workbuffers) const {
    const auto& stream = context.getThreadContext().stream();
    const auto& memoryManager = *memory_manager_;
    auto& mutableBuffer = workbuffers.mutable_buffers.at(0);
    auto& cancellationToken = context.getCancellationToken();
    auto& profiler = context.getProfiler();
    profiler.SetStream(stream);

    // First iteration
    for (const auto inputIdx : invariant_inputs_) {
        const auto paramIdx = inputs_parameters_map_.at(inputIdx);
        copyParam(stream, mutableBuffer, inputTensors, 0, inputIdx, paramIdx);
    }
    for (const auto& [inputIdx, paramIdx] : inputs_parameters_map_) {
        if (portmap_inputs_.count(inputIdx) == 0) {
            copyParam(stream, mutableBuffer, inputTensors, 0, inputIdx, paramIdx);
        }
    }

    const auto& execSequence = profiler.CreateExecSequence(this);
    for (int64_t iter = 0; iter < num_iterations_; ++iter) {
        cancellationToken.Check();

        // Input mapping of ports
        for (auto& it : portmap_inputs_) {
            const auto& inputIdx = it.first;
            const auto& paramIdx = inputs_parameters_map_.at(inputIdx);
            copyParam(stream, mutableBuffer, inputTensors, iter, inputIdx, paramIdx);
        }

        // Inner loop
        for (const auto& op : execSequence) {
            auto inTensors = memoryManager.inputTensorPointers(*op, mutableBuffer);
            auto outTensors = memoryManager.outputTensorPointers(*op, mutableBuffer);
            auto workBuffers = memoryManager.workBuffers(*op, mutableBuffer);
            op->Execute(context, inTensors, outTensors, workBuffers);
        }

        // Back-edge mapping
        for (auto& [resultIdx, paramIdx] : results_parameters_map_) {
            copyBackEdge(stream, mutableBuffer, resultIdx, paramIdx);
        }

        // Output mapping of ports
        for (const auto& [resultIdx, outputIdx] : results_outputs_map_) {
            if (portmap_outputs_.count(outputIdx) > 0) {
                copyResult(stream, mutableBuffer, outputTensors, iter, resultIdx, outputIdx);
            }
        }

        // Copy data to output
        if (iterations_results_map_.count(iter) > 0) {
            for (const auto& resultIdx : iterations_results_map_.at(iter)) {
                const auto& outputIdx = results_outputs_map_.at(resultIdx);
                copyResult(stream, mutableBuffer, outputTensors, iter, resultIdx, outputIdx);
            }
        }
    }
}

WorkbufferRequest TensorIteratorOp::GetWorkBufferRequest() const {
    std::vector<WorkbufferRequest::size_in_bytes_t> immutable_sizes;
    immutable_sizes.reserve(kernelmap_inputs_.size() + kernelmap_outputs_.size());
    for (const auto& kernel_map : kernelmap_inputs_) {
        immutable_sizes.push_back(kernel_map.second.getImmutableWorkbufferSize());
    }
    for (const auto& kernel_map : kernelmap_outputs_) {
        immutable_sizes.push_back(kernel_map.second.getImmutableWorkbufferSize());
    }
    return {immutable_sizes, SubGraph::GetWorkBufferRequest().mutable_sizes};
}

void TensorIteratorOp::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
    Expects(buffers.size() == kernelmap_inputs_.size() + kernelmap_outputs_.size());
    unsigned nextBufferIdx = 0;
    for (auto& kernel_map : kernelmap_inputs_) {
        auto& slice = kernel_map.second;
        slice.setImmutableWorkbuffer(buffers[nextBufferIdx++].get());
    }
    for (auto& kernel_map : kernelmap_outputs_) {
        auto& insert = kernel_map.second;
        insert.setImmutableWorkbuffer(buffers[nextBufferIdx++].get());
    }
}

void TensorIteratorOp::copyParam(const CUDA::Stream& stream,
                                 const CUDA::DevicePointer<void*> mutableBuffer,
                                 const IOperationExec::Inputs& inputTensors,
                                 const std::int64_t iter,
                                 const uint64_t inputIdx,
                                 const uint64_t paramIdx) const {
    auto& memoryManager = *memory_manager_;
    const std::size_t inputSize = inputs_info_[inputIdx].size_;
    const std::size_t paramSize = params_info_[paramIdx].size_;
    if (portmap_inputs_.count(inputIdx) == 0) {
        auto& input = inputTensors[inputIdx];
        const auto& param = params_[paramIdx];
        auto outputTensors = memoryManager.outputTensorPointers(*param, mutableBuffer);
        Expects(inputSize == paramSize);
        stream.transfer(outputTensors[0], input, inputSize);
    } else {
        const auto& portMap = portmap_inputs_.at(inputIdx);
        const auto& param = params_[paramIdx];
        auto outputTensors = memoryManager.outputTensorPointers(*param, mutableBuffer);
        const auto inputShape = inputs_info_[inputIdx].shape_;

        const auto& slice = kernelmap_inputs_.at(inputIdx);
        std::size_t start;
        if (portMap.start < 0) {
            start = inputShape[portMap.axis] + portMap.start;
        } else {
            start = portMap.start;
        }
        start += iter * portMap.stride;
        auto input = inputTensors[inputIdx];
        slice(stream.get(), input.get(), outputTensors[0].get(), start);
    }
}

void TensorIteratorOp::copyBackEdge(const CUDA::Stream& stream,
                                    CUDA::DevicePointer<void*> mutableBuffer,
                                    const uint64_t resultIdx,
                                    const uint64_t paramIdx) const {
    auto& memoryManager = *memory_manager_;
    const auto& result = results_[resultIdx];
    const auto& param = params_[paramIdx];
    auto paramTensors = memoryManager.outputTensorPointers(*param, mutableBuffer);
    auto resultTensors = memoryManager.inputTensorPointers(*result, mutableBuffer);
    const std::size_t paramSize = params_info_[paramIdx].size_;
    const std::size_t resultSize = results_info_[resultIdx].size_;
    Expects(paramSize == resultSize);
    stream.transfer(paramTensors[0], resultTensors[0], paramSize);
}

void TensorIteratorOp::copyResult(const CUDA::Stream& stream,
                                  CUDA::DevicePointer<void*> mutableBuffer,
                                  const IOperationExec::Outputs& outputTensors,
                                  const std::int64_t iter,
                                  const std::size_t resultIdx,
                                  const std::size_t outputIdx) const {
    auto& memoryManager = *memory_manager_;
    const auto resultSize = results_info_[resultIdx].size_;
    const std::size_t outputSize = outputs_info_[outputIdx].size_;
    if (portmap_outputs_.count(outputIdx) == 0) {
        const auto result = results_[resultIdx];
        auto inTensors = memoryManager.inputTensorPointers(*result, mutableBuffer);
        const auto output = outputTensors[outputIdx];
        Expects(resultSize == outputSize);
        stream.transfer(output, inTensors[0], outputSize);
    } else {
        auto output = outputTensors[outputIdx];
        const auto& result = results_[resultIdx];
        auto inputTensors = memoryManager.inputTensorPointers(*result, mutableBuffer);
        const auto portMap = portmap_outputs_.at(outputIdx);
        const auto outputShape = outputs_info_[outputIdx].shape_;

        const auto& insert = kernelmap_outputs_.at(outputIdx);
        std::size_t start;
        if (portMap.start < 0) {
            start = outputShape[portMap.axis] + portMap.start;
        } else {
            start = portMap.start;
        }
        start += iter * portMap.stride;
        insert(stream.get(), inputTensors[0].get(), output.get(), start);
    }
}

OPERATION_REGISTER(TensorIteratorOp, TensorIterator);

}  // namespace nvidia_gpu
}  // namespace ov
