// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor_iterator.hpp"

#include <cstdint>
#include <cuda_op_buffers_extractor.hpp>
#include <cuda_iexecution_delegator.hpp>
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

    updateExecSequence();

    // Input mapping of ports
    slices_.reserve(portmap_inputs_.size());
    for (const auto& it : portmap_inputs_) {
        const auto& inputIdx = it.first;
        const auto& paramIdx = inputs_parameters_map_.at(inputIdx);
        slices_.emplace_back(*this, inputIdx, paramIdx);
    }

    // Back-edge mapping
    transfers_.reserve(results_parameters_map_.size());
    for (const auto& [resultIdx, paramIdx] : results_parameters_map_) {
        transfers_.emplace_back(*this, resultIdx, paramIdx);
    }

    // Output mapping of ports
    inserts_.reserve(results_outputs_map_.size());
    for (const auto& [resultIdx, outputIdx] : results_outputs_map_) {
        if (portmap_outputs_.count(outputIdx) > 0) {
            inserts_.emplace_back(*this, resultIdx, outputIdx);
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
    auto& executionDelegator = context.getExecutionDelegator();
    executionDelegator.set_stream(stream);

    // First iteration
    for (const auto inputIdx : invariant_inputs_) {
        const auto paramIdx = inputs_parameters_map_.at(inputIdx);
        transferParam(stream, mutableBuffer, inputTensors, 0, inputIdx, paramIdx);
    }
    for (const auto& [inputIdx, paramIdx] : inputs_parameters_map_) {
        if (portmap_inputs_.count(inputIdx) == 0) {
            transferParam(stream, mutableBuffer, inputTensors, 0, inputIdx, paramIdx);
        }
    }

    for (int64_t iter = 0; iter < num_iterations_; ++iter) {
        // Input mapping of ports
        for (const auto& slice : slices_) {
            slice(stream, inputTensors, mutableBuffer, iter);
        }

        // Inner loop
        executionDelegator.execute_sequence(this, memoryManager, mutableBuffer, context);

        // Back-edge mapping
        for (const auto& transfer : transfers_) {
            transfer(stream, mutableBuffer);
        }

        // Output mapping of ports
        for (const auto& insert : inserts_) {
            insert(stream, mutableBuffer, outputTensors, iter);
        }

        // Copy data to output
        if (iterations_results_map_.count(iter) > 0) {
            for (const auto& resultIdx : iterations_results_map_.at(iter)) {
                const auto& outputIdx = results_outputs_map_.at(resultIdx);
                transferResult(stream, mutableBuffer, outputTensors, iter, resultIdx, outputIdx);
            }
        }
    }
}

CudaGraphCompatibility TensorIteratorOp::GetCudaGraphCompatibility() const {
    // This implementation is CUDA graph compatible only if this is the standard TI with output only of the last
    // iteration (which is handled outside of the iterations loop)
    if (iterations_results_map_.size() != 1 || iterations_results_map_.count(num_iterations_ - 1) == 0) {
        return CudaGraphCompatibility::NONE;
    }
    return SubGraph::GetCudaGraphCompatibility() == CudaGraphCompatibility::NONE ? CudaGraphCompatibility::NONE
                                                                                 : CudaGraphCompatibility::SPECIAL;
}

void TensorIteratorOp::Capture(InferenceRequestContext& context,
                               Inputs inputTensors,
                               Outputs outputTensors,
                               const Workbuffers& workbuffers) const {
    const auto& stream = context.getThreadContext().stream();
    const auto& memoryManager = *memory_manager_;
    auto& mutableBuffer = workbuffers.mutable_buffers.at(0);
    auto& executionDelegator = context.getExecutionDelegator();
    executionDelegator.set_stream(stream);
    auto& graphInfo = context.getCudaGraphContext().get_current_graph_info();

    CUDA::GraphCapture capture{stream};
    {
        auto scope = capture.getScope();
        // Input mapping of ports
        for (auto& slice : slices_) {
            slice.add_kernel_node(graphInfo, stream, mutableBuffer, inputTensors);
        }

        // Inner loop
        executionDelegator.capture_sequence(this, memoryManager, mutableBuffer, context);

        // Back-edge mapping
        for (auto& transfer : transfers_) {
            transfer.add_transfer_node(graphInfo, stream, mutableBuffer);
        }

        // Output mapping of ports
        for (auto& insert : inserts_) {
            insert.add_kernel_node(graphInfo, stream, mutableBuffer, outputTensors);
        }
    }
    graphInfo.set_graph(capture.getGraph());
}

void TensorIteratorOp::ExecuteGraph(InferenceRequestContext& context,
                                    Inputs inputTensors,
                                    Outputs outputTensors,
                                    const Workbuffers& workbuffers) const {
    const auto& stream = context.getThreadContext().stream();
    const auto& memoryManager = *memory_manager_;
    const auto& mutableBuffer = workbuffers.mutable_buffers.at(0);

    // First iteration; this part doesn't use CUDA graphs yet
    for (const auto inputIdx : invariant_inputs_) {
        const auto paramIdx = inputs_parameters_map_.at(inputIdx);
        transferParam(stream, mutableBuffer, inputTensors, 0, inputIdx, paramIdx);
    }
    for (const auto& [inputIdx, paramIdx] : inputs_parameters_map_) {
        if (portmap_inputs_.count(inputIdx) == 0) {
            transferParam(stream, mutableBuffer, inputTensors, 0, inputIdx, paramIdx);
        }
    }

    auto& graphInfo = context.getCudaGraphContext().get_current_graph_info();
    OPENVINO_ASSERT(graphInfo.get_kernels_count() == slices_.size() + inserts_.size(),
                    "CudaGraphContext/TensorIteratorOp slices or inserts count incosistency");

    // TI body loop
    for (int64_t iter = 0; iter < num_iterations_; ++iter) {
        for (std::size_t i = 0; i < slices_.size(); ++i) {
            slices_[i].update_kernel_node(graphInfo, i, mutableBuffer, inputTensors, iter);
        }
        for (std::size_t i = 0; i < inserts_.size(); ++i) {
            inserts_[i].update_kernel_node(graphInfo, i + slices_.size(), mutableBuffer, outputTensors, iter);
        }
        graphInfo.launch(stream);
    }

    // Copy data to output; this part doesn't use CUDA graphs yet
    if (iterations_results_map_.count(num_iterations_ - 1) > 0) {
        for (const auto& resultIdx : iterations_results_map_.at(num_iterations_ - 1)) {
            const auto& outputIdx = results_outputs_map_.at(resultIdx);
            transferResult(stream, mutableBuffer, outputTensors, num_iterations_ - 1, resultIdx, outputIdx);
        }
    }
}

TensorIteratorOp::SliceLauncher::SliceLauncher(const TensorIteratorOp& ti, uint64_t inputIdx, uint64_t paramIdx)
    : input_idx_{inputIdx},
      param_{*ti.params_[paramIdx]},
      memory_manager_{*ti.memory_manager_},
      slice_{ti.kernelmap_inputs_.at(inputIdx)} {
    OPENVINO_ASSERT(ti.portmap_inputs_.count(inputIdx) != 0, "Node name: ", ti.GetName());
    const auto& portMap = ti.portmap_inputs_.at(input_idx_);
    const auto& inputShape = ti.inputs_info_[input_idx_].shape_;
    start_ = portMap.start < 0 ? inputShape[portMap.axis] + portMap.start : portMap.start;
    stride_ = portMap.stride;
}

void TensorIteratorOp::SliceLauncher::add_kernel_node(CudaGraphInfo& info,
                                                      const CUDA::Stream& stream,
                                                      CUDA::DevicePointer<void*> mutableBuffer,
                                                      const IOperationExec::Inputs& inputTensors) {
    const auto* src = inputTensors[input_idx_].get();
    auto* dst = memory_manager_.outputTensorPointers(param_, mutableBuffer)[0].get();
    info.add_kernel(stream,
                    slice_.getKernel(),
                    slice_.getNumBlocks(),
                    slice_.getThreadsPerBlock(),
                    slice_.getPropsPtr(),
                    start_,
                    slice_.getSize(),
                    src,
                    dst);
}

TensorIteratorOp::TransferLauncher::TransferLauncher(const TensorIteratorOp& ti, uint64_t resultIdx, uint64_t paramIdx)
    : param_{*ti.params_[paramIdx]}, result_{*ti.results_[resultIdx]}, memory_manager_{*ti.memory_manager_} {
    param_size_ = ti.params_info_[paramIdx].size_;
    const auto resultSize = ti.results_info_[resultIdx].size_;
    OPENVINO_ASSERT(param_size_ == resultSize, "Node name: ", ti.GetName());
}

void TensorIteratorOp::TransferLauncher::add_transfer_node(CudaGraphInfo& info,
                                                           const CUDA::Stream& stream,
                                                           CUDA::DevicePointer<void*> mutableBuffer) {
    const auto& paramTensors = memory_manager_.outputTensorPointers(param_, mutableBuffer);
    auto dst = paramTensors[0];
    const auto& resultTensors = memory_manager_.inputTensorPointers(result_, mutableBuffer);
    const auto src = resultTensors[0];
    info.add_transfer(stream, dst, src, param_size_);
}

TensorIteratorOp::InsertLauncher::InsertLauncher(const TensorIteratorOp& ti,
                                                 const std::size_t resultIdx,
                                                 const std::size_t outputIdx)
    : output_idx_{outputIdx},
      result_{*ti.results_[resultIdx]},
      memory_manager_{*ti.memory_manager_},
      insert_{ti.kernelmap_outputs_.at(outputIdx)} {
    OPENVINO_ASSERT(ti.portmap_outputs_.count(outputIdx) != 0, "Node name: ", ti.GetName());
    const auto& portMap = ti.portmap_outputs_.at(output_idx_);
    const auto& outputShape = ti.outputs_info_[output_idx_].shape_;
    start_ = portMap.start < 0 ? outputShape[portMap.axis] + portMap.start : portMap.start;
    stride_ = portMap.stride;
}

void TensorIteratorOp::InsertLauncher::add_kernel_node(CudaGraphInfo& info,
                                                       const CUDA::Stream& stream,
                                                       CUDA::DevicePointer<void*> mutableBuffer,
                                                       const IOperationExec::Outputs& outputTensors) {
    const auto* src = memory_manager_.inputTensorPointers(result_, mutableBuffer)[0].get();
    auto* dst = outputTensors[output_idx_].get();
    info.add_kernel(stream,
                    insert_.getKernel(),
                    insert_.getNumBlocks(),
                    insert_.getThreadsPerBlock(),
                    insert_.getPropsPtr(),
                    start_,
                    insert_.getSize(),
                    src,
                    dst);
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
    OPENVINO_ASSERT(buffers.size() == kernelmap_inputs_.size() + kernelmap_outputs_.size(), "Node name: ", GetName());
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

void TensorIteratorOp::transferParam(const CUDA::Stream& stream,
                                     const CUDA::DevicePointer<void*> mutableBuffer,
                                     const IOperationExec::Inputs& inputTensors,
                                     const std::int64_t iter,
                                     const uint64_t inputIdx,
                                     const uint64_t paramIdx) const {
    OPENVINO_ASSERT(portmap_inputs_.count(inputIdx) == 0, "Node name: ", GetName());
    auto& memoryManager = *memory_manager_;
    const std::size_t inputSize = inputs_info_[inputIdx].size_;
    const std::size_t paramSize = params_info_[paramIdx].size_;

    auto& input = inputTensors[inputIdx];
    const auto& param = params_[paramIdx];
    auto outputTensors = memoryManager.outputTensorPointers(*param, mutableBuffer);
    OPENVINO_ASSERT(inputSize == paramSize, "Node name: ", GetName());

    stream.transfer(outputTensors[0], input, inputSize);
}

void TensorIteratorOp::transferResult(const CUDA::Stream& stream,
                                      CUDA::DevicePointer<void*> mutableBuffer,
                                      const IOperationExec::Outputs& outputTensors,
                                      const std::int64_t iter,
                                      const std::size_t resultIdx,
                                      const std::size_t outputIdx) const {
    OPENVINO_ASSERT(portmap_outputs_.count(outputIdx) == 0, "Node name: ", GetName());
    auto& memoryManager = *memory_manager_;
    const auto resultSize = results_info_[resultIdx].size_;
    const std::size_t outputSize = outputs_info_[outputIdx].size_;

    const auto result = results_[resultIdx];
    auto inTensors = memoryManager.inputTensorPointers(*result, mutableBuffer);
    const auto output = outputTensors[outputIdx];
    OPENVINO_ASSERT(resultSize == outputSize, "Node name: ", GetName());

    stream.transfer(output, inTensors[0], outputSize);
}

void TensorIteratorOp::updateExecSequence() {
    std::vector<OperationBase::Ptr> newExecSequence;
    for (const auto& op : exec_sequence_) {
        if (!dynamic_cast<const ParameterOp*>(op.get()) && !dynamic_cast<const ResultOp*>(op.get())) {
            newExecSequence.emplace_back(op);
        }
    }
    exec_sequence_ = std::move(newExecSequence);
}

OPERATION_REGISTER(TensorIteratorOp, TensorIterator);

}  // namespace nvidia_gpu
}  // namespace ov
