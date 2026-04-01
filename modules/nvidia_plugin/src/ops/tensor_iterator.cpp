// Copyright (C) 2018-2026 Intel Corporation
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
    const auto& dynBufCtx = context.getDynamicBufferContext();
    auto resolved = resolveInternalPointers(mutableBuffer, dynBufCtx);
    executionDelegator.set_stream(stream);

    // First iteration
    for (const auto inputIdx : invariant_inputs_) {
        const auto paramIdx = inputs_parameters_map_.at(inputIdx);
        OPENVINO_ASSERT(inputs_info_[inputIdx].size_ == params_info_[paramIdx].size_, "Node name: ", GetName());
        stream.transfer(resolved.param_outputs[paramIdx], inputTensors[inputIdx], inputs_info_[inputIdx].size_);
    }
    for (const auto& [inputIdx, paramIdx] : inputs_parameters_map_) {
        if (portmap_inputs_.count(inputIdx) == 0) {
            OPENVINO_ASSERT(inputs_info_[inputIdx].size_ == params_info_[paramIdx].size_, "Node name: ", GetName());
            stream.transfer(resolved.param_outputs[paramIdx], inputTensors[inputIdx], inputs_info_[inputIdx].size_);
        }
    }

    for (int64_t iter = 0; iter < num_iterations_; ++iter) {
        // Input mapping of ports
        for (const auto& slice : slices_) {
            slice(stream, inputTensors[slice.inputIdx()].get(), resolved.param_outputs[slice.paramIdx()].get(), iter);
        }

        // Inner loop
        executionDelegator.execute_sequence(this, memoryManager, mutableBuffer, context);

        // Back-edge mapping
        for (const auto& transfer : transfers_) {
            transfer(stream, resolved.result_inputs[transfer.resultIdx()].get(), resolved.param_outputs[transfer.paramIdx()].get());
        }

        // Output mapping of ports
        for (const auto& insert : inserts_) {
            insert(stream, resolved.result_inputs[insert.resultIdx()].get(), outputTensors[insert.outputIdx()].get(), iter);
        }

        // Copy data to output
        if (iterations_results_map_.count(iter) > 0) {
            for (const auto& resultIdx : iterations_results_map_.at(iter)) {
                const auto& outputIdx = results_outputs_map_.at(resultIdx);
                OPENVINO_ASSERT(results_info_[resultIdx].size_ == outputs_info_[outputIdx].size_, "Node name: ", GetName());
                stream.transfer(outputTensors[outputIdx], resolved.result_inputs[resultIdx], outputs_info_[outputIdx].size_);
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
    if (!is_compatibility_analyzed_) {
        graph_compatibility_ = CudaGraphCompatibility::NONE;
        for (const auto& op : exec_sequence_) {
            auto opCompatability = op->GetCudaGraphCompatibility();
            if (opCompatability == CudaGraphCompatibility::SPECIAL || opCompatability == CudaGraphCompatibility::FULL) {
                graph_compatibility_ = CudaGraphCompatibility::SPECIAL;
                break;
            }
        }
        is_compatibility_analyzed_ = true;
    }
    return graph_compatibility_;
}

void TensorIteratorOp::initializeRunner() {
    // For better performance nested topology runners are not used if all operations in execution sequence are
    // fully CUDA graph compatible
    if (std::any_of(exec_sequence_.begin(), exec_sequence_.end(), [](const auto& op) {
            return op->GetCudaGraphCompatibility() != CudaGraphCompatibility::FULL;
        })) {
        SubGraph::initializeRunner();
    }
}

TensorIteratorOp::SliceLauncher::SliceLauncher(const TensorIteratorOp& ti, uint64_t inputIdx, uint64_t paramIdx)
    : input_idx_{inputIdx},
      param_idx_{paramIdx},
      slice_{ti.kernelmap_inputs_.at(inputIdx)} {
    OPENVINO_ASSERT(ti.portmap_inputs_.count(inputIdx) != 0, "Node name: ", ti.GetName());
    const auto& portMap = ti.portmap_inputs_.at(input_idx_);
    const auto& inputShape = ti.inputs_info_[input_idx_].shape_;
    start_ = portMap.start < 0 ? inputShape[portMap.axis] + portMap.start : portMap.start;
    stride_ = portMap.stride;
}

void TensorIteratorOp::SliceLauncher::addKernelNode(ICudaGraphInfo& info,
                                                    const CUDA::Stream& stream,
                                                    const void* src, void* dst) {
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
    : param_idx_{paramIdx}, result_idx_{resultIdx} {
    param_size_ = ti.params_info_[paramIdx].size_;
    const auto resultSize = ti.results_info_[resultIdx].size_;
    OPENVINO_ASSERT(param_size_ == resultSize, "Node name: ", ti.GetName());
}

void TensorIteratorOp::TransferLauncher::addTransferNode(ICudaGraphInfo& info,
                                                         const CUDA::Stream& stream,
                                                         const void* src, void* dst) {
    info.add_transfer(stream,
                      CUDA::DevicePointer<void*>{dst},
                      CUDA::DevicePointer<const void*>{src},
                      param_size_);
}

TensorIteratorOp::InsertLauncher::InsertLauncher(const TensorIteratorOp& ti,
                                                 const std::size_t resultIdx,
                                                 const std::size_t outputIdx)
    : output_idx_{outputIdx},
      result_idx_{resultIdx},
      insert_{ti.kernelmap_outputs_.at(outputIdx)} {
    OPENVINO_ASSERT(ti.portmap_outputs_.count(outputIdx) != 0, "Node name: ", ti.GetName());
    const auto& portMap = ti.portmap_outputs_.at(output_idx_);
    const auto& outputShape = ti.outputs_info_[output_idx_].shape_;
    start_ = portMap.start < 0 ? outputShape[portMap.axis] + portMap.start : portMap.start;
    stride_ = portMap.stride;
}

void TensorIteratorOp::InsertLauncher::addKernelNode(ICudaGraphInfo& info,
                                                     const CUDA::Stream& stream,
                                                     const void* src, void* dst) {
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

void TensorIteratorOp::CaptureSingle(InferenceRequestContext& context,
                                     Inputs inputTensors,
                                     Outputs outputTensors,
                                     const Workbuffers& workbuffers) const {
    const auto& stream = context.getThreadContext().stream();
    const auto& memoryManager = *memory_manager_;
    auto& mutableBuffer = workbuffers.mutable_buffers.at(0);
    auto& executionDelegator = context.getExecutionDelegator();
    const auto& dynBufCtx = context.getDynamicBufferContext();
    auto resolved = resolveInternalPointers(mutableBuffer, dynBufCtx);
    executionDelegator.set_stream(stream);
    auto& graphInfo = context.getCurrentCudaGraphInfo();
    OPENVINO_ASSERT(!graphInfo.is_nested(), "For single-graph mode graphInfo shouldn't be nested");

    CUDA::GraphCapture capture{stream};
    {
        auto scope = capture.getScope();
        // Input mapping of ports
        for (auto& slice : slices_) {
            slice.addKernelNode(graphInfo, stream,
                                inputTensors[slice.inputIdx()].get(),
                                resolved.param_outputs[slice.paramIdx()].get());
        }

        // Inner loop
        executionDelegator.capture_sequence(this, memoryManager, mutableBuffer, context);

        // Back-edge mapping
        for (auto& transfer : transfers_) {
            transfer.addTransferNode(graphInfo, stream,
                                     resolved.result_inputs[transfer.resultIdx()].get(),
                                     resolved.param_outputs[transfer.paramIdx()].get());
        }

        // Output mapping of ports
        for (auto& insert : inserts_) {
            insert.addKernelNode(graphInfo, stream,
                                 resolved.result_inputs[insert.resultIdx()].get(),
                                 outputTensors[insert.outputIdx()].get());
        }
    }
    graphInfo.set_current_graph(capture.getGraph());
}

void TensorIteratorOp::ExecuteGraphSingle(InferenceRequestContext& context,
                                          Inputs inputTensors,
                                          Outputs outputTensors,
                                          const Workbuffers& workbuffers) const {
    const auto& stream = context.getThreadContext().stream();
    const auto& mutableBuffer = workbuffers.mutable_buffers.at(0);
    const auto& dynBufCtx = context.getDynamicBufferContext();
    auto resolved = resolveInternalPointers(mutableBuffer, dynBufCtx);

    // First iteration
    for (const auto inputIdx : invariant_inputs_) {
        const auto paramIdx = inputs_parameters_map_.at(inputIdx);
        OPENVINO_ASSERT(inputs_info_[inputIdx].size_ == params_info_[paramIdx].size_, "Node name: ", GetName());
        stream.transfer(resolved.param_outputs[paramIdx], inputTensors[inputIdx], inputs_info_[inputIdx].size_);
    }
    for (const auto& [inputIdx, paramIdx] : inputs_parameters_map_) {
        if (portmap_inputs_.count(inputIdx) == 0) {
            OPENVINO_ASSERT(inputs_info_[inputIdx].size_ == params_info_[paramIdx].size_, "Node name: ", GetName());
            stream.transfer(resolved.param_outputs[paramIdx], inputTensors[inputIdx], inputs_info_[inputIdx].size_);
        }
    }

    auto& graphInfo = context.getCurrentCudaGraphInfo();
    OPENVINO_ASSERT(graphInfo.get_kernels_count() == slices_.size() + inserts_.size(),
                    "CudaGraphContext/TensorIteratorOp slices or inserts count incosistency");

    // TI body loop
    for (int64_t iter = 0; iter < num_iterations_; ++iter) {
        for (std::size_t i = 0; i < slices_.size(); ++i) {
            slices_[i].updateKernelNode(graphInfo, i,
                                        inputTensors[slices_[i].inputIdx()].get(),
                                        resolved.param_outputs[slices_[i].paramIdx()].get(),
                                        iter);
        }
        for (std::size_t i = 0; i < inserts_.size(); ++i) {
            inserts_[i].updateKernelNode(graphInfo, i + slices_.size(),
                                         resolved.result_inputs[inserts_[i].resultIdx()].get(),
                                         outputTensors[inserts_[i].outputIdx()].get(),
                                         iter);
        }
        graphInfo.launch(stream);
    }

    // Copy data to output
    if (iterations_results_map_.count(num_iterations_ - 1) > 0) {
        for (const auto& resultIdx : iterations_results_map_.at(num_iterations_ - 1)) {
            const auto& outputIdx = results_outputs_map_.at(resultIdx);
            OPENVINO_ASSERT(results_info_[resultIdx].size_ == outputs_info_[outputIdx].size_, "Node name: ", GetName());
            stream.transfer(outputTensors[outputIdx], resolved.result_inputs[resultIdx], outputs_info_[outputIdx].size_);
        }
    }
}

void TensorIteratorOp::CaptureMulti(InferenceRequestContext& context,
                                    Inputs inputTensors,
                                    Outputs outputTensors,
                                    const Workbuffers& workbuffers) const {
    const auto& stream = context.getThreadContext().stream();
    auto& mutableBuffer = workbuffers.mutable_buffers.at(0);
    auto& executionDelegator = context.getExecutionDelegator();
    const auto& dynBufCtx = context.getDynamicBufferContext();
    auto resolved = resolveInternalPointers(mutableBuffer, dynBufCtx);
    executionDelegator.set_stream(stream);
    auto& graphPack = context.getCurrentCudaGraphInfo();
    OPENVINO_ASSERT(graphPack.is_nested(), "For multi-graph mode graphPack should be nested");

    graphPack.add(CudaGraphInfo::create());
    CUDA::GraphCapture capture{stream};
    {
        auto scope = capture.getScope();
        // Input mapping of ports
        for (auto& slice : slices_) {
            slice.addKernelNode(graphPack, stream,
                                inputTensors[slice.inputIdx()].get(),
                                resolved.param_outputs[slice.paramIdx()].get());
        }
    }
    graphPack.set_current_graph(capture.getGraph());

    auto& bodyGraphInfo = graphPack.add(CudaGraphPack::create());
    context.setCurrentCudaGraphInfo(bodyGraphInfo);
    // Inner loop
    runner_->Capture(context, workbuffers);

    graphPack.add(CudaGraphInfo::create());
    CUDA::GraphCapture capture2{stream};
    {
        auto scope = capture2.getScope();
        // Back-edge mapping
        for (auto& transfer : transfers_) {
            transfer.addTransferNode(graphPack, stream,
                                     resolved.result_inputs[transfer.resultIdx()].get(),
                                     resolved.param_outputs[transfer.paramIdx()].get());
        }

        // Output mapping of ports
        for (auto& insert : inserts_) {
            insert.addKernelNode(graphPack, stream,
                                 resolved.result_inputs[insert.resultIdx()].get(),
                                 outputTensors[insert.outputIdx()].get());
        }
    }
    graphPack.set_current_graph(capture2.getGraph());
}

void TensorIteratorOp::ExecuteGraphMulti(InferenceRequestContext& context,
                                         Inputs inputTensors,
                                         Outputs outputTensors,
                                         const Workbuffers& workbuffers) const {
    const auto& stream = context.getThreadContext().stream();
    const auto& mutableBuffer = workbuffers.mutable_buffers.at(0);
    const auto& dynBufCtx = context.getDynamicBufferContext();
    auto resolved = resolveInternalPointers(mutableBuffer, dynBufCtx);

    // First iteration
    for (const auto inputIdx : invariant_inputs_) {
        const auto paramIdx = inputs_parameters_map_.at(inputIdx);
        OPENVINO_ASSERT(inputs_info_[inputIdx].size_ == params_info_[paramIdx].size_, "Node name: ", GetName());
        stream.transfer(resolved.param_outputs[paramIdx], inputTensors[inputIdx], inputs_info_[inputIdx].size_);
    }
    for (const auto& [inputIdx, paramIdx] : inputs_parameters_map_) {
        if (portmap_inputs_.count(inputIdx) == 0) {
            OPENVINO_ASSERT(inputs_info_[inputIdx].size_ == params_info_[paramIdx].size_, "Node name: ", GetName());
            stream.transfer(resolved.param_outputs[paramIdx], inputTensors[inputIdx], inputs_info_[inputIdx].size_);
        }
    }

    auto& graphPack = context.getCurrentCudaGraphInfo();
    OPENVINO_ASSERT(graphPack.get_kernels_count() == slices_.size() + inserts_.size(),
                    "CudaGraphContext/TensorIteratorOp slices or inserts count incosistency");

    OPENVINO_ASSERT(graphPack.get_graphs_count() == 3, "Current graphPack should contain 3 sub-elements");

    graphPack.select_current_graph(0);
    auto& preInfo = graphPack.get_current_graph();

    graphPack.select_current_graph(1);
    auto& bodyContext = graphPack.get_current_graph();

    graphPack.select_current_graph(2);
    auto& postInfo = graphPack.get_current_graph();

    // TI body loop
    for (int64_t iter = 0; iter < num_iterations_; ++iter) {
        for (std::size_t i = 0; i < slices_.size(); ++i) {
            slices_[i].updateKernelNode(preInfo, i,
                                        inputTensors[slices_[i].inputIdx()].get(),
                                        resolved.param_outputs[slices_[i].paramIdx()].get(),
                                        iter);
        }
        preInfo.launch(stream);

        context.setCurrentCudaGraphInfo(bodyContext);
        runner_->Run(context, workbuffers);

        for (std::size_t i = 0; i < inserts_.size(); ++i) {
            inserts_[i].updateKernelNode(postInfo, i,
                                         resolved.result_inputs[inserts_[i].resultIdx()].get(),
                                         outputTensors[inserts_[i].outputIdx()].get(),
                                         iter);
        }
        postInfo.launch(stream);
    }
    // Copy data to output
    if (iterations_results_map_.count(num_iterations_ - 1) > 0) {
        for (const auto& resultIdx : iterations_results_map_.at(num_iterations_ - 1)) {
            const auto& outputIdx = results_outputs_map_.at(resultIdx);
            OPENVINO_ASSERT(results_info_[resultIdx].size_ == outputs_info_[outputIdx].size_, "Node name: ", GetName());
            stream.transfer(outputTensors[outputIdx], resolved.result_inputs[resultIdx], outputs_info_[outputIdx].size_);
        }
    }
}

TensorIteratorOp::ResolvedPointers TensorIteratorOp::resolveInternalPointers(
    CUDA::DevicePointer<void*> mutableBuffer,
    const DynamicBufferContext& dynBufCtx) const {
    const auto& mm = *memory_manager_;
    ResolvedPointers resolved;
    resolved.param_outputs.reserve(params_.size());
    resolved.result_inputs.reserve(results_.size());
    for (std::size_t i = 0; i < params_.size(); ++i) {
        resolved.param_outputs.push_back(mm.outputTensorPointers(*params_[i], mutableBuffer, dynBufCtx)[0]);
    }
    for (std::size_t i = 0; i < results_.size(); ++i) {
        resolved.result_inputs.push_back(mm.inputTensorPointers(*results_[i], mutableBuffer, dynBufCtx)[0]);
    }
    return resolved;
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
