// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_dynamic_operation.hpp"

#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>

#include "cuda_operation_registry.hpp"
#include "ops/parameter.hpp"
#include "ops/result.hpp"

namespace ov {
namespace nvidia_gpu {

DynamicOperation::DynamicOperation(const CreationContext& context,
                                   const std::shared_ptr<ov::Node>& node,
                                   IndexCollection&& inputIds,
                                   IndexCollection&& outputIds)
    : OperationBase(context, *node, std::move(inputIds), IndexCollection{}),
      original_node_{node},
      creation_context_{context},
      dynamic_output_ids_{std::move(outputIds)} {}

void DynamicOperation::Execute(const InferenceRequestContext& context,
                               Inputs inputTensors,
                               Outputs /*outputTensors*/,
                               const Workbuffers& /*workbuffers*/) const {
    auto& dynBufCtx = const_cast<InferenceRequestContext&>(context).getDynamicBufferContext();
    const auto& stream = context.getThreadContext().stream();

    if (auto paramNode = std::dynamic_pointer_cast<ov::op::v0::Parameter>(original_node_)) {
        executeParameter(*paramNode, context, stream, dynBufCtx);
        return;
    }

    if (auto resultNode = std::dynamic_pointer_cast<ov::op::v0::Result>(original_node_)) {
        executeResult(*resultNode, context, stream, dynBufCtx);
        return;
    }

    // 1. Collect actual input shapes: from DynamicBufferContext for dynamic inputs,
    //    from the original node for static inputs (e.g. Constant).
    ShapeKey key;
    key.input_shapes.reserve(input_ids_.size());
    for (size_t i = 0; i < input_ids_.size(); ++i) {
        BufferID bufId = input_ids_[i].GetBuffer().GetId();
        if (dynBufCtx.hasShape(bufId)) {
            key.input_shapes.push_back(dynBufCtx.getShape(bufId));
        } else {
            key.input_shapes.push_back(original_node_->get_input_shape(i));
        }
    }

    // 2. Lookup or create cached static operation
    std::shared_ptr<CachedOperation> cached;
    {
        std::lock_guard<std::mutex> lock{cache_mutex_};
        auto* found = shape_cache_.find(key);
        if (found) {
            cached = *found;
        } else {
            cached = createCachedOperation(key);
            shape_cache_.insert(key, cached);
        }
    }

    // 3. Resolve input pointers (check DynamicBufferContext for overrides)
    std::vector<CUDA::DevicePointer<const void*>> input_ptrs;
    input_ptrs.reserve(input_ids_.size());
    for (size_t i = 0; i < input_ids_.size(); ++i) {
        auto dynBuf = dynBufCtx.getDynamicBuffer(input_ids_[i].GetBuffer().GetId());
        if (dynBuf) {
            input_ptrs.emplace_back(dynBuf->get());
        } else {
            input_ptrs.push_back(inputTensors[i]);
        }
    }

    // 4. Allocate dynamic memory for outputs via stream-ordered allocation
    std::vector<CUDA::Allocation> output_allocs;
    std::vector<CUDA::DevicePointer<void*>> output_ptrs;
    output_allocs.reserve(cached->output_sizes.size());
    output_ptrs.reserve(cached->output_sizes.size());
    for (size_t i = 0; i < cached->output_sizes.size(); ++i) {
        size_t sz = std::max(cached->output_sizes[i], size_t{1});
        auto alloc = stream.malloc(sz);
        output_ptrs.emplace_back(alloc.get());
        output_allocs.push_back(std::move(alloc));
    }

    // 5. Allocate dynamic memory for mutable workbuffers
    Workbuffers dyn_workbuffers;
    std::vector<CUDA::Allocation> wb_allocs;
    for (size_t sz : cached->workbuffer_request.mutable_sizes) {
        auto alloc = stream.malloc(std::max(sz, size_t{1}));
        dyn_workbuffers.mutable_buffers.emplace_back(alloc.get());
        wb_allocs.push_back(std::move(alloc));
    }
    // Immutable workbuffers come from the cached operation (persistent)
    dyn_workbuffers.immutable_buffers.reserve(cached->immutable_wb_ptrs.size());
    for (const auto& ptr : cached->immutable_wb_ptrs) {
        dyn_workbuffers.immutable_buffers.emplace_back(ptr.get());
    }

    // 6. Delegate execution to cached static operation
    cached->operation->Execute(context, input_ptrs, output_ptrs, dyn_workbuffers);

    // 7. Register output shapes and dynamic buffers
    for (size_t i = 0; i < dynamic_output_ids_.size(); ++i) {
        BufferID outId = dynamic_output_ids_[i].GetBuffer().GetId();
        dynBufCtx.registerDynamicBuffer(outId, std::move(output_allocs[i]), cached->output_shapes[i]);
    }

    // 8. Release dynamic buffers whose last consumer is this operation
    for (BufferID id : release_ids_) {
        dynBufCtx.releaseDynamicBuffer(id);
    }
}

void DynamicOperation::executeParameter(const ov::op::v0::Parameter& paramNode,
                                         const InferenceRequestContext& context,
                                         const CUDA::Stream& stream,
                                         DynamicBufferContext& dynBufCtx) const {
    auto tensor = context.getTensorMappingContext().get_input_tensor(
        ParameterOp::GetInputTensorName(paramNode));
    auto shape = tensor->get_shape();
    size_t byte_size = tensor->get_byte_size();
    auto alloc = stream.malloc(std::max(byte_size, size_t{1}));
    stream.upload(CUDA::DevicePointer<void*>{alloc.get()}, tensor->data(), byte_size);
    BufferID outBufId = dynamic_output_ids_[0].GetBuffer().GetId();
    dynBufCtx.registerDynamicBuffer(outBufId, std::move(alloc), shape);
}

void DynamicOperation::executeResult(const ov::op::v0::Result& resultNode,
                                      const InferenceRequestContext& context,
                                      const CUDA::Stream& stream,
                                      DynamicBufferContext& dynBufCtx) const {
    BufferID inputBufId = input_ids_[0].GetBuffer().GetId();
    auto dynBuf = dynBufCtx.getDynamicBuffer(inputBufId);
    if (!dynBuf) {
        return;
    }
    auto shape = dynBufCtx.getShape(inputBufId);
    auto elemType = resultNode.get_output_element_type(0);
    auto names = ResultOp::GetOutputTensorName(resultNode);
    std::shared_ptr<ov::Tensor> tensor;
    for (const auto& name : names) {
        if (context.getTensorMappingContext().has_output_tensor(name)) {
            tensor = context.getTensorMappingContext().get_output_tensor(name);
            break;
        }
    }
    OPENVINO_ASSERT(tensor, "Output tensor not found for Result: ", resultNode.get_friendly_name());
    *tensor = ov::Tensor(elemType, shape);
    stream.download(tensor->data(),
                    CUDA::DevicePointer<const void*>{dynBuf->get()},
                    tensor->get_byte_size());
}

std::shared_ptr<CachedOperation> DynamicOperation::createCachedOperation(const ShapeKey& key) const {
    // 1. Create temporary Parameter nodes with concrete shapes
    ov::OutputVector new_inputs;
    new_inputs.reserve(original_node_->get_input_size());
    for (size_t i = 0; i < original_node_->get_input_size(); ++i) {
        auto param = std::make_shared<ov::op::v0::Parameter>(
            original_node_->get_input_element_type(i),
            key.input_shapes[i]);
        new_inputs.push_back(param->output(0));
    }

    // 2. Clone node with concrete shapes and infer output types/shapes
    auto cloned = original_node_->clone_with_new_inputs(new_inputs);
    cloned->validate_and_infer_types();

    // 3. Collect output shapes and sizes
    std::vector<ov::Shape> output_shapes;
    std::vector<size_t> output_sizes;
    output_shapes.reserve(cloned->get_output_size());
    output_sizes.reserve(cloned->get_output_size());
    for (size_t i = 0; i < cloned->get_output_size(); ++i) {
        const auto& shape = cloned->get_output_shape(i);
        output_shapes.push_back(shape);
        output_sizes.push_back(cloned->get_output_element_type(i).size() *
                               std::max(size_t{1}, ov::shape_size(shape)));
    }

    // 4. Create dummy TensorIDs for the inner operation
    IndexCollection dummy_in, dummy_out;
    for (size_t i = 0; i < cloned->get_input_size(); ++i) {
        dummy_in.push_back(TensorID{static_cast<BufferID>(i)});
    }
    for (size_t i = 0; i < cloned->get_output_size(); ++i) {
        dummy_out.push_back(TensorID{static_cast<BufferID>(cloned->get_input_size() + i)});
    }

    // 5. Create the static operation via registry
    auto operation = OperationRegistry::getInstance().createOperation(
        creation_context_, cloned, std::move(dummy_in), std::move(dummy_out));

    // 6. Handle workbuffers
    WorkbufferRequest wb_request = operation->GetWorkBufferRequest();

    // Allocate persistent memory for immutable workbuffers and initialize.
    // Uses DefaultStream because these buffers are shared across inference
    // requests on different streams. DefaultStream (stream 0) has implicit
    // synchronization with all other streams, ensuring initialization
    // completes before any inference stream reads the data.
    std::vector<CUDA::DefaultAllocation> immutable_wb_allocs;
    std::vector<CUDA::DevicePointer<void*>> immutable_wb_ptrs;
    if (!wb_request.immutable_sizes.empty()) {
        IOperationExec::Buffers init_buffers;
        for (size_t sz : wb_request.immutable_sizes) {
            auto alloc = CUDA::DefaultStream::stream().malloc(std::max(sz, size_t{1}));
            immutable_wb_ptrs.emplace_back(alloc.get());
            init_buffers.emplace_back(alloc.get());
            immutable_wb_allocs.push_back(std::move(alloc));
        }
        operation->InitSharedImmutableWorkbuffers(init_buffers);
    }

    return std::make_shared<CachedOperation>(CachedOperation{std::move(operation),
                                                              std::move(output_shapes),
                                                              std::move(output_sizes),
                                                              std::move(wb_request),
                                                              std::move(immutable_wb_allocs),
                                                              std::move(immutable_wb_ptrs)});
}

}  // namespace nvidia_gpu
}  // namespace ov
