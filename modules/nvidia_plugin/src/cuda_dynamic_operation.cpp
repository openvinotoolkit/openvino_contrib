// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_dynamic_operation.hpp"

#include <openvino/op/constant.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/op/util/assign_base.hpp>
#include <openvino/op/util/read_value_base.hpp>
#include <openvino/op/util/variable_extension.hpp>

#include "cuda_operation_registry.hpp"
#include "cuda_variable_state.hpp"
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
    auto& shapeCtx = const_cast<InferenceRequestContext&>(context).getShapeContext();
    auto& dynBufCtx = const_cast<InferenceRequestContext&>(context).getDynamicBufferContext();
    const auto& stream = context.getThreadContext().stream();

    if (auto paramNode = std::dynamic_pointer_cast<ov::op::v0::Parameter>(original_node_)) {
        executeParameter(*paramNode, context, stream, shapeCtx, dynBufCtx);
        return;
    }

    if (auto resultNode = std::dynamic_pointer_cast<ov::op::v0::Result>(original_node_)) {
        executeResult(*resultNode, context, stream, shapeCtx, dynBufCtx);
        return;
    }

    if (auto readValueNode = std::dynamic_pointer_cast<ov::op::util::ReadValueBase>(original_node_)) {
        executeReadValue(*readValueNode, context, stream, inputTensors, shapeCtx, dynBufCtx);
        return;
    }

    if (auto assignNode = std::dynamic_pointer_cast<ov::op::util::AssignBase>(original_node_)) {
        executeAssign(*assignNode, context, stream, inputTensors, shapeCtx, dynBufCtx);
        return;
    }

    // 1. Collect actual input shapes: from ShapeContext for dynamic inputs,
    //    from the original node for static inputs (e.g. Constant).
    ShapeKey key;
    key.input_shapes.reserve(input_ids_.size());
    for (size_t i = 0; i < input_ids_.size(); ++i) {
        BufferID bufId = input_ids_[i].GetBuffer().GetId();
        if (shapeCtx.hasShape(bufId)) {
            key.input_shapes.push_back(shapeCtx.getShape(bufId));
        } else if (original_node_->get_input_partial_shape(i).is_static()) {
            key.input_shapes.push_back(original_node_->get_input_shape(i));
        } else {
            OPENVINO_THROW("DynamicOperation '", GetName(), "': input ", i,
                           " has dynamic shape ", original_node_->get_input_partial_shape(i),
                           " but no shape registered in ShapeContext (bufId=", bufId, ")");
        }
    }

    // 2. Resolve input pointers (check DynamicBufferContext for overrides).
    std::vector<CUDA::DevicePointer<const void*>> input_ptrs;
    input_ptrs.reserve(input_ids_.size());
    for (size_t i = 0; i < input_ids_.size(); ++i) {
        auto dynBuf = dynBufCtx.getDynamicOutput(input_ids_[i].GetBuffer().GetId());
        if (dynBuf) {
            input_ptrs.emplace_back(dynBuf->get());
        } else {
            input_ptrs.push_back(inputTensors[i]);
        }
    }

    // 2b. Include small integer tensor values in cache key.
    //     Operations like Broadcast, Reshape need input VALUES (not just shapes)
    //     to determine output shapes. Without this, cache hits return stale
    //     operations when shapes stay the same but values change between inferences.
    if (needs_value_cache_) {
        bool synced = false;
        key.input_values.resize(input_ids_.size());
        for (size_t i = 0; i < input_ids_.size(); ++i) {
            auto source_node = original_node_->get_input_node_shared_ptr(i);
            if (std::dynamic_pointer_cast<ov::op::v0::Constant>(source_node)) continue;
            const auto& shape = key.input_shapes[i];
            auto elem_type = original_node_->get_input_element_type(i);
            size_t num_elements = ov::shape_size(shape);
            bool is_shape_value = (shape.size() <= 1 && num_elements > 0 && num_elements <= 64 &&
                (elem_type == ov::element::i32 || elem_type == ov::element::i64));
            if (is_shape_value) {
                if (!synced) { stream.synchronize(); synced = true; }
                size_t byte_size = elem_type.size() * num_elements;
                std::vector<uint8_t> host_data(byte_size);
                stream.download(host_data.data(),
                                CUDA::DevicePointer<const void*>{input_ptrs[i].get()},
                                byte_size);
                auto& vals = key.input_values[i];
                vals.resize(num_elements);
                if (elem_type == ov::element::i64) {
                    auto* p = reinterpret_cast<const int64_t*>(host_data.data());
                    std::copy(p, p + num_elements, vals.begin());
                } else {
                    auto* p = reinterpret_cast<const int32_t*>(host_data.data());
                    for (size_t e = 0; e < num_elements; ++e) vals[e] = p[e];
                }
            }
        }
    }

    // 3. Lookup or create cached static operation
    std::shared_ptr<CachedOperation> cached;
    {
        std::lock_guard<std::mutex> lock{cache_mutex_};
        auto* found = shape_cache_.find(key);
        if (found) {
            cached = *found;
        } else {
            cached = createCachedOperation(key, input_ptrs, stream);
            shape_cache_.insert(key, cached);
        }
    }

    // 4. Allocate dynamic memory for outputs via stream-ordered allocation
    std::vector<CUDA::Allocation> output_allocs;
    output_allocs.reserve(cached->output_sizes.size());
    for (size_t i = 0; i < cached->output_sizes.size(); ++i) {
        size_t sz = std::max(cached->output_sizes[i], size_t{1});
        output_allocs.push_back(stream.malloc(sz));
    }

    // 5. Execute the cached operation (skip if no-op, e.g. zero-element outputs)
    bool isReshapeLike = ov::is_type<ov::op::v1::Reshape>(original_node_) ||
                         ov::is_type<ov::op::v0::Squeeze>(original_node_) ||
                         ov::is_type<ov::op::v0::Unsqueeze>(original_node_);

    if (isReshapeLike) {
        // Reshape-like operations are zero-copy: same data, different shape.
        // The registry creates a NopOp for them (empty Execute), so we must
        // copy data from input to output ourselves.
        OPENVINO_ASSERT(!input_ptrs.empty() && !output_allocs.empty());
        stream.transfer(CUDA::DevicePointer<void*>{output_allocs[0].get()},
                        CUDA::DevicePointer<const void*>{input_ptrs[0].get()},
                        cached->output_sizes[0]);
    } else if (cached->operation) {
        std::vector<CUDA::DevicePointer<void*>> output_ptrs;
        output_ptrs.reserve(output_allocs.size());
        for (auto& alloc : output_allocs) {
            output_ptrs.emplace_back(alloc.get());
        }

        // Allocate dynamic memory for mutable workbuffers
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

        cached->operation->Execute(context, input_ptrs, output_ptrs, dyn_workbuffers);
    }

    // 6. Register output shapes and dynamic buffers
    for (size_t i = 0; i < dynamic_output_ids_.size(); ++i) {
        BufferID outId = dynamic_output_ids_[i].GetBuffer().GetId();
        shapeCtx.setShape(outId, cached->output_shapes[i]);
        dynBufCtx.registerDynamicOutput(outId, std::move(output_allocs[i]));
    }

    // 7. Release dynamic buffers whose last consumer is this operation
    for (BufferID id : release_ids_) {
        dynBufCtx.releaseDynamicBuffer(id);
    }
}

void DynamicOperation::executeParameter(const ov::op::v0::Parameter& paramNode,
                                         const InferenceRequestContext& context,
                                         const CUDA::Stream& stream,
                                         ShapeContext& shapeCtx,
                                         DynamicBufferContext& dynBufCtx) const {
    auto tensor = context.getTensorMappingContext().get_input_tensor(
        ParameterOp::GetInputTensorName(paramNode));
    auto shape = tensor->get_shape();
    size_t byte_size = tensor->get_byte_size();
    auto alloc = stream.malloc(std::max(byte_size, size_t{1}));
    stream.upload(CUDA::DevicePointer<void*>{alloc.get()}, tensor->data(), byte_size);
    BufferID outBufId = dynamic_output_ids_[0].GetBuffer().GetId();
    shapeCtx.setShape(outBufId, shape);
    dynBufCtx.registerDynamicOutput(outBufId, std::move(alloc));
}

void DynamicOperation::executeResult(const ov::op::v0::Result& resultNode,
                                      const InferenceRequestContext& context,
                                      const CUDA::Stream& stream,
                                      ShapeContext& shapeCtx,
                                      DynamicBufferContext& dynBufCtx) const {
    BufferID inputBufId = input_ids_[0].GetBuffer().GetId();
    auto dynBuf = dynBufCtx.getDynamicOutput(inputBufId);
    if (!dynBuf) {
        return;
    }
    auto shape = shapeCtx.getShape(inputBufId);
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

void DynamicOperation::executeReadValue(const ov::op::util::ReadValueBase& readValueNode,
                                         const InferenceRequestContext& context,
                                         const CUDA::Stream& stream,
                                         Inputs inputTensors,
                                         ShapeContext& shapeCtx,
                                         DynamicBufferContext& dynBufCtx) const {
    OPENVINO_ASSERT(context.hasVariableContext(), "ReadValue requires VariableContext");
    auto& varCtx = context.getVariableContext();
    auto variable_id = dynamic_cast<const ov::op::util::VariableExtension&>(readValueNode).get_variable_id();
    auto state = varCtx.get_variable_state(variable_id);

    ov::Shape shape;
    size_t byte_size;

    if (state->is_reset_state()) {
        // First inference or after reset
        if (!input_ids_.empty() && !inputTensors.empty()) {
            // Use init_value shape from the first input
            BufferID initBufId = input_ids_[0].GetBuffer().GetId();
            if (shapeCtx.hasShape(initBufId)) {
                shape = shapeCtx.getShape(initBufId);
            } else if (readValueNode.get_input_partial_shape(0).is_static()) {
                shape = readValueNode.get_input_shape(0);
            } else {
                shape = state->shape();
            }
            byte_size = readValueNode.get_output_element_type(0).size() *
                        std::max(ov::shape_size(shape), size_t{1});

            auto alloc = stream.malloc(std::max(byte_size, size_t{1}));
            // Copy init_value to output
            auto dynBuf = dynBufCtx.getDynamicOutput(initBufId);
            if (dynBuf) {
                stream.transfer(CUDA::DevicePointer<void*>{alloc.get()},
                                CUDA::DevicePointer<const void*>{dynBuf->get()},
                                byte_size);
            } else {
                stream.transfer(CUDA::DevicePointer<void*>{alloc.get()},
                                CUDA::DevicePointer<const void*>{inputTensors[0].get()},
                                byte_size);
            }
            BufferID outBufId = dynamic_output_ids_[0].GetBuffer().GetId();
            shapeCtx.setShape(outBufId, shape);
            dynBufCtx.registerDynamicOutput(outBufId, std::move(alloc));
        } else {
            // No init_value: zero-fill
            shape = state->shape();
            byte_size = readValueNode.get_output_element_type(0).size() *
                        std::max(ov::shape_size(shape), size_t{1});
            auto alloc = stream.malloc(std::max(byte_size, size_t{1}));
            stream.memset(alloc, 0, byte_size);
            BufferID outBufId = dynamic_output_ids_[0].GetBuffer().GetId();
            shapeCtx.setShape(outBufId, shape);
            dynBufCtx.registerDynamicOutput(outBufId, std::move(alloc));
        }
    } else {
        // Normal case: copy from state buffer
        shape = state->shape();
        byte_size = state->device_buffer_byte_size();
        auto alloc = stream.malloc(std::max(byte_size, size_t{1}));
        if (byte_size > 0) {
            state->read_device_state(stream, CUDA::DevicePointer<void*>{alloc.get()}, byte_size);
        }
        BufferID outBufId = dynamic_output_ids_[0].GetBuffer().GetId();
        shapeCtx.setShape(outBufId, shape);
        dynBufCtx.registerDynamicOutput(outBufId, std::move(alloc));
    }
}

void DynamicOperation::executeAssign(const ov::op::util::AssignBase& assignNode,
                                      const InferenceRequestContext& context,
                                      const CUDA::Stream& stream,
                                      Inputs inputTensors,
                                      ShapeContext& shapeCtx,
                                      DynamicBufferContext& dynBufCtx) const {
    OPENVINO_ASSERT(context.hasVariableContext(), "Assign requires VariableContext");
    auto& varCtx = context.getVariableContext();
    auto variable_id = dynamic_cast<const ov::op::util::VariableExtension&>(assignNode).get_variable_id();
    auto state = varCtx.get_variable_state(variable_id);

    // Resolve input pointer (from DynamicBufferContext or regular memory)
    BufferID inputBufId = input_ids_[0].GetBuffer().GetId();
    auto dynBuf = dynBufCtx.getDynamicOutput(inputBufId);

    const void* raw_ptr = nullptr;
    if (dynBuf) {
        raw_ptr = dynBuf->get();
    } else if (!inputTensors.empty()) {
        raw_ptr = inputTensors[0].get();
    }
    OPENVINO_ASSERT(raw_ptr != nullptr, "Assign '", GetName(), "': could not resolve input pointer");

    // Get input shape
    ov::Shape shape;
    if (shapeCtx.hasShape(inputBufId)) {
        shape = shapeCtx.getShape(inputBufId);
    } else if (assignNode.get_input_partial_shape(0).is_static()) {
        shape = assignNode.get_input_shape(0);
    } else {
        OPENVINO_THROW("Assign '", GetName(), "': input has dynamic shape but no shape in ShapeContext");
    }

    // Update variable state (D2D copy)
    state->update_device_state(stream, CUDA::DevicePointer<const void*>{raw_ptr}, shape);

    // Release input dynamic buffer if this is the last consumer
    for (BufferID id : release_ids_) {
        dynBufCtx.releaseDynamicBuffer(id);
    }
}

std::shared_ptr<CachedOperation> DynamicOperation::createCachedOperation(
        const ShapeKey& key,
        const std::vector<CUDA::DevicePointer<const void*>>& input_ptrs,
        const CUDA::Stream& stream) const {
    // 1. Build input vector for cloning. Three categories of inputs:
    //    a) Constant inputs (e.g. axis for Gather) → preserve original Constant
    //    b) Data inputs → replace with Parameter having concrete shape
    //    c) Shape-value inputs (e.g. target_shape for Broadcast/Reshape) →
    //       initially Parameters; promoted to Constants on retry if needed
    ov::OutputVector new_inputs;
    new_inputs.reserve(original_node_->get_input_size());
    for (size_t i = 0; i < original_node_->get_input_size(); ++i) {
        auto source_node = original_node_->get_input_node_shared_ptr(i);
        if (std::dynamic_pointer_cast<ov::op::v0::Constant>(source_node)) {
            new_inputs.push_back(original_node_->input_value(i));
        } else {
            auto param = std::make_shared<ov::op::v0::Parameter>(
                original_node_->get_input_element_type(i),
                key.input_shapes[i]);
            new_inputs.push_back(param->output(0));
        }
    }

    // 2. Clone node with concrete shapes and infer output types/shapes
    auto cloned = original_node_->clone_with_new_inputs(new_inputs);
    cloned->validate_and_infer_types();

    // 3. Check if all output shapes resolved to static.
    //    Operations like Broadcast, Reshape, Squeeze need input VALUES (not just
    //    shapes) to determine output shapes. If outputs remain dynamic, retry by
    //    downloading small integer tensor values from GPU and creating Constants.
    bool has_dynamic_output = false;
    for (size_t i = 0; i < cloned->get_output_size(); ++i) {
        if (cloned->get_output_partial_shape(i).is_dynamic()) {
            has_dynamic_output = true;
            break;
        }
    }

    if (has_dynamic_output) {
        // Mark this operation as needing value-based cache keys for future calls.
        needs_value_cache_ = true;

        // Synchronize stream to ensure all prior GPU ops have completed their writes
        stream.synchronize();

        new_inputs.clear();
        for (size_t i = 0; i < original_node_->get_input_size(); ++i) {
            auto source_node = original_node_->get_input_node_shared_ptr(i);
            if (std::dynamic_pointer_cast<ov::op::v0::Constant>(source_node)) {
                new_inputs.push_back(original_node_->input_value(i));
            } else {
                const auto& shape = key.input_shapes[i];
                auto elem_type = original_node_->get_input_element_type(i);
                size_t num_elements = ov::shape_size(shape);
                size_t byte_size = elem_type.size() * std::max(num_elements, size_t{1});

                // Small integer tensors are likely shape/index values needed
                // for shape inference (Broadcast target_shape, Reshape pattern, etc.)
                bool is_shape_value = (shape.size() <= 1 && num_elements <= 64 &&
                    (elem_type == ov::element::i32 || elem_type == ov::element::i64));

                if (is_shape_value && num_elements > 0) {
                    std::vector<uint8_t> host_data(byte_size);
                    stream.download(host_data.data(),
                                    CUDA::DevicePointer<const void*>{input_ptrs[i].get()},
                                    byte_size);
                    // Debug: print downloaded values
                    auto const_node = std::make_shared<ov::op::v0::Constant>(
                        elem_type, shape, host_data.data());
                    new_inputs.push_back(const_node->output(0));
                } else {
                    auto param = std::make_shared<ov::op::v0::Parameter>(elem_type, shape);
                    new_inputs.push_back(param->output(0));
                }
            }
        }
        cloned = original_node_->clone_with_new_inputs(new_inputs);
        cloned->validate_and_infer_types();
    }

    // 4. Collect output shapes and sizes
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

    // 5. Check if any OUTPUT has zero elements. If so, the inner operation
    //    should be a no-op because CUDA kernels generally can't handle zero-extent
    //    tensors. We still compute correct output shapes via
    //    validate_and_infer_types() and register them in ShapeContext so that
    //    downstream shape computations (ShapeOf, Reshape, etc.) work correctly.
    //    NOTE: We do NOT check inputs — operations like Concat can have a
    //    zero-element input but still produce a valid non-zero output.
    bool has_zero_element_output = false;
    for (const auto& shape : output_shapes) {
        if (ov::shape_size(shape) == 0) { has_zero_element_output = true; break; }
    }
    if (has_zero_element_output) {
        return std::make_shared<CachedOperation>(CachedOperation{nullptr,
                                                                  std::move(output_shapes),
                                                                  std::move(output_sizes),
                                                                  WorkbufferRequest{},
                                                                  {},
                                                                  {}});
    }

    // 6. Create dummy TensorIDs for the inner operation
    IndexCollection dummy_in, dummy_out;
    for (size_t i = 0; i < cloned->get_input_size(); ++i) {
        dummy_in.push_back(TensorID{static_cast<BufferID>(i)});
    }
    for (size_t i = 0; i < cloned->get_output_size(); ++i) {
        dummy_out.push_back(TensorID{static_cast<BufferID>(cloned->get_input_size() + i)});
    }

    // 7. Create the static operation via registry
    auto operation = OperationRegistry::getInstance().createOperation(
        creation_context_, cloned, std::move(dummy_in), std::move(dummy_out));

    // 8. Handle workbuffers
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
