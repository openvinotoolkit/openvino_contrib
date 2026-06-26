// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_dynamic_operation.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
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

namespace {
// Small integer 0-D/1-D tensors are likely shape/index values (Broadcast
// target_shape, Reshape pattern, ...) that drive output-shape inference.
bool isShapeValueInput(const ov::Shape& shape, const ov::element::Type& elem_type) {
    const size_t num_elements = ov::shape_size(shape);
    return shape.size() <= 1 && num_elements > 0 && num_elements <= 64 &&
           (elem_type == ov::element::i32 || elem_type == ov::element::i64);
}

// Reshape/Squeeze/Unsqueeze are zero-copy metadata ops, modeled by the registry
// as a NopOp (empty Execute), so DynamicOperation copies the data itself.
bool isReshapeLike(const std::shared_ptr<ov::Node>& node) {
    return ov::is_type<ov::op::v1::Reshape>(node) ||
           ov::is_type<ov::op::v0::Squeeze>(node) ||
           ov::is_type<ov::op::v0::Unsqueeze>(node);
}

// Download a small integer shape/index input (see isShapeValueInput) into host
// int64 values. Used both to make the cache key value-sensitive (prepare 2b)
// and to constant-fold the values during shape inference (createCachedOperation).
// No explicit synchronization is needed: the copy is stream-ordered after the
// producing op (same per-request stream), and a device-to-pageable-host copy
// returns to the host only once it has completed.
std::vector<int64_t> downloadShapeValues(const CUDA::Stream& stream,
                                         CUDA::DevicePointer<const void*> src,
                                         const ov::Shape& shape,
                                         const ov::element::Type& elem_type) {
    const size_t num_elements = ov::shape_size(shape);
    std::vector<uint8_t> bytes(elem_type.size() * num_elements);
    stream.download(bytes.data(), src, bytes.size());
    std::vector<int64_t> values(num_elements);
    if (elem_type == ov::element::i64) {
        const auto* p = reinterpret_cast<const int64_t*>(bytes.data());
        std::copy(p, p + num_elements, values.begin());
    } else {
        const auto* p = reinterpret_cast<const int32_t*>(bytes.data());
        std::copy(p, p + num_elements, values.begin());
    }
    return values;
}
}  // namespace

DynamicOperation::DynamicOperation(const CreationContext& context,
                                   const std::shared_ptr<ov::Node>& node,
                                   IndexCollection&& inputIds,
                                   IndexCollection&& outputIds)
    : OperationBase(context, *node, std::move(inputIds), IndexCollection{}),
      original_node_{node},
      creation_context_{context},
      dynamic_output_ids_{std::move(outputIds)} {
    has_dynamic_output_ = computeHasDynamicOutput();
}

bool DynamicOperation::computeHasDynamicOutput() const {
    // Value-dependence is a structural property of the node: clone it with concrete
    // placeholder input shapes (dynamic dims -> 1) and check whether the output
    // stays dynamic. If it does, the output depends on input VALUES (Broadcast
    // target shape, Reshape pattern, ...) and the cache key must fold those values
    // in. Done once here (single-threaded) instead of lazily in the const Execute()
    // path, so the very first cache key is already value-sensitive and the shared
    // flag is never written concurrently.
    try {
        ov::OutputVector new_inputs;
        new_inputs.reserve(original_node_->get_input_size());
        for (size_t i = 0; i < original_node_->get_input_size(); ++i) {
            if (std::dynamic_pointer_cast<ov::op::v0::Constant>(original_node_->get_input_node_shared_ptr(i))) {
                new_inputs.push_back(original_node_->input_value(i));
                continue;
            }
            const auto& pshape = original_node_->get_input_partial_shape(i);
            if (pshape.rank().is_dynamic()) {
                // No concrete shape available -- be conservative and always fold
                // values (safe: never under-keys, at worst an extra cache entry).
                return true;
            }
            ov::Shape concrete;
            concrete.reserve(pshape.size());
            for (const auto& d : pshape) {
                concrete.push_back(d.is_static() ? d.get_length() : 1);
            }
            new_inputs.push_back(
                std::make_shared<ov::op::v0::Parameter>(original_node_->get_input_element_type(i), concrete)
                    ->output(0));
        }
        auto cloned = original_node_->clone_with_new_inputs(new_inputs);
        cloned->validate_and_infer_types();
        for (size_t i = 0; i < cloned->get_output_size(); ++i) {
            if (cloned->get_output_partial_shape(i).is_dynamic()) {
                return true;
            }
        }
        return false;
    } catch (...) {
        // If the probe can't run (e.g. an op needing a runtime context), assume
        // value-dependent so the key never under-keys.
        return true;
    }
}

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

    if (auto readValueNode = std::dynamic_pointer_cast<ov::op::util::ReadValueBase>(original_node_)) {
        executeReadValue(*readValueNode, context, stream, inputTensors, dynBufCtx);
        return;
    }

    if (auto assignNode = std::dynamic_pointer_cast<ov::op::util::AssignBase>(original_node_)) {
        executeAssign(*assignNode, context, stream, inputTensors, dynBufCtx);
        return;
    }

    // Cached path: prepare once (input shapes/values, pointers, cached static op,
    // output buffers), then execute (reshape-copy or kernel) and finalize
    // (register outputs, release dead buffers).
    auto ctx = prepareCachedOperationContext(context, stream, inputTensors, dynBufCtx);
    executeCachedOperation(context, stream, ctx);
    finalizeOutputs(dynBufCtx, ctx);
}

DynamicOperation::CachedOperationContext DynamicOperation::prepareCachedOperationContext(
    const InferenceRequestContext& context,
    const CUDA::Stream& stream,
    Inputs inputTensors,
    DynamicBufferContext& dynBufCtx) const {
    // 1. Collect actual input shapes: from DynamicBufferContext for dynamic
    //    inputs, from the original node for static inputs (e.g. Constant).
    ShapeKey key;
    key.input_shapes.reserve(input_ids_.size());
    for (size_t i = 0; i < input_ids_.size(); ++i) {
        BufferID bufId = input_ids_[i].GetBuffer().GetId();
        if (dynBufCtx.hasShape(bufId)) {
            key.input_shapes.push_back(dynBufCtx.getShape(bufId));
        } else if (original_node_->get_input_partial_shape(i).is_static()) {
            key.input_shapes.push_back(original_node_->get_input_shape(i));
        } else {
            OPENVINO_THROW("DynamicOperation '", GetName(), "': input ", i, " has dynamic shape ",
                           original_node_->get_input_partial_shape(i),
                           " but no shape registered in DynamicBufferContext (bufId=", bufId, ")");
        }
    }

    // 2. Resolve input pointers (DynamicBufferContext override, else static tensor).
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

    // 2b. If this op's output shape is dynamic even with concrete input shapes
    //     (it depends on input VALUES — e.g. Broadcast target shape, Reshape
    //     pattern), fold the small integer input values into the cache key so
    //     shape changes driven by values are not masked by a stale cache hit.
    if (has_dynamic_output_) {
        key.input_values.resize(input_ids_.size());
        for (size_t i = 0; i < input_ids_.size(); ++i) {
            if (std::dynamic_pointer_cast<ov::op::v0::Constant>(original_node_->get_input_node_shared_ptr(i))) {
                continue;
            }
            const auto& shape = key.input_shapes[i];
            auto elem_type = original_node_->get_input_element_type(i);
            if (!isShapeValueInput(shape, elem_type)) {
                continue;
            }
            key.input_values[i] = downloadShapeValues(stream, input_ptrs[i], shape, elem_type);
        }
    }

    // 3. Lookup or create cached static operation in the model-global cache.
    auto& cache = const_cast<InferenceRequestContext&>(context).getDynamicOperationCache();
    auto cached = cache.getOrCreate(original_node_.get(), key, [this, &key, &input_ptrs, &stream] {
        return createCachedOperation(key, input_ptrs, stream);
    });

    // 4. Allocate dynamic memory for outputs via stream-ordered allocation.
    std::vector<CUDA::Allocation> output_allocs;
    output_allocs.reserve(cached->output_sizes.size());
    for (size_t sz : cached->output_sizes) {
        output_allocs.push_back(stream.malloc(std::max(sz, size_t{1})));
    }

    return CachedOperationContext{std::move(cached), std::move(input_ptrs), std::move(output_allocs)};
}

void DynamicOperation::executeCachedOperation(const InferenceRequestContext& context,
                                              const CUDA::Stream& stream,
                                              const CachedOperationContext& ctx) const {
    if (isReshapeLike(original_node_)) {
        // Reshape/Squeeze/Unsqueeze are zero-copy metadata ops (NopOp): copy the
        // data input -> output so downstream consumers read it at the newly
        // allocated output buffer.
        OPENVINO_ASSERT(!ctx.input_ptrs.empty() && !ctx.output_allocs.empty(),
                        "Reshape-like op '", GetName(), "' has no inputs or outputs");
        stream.transfer(CUDA::DevicePointer<void*>{ctx.output_allocs[0].get()},
                        CUDA::DevicePointer<const void*>{ctx.input_ptrs[0].get()},
                        ctx.cached->output_sizes[0]);
        return;
    }

    // A null operation models a zero-element output: nothing to execute.
    if (!ctx.cached->operation) {
        return;
    }

    std::vector<CUDA::DevicePointer<void*>> output_ptrs;
    output_ptrs.reserve(ctx.output_allocs.size());
    for (auto& alloc : ctx.output_allocs) {
        output_ptrs.emplace_back(alloc.get());
    }

    Workbuffers dyn_workbuffers;
    std::vector<CUDA::Allocation> wb_allocs;
    for (size_t sz : ctx.cached->workbuffer_request.mutable_sizes) {
        auto alloc = stream.malloc(std::max(sz, size_t{1}));
        dyn_workbuffers.mutable_buffers.emplace_back(alloc.get());
        wb_allocs.push_back(std::move(alloc));
    }
    dyn_workbuffers.immutable_buffers.reserve(ctx.cached->immutable_wb_ptrs.size());
    for (const auto& ptr : ctx.cached->immutable_wb_ptrs) {
        dyn_workbuffers.immutable_buffers.emplace_back(ptr.get());
    }

    ctx.cached->operation->Execute(context, ctx.input_ptrs, output_ptrs, dyn_workbuffers);
}

void DynamicOperation::finalizeOutputs(DynamicBufferContext& dynBufCtx, CachedOperationContext& ctx) const {
    // Register output shapes and dynamic buffers.
    for (size_t i = 0; i < dynamic_output_ids_.size(); ++i) {
        BufferID outId = dynamic_output_ids_[i].GetBuffer().GetId();
        dynBufCtx.registerDynamicBuffer(outId, std::move(ctx.output_allocs[i]), ctx.cached->output_shapes[i]);
    }

    // Release dynamic buffers whose last consumer is this operation.
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

void DynamicOperation::executeReadValue(const ov::op::util::ReadValueBase& readValueNode,
                                         const InferenceRequestContext& context,
                                         const CUDA::Stream& stream,
                                         Inputs inputTensors,
                                         DynamicBufferContext& dynBufCtx) const {
    OPENVINO_ASSERT(context.hasVariableContext(), "ReadValue requires VariableContext");
    auto& varCtx = context.getVariableContext();
    auto variable_id = dynamic_cast<const ov::op::util::VariableExtension&>(readValueNode).get_variable_id();
    auto state = varCtx.get_variable_state(variable_id);
    BufferID outBufId = dynamic_output_ids_[0].GetBuffer().GetId();

    if (!state->is_reset_state()) {
        // Subsequent inferences: copy from the saved variable-state buffer.
        auto shape = state->shape();
        size_t byte_size = state->device_buffer_byte_size();
        auto alloc = stream.malloc(std::max(byte_size, size_t{1}));
        if (byte_size > 0) {
            state->read_device_state(stream, CUDA::DevicePointer<void*>{alloc.get()}, byte_size);
        }
        dynBufCtx.registerDynamicBuffer(outBufId, std::move(alloc), shape);
        return;
    }

    // First inference or after reset: output the init_value or zeros.
    bool has_init = !input_ids_.empty() && !inputTensors.empty();
    ov::Shape shape;
    if (has_init) {
        BufferID initBufId = input_ids_[0].GetBuffer().GetId();
        shape = dynBufCtx.hasShape(initBufId) ? dynBufCtx.getShape(initBufId)
                                              : readValueNode.get_input_shape(0);
    } else {
        shape = state->shape();
    }

    auto elem_type = readValueNode.get_output_element_type(0);
    size_t byte_size = elem_type.size() * std::max(ov::shape_size(shape), size_t{1});
    auto alloc = stream.malloc(std::max(byte_size, size_t{1}));

    if (has_init) {
        BufferID initBufId = input_ids_[0].GetBuffer().GetId();
        auto dynBuf = dynBufCtx.getDynamicBuffer(initBufId);
        const void* src = dynBuf ? dynBuf->get() : inputTensors[0].get();
        stream.transfer(CUDA::DevicePointer<void*>{alloc.get()},
                        CUDA::DevicePointer<const void*>{src}, byte_size);
    } else {
        stream.memset(alloc, 0, byte_size);
    }

    dynBufCtx.registerDynamicBuffer(outBufId, std::move(alloc), shape);
}

void DynamicOperation::executeAssign(const ov::op::util::AssignBase& assignNode,
                                      const InferenceRequestContext& context,
                                      const CUDA::Stream& stream,
                                      Inputs inputTensors,
                                      DynamicBufferContext& dynBufCtx) const {
    OPENVINO_ASSERT(context.hasVariableContext(), "Assign requires VariableContext");
    auto& varCtx = context.getVariableContext();
    auto variable_id = dynamic_cast<const ov::op::util::VariableExtension&>(assignNode).get_variable_id();
    auto state = varCtx.get_variable_state(variable_id);

    BufferID inputBufId = input_ids_[0].GetBuffer().GetId();
    auto dynBuf = dynBufCtx.getDynamicBuffer(inputBufId);
    const void* raw_ptr = nullptr;
    if (dynBuf) {
        raw_ptr = dynBuf->get();
    } else if (!inputTensors.empty()) {
        raw_ptr = inputTensors[0].get();
    }
    OPENVINO_ASSERT(raw_ptr != nullptr, "Assign '", GetName(), "': could not resolve input pointer");

    ov::Shape shape;
    if (dynBufCtx.hasShape(inputBufId)) {
        shape = dynBufCtx.getShape(inputBufId);
    } else if (assignNode.get_input_partial_shape(0).is_static()) {
        shape = assignNode.get_input_shape(0);
    } else {
        OPENVINO_THROW("Assign '", GetName(), "': input has dynamic shape but no shape registered");
    }

    state->update_device_state(stream, CUDA::DevicePointer<const void*>{raw_ptr}, shape);

    for (BufferID id : release_ids_) {
        dynBufCtx.releaseDynamicBuffer(id);
    }
}

std::shared_ptr<CachedOperation>
DynamicOperation::createCachedOperation(
        const ShapeKey& key,
        const std::vector<CUDA::DevicePointer<const void*>>& input_ptrs,
        const CUDA::Stream& stream) const {
    // 1. Build the clone inputs: Constants preserved, others -> concrete-shape Parameters.
    ov::OutputVector new_inputs;
    new_inputs.reserve(original_node_->get_input_size());
    for (size_t i = 0; i < original_node_->get_input_size(); ++i) {
        if (std::dynamic_pointer_cast<ov::op::v0::Constant>(original_node_->get_input_node_shared_ptr(i))) {
            new_inputs.push_back(original_node_->input_value(i));
        } else {
            new_inputs.push_back(
                std::make_shared<ov::op::v0::Parameter>(original_node_->get_input_element_type(i),
                                                        key.input_shapes[i])
                    ->output(0));
        }
    }

    auto cloned = original_node_->clone_with_new_inputs(new_inputs);
    cloned->validate_and_infer_types();

    // 2. Value-dependent ops (has_dynamic_output_, determined at construction) need
    //    input VALUES (Broadcast target shape, Reshape pattern, ...): re-clone with
    //    the small integer inputs folded in as Constants so the output shape infers.
    if (has_dynamic_output_) {
        new_inputs.clear();
        for (size_t i = 0; i < original_node_->get_input_size(); ++i) {
            if (std::dynamic_pointer_cast<ov::op::v0::Constant>(original_node_->get_input_node_shared_ptr(i))) {
                new_inputs.push_back(original_node_->input_value(i));
                continue;
            }
            const auto& shape = key.input_shapes[i];
            auto elem_type = original_node_->get_input_element_type(i);
            if (isShapeValueInput(shape, elem_type) && i < input_ptrs.size()) {
                // Reuse the values already downloaded for the cache key (always
                // populated for value-dependent ops); download as a safe fallback.
                std::vector<int64_t> values =
                    (i < key.input_values.size() && !key.input_values[i].empty())
                        ? key.input_values[i]
                        : downloadShapeValues(stream, input_ptrs[i], shape, elem_type);
                new_inputs.push_back(
                    std::make_shared<ov::op::v0::Constant>(elem_type, shape, values)->output(0));
            } else {
                new_inputs.push_back(std::make_shared<ov::op::v0::Parameter>(elem_type, shape)->output(0));
            }
        }
        cloned = original_node_->clone_with_new_inputs(new_inputs);
        cloned->validate_and_infer_types();
    }

    // 3. Collect output shapes and sizes.
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

    // 4. Zero-element outputs: CUDA kernels generally cannot handle zero-extent
    //    tensors, so the inner op is a no-op. We still keep the (correct) output
    //    shapes for downstream shape computations.
    for (const auto& shape : output_shapes) {
        if (ov::shape_size(shape) == 0) {
            return std::make_shared<CachedOperation>(
                CachedOperation{nullptr, std::move(output_shapes), std::move(output_sizes),
                                WorkbufferRequest{}, {}, {}});
        }
    }

    // 5. Create dummy TensorIDs for the inner operation.
    IndexCollection dummy_in, dummy_out;
    for (size_t i = 0; i < cloned->get_input_size(); ++i) {
        dummy_in.push_back(TensorID{static_cast<BufferID>(i)});
    }
    for (size_t i = 0; i < cloned->get_output_size(); ++i) {
        dummy_out.push_back(TensorID{static_cast<BufferID>(cloned->get_input_size() + i)});
    }

    // 6. Create the static operation via the registry.
    auto operation = OperationRegistry::getInstance().createOperation(
        creation_context_, cloned, std::move(dummy_in), std::move(dummy_out));

    // 7. Allocate + initialize persistent immutable workbuffers (shared across
    //    inference requests, hence DefaultStream).
    WorkbufferRequest wb_request = operation->GetWorkBufferRequest();
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
