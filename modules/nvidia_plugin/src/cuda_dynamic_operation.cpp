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

#include <openvino/op/paged_attention.hpp>

#include <cmath>

#include "ops/converters.hpp"
#include "cuda_operation_registry.hpp"
#include "cuda_variable_state.hpp"
#include "kernels/paged_attention.hpp"
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

    if (ov::is_type<ov::op::PagedAttentionExtension>(original_node_)) {
        executePagedAttention(context, stream, inputTensors, dynBufCtx);
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
    // Logical byte size (0 for an empty state, e.g. an initial KV-cache with a
    // zero-length sequence). Only the allocation is bumped to 1 byte to avoid a
    // zero-size malloc; the copy/memset must use the real size so an empty init
    // tensor is never read past its end.
    size_t byte_size = elem_type.size() * ov::shape_size(shape);
    auto alloc = stream.malloc(std::max(byte_size, size_t{1}));

    if (has_init) {
        if (byte_size > 0) {
            BufferID initBufId = input_ids_[0].GetBuffer().GetId();
            auto dynBuf = dynBufCtx.getDynamicBuffer(initBufId);
            const void* src = dynBuf ? dynBuf->get() : inputTensors[0].get();
            stream.transfer(CUDA::DevicePointer<void*>{alloc.get()},
                            CUDA::DevicePointer<const void*>{src}, byte_size);
        }
    } else if (byte_size > 0) {
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

void DynamicOperation::executePagedAttention(const InferenceRequestContext& context,
                                             const CUDA::Stream& stream,
                                             Inputs inputTensors,
                                             DynamicBufferContext& dynBufCtx) const {
    OPENVINO_ASSERT(context.hasPagedAttentionContext(), "PagedAttention requires a PagedAttentionContext");
    auto& paCtx = const_cast<InferenceRequestContext&>(context).getPagedAttentionContext();

    // Device pointer / shape of input `i` (from the registered dynamic buffer, else
    // the framework-provided input tensor / the node's static shape).
    const auto inputPtr = [&](std::size_t i) -> const void* {
        auto dynBuf = dynBufCtx.getDynamicBuffer(input_ids_[i].GetBuffer().GetId());
        return dynBuf ? dynBuf->get() : inputTensors[i].get();
    };
    const auto inputShape = [&](std::size_t i) -> ov::Shape {
        const BufferID id = input_ids_[i].GetBuffer().GetId();
        return dynBufCtx.hasShape(id) ? dynBufCtx.getShape(id) : original_node_->get_input_shape(i);
    };
    const auto constFloat = [&](std::size_t i, float dflt) -> float {
        if (auto c = ov::as_type_ptr<ov::op::v0::Constant>(original_node_->get_input_node_shared_ptr(i))) {
            const auto v = c->cast_vector<float>();
            if (!v.empty()) return v[0];
        }
        return dflt;
    };
    const auto constInt = [&](std::size_t i, int dflt) -> int {
        if (auto c = ov::as_type_ptr<ov::op::v0::Constant>(original_node_->get_input_node_shared_ptr(i))) {
            const auto v = c->cast_vector<int64_t>();
            if (!v.empty()) return static_cast<int>(v[0]);
        }
        return dflt;
    };

    // Head geometry (rt_info) and derived counts.
    const auto& rt = original_node_->get_rt_info();
    const int num_kv_heads = static_cast<int>(rt.at("num_k_heads").as<std::size_t>());
    const int head_size = static_cast<int>(rt.at("k_head_size").as<std::size_t>());
    const int v_head_size = static_cast<int>(rt.at("v_head_size").as<std::size_t>());
    const auto q_shape = inputShape(0);  // [num_tokens, num_heads * head_size]
    const int num_tokens = static_cast<int>(q_shape[0]);
    const int num_heads = static_cast<int>(q_shape[1] / head_size);
    constexpr int block_size = 32;

    const auto elem_type = original_node_->get_input_element_type(0);
    const auto kernel_type = convertDataType<ov::nvidia_gpu::kernel::Type_t>(elem_type);

    if (num_tokens == 0) {
        ov::Shape out_shape{0, static_cast<size_t>(num_heads * v_head_size)};
        auto alloc = stream.malloc(1);
        dynBufCtx.registerDynamicBuffer(dynamic_output_ids_[0].GetBuffer().GetId(), std::move(alloc), out_shape);
        for (BufferID id : release_ids_) dynBufCtx.releaseDynamicBuffer(id);
        return;
    }

    // Scheduling metadata (small) — copied to host to build slot_mapping/row_to_subseq.
    const auto downloadI32 = [&](std::size_t i) {
        const std::size_t n = ov::shape_size(inputShape(i));
        std::vector<int32_t> v(n);
        if (n) stream.download(v.data(), CUDA::DevicePointer<const void*>{inputPtr(i)}, n * sizeof(int32_t));
        return v;
    };
    const auto past_lens = downloadI32(5);
    const auto subsequence_begins = downloadI32(6);
    const auto block_indices = downloadI32(7);
    const auto block_indices_begins = downloadI32(8);
    const int nseq = static_cast<int>(past_lens.size());

    std::vector<int32_t> slot_mapping(num_tokens), row_to_subseq(num_tokens);
    for (int i = 0; i < nseq; ++i) {
        const int blk_begin = block_indices_begins[i];
        for (int row = subsequence_begins[i]; row < subsequence_begins[i + 1]; ++row) {
            row_to_subseq[row] = i;
            const int pos = past_lens[i] + (row - subsequence_begins[i]);
            const int bn = block_indices[blk_begin + pos / block_size];
            slot_mapping[row] = bn * block_size + (pos % block_size);
        }
    }

    // Persistent device caches, sized from the (uploaded) cache Parameter shapes.
    const std::size_t kc_bytes = ov::shape_size(inputShape(3)) * elem_type.size();
    const std::size_t vc_bytes = ov::shape_size(inputShape(4)) * elem_type.size();
    void* key_cache = paCtx.get_or_allocate(input_ids_[3].GetBuffer().GetId(), kc_bytes, stream);
    void* value_cache = paCtx.get_or_allocate(input_ids_[4].GetBuffer().GetId(), vc_bytes, stream);

    // Upload slot_mapping / row_to_subseq.
    const std::size_t idx_bytes = static_cast<std::size_t>(num_tokens) * sizeof(int32_t);
    auto d_slot = stream.malloc(idx_bytes);
    auto d_r2s = stream.malloc(idx_bytes);
    stream.upload(CUDA::DevicePointer<void*>{d_slot.get()}, slot_mapping.data(), idx_bytes);
    stream.upload(CUDA::DevicePointer<void*>{d_r2s.get()}, row_to_subseq.data(), idx_bytes);

    // scale (input 9, default 1/sqrt(head_size)), sliding_window (input 10), alibi (input 11).
    float scale = constFloat(9, 0.0f);
    if (scale == 0.0f) scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    const int sliding_window = constInt(10, 0);
    // alibi_slopes (input 11): a per-head f32 vector, or empty. Allocate unconditionally
    // (CUDA::Allocation is not default-constructible); the kernel only reads it when
    // has_alibi is set.
    auto d_alibi = stream.malloc(std::max(static_cast<std::size_t>(num_heads) * sizeof(float), std::size_t{1}));
    const float* alibi_ptr = static_cast<const float*>(d_alibi.get());
    bool has_alibi = false;
    if (auto al = ov::as_type_ptr<ov::op::v0::Constant>(original_node_->get_input_node_shared_ptr(11))) {
        const auto v = al->cast_vector<float>();
        if (v.size() == static_cast<std::size_t>(num_heads)) {
            stream.upload(CUDA::DevicePointer<void*>{d_alibi.get()}, v.data(), v.size() * sizeof(float));
            has_alibi = true;
        }
    }

    // Kernel A: scatter the new K/V (inputs 1/2) into the persistent cache.
    kernel::reshape_and_cache(stream.get(),
                              kernel_type,
                              inputPtr(1),
                              inputPtr(2),
                              key_cache,
                              value_cache,
                              static_cast<const int32_t*>(d_slot.get()),
                              num_tokens,
                              num_kv_heads,
                              head_size,
                              v_head_size,
                              block_size);

    // Output [num_tokens, num_heads * v_head_size].
    ov::Shape out_shape{static_cast<size_t>(num_tokens), static_cast<size_t>(num_heads * v_head_size)};
    auto out_alloc = stream.malloc(std::max(ov::shape_size(out_shape) * elem_type.size(), std::size_t{1}));

    const kernel::PagedAttentionParams params{num_heads, num_kv_heads, head_size, v_head_size,
                                              block_size, scale,       sliding_window, has_alibi};
    kernel::paged_attention(stream.get(),
                            kernel_type,
                            inputPtr(0),
                            key_cache,
                            value_cache,
                            static_cast<const int32_t*>(inputPtr(7)),
                            static_cast<const int32_t*>(inputPtr(8)),
                            static_cast<const int32_t*>(inputPtr(5)),
                            static_cast<const int32_t*>(inputPtr(6)),
                            static_cast<const int32_t*>(d_r2s.get()),
                            alibi_ptr,
                            out_alloc.get(),
                            params,
                            num_tokens);

    dynBufCtx.registerDynamicBuffer(dynamic_output_ids_[0].GetBuffer().GetId(), std::move(out_alloc), out_shape);
    for (BufferID id : release_ids_) dynBufCtx.releaseDynamicBuffer(id);
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
