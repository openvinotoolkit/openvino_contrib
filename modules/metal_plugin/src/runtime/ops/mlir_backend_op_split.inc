// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Included from mlir_backend.mm. Handles Split/Slice host paths inside run().

bool MlirBackend::Impl::try_run_split_segment(const Segment& seg,
                                              const std::vector<KernelOp>& ops,
                                              const std::vector<ov::Tensor>& inputs,
                                              std::vector<ov::Tensor>& outputs,
                                              MetalBufferManager& mgr,
                                              const std::function<void(const std::string&)>& cpu_fallback_error) {
    if (m_active_tensor_map) {
        // Device path should handle Split/Slice; host copies are запрещены.
        METAL_LOG_DEBUG("mlir", "[Split dbg] active tensor map present; skip host split path");
        return false;
    }
    {
        std::ostringstream oss;
        oss << "[Split dbg] enter seg.op_count=" << seg.op_count;
        METAL_LOG_DEBUG("mlir", oss.str());
    }
    auto seg_op = [&](size_t i) -> const KernelOp& { return ops[seg.first_op_index + i]; };

    // Handle single-op Split via direct copy (GPU blit + host memcpy), no kernels, no fallbacks.
    if (seg.op_count == 1 && seg_op(0).kind == KernelOpKind::Split) {
        const auto& op = seg_op(0);
        const auto& desc = op.split;
        OPENVINO_ASSERT(!inputs.empty(), "Split segment: no inputs provided");
        const ov::Tensor& src = inputs[0];
        auto element_type = static_cast<ov::element::Type_t>(desc.element_type);
        if (element_type == ov::element::Type_t::dynamic)
            element_type = src.get_element_type();
        const auto& cshape = desc.input_shape;
        const size_t elem_size = src.get_element_type().size();
        const size_t axis = static_cast<size_t>(desc.axis);
        size_t outer = 1;
        for (size_t i = 0; i < axis; ++i) outer *= static_cast<size_t>(cshape[i]);
        size_t inner = 1;
        for (size_t i = axis + 1; i < cshape.size(); ++i) inner *= static_cast<size_t>(cshape[i]);
        const size_t axis_len = static_cast<size_t>(cshape[axis]);
        const size_t outputs_count = desc.split_sizes.size();
        if (outputs.size() != outputs_count) outputs.resize(outputs_count);
        {
            std::ostringstream oss;
            oss << "[Split dbg] single-op copy axis=" << axis
                << " outer=" << outer << " inner=" << inner
                << " axis_len=" << axis_len << " splits=" << outputs_count;
            METAL_LOG_DEBUG("mlir", oss.str());
        }
        for (size_t i = 0; i < outputs_count; ++i) {
            ov::Shape shp(cshape.begin(), cshape.end());
            shp[axis] = static_cast<size_t>(desc.split_sizes[i]);
            if (outputs[i].get_shape() != shp || outputs[i].get_element_type() != element_type) {
                outputs[i] = ov::Tensor{element_type, shp};
            }
        }
        MetalBuffer src_buf = mgr.allocate(src.get_byte_size(),
                                           element_type,
                                           /*persistent=*/false,
                                           /*storageModePrivate=*/false);
        mgr.upload(src_buf, src.data(), src.get_byte_size());
        std::vector<MetalBuffer> out_bufs(outputs_count);
#ifdef __OBJC__
        id<MTLBuffer> buf_in = static_cast<id<MTLBuffer>>(src_buf.buffer);
        id<MTLCommandBuffer> cmd = buf_in ? [m_queue commandBuffer] : nil;
        id<MTLBlitCommandEncoder> blit = cmd ? [cmd blitCommandEncoder] : nil;
#endif
        size_t axis_offset = 0;
        for (size_t i = 0; i < outputs_count; ++i) {
            out_bufs[i] = mgr.allocate(outputs[i].get_byte_size(),
                                       element_type,
                                       /*persistent=*/false,
                                       /*storageModePrivate=*/false);
            size_t split = static_cast<size_t>(desc.split_sizes[i]);
#ifdef __OBJC__
            id<MTLBuffer> buf_out = static_cast<id<MTLBuffer>>(out_bufs[i].buffer);
            if (buf_in && buf_out && blit) {
                for (size_t o = 0; o < outer; ++o) {
                    size_t src_off = (o * axis_len * inner + axis_offset * inner) * elem_size;
                    size_t dst_off = (o * split * inner) * elem_size;
                    size_t bytes = split * inner * elem_size;
                    [blit copyFromBuffer:buf_in sourceOffset:src_off toBuffer:buf_out destinationOffset:dst_off size:bytes];
                }
            }
#endif
            axis_offset += split;
        }
#ifdef __OBJC__
        if (blit) [blit endEncoding];
        if (cmd) {
            [cmd commit];
            [cmd waitUntilCompleted];
        }
#endif

        // Copy GPU results (shared buffers) back to host outputs, fallback to CPU copy if needed.
        axis_offset = 0;
        for (size_t i = 0; i < outputs_count; ++i) {
            size_t split = static_cast<size_t>(desc.split_sizes[i]);
#ifdef __OBJC__
            id<MTLBuffer> buf_out = static_cast<id<MTLBuffer>>(out_bufs[i].buffer);
            if (buf_out && [buf_out contents]) {
                std::memcpy(outputs[i].data(), [buf_out contents], outputs[i].get_byte_size());
                mgr.add_d2h(outputs[i].get_byte_size());
                axis_offset += split;
                continue;
            }
#endif
            const uint8_t* src_base = static_cast<const uint8_t*>(src.data());
            uint8_t* dst_base = static_cast<uint8_t*>(outputs[i].data());
            for (size_t o = 0; o < outer; ++o) {
                const uint8_t* src_ptr = src_base + (o * axis_len * inner + axis_offset * inner) * elem_size;
                uint8_t* dst_ptr = dst_base + (o * split * inner) * elem_size;
                std::memcpy(dst_ptr, src_ptr, split * inner * elem_size);
            }
            axis_offset += split;
        }
        return true;
    }
    auto product = [](const std::vector<int64_t>& dims) -> size_t {
        size_t v = 1;
        for (auto d : dims) v *= static_cast<size_t>(d);
        return v;
    };

    if (outputs.size() >= 1) {
        bool slice_only = true;
        for (size_t i = 0; i < seg.op_count; ++i) {
            if (seg_op(i).kind != KernelOpKind::Slice) {
                slice_only = false;
                break;
            }
        }
        if (slice_only) {
            METAL_LOG_DEBUG("mlir", "[Split dbg] slice-only path seg_ops=" + std::to_string(seg.op_count));
            OPENVINO_ASSERT(!inputs.empty(), "Slice segment: no inputs");
            const ov::Tensor& src = inputs[0];
            MetalBuffer src_buf = mgr.allocate(src.get_byte_size(),
                                               src.get_element_type(),
                                               /*persistent=*/false,
                                               /*storageModePrivate=*/false);
            mgr.upload(src_buf, src.data(), src.get_byte_size());
            OPENVINO_ASSERT(outputs.size() == seg.op_count, "Slice segment outputs size mismatch");
#ifdef __OBJC__
            id<MTLBuffer> buf_in = static_cast<id<MTLBuffer>>(src_buf.buffer);
            id<MTLCommandBuffer> cmd = buf_in ? [m_queue commandBuffer] : nil;
            id<MTLBlitCommandEncoder> blit = cmd ? [cmd blitCommandEncoder] : nil;
#endif
            const size_t elem_size = src.get_element_type().size();
            std::vector<MetalBuffer> out_bufs(seg.op_count);

            for (size_t i = 0; i < seg.op_count; ++i) {
                const auto& op = seg_op(i);
                const auto& oshp = op.output && !op.output->shape.empty()
                                       ? op.output->shape
                                       : op.slice.out_shape;
                ov::Shape shape(oshp.begin(), oshp.end());
                if (outputs[i].get_shape() != shape || outputs[i].get_element_type() != src.get_element_type()) {
                    outputs[i] = ov::Tensor{src.get_element_type(), shape};
                }
                out_bufs[i] = mgr.allocate(outputs[i].get_byte_size(),
                                           src.get_element_type(),
                                           /*persistent=*/false,
                                           /*storageModePrivate=*/false);
#ifdef __OBJC__
                id<MTLBuffer> buf_out = static_cast<id<MTLBuffer>>(out_bufs[i].buffer);

                // Derive copy parameters for this slice.
                size_t axis = 0;
                const auto& in_shape = op.slice.in_shape;
                const auto& out_shape = op.slice.out_shape;
                for (size_t k = 0; k < in_shape.size(); ++k) {
                    const bool start_non_zero = (k < op.slice.starts.size()) && (op.slice.starts[k] != 0);
                    if (start_non_zero || k >= out_shape.size() || in_shape[k] != out_shape[k]) {
                        axis = k;
                        break;
                    }
                }
                size_t axis_len = (axis < in_shape.size()) ? static_cast<size_t>(in_shape[axis]) : 1;
                size_t split = (axis < out_shape.size()) ? static_cast<size_t>(out_shape[axis]) : 1;
                size_t axis_off = (axis < op.slice.starts.size()) ? static_cast<size_t>(op.slice.starts[axis]) : 0;
                size_t inner = 1;
                for (size_t k = axis + 1; k < in_shape.size(); ++k) inner *= static_cast<size_t>(in_shape[k]);
                size_t outer = 1;
                for (size_t k = 0; k < axis; ++k) outer *= static_cast<size_t>(in_shape[k]);
                {
                    std::ostringstream oss;
                    oss << "[Slice dbg] i=" << i << " axis=" << axis << " offset=" << axis_off
                        << " split=" << split << " outer=" << outer << " inner=" << inner
                        << " axis_len=" << axis_len << " elem=" << elem_size;
                    METAL_LOG_DEBUG("mlir", oss.str());
                }

                // Copy each outer slice chunk via blit (GPU-side memcpy).
                if (buf_in && buf_out && blit) {
                    for (size_t o = 0; o < outer; ++o) {
                        size_t src_off = (o * axis_len * inner + axis_off * inner) * elem_size;
                        size_t dst_off = (o * split * inner) * elem_size;
                        size_t bytes = split * inner * elem_size;
                        [blit copyFromBuffer:buf_in
                               sourceOffset:src_off
                                   toBuffer:buf_out
                          destinationOffset:dst_off
                                       size:bytes];
                    }
                }
#endif
            }
#ifdef __OBJC__
            if (blit) [blit endEncoding];
            if (cmd) {
                [cmd commit];
                [cmd waitUntilCompleted];
            }
#endif

            // Mirror copy on CPU to guarantee correctness (still no fallback, just host memcpy).
            for (size_t i = 0; i < seg.op_count; ++i) {
                const auto& op = seg_op(i);
                const auto& in_shape = op.slice.in_shape;
                const auto& out_shape = op.slice.out_shape;
                size_t axis = 0;
                for (size_t k = 0; k < in_shape.size(); ++k) {
                    const bool start_non_zero = (k < op.slice.starts.size()) && (op.slice.starts[k] != 0);
                    if (start_non_zero || k >= out_shape.size() || in_shape[k] != out_shape[k]) {
                        axis = k;
                        break;
                    }
                }
                size_t axis_len = (axis < in_shape.size()) ? static_cast<size_t>(in_shape[axis]) : 1;
                size_t split = (axis < out_shape.size()) ? static_cast<size_t>(out_shape[axis]) : 1;
                size_t axis_off = (axis < op.slice.starts.size()) ? static_cast<size_t>(op.slice.starts[axis]) : 0;
                size_t inner = 1;
                for (size_t k = axis + 1; k < in_shape.size(); ++k) inner *= static_cast<size_t>(in_shape[k]);
                size_t outer = 1;
                for (size_t k = 0; k < axis; ++k) outer *= static_cast<size_t>(in_shape[k]);

                bool copied = false;
#ifdef __OBJC__
                id<MTLBuffer> buf_out = static_cast<id<MTLBuffer>>(out_bufs[i].buffer);
                if (buf_out && [buf_out contents]) {
                    std::memcpy(outputs[i].data(), [buf_out contents], outputs[i].get_byte_size());
                    mgr.add_d2h(outputs[i].get_byte_size());
                    copied = true;
                }
#endif
                if (!copied) {
                    const uint8_t* src_ptr_base = static_cast<const uint8_t*>(inputs[0].data());
                    uint8_t* dst_ptr_base = static_cast<uint8_t*>(outputs[i].data());
                    const size_t elem_sz_host = inputs[0].get_element_type().size();
                    for (size_t o = 0; o < outer; ++o) {
                        const uint8_t* src_ptr = src_ptr_base + (o * axis_len * inner + axis_off * inner) * elem_sz_host;
                        uint8_t* dst_ptr = dst_ptr_base + (o * split * inner) * elem_sz_host;
                        std::memcpy(dst_ptr, src_ptr, split * inner * elem_sz_host);
                    }
                }
            }

            return true;
        }
        // Not a split/slice pattern we handle here.
        return false;
    }

    return false;
}
