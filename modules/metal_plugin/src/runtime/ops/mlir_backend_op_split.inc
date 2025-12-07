// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Included from mlir_backend.mm. Handles Split/Slice host paths inside run().

bool MlirBackend::Impl::try_run_split_segment(const Segment& seg,
                                              const std::vector<KernelOp>& ops,
                                              const std::vector<ov::Tensor>& inputs,
                                              std::vector<ov::Tensor>& outputs,
                                              const std::function<void(const std::string&)>& cpu_fallback_error) {
    auto seg_op = [&](size_t i) -> const KernelOp& { return ops[seg.first_op_index + i]; };

    // Single Split op → dedicated GPU kernel (already compiled as seg pipeline).
    if (seg.op_count == 1 && seg_op(0).kind == KernelOpKind::Split) {
        const auto& op = seg_op(0);
        const auto& desc = op.split;
        const size_t outputs_count = desc.split_sizes.size();
        if (inputs.empty()) cpu_fallback_error("no inputs");
        if (outputs.size() != outputs_count) {
            if (!m_allow_partial_offload) cpu_fallback_error("split outputs mismatch");
            ensure_fallback(m_original_model);
            // reuse local run_fallback defined in run(); pass via lambda parameter to avoid name clash
            // caller should supply cpu_fallback_error that uses run_fallback if needed
            // nothing more to do here
            return true;
        }

        auto element_type = static_cast<ov::element::Type_t>(desc.element_type);
        if (element_type == ov::element::Type_t::dynamic)
            element_type = inputs[0].get_element_type();

        // Prepare outputs shapes
        for (size_t i = 0; i < outputs_count; ++i) {
            ov::Shape shp;
            shp.reserve(desc.input_shape.size());
            for (size_t d = 0; d < desc.input_shape.size(); ++d) {
                size_t val = static_cast<size_t>(desc.input_shape[d]);
                if (d == static_cast<size_t>(desc.axis)) val = desc.split_sizes[i];
                shp.push_back(val);
            }
            if (outputs[i].get_shape() != shp || outputs[i].get_element_type() != element_type) {
                outputs[i] = ov::Tensor{element_type, shp};
            }
        }

        id<MTLBuffer> buf_in = [m_device newBufferWithBytesNoCopy:const_cast<void*>(inputs[0].data())
                                                        length:inputs[0].get_byte_size()
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];
        std::vector<id<MTLBuffer>> out_bufs(outputs_count, nil);
        for (size_t i = 0; i < outputs_count; ++i) {
            out_bufs[i] = [m_device newBufferWithBytesNoCopy:const_cast<void*>(outputs[i].data())
                                                      length:outputs[i].get_byte_size()
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
        }

        id<MTLCommandBuffer> cmd = [m_queue commandBuffer];
        const NSUInteger threads_per_tg = 128;
        struct SplitParams {
            uint32_t axis_offset;
            uint32_t split_size;
            uint32_t inner;
            uint32_t outer;
            uint32_t axis_len;
        };
        for (size_t i = 0; i < outputs_count; ++i) {
            SplitParams p{};
            p.axis_offset = static_cast<uint32_t>(desc.axis_offsets[i]);
            p.split_size = static_cast<uint32_t>(desc.split_sizes[i]);
            p.inner = static_cast<uint32_t>(desc.inner);
            // outer = product of dims before axis
            uint64_t outer = 1;
            for (size_t k = 0; k < static_cast<size_t>(desc.axis); ++k)
                outer *= static_cast<uint64_t>(desc.input_shape[k]);
            p.outer = static_cast<uint32_t>(outer);
            p.axis_len = static_cast<uint32_t>(desc.input_shape[static_cast<size_t>(desc.axis)]);
            id<MTLBuffer> param_buf = [m_device newBufferWithBytes:&p
                                                            length:sizeof(SplitParams)
                                                           options:MTLResourceStorageModeShared];

            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:m_pipelines[seg.first_op_index + 0]];
            [enc setBuffer:buf_in offset:0 atIndex:0];
            [enc setBuffer:out_bufs[i] offset:0 atIndex:1];
            [enc setBuffer:param_buf offset:0 atIndex:2];

            NSUInteger threads = static_cast<NSUInteger>(p.outer * p.split_size * p.inner);
            MTLSize grid = MTLSizeMake(threads, 1, 1);
            MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [param_buf release];
        }
        [cmd commit];
        [cmd waitUntilCompleted];
        [buf_in release];
        for (auto b : out_bufs) if (b) [b release];
        return true;
    }

    auto product = [](const std::vector<int64_t>& dims) -> size_t {
        size_t v = 1;
        for (auto d : dims) v *= static_cast<size_t>(d);
        return v;
    };

    if (outputs.size() > 1) {
        bool slice_only = true;
        for (size_t i = 0; i < seg.op_count; ++i) {
            if (seg_op(i).kind != KernelOpKind::Slice) {
                slice_only = false;
                break;
            }
        }
        if (slice_only) {
            if (inputs.empty()) cpu_fallback_error("no inputs");
            const ov::Tensor& src = inputs[0];
            id<MTLBuffer> buf_in = [m_device newBufferWithBytesNoCopy:const_cast<void*>(src.data())
                                                        length:src.get_byte_size()
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];
            id<MTLCommandBuffer> cmd = [m_queue commandBuffer];
            OPENVINO_ASSERT(outputs.size() == seg.op_count, "Slice segment outputs size mismatch");
            for (size_t i = 0; i < seg.op_count; ++i) {
                const auto& op = seg_op(i);
                const auto& oshp = op.output && !op.output->shape.empty()
                                       ? op.output->shape
                                       : op.slice.out_shape;
                ov::Shape shape(oshp.begin(), oshp.end());
                if (outputs[i].get_shape() != shape || outputs[i].get_element_type() != src.get_element_type()) {
                    outputs[i] = ov::Tensor{src.get_element_type(), shape};
                }
                id<MTLBuffer> buf_out = [m_device newBufferWithBytesNoCopy:const_cast<void*>(outputs[i].data())
                                                              length:outputs[i].get_byte_size()
                                                             options:MTLResourceStorageModeShared
                                                         deallocator:nil];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:m_pipelines[seg.first_op_index + i]];
                [enc setBuffer:buf_in offset:0 atIndex:0];
                [enc setBuffer:buf_out offset:0 atIndex:1];
                MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(product(oshp)), 1, 1);
                MTLSize tg = MTLSizeMake(128, 1, 1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc endEncoding];
                [buf_out release];
            }
            [cmd commit];
            [cmd waitUntilCompleted];
            [buf_in release];
            return true;
        }
        if (m_allow_partial_offload) {
            ensure_fallback(m_original_model);
            return true;
        }
        cpu_fallback_error("multiple outputs not supported");
    }

    return false;
}
