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
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:m_pipelines[seg.first_op_index + 0]];
        [enc setBuffer:buf_in offset:0 atIndex:0];
        for (size_t i = 0; i < outputs_count; ++i) {
            [enc setBuffer:out_bufs[i] offset:0 atIndex:(1 + i)];
        }

        auto total_elems = [&]() {
            size_t t = 1;
            for (auto d : desc.input_shape) t *= static_cast<size_t>(d);
            return t;
        }();
        const NSUInteger threads_per_tg = 128;
        MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(total_elems), 1, 1);
        MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        [buf_in release];
        for (auto b : out_bufs) if (b) [b release];
        return true;
    }

    // Host slicing path for Slice-only segments (multi-output Split expansion)
    auto run_host_slices = [&](const Segment& s) {
        if (inputs.empty()) cpu_fallback_error("no inputs");
        const ov::Tensor& src = inputs[0];
        const bool is_f16 = src.get_element_type() == ov::element::f16;

        auto product = [](const std::vector<int64_t>& dims) -> size_t {
            size_t v = 1;
            for (auto d : dims) v *= static_cast<size_t>(d);
            return v;
        };

        auto copy_slice = [&](const KernelOp& op, ov::Tensor& dst) {
            const auto& desc = op.slice;
            const size_t rank = desc.out_shape.size();
            std::vector<size_t> out_strides(rank, 1);
            for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
                out_strides[i] = out_strides[i + 1] * static_cast<size_t>(desc.out_shape[i + 1]);
            }
            const size_t elems = product(desc.out_shape);
            if (is_f16) {
                const auto* src_ptr = src.data<const ov::float16>();
                auto* dst_ptr = dst.data<ov::float16>();
                for (size_t idx = 0; idx < elems; ++idx) {
                    size_t tmp = idx;
                    int64_t in_index = 0;
                    for (size_t d = 0; d < rank; ++d) {
                        size_t coord = tmp / out_strides[d];
                        tmp -= coord * out_strides[d];
                        int64_t in_coord = desc.starts[d] + static_cast<int64_t>(coord) * desc.steps[d];
                        in_index += in_coord * desc.in_strides[d];
                    }
                    dst_ptr[idx] = src_ptr[in_index];
                }
            } else {
                const auto* src_ptr = src.data<const float>();
                auto* dst_ptr = dst.data<float>();
                for (size_t idx = 0; idx < elems; ++idx) {
                    size_t tmp = idx;
                    int64_t in_index = 0;
                    for (size_t d = 0; d < rank; ++d) {
                        size_t coord = tmp / out_strides[d];
                        tmp -= coord * out_strides[d];
                        int64_t in_coord = desc.starts[d] + static_cast<int64_t>(coord) * desc.steps[d];
                        in_index += in_coord * desc.in_strides[d];
                    }
                    dst_ptr[idx] = src_ptr[in_index];
                }
            }
        };

        OPENVINO_ASSERT(outputs.size() == s.op_count, "Split host path expects one output per Slice op");
        for (size_t i = 0; i < s.op_count; ++i) {
            const auto& op = seg_op(i);
            OPENVINO_ASSERT(op.kind == KernelOpKind::Slice, "Split host path supports Slice-only segment");
            auto& out = outputs[i];
            const auto& oshp = op.output && !op.output->shape.empty()
                                   ? op.output->shape
                                   : op.slice.out_shape;
            ov::Shape shape;
            shape.assign(oshp.begin(), oshp.end());
            if (out.get_shape() != shape || out.get_element_type() != src.get_element_type()) {
                out = ov::Tensor{src.get_element_type(), shape};
            }
            copy_slice(op, out);
        }
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
            run_host_slices(seg);
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
