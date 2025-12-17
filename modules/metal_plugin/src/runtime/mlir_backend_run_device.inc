bool MlirBackend::Impl::run_device(MetalTensorMap& tensors, MetalBufferManager& mgr) {
#ifdef __OBJC__
    @autoreleasepool {
        struct ManagerGuard {
            MlirBackend::Impl* self;
            void set(MetalBufferManager* m) { if (self) { self->m_active_mgr = m; self->m_owned_mgr.reset(); } }
            ~ManagerGuard() { if (self) self->m_active_mgr = nullptr; }
        } guard{this};
        guard.set(&mgr);
        struct TensorMapGuard {
            MlirBackend::Impl* self;
            MetalTensorMap* prev = nullptr;
            void set(MetalTensorMap* m) { if (self) { prev = self->m_active_tensor_map; self->m_active_tensor_map = m; } }
            ~TensorMapGuard() { if (self) self->m_active_tensor_map = prev; }
        } tm_guard{this};
        tm_guard.set(&tensors);

        // Special-case reshape-only graphs (no pipelines built).
        auto reshape_only = [&]() -> std::shared_ptr<const ov::op::v1::Reshape> {
            std::shared_ptr<const ov::op::v1::Reshape> reshape;
            for (const auto& node : m_model->get_ordered_ops()) {
                if (ov::is_type<ov::op::v0::Parameter>(node.get()) ||
                    ov::is_type<ov::op::v0::Result>(node.get()) ||
                    ov::is_type<ov::op::v0::Constant>(node.get())) {
                    continue;
                }
                if (auto r = ov::as_type_ptr<const ov::op::v1::Reshape>(node)) {
                    if (reshape)
                        return nullptr;  // more than one compute op
                    reshape = r;
                    continue;
                }
                return nullptr;  // any other op → not reshape-only
            }
            return reshape;
        };

        if (m_segments.empty() || m_ops.empty() || m_pipelines.empty()) {
            bool reshape_ok = false;
            bool layout_ok = false;
            {
                std::ostringstream os;
                os << "[METAL MLIR] run_device layout-only scan: ";
                for (const auto& node : m_model->get_ordered_ops()) {
                    if (ov::is_type<ov::op::v0::Parameter>(node.get()) ||
                        ov::is_type<ov::op::v0::Result>(node.get()) ||
                        ov::is_type<ov::op::v0::Constant>(node.get())) {
                        continue;
                    }
                    os << node->get_type_info().name << " ";
                }
                METAL_LOG_DEBUG("mlir", os.str());
            }
            auto get_input_dev = [&](size_t idx) -> std::optional<MetalTensor> {
                if (tensors.has_input_device(idx))
                    return tensors.get_input_device(idx);
                if (tensors.has_input_host(idx)) {
                    const auto& h = tensors.get_input_host(idx);
                    ov::element::Type et = h.get_element_type() == ov::element::f16 ? ov::element::f32 : h.get_element_type();
                    MetalBuffer b = mgr.allocate(h.get_byte_size(), et, /*persistent=*/false, /*storageModePrivate=*/true);
                    mgr.upload(b, h.data(), h.get_byte_size());
                    MetalTensor t{b, h.get_shape(), h.get_element_type()};
                    return t;
                }
                return std::nullopt;
            };

            if (auto r = reshape_only()) {
                auto in_opt = get_input_dev(0);
                if (!in_opt) {
                    METAL_LOG_ERROR("mlir", "[METAL MLIR] run_device reshape-only: input0 missing");
                    return false;
                }
                const auto& in = *in_opt;
                ov::Shape out_shape;
                if (r->get_output_partial_shape(0).is_static()) {
                    out_shape = r->get_output_partial_shape(0).to_shape();
                } else {
                    auto pattern_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(r->get_input_node_shared_ptr(1));
                    if (!pattern_const) {
                        METAL_LOG_ERROR("mlir", "[METAL MLIR] reshape-only: pattern is not constant");
                        return false;
                    }
                    auto pattern = pattern_const->cast_vector<int64_t>();
                    const bool special_zero = r->get_special_zero();
                    out_shape.resize(pattern.size());
                    size_t known_product = 1;
                    int64_t minus_one_idx = -1;
                    for (size_t i = 0; i < pattern.size(); ++i) {
                        int64_t dim = pattern[i];
                        if (dim == 0 && special_zero) {
                            dim = static_cast<int64_t>(i < in.shape.size() ? in.shape[i] : 0);
                        } else if (dim == -1) {
                            minus_one_idx = static_cast<int64_t>(i);
                            continue;
                        }
                        if (dim < 0) {
                            METAL_LOG_STREAM(::ov::metal_plugin::MetalLogLevel::Error, "mlir",
                                             "[METAL MLIR] reshape-only: invalid dim " << dim);
                            return false;
                        }
                        out_shape[i] = static_cast<size_t>(dim);
                        known_product *= dim > 0 ? static_cast<size_t>(dim) : 1;
                    }
                    if (minus_one_idx >= 0) {
                        size_t in_elems = ov::shape_size(in.shape);
                        size_t inferred = in_elems / known_product;
                        out_shape[static_cast<size_t>(minus_one_idx)] = inferred;
                    }
                }
                size_t bytes = ov::shape_size(out_shape) * in.buf.type.size();
                MetalBuffer dst = mgr.allocate(bytes, in.buf.type, /*persistent=*/false, /*storageModePrivate=*/false);
#ifdef __OBJC__
                id<MTLBuffer> src_buf = static_cast<id<MTLBuffer>>(in.buf.buffer);
                id<MTLBuffer> dst_buf = static_cast<id<MTLBuffer>>(dst.buffer);
                if (!src_buf || !dst_buf) {
                    METAL_LOG_ERROR("mlir", "[METAL MLIR] reshape-only: null MTLBuffer");
                    return false;
                }
                id<MTLCommandBuffer> cb = [m_queue commandBuffer];
                id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                size_t copy_bytes = std::min(in.buf.size, dst.size);
                [blit copyFromBuffer:src_buf sourceOffset:0 toBuffer:dst_buf destinationOffset:0 size:copy_bytes];
                [blit endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
#else
                (void)in;
                METAL_LOG_ERROR("mlir", "[METAL MLIR] reshape-only: Metal unavailable");
                return false;
#endif
                MetalTensor out{dst, out_shape, in.expected_type == ov::element::dynamic ? in.buf.type : in.expected_type};
                tensors.bind_output_device(0, out);
                reshape_ok = true;
                return true;
            }

            // Layout-only fallback: if model has no compute kernels, copy input0->output0 on device.
            if (!m_model->outputs().empty() && m_model->output(0).get_partial_shape().is_static()) {
                auto in_opt = get_input_dev(0);
                if (!in_opt) {
                    METAL_LOG_ERROR("mlir", "[METAL MLIR] layout-only: input0 missing");
                    return false;
                }
                const auto& in = *in_opt;
                ov::Shape out_shape = m_model->output(0).get_partial_shape().to_shape();
                size_t bytes = ov::shape_size(out_shape) * in.buf.type.size();
                MetalBuffer dst = mgr.allocate(bytes, in.buf.type, /*persistent=*/false, /*storageModePrivate=*/false);
#ifdef __OBJC__
                id<MTLBuffer> src_buf = static_cast<id<MTLBuffer>>(in.buf.buffer);
                id<MTLBuffer> dst_buf = static_cast<id<MTLBuffer>>(dst.buffer);
                if (src_buf && dst_buf) {
                    id<MTLCommandBuffer> cb = [m_queue commandBuffer];
                    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                    size_t copy_bytes = std::min(in.buf.size, dst.size);
                    [blit copyFromBuffer:src_buf sourceOffset:0 toBuffer:dst_buf destinationOffset:0 size:copy_bytes];
                    [blit endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                    MetalTensor out{dst, out_shape, in.expected_type == ov::element::dynamic ? in.buf.type : in.expected_type};
                    tensors.bind_output_device(0, out);
                    layout_ok = true;
                    return true;
                }
#endif
            }

            if (!layout_ok) {
                auto in_opt = get_input_dev(0);
                if (in_opt) {
                    const auto& in = *in_opt;
                    ov::Shape out_shape;
                    if (!m_model->outputs().empty() && m_model->output(0).get_partial_shape().is_static()) {
                        out_shape = m_model->output(0).get_partial_shape().to_shape();
                    } else {
                        out_shape = in.shape;
                    }
                    size_t bytes = ov::shape_size(out_shape) * in.buf.type.size();
                    MetalBuffer dst = mgr.allocate(bytes, in.buf.type, /*persistent=*/false, /*storageModePrivate=*/true);
#ifdef __OBJC__
                    id<MTLBuffer> src_buf = static_cast<id<MTLBuffer>>(in.buf.buffer);
                    id<MTLBuffer> dst_buf = static_cast<id<MTLBuffer>>(dst.buffer);
                    if (src_buf && dst_buf) {
                        id<MTLCommandBuffer> cb = [m_queue commandBuffer];
                        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                        size_t copy_bytes = std::min(in.buf.size, dst.size);
                        [blit copyFromBuffer:src_buf sourceOffset:0 toBuffer:dst_buf destinationOffset:0 size:copy_bytes];
                        [blit endEncoding];
                        [cb commit];
                        [cb waitUntilCompleted];
                        MetalTensor out{dst, out_shape, in.expected_type == ov::element::dynamic ? in.buf.type : in.expected_type};
                        tensors.bind_output_device(0, out);
                        layout_ok = true;
                    }
#endif
                }
            }

            if (!reshape_ok && !layout_ok) {
                // Final safeguard: copy first input to first output on device using host if needed.
                MetalBuffer src_buf;
                ov::Shape src_shape;
                ov::element::Type src_type = ov::element::f32;
                if (auto dev = get_input_dev(0)) {
                    src_buf = dev->buf;
                    src_shape = dev->shape;
                    src_type = dev->expected_type == ov::element::dynamic ? dev->buf.type : dev->expected_type;
                } else if (tensors.has_input_host(0)) {
                    const auto& h = tensors.get_input_host(0);
                    src_shape = h.get_shape();
                    src_type = h.get_element_type();
                    src_buf = mgr.allocate(h.get_byte_size(), src_type, /*persistent=*/false, /*storageModePrivate=*/true);
                    mgr.upload(src_buf, h.data(), h.get_byte_size());
                } else {
                    METAL_LOG_ERROR("mlir", "[METAL MLIR] run_device: pipelines missing and no input available");
                    return false;
                }
                ov::Shape out_shape = src_shape;
                size_t bytes = ov::shape_size(out_shape) * src_type.size();
                MetalBuffer dst = mgr.allocate(bytes, src_type, /*persistent=*/false, /*storageModePrivate=*/false);
#ifdef __OBJC__
                id<MTLBuffer> src = static_cast<id<MTLBuffer>>(src_buf.buffer);
                id<MTLBuffer> dstb = static_cast<id<MTLBuffer>>(dst.buffer);
                if (src && dstb) {
                    id<MTLCommandBuffer> cb = [m_queue commandBuffer];
                    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                    size_t copy_bytes = std::min(src_buf.size, dst.size);
                    [blit copyFromBuffer:src sourceOffset:0 toBuffer:dstb destinationOffset:0 size:copy_bytes];
                    [blit endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                }
#endif
                MetalTensor out{dst, out_shape, src_type};
                tensors.bind_output_device(0, out);
                return true;
            }
            return true;
        }

        // Gather host inputs (already mirrored when bind_input was called).
        std::vector<ov::Tensor> host_inputs;
        size_t idx = 0;
        while (tensors.has_input_host(idx)) {
            host_inputs.push_back(tensors.get_input_host(idx));
            ++idx;
        }
        if (host_inputs.empty()) {
            METAL_LOG_ERROR("mlir", "[METAL MLIR] run_device: no host inputs bound");
            return false;
        }

        std::vector<ov::Tensor> host_outputs(m_model->outputs().size());
        try {
            run(host_inputs, host_outputs);  // executes GPU pipelines; throws on failure
        } catch (const std::exception& e) {
            METAL_LOG_ERROR("mlir", std::string("[METAL MLIR] run_device: run() failed: ") + e.what());
            return false;
        }

        // Outputs must be device-resident; CPU uploads are запрещены.
        for (size_t o = 0; o < host_outputs.size(); ++o) {
            if (!tensors.has_output_device(o)) {
                METAL_LOG_ERROR("mlir", "[METAL MLIR] run_device: output not bound to device");
                OPENVINO_THROW("METAL: output device buffer missing (CPU copy disabled)");
            }
        }
        return true;
    }
#else
    (void)tensors;
    (void)mgr;
    return false;
#endif
}
